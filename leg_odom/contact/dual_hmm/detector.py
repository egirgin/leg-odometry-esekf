"""
Dual HMM contact detector: fused 1D GRF load HMM + kinematic GMM/HMM, optional energy on the kin branch.

See :mod:`leg_odom.contact.gmm_hmm_core` for shared GMM ordering and pretrained I/O.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate
from leg_odom.contact.dual_hmm.hmm_kin_energy import KinGaussianHmmEnergy
from leg_odom.contact.dual_hmm.paths import resolve_pretrained_dual_hmm_path
from leg_odom.contact.dual_hmm.spec import parse_dual_kinematic_feature_fields
from leg_odom.contact.gmm_hmm_core.fitting import (
    fit_gmm_ordered,
    fit_offline_dual_per_leg,
    flat_ordering_component_index,
    load_pretrained_dual_hmm_npz,
)
from leg_odom.contact.gmm_hmm_core.hmm_gaussian import TwoStateGaussianHMM
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.features import (
    DEFAULT_INSTANT_FEATURE_FIELDS,
    flatten_history_window,
    instant_vector_from_step,
    parse_instant_feature_fields,
)
from leg_odom.kinematics.base import BaseKinematics

_STANCE_P_FUSED_MIN = 0.5


class DualHmmContactDetector(BaseContactDetector):
    """
    Per-foot dual HMM: **always** 1D GRF load HMM + kinematic GMM/HMM, fused into one stance belief.

    For GRF-only or kin-only contact, use ``contact.detector: gmm`` instead.

    ``feature_dim`` = ``1 + D_kin`` (scalar load + flattened kin window). Offline mode forces kin
    ``N=1`` (YAML ``history_length`` ignored for fit and buffer).
    """

    def __init__(
        self,
        *,
        kin_feature_fields: tuple[str, ...],
        history_length: int = 1,
        trans_stay: float = 0.99,
        mode: Literal["offline", "online"] = "offline",
        use_energy: bool = False,
        kinematics_z_stationary_std_skip: float = 0.005,
        fit_interval: int = 250,
        window_size: int = 500,
        degeneracy_max_weight: float = 0.80,
        random_state: int = 42,
        pretrained_path: str | Path | None = None,
        load_initial_means: npt.NDArray[np.floating] | None = None,
        load_initial_covariances: npt.NDArray[np.floating] | None = None,
        kin_initial_means: npt.NDArray[np.floating] | None = None,
        kin_initial_covariances: npt.NDArray[np.floating] | None = None,
        energy_percentile: float = 98.0,
        energy_norm_window: int = 2000,
        energy_spike_min: float = 0.05,
        initial_gamma: float = 1.0,
        learning_rate_gamma: float = 0.1,
        gamma_min: float = 1.0,
        gamma_max: float = 15.0,
        gamma_learning_multiplier_high: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self._use_energy = bool(use_energy)
        self.last_energy_normalized: float = 0.0
        self._z_skip = float(kinematics_z_stationary_std_skip)
        self._mode: Literal["offline", "online"] = mode
        n_arg = int(history_length)
        if n_arg < 1:
            raise ValueError("history_length must be >= 1")
        self._N = 1 if mode == "offline" else n_arg
        self._kin_spec = parse_dual_kinematic_feature_fields(kin_feature_fields)
        self._D_kin = self._N * self._kin_spec.instant_dim
        self._spec_grf = parse_instant_feature_fields(("grf_n",))
        self._trans_stay = float(trans_stay)
        self._fit_interval = int(fit_interval)
        self._window_size = int(window_size)
        self._degen_w = float(degeneracy_max_weight)
        self._rng = int(random_state)
        self._verbose = bool(verbose)

        self._gamma_lr = float(learning_rate_gamma)
        self._gamma_min = float(gamma_min)
        self._gamma_max = float(gamma_max)
        self._gamma_mult_hi = float(gamma_learning_multiplier_high)
        self._energy_pct = float(energy_percentile)
        self._energy_win = int(energy_norm_window)
        self._energy_spike_min = float(energy_spike_min)

        self._instant_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=self._N)
        self._grf_window: deque[float] = deque(maxlen=self._window_size)
        self._kin_flat_window: deque[npt.NDArray[np.float64]] = deque(maxlen=self._window_size)
        self._clock = 0

        self._load_hmm = TwoStateGaussianHMM(self._trans_stay)
        self._kin_hmm = KinGaussianHmmEnergy(self._trans_stay, initial_gamma=float(initial_gamma))

        # Fallback / last-good emissions (load: 2x1, kin: 2xD_kin)
        self._load_fb_m: npt.NDArray[np.float64] | None = None
        self._load_fb_c: npt.NDArray[np.float64] | None = None
        self._load_lg_m: npt.NDArray[np.float64] | None = None
        self._load_lg_c: npt.NDArray[np.float64] | None = None
        self._kin_fb_m: npt.NDArray[np.float64] | None = None
        self._kin_fb_c: npt.NDArray[np.float64] | None = None
        self._kin_lg_m: npt.NDArray[np.float64] | None = None
        self._kin_lg_c: npt.NDArray[np.float64] | None = None

        if mode == "offline":
            if load_initial_means is None or load_initial_covariances is None:
                raise ValueError("offline dual_hmm requires load_initial_means/covariances")
            if kin_initial_means is None or kin_initial_covariances is None:
                raise ValueError("offline dual_hmm requires kin_initial_means/covariances")
            lm = np.asarray(load_initial_means, dtype=np.float64)
            lc = np.asarray(load_initial_covariances, dtype=np.float64)
            if lm.shape != (2, 1) or lc.shape != (2, 1, 1):
                raise ValueError(f"load GMM shape mismatch: means {lm.shape}, covs {lc.shape}")
            self._apply_load_emissions(lm, lc)
            self._load_fb_m, self._load_fb_c = lm.copy(), lc.copy()
            km = np.asarray(kin_initial_means, dtype=np.float64)
            kc = np.asarray(kin_initial_covariances, dtype=np.float64)
            if km.shape != (2, self._D_kin) or kc.shape != (2, self._D_kin, self._D_kin):
                raise ValueError(f"kin GMM shape mismatch: means {km.shape}, covs {kc.shape}")
            self._apply_kin_emissions(km, kc)
            self._kin_fb_m, self._kin_fb_c = km.copy(), kc.copy()
        else:
            if not pretrained_path:
                raise ValueError("online dual_hmm requires pretrained_path (dual .npz)")
            p = resolve_pretrained_dual_hmm_path(pretrained_path)
            lm, lc, km, kc = load_pretrained_dual_hmm_npz(
                p,
                expected_kin_feature_dim=self._D_kin,
                expected_kin_history_length=self._N,
                expected_kin_instant_dim=self._kin_spec.instant_dim,
            )
            self._apply_load_emissions(lm, lc)
            self._load_fb_m, self._load_fb_c = lm.copy(), lc.copy()
            self._apply_kin_emissions(km, kc)
            self._kin_fb_m, self._kin_fb_c = km.copy(), kc.copy()

        self._prev_v: npt.NDArray[np.float64] | None = None
        self._energy_hist: deque[float] = deque(maxlen=max(16, self._energy_win))
        self._prev_state_load: int | None = None
        self._prev_state_kin: int | None = None

    @property
    def feature_dim(self) -> int:
        return int(1 + self._D_kin)

    @property
    def history_length(self) -> int:
        return int(self._N)

    def _apply_load_emissions(self, m: npt.NDArray[np.float64], c: npt.NDArray[np.float64]) -> None:
        m = np.asarray(m, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        self._load_hmm.update_dists(
            mu_swing=m[1].reshape(1),
            cov_swing=c[1].reshape(1, 1),
            mu_stance=m[0].reshape(1),
            cov_stance=c[0].reshape(1, 1),
        )

    def _apply_kin_emissions(self, m: npt.NDArray[np.float64], c: npt.NDArray[np.float64]) -> None:
        m = np.asarray(m, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        self._kin_hmm.update_dists(
            mu_swing=m[1],
            cov_swing=c[1],
            mu_stance=m[0],
            cov_stance=c[0],
        )

    def _revert_load(self) -> None:
        if self._load_lg_m is not None and self._load_lg_c is not None:
            self._apply_load_emissions(self._load_lg_m, self._load_lg_c)
        elif self._load_fb_m is not None and self._load_fb_c is not None:
            self._apply_load_emissions(self._load_fb_m, self._load_fb_c)

    def _revert_kin(self) -> None:
        if self._kin_lg_m is not None and self._kin_lg_c is not None:
            self._apply_kin_emissions(self._kin_lg_m, self._kin_lg_c)
        elif self._kin_fb_m is not None and self._kin_fb_c is not None:
            self._apply_kin_emissions(self._kin_fb_m, self._kin_fb_c)

    def _maybe_refit_online(self) -> None:
        if self._mode != "online":
            return
        if self._fit_interval <= 0:
            return
        if self._clock == 0 or (self._clock % self._fit_interval) != 0:
            return
        need_grf = len(self._grf_window) < self._window_size
        need_kin = len(self._kin_flat_window) < self._window_size
        if need_grf or need_kin:
            return

        Xl = np.array(self._grf_window, dtype=np.float64).reshape(-1, 1)
        mo_l, co_l, bad_l = fit_gmm_ordered(Xl, self._spec_grf, 1, random_state=self._rng)
        if not bad_l and mo_l.shape == (2, 1):
            g_tmp = GaussianMixture(n_components=2, covariance_type="full", random_state=self._rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                try:
                    g_tmp.fit(Xl)
                except (ValueError, np.linalg.LinAlgError):
                    self._revert_load()
                    return
            if float(np.max(g_tmp.weights_)) >= self._degen_w:
                self._revert_load()
            else:
                self._apply_load_emissions(mo_l, co_l)
                self._load_lg_m, self._load_lg_c = mo_l.copy(), co_l.copy()
        else:
            self._revert_load()

        Xk = np.stack([np.asarray(v, dtype=np.float64) for v in self._kin_flat_window], axis=0)
        j = flat_ordering_component_index(self._kin_spec, self._N)
        if float(np.std(Xk[:, j])) < self._z_skip:
            if self._verbose:
                print("[dual_hmm] kin window mostly stationary; skip refit")
            return
        mo_k, co_k, bad_k = fit_gmm_ordered(Xk, self._kin_spec, self._N, random_state=self._rng)
        if not bad_k and mo_k.shape[1] == self._D_kin:
            g_tmp = GaussianMixture(n_components=2, covariance_type="full", random_state=self._rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                try:
                    g_tmp.fit(Xk)
                except (ValueError, np.linalg.LinAlgError):
                    self._revert_kin()
                    return
            if float(np.max(g_tmp.weights_)) >= self._degen_w:
                self._revert_kin()
            else:
                self._apply_kin_emissions(mo_k, co_k)
                self._kin_lg_m, self._kin_lg_c = mo_k.copy(), co_k.copy()
        else:
            self._revert_kin()

    def _energy_spike(self, v_foot: npt.NDArray[np.float64]) -> float:
        v = np.asarray(v_foot, dtype=np.float64).reshape(3)
        if self._prev_v is None:
            self._prev_v = v.copy()
            return 0.0
        dv = v - self._prev_v
        self._prev_v = v.copy()
        raw = float(np.dot(dv, dv))
        self._energy_hist.append(raw)
        if len(self._energy_hist) < 8:
            return float(np.clip(raw / max(raw, 1e-12), 0.0, 1.0))
        pct = float(np.percentile(np.asarray(self._energy_hist, dtype=np.float64), self._energy_pct))
        denom = max(pct, 1e-12)
        return float(np.clip(raw / denom, 0.0, 1.0))

    def _update_gamma_online(self, pseudo_y: float, a_ij_t: float, energy_spike: float) -> None:
        if energy_spike <= self._energy_spike_min:
            return
        alpha = 1.0 - self._kin_hmm.base_switch
        a_clamped = float(np.clip(a_ij_t, 1e-5, 1.0 - 1e-5))
        mult = self._gamma_mult_hi if pseudo_y >= 0.5 else 1.0
        dL_da = (a_clamped - pseudo_y) / (a_clamped * (1.0 - a_clamped))
        da_dGamma = alpha * energy_spike * np.exp(-self._kin_hmm.gamma * energy_spike)
        grad = dL_da * da_dGamma
        self._kin_hmm.gamma = float(
            np.clip(
                self._kin_hmm.gamma - (self._gamma_lr * mult) * grad,
                self._gamma_min,
                self._gamma_max,
            )
        )

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        grf = float(step.grf_n)
        self._grf_window.append(grf)

        inst = instant_vector_from_step(step, self._kin_spec)
        self._instant_buf.append(inst)
        nbuf = len(self._instant_buf)
        flat_kin: npt.NDArray[np.float64] | None = None
        if nbuf < self._N:
            self.last_energy_normalized = 0.0
            self._clock += 1
            self._maybe_refit_online()
            return ContactEstimate(stance=True, p_stance=1.0)
        win = np.stack([np.asarray(x, dtype=np.float64) for x in self._instant_buf], axis=0)
        flat_kin = flatten_history_window(win)
        self._kin_flat_window.append(flat_kin.copy())

        self._clock += 1
        self._maybe_refit_online()

        self.last_energy_normalized = 0.0
        energy_val = 0.0
        if self._use_energy:
            energy_val = self._energy_spike(step.v_foot_body)
            self.last_energy_normalized = float(energy_val)

        xv = np.array([grf], dtype=np.float64)
        p_load, _stance_l = self._load_hmm.update(xv)
        state_load = 1 if p_load > 0.5 else 0

        assert flat_kin is not None
        p_kin, _stance_k, current_switch_prob = self._kin_hmm.update(
            flat_kin,
            energy_spike=energy_val,
            use_energy=self._use_energy,
        )
        state_kin = 1 if p_kin > 0.5 else 0

        if self._use_energy:
            if self._prev_state_load is not None and self._prev_state_kin is not None:
                ls = state_load != self._prev_state_load
                ks = state_kin != self._prev_state_kin
                if ls and ks:
                    self._update_gamma_online(1.0, current_switch_prob, energy_val)
                elif not ls and not ks:
                    self._update_gamma_online(0.0, current_switch_prob, energy_val)
        self._prev_state_load = state_load
        self._prev_state_kin = state_kin

        fused_unnorm_sw = (1.0 - p_load) * (1.0 - p_kin)
        fused_unnorm_st = p_load * p_kin
        s = fused_unnorm_sw + fused_unnorm_st + 1e-12
        p_fused = fused_unnorm_st / s

        stance = p_fused >= _STANCE_P_FUSED_MIN
        return ContactEstimate(stance=stance, p_stance=float(p_fused))

    def reset(self) -> None:
        self._instant_buf.clear()
        self._grf_window.clear()
        self._kin_flat_window.clear()
        self._clock = 0
        self._prev_v = None
        self._energy_hist.clear()
        self._prev_state_load = None
        self._prev_state_kin = None
        self.last_energy_normalized = 0.0
        self._load_hmm.reset_belief()
        self._kin_hmm.reset_belief()
        if self._load_fb_m is not None and self._load_fb_c is not None:
            self._apply_load_emissions(self._load_fb_m, self._load_fb_c)
        if self._kin_fb_m is not None and self._kin_fb_c is not None:
            self._apply_kin_emissions(self._kin_fb_m, self._kin_fb_c)


def _dual_cfg_from_mapping(dm: Mapping[str, Any], *, prefix: str = "contact.dual_hmm") -> dict[str, Any]:
    """Normalize YAML block into DualHmmContactDetector kwargs (subset)."""
    if "grf_n" in str(dm.get("feature_fields", [])):
        raise ValueError(f"{prefix}.feature_fields must not include grf_n")

    def f(name: str, default: Any) -> Any:
        return dm[name] if name in dm else default

    fields_raw = dm.get("feature_fields", list(DEFAULT_INSTANT_FEATURE_FIELDS))
    if not isinstance(fields_raw, (list, tuple)):
        raise TypeError(f"{prefix}.feature_fields must be a list")
    kin_fields = tuple(str(x) for x in fields_raw)

    kw: dict[str, Any] = {
        "kin_feature_fields": kin_fields,
        "history_length": int(f("history_length", 1)),
        "trans_stay": float(f("trans_stay", 0.99)),
        "mode": str(f("mode", "offline")).lower(),
        "use_energy": bool(f("use_energy", False)),
        "kinematics_z_stationary_std_skip": float(f("kinematics_z_stationary_std_skip", 0.005)),
        "fit_interval": int(f("fit_interval", 250)),
        "window_size": int(f("window_size", 500)),
        "degeneracy_max_weight": float(f("degeneracy_max_weight", 0.80)),
        "random_state": int(f("random_state", 42)),
        "pretrained_path": dm.get("pretrained_path"),
        "energy_percentile": float(f("energy_percentile", 98.0)),
        "energy_norm_window": int(f("energy_norm_window", 2000)),
        "energy_spike_min": float(f("energy_spike_min", 0.05)),
        "initial_gamma": float(f("initial_gamma", 1.0)),
        "learning_rate_gamma": float(f("learning_rate_gamma", 0.1)),
        "gamma_min": float(f("gamma_min", 1.0)),
        "gamma_max": float(f("gamma_max", 15.0)),
        "gamma_learning_multiplier_high": float(f("gamma_learning_multiplier_high", 5.0)),
        "verbose": bool(f("verbose", False)),
    }
    return kw


def build_dual_hmm_detectors_from_cfg(
    cfg: Mapping[str, Any],
    *,
    recording: LegOdometrySequence | None = None,
    kin_model: BaseKinematics | None = None,
    workspace_root: Path | None = None,
) -> list[DualHmmContactDetector]:
    """Build one :class:`DualHmmContactDetector` per leg from ``contact.dual_hmm``."""
    _ = workspace_root  # reserved for future path resolution parity with neural
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        raise ValueError("contact config missing")
    dm = block.get("dual_hmm")
    if not isinstance(dm, Mapping):
        raise ValueError("contact.detector is dual_hmm but contact.dual_hmm mapping is missing")

    kw = _dual_cfg_from_mapping(dm)
    mode = kw.pop("mode")
    if mode not in ("offline", "online"):
        raise ValueError(f"contact.dual_hmm.mode must be offline|online, got {mode!r}")
    kw["mode"] = mode

    pretrained_path = kw.pop("pretrained_path")

    per_leg_dual: list[
        tuple[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ]
    ] | None = None

    if mode == "offline":
        if recording is None or kin_model is None:
            raise ValueError("offline dual_hmm requires recording and kin_model in build_contact_stack")
        kf = kw["kin_feature_fields"]
        rs0 = int(kw["random_state"])
        hl_off = 1
        per_leg_dual = fit_offline_dual_per_leg(
            recording,
            kin_model,
            kin_feature_fields=kf,
            history_length=hl_off,
            random_state=rs0,
        )

    n_legs = int(kin_model.n_legs) if kin_model is not None else 4
    detectors: list[DualHmmContactDetector] = []
    for leg in range(n_legs):
        leg_kw = dict(kw)
        if mode == "offline":
            leg_kw["history_length"] = 1
            assert per_leg_dual is not None
            (lm, lc), (km, kc) = per_leg_dual[leg]
            leg_kw["load_initial_means"], leg_kw["load_initial_covariances"] = lm, lc
            leg_kw["kin_initial_means"], leg_kw["kin_initial_covariances"] = km, kc
        else:
            if not pretrained_path:
                raise ValueError("contact.dual_hmm.pretrained_path is required for online mode")
            leg_kw["pretrained_path"] = str(pretrained_path)
        detectors.append(DualHmmContactDetector(**leg_kw))
    return detectors
