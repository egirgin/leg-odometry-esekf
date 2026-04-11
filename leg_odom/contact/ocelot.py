"""
Ocelot contact detector: FSM on foot load + optional 1D sklearn GMM (isolated from GMM+HMM) + GLRT.

Runs **inside the EKF loop** only: GLRT needs nominal body velocity and attitude
(:attr:`ContactDetectorStepInput.v_body_world`, :attr:`ContactDetectorStepInput.R_wb`).

**Naming:** detector ``ocelot`` is unrelated to ``dataset.kind: ocelot`` (lowstate layout).
"""

from __future__ import annotations

import warnings
from collections import deque
from enum import Enum
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.io.columns import FOOT_FORCE_COLS

# --- Hard-coded FSM timing (legacy FSM_CONFIG) ---------------------------------------------------
DEBOUNCE_ON = 2
DEBOUNCE_OFF = 2
SLIP_SPEED_OFF = 0.1

# GLRT: sliding window length in seconds (matches legacy STATISTICAL_GLRT_CONFIG_HIGH_RATE wlen_s).
WLEN_S = 0.01
MIN_WIN_GLRT = 3
DEFAULT_NOISE_STD_DEV = 0.45
P_STANCE_FLOOR = 1e-6

# Minimum samples to fit 2-component GMM on force.
_MIN_GMM_SAMPLES = 200


class ContactState(Enum):
    SWING = 0
    TOUCHDOWN = 1
    STANCE = 2
    LIFTOFF = 3


class ForceFSM:
    """Debounced force FSM (one leg). Thresholds are scalars for this leg."""

    def __init__(
        self,
        *,
        force_on: float,
        force_off: float,
        debounce_on: int = DEBOUNCE_ON,
        debounce_off: int = DEBOUNCE_OFF,
        slip_speed_off: float = SLIP_SPEED_OFF,
    ) -> None:
        self._fo = float(force_on)
        self._fx = float(force_off)
        self._deb_on = int(debounce_on)
        self._deb_off = int(debounce_off)
        self._slip = float(slip_speed_off)
        self.state = ContactState.SWING
        self.td_counter = 0
        self.lo_counter = 0

    def update(self, f_mag: float, foot_speed: float) -> ContactState:
        if self.state == ContactState.SWING:
            if f_mag > self._fo:
                self.state = ContactState.TOUCHDOWN
                self.td_counter = 1
        elif self.state == ContactState.TOUCHDOWN:
            if f_mag > self._fo:
                self.td_counter += 1
                if self.td_counter >= self._deb_on:
                    self.state = ContactState.STANCE
            else:
                self.state = ContactState.SWING
                self.td_counter = 0
        elif self.state == ContactState.STANCE:
            if f_mag < self._fx or (foot_speed > self._slip):
                self.state = ContactState.LIFTOFF
                self.lo_counter = 1
        elif self.state == ContactState.LIFTOFF:
            if f_mag < self._fx:
                self.lo_counter += 1
                if self.lo_counter >= self._deb_off:
                    self.state = ContactState.SWING
            else:
                self.state = ContactState.STANCE
                self.lo_counter = 0
        return self.state


def _fit_gmm_two_component_1d(data_1d: npt.NDArray[np.float64], *, random_state: int = 0):
    """Return (GaussianMixture, stance_row_index) or (None, None) if fit fails."""
    x = np.asarray(data_1d, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < _MIN_GMM_SAMPLES:
        return None, None
    p5, p95 = np.percentile(x, [5, 95])
    xf = x[(x > p5) & (x < p95)]
    if xf.size < _MIN_GMM_SAMPLES:
        xf = x
    xm = xf.reshape(-1, 1)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                gmm = GaussianMixture(
                    n_components=2, random_state=random_state, covariance_type="full", max_iter=200
                ).fit(xm)
            except (ValueError, np.linalg.LinAlgError):
                return None, None
    means = gmm.means_.flatten()
    if abs(float(means[0] - means[1])) < 10.0:
        return None, None
    stance_idx = int(np.argmax(means))
    return gmm, stance_idx


def _p_gmm_stance(gmm: GaussianMixture, stance_idx: int, f: float) -> float:
    """Responsibility of stance (higher-mean) component at scalar force ``f``."""
    try:
        proba = gmm.predict_proba(np.array([[float(f)]], dtype=np.float64))
        return float(np.clip(proba[0, stance_idx], 0.0, 1.0))
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def _p_linear(f: float, force_on: float, force_max: float) -> float:
    d = max(float(force_max) - float(force_on), 1e-6)
    return float(np.clip((float(f) - float(force_on)) / d, 0.0, 1.0))


class GLRTLeg:
    """Single-leg GLRT on world-frame foot velocity (legacy GLRTStatistical, one buffer)."""

    def __init__(
        self,
        *,
        buf_len: int,
        noise_std_dev: float,
        min_win: int = MIN_WIN_GLRT,
    ) -> None:
        self.buf_len = max(int(buf_len), 2)
        self.min_win = int(min_win)
        v = float(noise_std_dev) ** 2
        self._sigma0_inv = np.identity(3, dtype=np.float64) / (v + 1e-9)
        self._chi2 = float(st.chi2.ppf(0.95, df=3))
        self._buf: list[npt.NDArray[np.float64]] = []

    def reset(self) -> None:
        self._buf.clear()

    def update(
        self,
        v_foot_world: npt.NDArray[np.float64],
    ) -> tuple[bool, float, float]:
        v = np.asarray(v_foot_world, dtype=np.float64).reshape(3)
        self._buf.append(v.copy())
        if len(self._buf) > self.buf_len:
            self._buf.pop(0)
        speed = float(np.linalg.norm(v))
        if len(self._buf) < max(self.min_win, 2):
            return False, 0.0, speed
        s = np.vstack(self._buf)
        mu_hat = np.mean(s, axis=0)
        t_glrt = float(mu_hat.T @ self._sigma0_inv @ mu_hat)
        n = len(self._buf)
        t_scaled = t_glrt * n
        ok = bool(t_scaled < self._chi2)
        q = float(np.clip(1.0 - (t_scaled / self._chi2), 0.0, 1.0))
        return ok, q, speed


class OcelotContactDetector(BaseContactDetector):
    """
    Per-leg Ocelot detector. User supplies one ``force_on`` / ``force_off`` (shared across legs);
    ``force_max`` comes from data (offline: full-sequence max; online: max over last ``window_size``
    samples).
    """

    def __init__(
        self,
        *,
        leg_id: int,
        use_fsm: bool,
        use_glrt: bool,
        fsm_gmm_mode: str,
        force_on: float,
        force_off: float,
        force_max_init: float,
        window_size: int,
        fit_interval: int,
        noise_std_dev: float,
        glrt_buf_len: int,
        gmm_model: GaussianMixture | None,
        gmm_stance_idx: int | None,
        random_state: int = 0,
    ) -> None:
        self._leg = int(leg_id)
        self._use_fsm = bool(use_fsm)
        self._use_glrt = bool(use_glrt)
        self._mode = str(fsm_gmm_mode).lower().strip()
        if self._mode not in ("offline", "online"):
            raise ValueError("fsm_gmm_mode must be offline or online")
        self._fo = float(force_on)
        self._fx = float(force_off)
        self._wsize = max(1, int(window_size))
        self._fit_iv = max(1, int(fit_interval))
        self._noise = float(noise_std_dev)
        self._rs = int(random_state)

        self._fmax = float(force_max_init)
        self._fsm = ForceFSM(force_on=self._fo, force_off=self._fx) if self._use_fsm else None
        self._glrt = GLRTLeg(buf_len=glrt_buf_len, noise_std_dev=self._noise) if self._use_glrt else None

        self._gmm = gmm_model
        self._stance_idx = gmm_stance_idx
        self._force_win: deque[float] = deque(maxlen=self._wsize)
        self._step = 0

    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def history_length(self) -> int:
        return 1

    def reset(self) -> None:
        self._step = 0
        self._force_win.clear()
        if self._fsm is not None:
            self._fsm = ForceFSM(force_on=self._fo, force_off=self._fx)
        if self._glrt is not None:
            self._glrt.reset()

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        f = float(step.grf_n)
        p_fsm = 1.0
        stance_fsm = True
        p_gmm_part = 1.0

        if self._use_fsm:
            if self._mode == "online":
                self._force_win.append(f)
                self._fmax = max(self._force_win) if self._force_win else max(f, self._fo)
                self._step += 1
                if self._step % self._fit_iv == 0 and len(self._force_win) >= 50:
                    arr = np.array(self._force_win, dtype=np.float64)
                    gm, si = _fit_gmm_two_component_1d(arr, random_state=self._rs + self._leg)
                    if gm is not None:
                        self._gmm, self._stance_idx = gm, si
            # offline: force_max fixed at init from full-sequence max

            if self._gmm is not None and self._stance_idx is not None:
                p_gmm_part = _p_gmm_stance(self._gmm, self._stance_idx, f)

            p_lin = _p_linear(f, self._fo, self._fmax)
            p_fsm = max(p_lin * p_gmm_part, P_STANCE_FLOOR)

            assert self._fsm is not None
            foot_speed = 0.0
            st = self._fsm.update(f, foot_speed)
            stance_fsm = st == ContactState.STANCE

        p_glrt = 1.0
        stance_glrt = True
        if self._use_glrt:
            if step.v_body_world is None or step.R_wb is None:
                raise ValueError(
                    "Ocelot GLRT requires ContactDetectorStepInput.v_body_world and R_wb (EKF loop)."
                )
            vb = np.asarray(step.v_foot_body, dtype=np.float64).reshape(3)
            r = np.asarray(step.R_wb, dtype=np.float64).reshape(3, 3)
            vw = np.asarray(step.v_body_world, dtype=np.float64).reshape(3)
            v_fw = vw + r @ vb
            assert self._glrt is not None
            stance_glrt, p_glrt, _sp = self._glrt.update(v_fw)

        if self._use_fsm and self._use_glrt:
            stance = bool(stance_fsm and stance_glrt)
            p = max(p_fsm * p_glrt, P_STANCE_FLOOR)
        elif self._use_fsm:
            stance = bool(stance_fsm)
            p = p_fsm
        elif self._use_glrt:
            stance = bool(stance_glrt)
            p = max(p_glrt, P_STANCE_FLOOR)
        else:
            stance = False
            p = P_STANCE_FLOOR

        return ContactEstimate(stance=stance, p_stance=float(p))


def _offline_force_max_per_leg(frames: pd.DataFrame) -> npt.NDArray[np.float64]:
    out = np.zeros(4, dtype=np.float64)
    for i in range(4):
        col = FOOT_FORCE_COLS[i]
        if col not in frames.columns:
            out[i] = 1.0
            continue
        s = pd.to_numeric(frames[col], errors="coerce")
        mx = float(np.nanmax(s.to_numpy())) if len(s) else 1.0
        out[i] = max(mx, 1.0)
    return out


def _offline_gmm_per_leg(frames: pd.DataFrame, *, random_state: int) -> tuple[list[GaussianMixture | None], list[int | None]]:
    gmms: list[GaussianMixture | None] = []
    stances: list[int | None] = []
    for i in range(4):
        col = FOOT_FORCE_COLS[i]
        if col not in frames.columns:
            gmms.append(None)
            stances.append(None)
            continue
        s = pd.to_numeric(frames[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        g, si = _fit_gmm_two_component_1d(s, random_state=random_state + i)
        gmms.append(g)
        stances.append(si)
    return gmms, stances


def build_ocelot_detectors_from_cfg(
    cfg: Mapping[str, Any],
    *,
    recording: LegOdometrySequence | None = None,
    kin_model: Any = None,
) -> list[OcelotContactDetector]:
    """
    Build four per-leg :class:`OcelotContactDetector` instances.

    ``recording`` is required (offline GMM / force_max and rate). ``kin_model`` is unused.
    """
    del kin_model
    if recording is None:
        raise ValueError("contact.detector ocelot requires recording in build_contact_stack")

    c = cfg.get("contact")
    if not isinstance(c, Mapping):
        raise ValueError("contact must be a mapping")
    o = c.get("ocelot")
    if not isinstance(o, Mapping):
        raise ValueError("contact.ocelot must be a mapping when contact.detector is ocelot")

    use_fsm = bool(o.get("use_fsm", True))
    use_glrt = bool(o.get("use_glrt", True))
    if not use_fsm and not use_glrt:
        raise ValueError("contact.ocelot: at least one of use_fsm or use_glrt must be true")

    mode = str(o.get("fsm_gmm_mode", "offline")).lower().strip()
    fo = o.get("force_on", 25.0)
    fx = o.get("force_off", 15.0)
    if isinstance(fo, (list, tuple)) or isinstance(fx, (list, tuple)):
        raise ValueError("contact.ocelot.force_on and force_off must be single numbers (same for all legs)")
    force_on = float(fo)
    force_off = float(fx)
    window_size = int(o.get("window_size", 500))
    fit_interval = int(o.get("fit_interval", 250))
    noise_std_dev = float(o.get("noise_std_dev", DEFAULT_NOISE_STD_DEV))
    rh = o.get("rate_hz", None)
    rate_hz = float(recording.median_rate_hz) if rh is None or rh == "" else float(rh)
    glrt_buf_len = max(int(round(WLEN_S * rate_hz)), 2)
    random_state = int(o.get("random_state", 42))

    fmax_off = _offline_force_max_per_leg(recording.frames)
    gmms: list[GaussianMixture | None]
    stix: list[int | None]
    if use_fsm and mode == "offline":
        gmms, stix = _offline_gmm_per_leg(recording.frames, random_state=random_state)
    else:
        gmms = [None] * 4
        stix = [None] * 4

    dets: list[OcelotContactDetector] = []
    for leg in range(4):
        if use_fsm and mode == "offline":
            fmax0 = float(fmax_off[leg])
        elif use_fsm and mode == "online":
            fmax0 = max(force_on, 1.0)
        else:
            fmax0 = 1.0
        dets.append(
            OcelotContactDetector(
                leg_id=leg,
                use_fsm=use_fsm,
                use_glrt=use_glrt,
                fsm_gmm_mode=mode,
                force_on=force_on,
                force_off=force_off,
                force_max_init=fmax0,
                window_size=window_size,
                fit_interval=fit_interval,
                noise_std_dev=noise_std_dev,
                glrt_buf_len=glrt_buf_len,
                gmm_model=gmms[leg] if use_fsm else None,
                gmm_stance_idx=stix[leg] if use_fsm else None,
                random_state=random_state,
            )
        )
    return dets
