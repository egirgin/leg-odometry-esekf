"""
Per-foot contact detector: 2-GMM emissions + 2-state Gaussian HMM filter.

**Modes**

- ``offline``: mixture parameters come from a **whole-sequence** fit (no .npz). Emissions are
  always **instant** Gaussians in ``ℝ^d`` (YAML ``history_length`` is **ignored**; internally
  ``N=1``). Use ``online`` with a pretrained ``.npz`` if you need ``history_length > 1``.
- ``online``: start from a **pretrained** ``.npz`` (initial emissions + fault-tolerant fallback).
  A sliding buffer periodically refits the GMM; degenerate fits revert to last good or pretrained.

**ZUPT**

The EKF forms ``R_foot`` from ``p_stance`` in :mod:`leg_odom.filters.zupt_measurement`. ZUPT is
applied only when ``stance`` is True.

**Warm-up**

- ``online``: until ``history_length`` instants are collected, returns ``stance=True``,
  ``p_stance=1.0``, and does **not** advance the HMM.
- ``offline``: no warm-up branch; no sliding refit; instant-only HMM (see above).
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
from leg_odom.features import (
    DEFAULT_INSTANT_FEATURE_FIELDS,
    InstantFeatureSpec,
    flatten_history_window,
    instant_vector_from_step,
    parse_instant_feature_fields,
)
from leg_odom.contact.gmm_hmm_core.fitting import fit_gmm_ordered, fit_offline_per_leg, load_pretrained_gmm_npz
from leg_odom.contact.gmm_hmm_core.hmm_gaussian import TwoStateGaussianHMM
from leg_odom.contact.gmm_hmm.paths import resolve_pretrained_gmm_path
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.kinematics.base import BaseKinematics


class GmmHmmContactDetector(BaseContactDetector):
    """``feature_dim == N * d_instant``, ``history_length == N``; offline always uses ``N == 1``."""

    def __init__(
        self,
        *,
        feature_fields: tuple[str, ...] | None = None,
        history_length: int = 1,
        trans_stay: float = 0.99,
        mode: Literal["offline", "online"] = "offline",
        pretrained_path: str | Path | None = None,
        initial_means: npt.NDArray[np.floating] | None = None,
        initial_covariances: npt.NDArray[np.floating] | None = None,
        fit_interval: int = 250,
        window_size: int = 500,
        degeneracy_max_weight: float = 0.80,
        random_state: int = 42,
    ) -> None:
        self._spec = parse_instant_feature_fields(feature_fields or DEFAULT_INSTANT_FEATURE_FIELDS)
        n_arg = int(history_length)
        if n_arg < 1:
            raise ValueError("history_length must be >= 1")
        self._N = 1 if mode == "offline" else n_arg
        self._D = self._N * self._spec.instant_dim
        self._trans_stay = float(trans_stay)
        self._mode: Literal["offline", "online"] = mode
        self._fit_interval = int(fit_interval)
        self._window_size = int(window_size)
        self._degen_w = float(degeneracy_max_weight) # lower values reject more and classify as stance.
        self._rng = int(random_state)

        self._instant_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=self._N)
        self._flat_window: deque[npt.NDArray[np.float64]] = deque(maxlen=self._window_size)
        self._hmm = TwoStateGaussianHMM(self._trans_stay)
        self._clock = 0

        self._means_stance_swing: npt.NDArray[np.float64] | None = None
        self._covs_stance_swing: npt.NDArray[np.float64] | None = None
        self._fallback_means: npt.NDArray[np.float64] | None = None
        self._fallback_covs: npt.NDArray[np.float64] | None = None
        self._last_good_means: npt.NDArray[np.float64] | None = None
        self._last_good_covs: npt.NDArray[np.float64] | None = None

        if mode == "offline":
            if initial_means is None or initial_covariances is None:
                raise ValueError("offline mode requires initial_means and initial_covariances from pre-fitting")
            m = np.asarray(initial_means, dtype=np.float64)
            c = np.asarray(initial_covariances, dtype=np.float64)
            if m.shape != (2, self._D) or c.shape != (2, self._D, self._D):
                raise ValueError(f"offline GMM shape mismatch: means {m.shape}, covs {c.shape}")
            self._apply_emission_params(m, c)
            self._fallback_means = m.copy()
            self._fallback_covs = c.copy()
        else:
            if pretrained_path is None:
                raise ValueError("online mode requires pretrained_path (fallback + initial emissions) from training")
            p = resolve_pretrained_gmm_path(pretrained_path)
            m, c = load_pretrained_gmm_npz(
                p,
                expected_feature_dim=self._D,
                expected_history_length=self._N,
                expected_instant_dim=self._spec.instant_dim,
            )
            self._fallback_means = m.copy()
            self._fallback_covs = c.copy()
            self._apply_emission_params(m, c)

    @property
    def spec(self) -> InstantFeatureSpec:
        return self._spec

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def history_length(self) -> int:
        return self._N

    def _apply_emission_params(
        self,
        means_stance_swing: npt.NDArray[np.float64],
        covs_stance_swing: npt.NDArray[np.float64],
    ) -> None:
        m = np.asarray(means_stance_swing, dtype=np.float64)
        c = np.asarray(covs_stance_swing, dtype=np.float64)
        self._means_stance_swing = m
        self._covs_stance_swing = c
        self._hmm.update_dists(
            mu_swing=m[1],
            cov_swing=c[1],
            mu_stance=m[0],
            cov_stance=c[0],
        )

    def _maybe_refit_online(self, flat_x: npt.NDArray[np.float64]) -> None:
        """Append to sliding window; optionally refit GMM when full (online only)."""
        if self._mode != "online":
            return
        self._flat_window.append(flat_x.copy())
        self._clock += 1
        if len(self._flat_window) < self._window_size:
            return
        if self._fit_interval <= 0 or (self._clock % self._fit_interval) != 0:
            return
        X = np.stack([np.asarray(v, dtype=np.float64) for v in self._flat_window], axis=0)
        mo, co, bad = fit_gmm_ordered(X, self._spec, self._N, random_state=self._rng)
        if bad or mo.shape[1] != self._D:
            self._revert_emissions()
            return
        # Calculate the GMM weights temporarily to check if the GMM is degenerate
        g_tmp = GaussianMixture(n_components=2, covariance_type="full", random_state=self._rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                g_tmp.fit(X)
            except (ValueError, np.linalg.LinAlgError):
                self._revert_emissions()
                return
        if float(np.max(g_tmp.weights_)) >= self._degen_w:
            self._revert_emissions()
            return
        # If the GMM is not degenerate, apply the emission parameters to HMM
        self._apply_emission_params(mo, co)
        self._last_good_means = mo.copy()
        self._last_good_covs = co.copy()

    def _revert_emissions(self) -> None:
        if self._last_good_means is not None and self._last_good_covs is not None:
            self._apply_emission_params(self._last_good_means, self._last_good_covs)
        elif self._fallback_means is not None and self._fallback_covs is not None:
            self._apply_emission_params(self._fallback_means, self._fallback_covs)

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        """
        Get the stance belief and state from the HMM.
        If online mode, update HMM belief and GMM emissions.
        If offline mode, use the pre-fitted GMM emissions.
        """
        inst = instant_vector_from_step(step, self._spec)
        self._instant_buf.append(inst)
        buf_list = [np.asarray(v, dtype=np.float64) for v in self._instant_buf]
        n = len(buf_list)

        if self._mode == "online" and n < self._N:
            p_w = 1.0
            return ContactEstimate(stance=True, p_stance=p_w)

        win = np.stack(buf_list, axis=0)
        flat_x = flatten_history_window(win)
        self._maybe_refit_online(flat_x)
        p_stance, stance = self._hmm.update(flat_x)
        return ContactEstimate(stance=stance, p_stance=float(p_stance))

    def reset(self) -> None:
        self._instant_buf.clear()
        self._flat_window.clear()
        self._hmm.reset_belief()
        self._clock = 0
        if self._fallback_means is not None and self._fallback_covs is not None:
            self._apply_emission_params(self._fallback_means, self._fallback_covs)


def build_gmm_hmm_detectors_from_cfg(
    cfg: Mapping[str, Any],
    *,
    recording: LegOdometrySequence | None = None,
    kin_model: BaseKinematics | None = None,
) -> list[GmmHmmContactDetector]:
    """Build one :class:`GmmHmmContactDetector` per leg from ``contact.gmm``."""
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        raise ValueError("contact config missing")
    gmm_cfg = block.get("gmm")
    if not isinstance(gmm_cfg, Mapping):
        raise ValueError("contact.detector is gmm but contact.gmm mapping is missing")

    fields_raw = gmm_cfg.get("feature_fields", list(DEFAULT_INSTANT_FEATURE_FIELDS))
    if not isinstance(fields_raw, (list, tuple)):
        raise TypeError("contact.gmm.feature_fields must be a list")
    feature_fields = tuple(str(x) for x in fields_raw)
    history_length_yaml = int(gmm_cfg.get("history_length", 1))
    trans_stay = float(gmm_cfg.get("trans_stay", 0.99))
    mode = str(gmm_cfg.get("mode", "offline")).lower()
    if mode not in ("offline", "online"):
        raise ValueError(f"contact.gmm.mode must be offline|online, got {mode!r}")
    history_length = 1 if mode == "offline" else history_length_yaml

    fit_interval = int(gmm_cfg.get("fit_interval", 250))
    window_size = int(gmm_cfg.get("window_size", 500))
    degeneracy_max_weight = float(gmm_cfg.get("degeneracy_max_weight", 0.80))
    random_state = int(gmm_cfg.get("random_state", 42))
    pretrained_path = gmm_cfg.get("pretrained_path")

    per_leg_params: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None
    if mode == "offline":
        if recording is None or kin_model is None:
            raise ValueError("offline GMM requires recording and kin_model in build_contact_stack")
        per_leg_params = fit_offline_per_leg(
            recording,
            kin_model,
            feature_fields=feature_fields,
            history_length=1,
            random_state=random_state,
        )

    n_legs = int(kin_model.n_legs) if kin_model is not None else 4
    detectors: list[GmmHmmContactDetector] = []
    for leg in range(n_legs):
        kw: dict[str, Any] = {
            "feature_fields": feature_fields,
            "history_length": history_length,
            "trans_stay": trans_stay,
            "mode": mode,
            "fit_interval": fit_interval,
            "window_size": window_size,
            "degeneracy_max_weight": degeneracy_max_weight,
            "random_state": random_state,
        }
        if mode == "offline":
            m, c = per_leg_params[leg]  # type: ignore[index]
            kw["initial_means"] = m
            kw["initial_covariances"] = c
        else:
            if not pretrained_path:
                raise ValueError("contact.gmm.pretrained_path is required for online mode")
            kw["pretrained_path"] = pretrained_path
        detectors.append(GmmHmmContactDetector(**kw))
    return detectors
