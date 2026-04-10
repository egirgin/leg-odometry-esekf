"""
Kinematics HMM with optional energy-modulated transitions (legacy ``KinematicsHMM``).

When ``use_energy`` is false, behavior matches a homogeneous transition matrix like
:class:`~leg_odom.contact.gmm_hmm_core.hmm_gaussian.TwoStateGaussianHMM`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


class KinGaussianHmmEnergy:
    """2-state Gaussian emissions + symmetric transitions; optional energy-dependent switching."""

    def __init__(self, base_trans_stay: float, initial_gamma: float = 1.0) -> None:
        ts = float(base_trans_stay)
        if not (0.0 < ts < 1.0):
            raise ValueError("base_trans_stay must be in (0, 1)")
        self.base_stay = ts
        self.base_switch = 1.0 - ts
        self.gamma = float(initial_gamma)
        self._static_trans = np.array([[ts, self.base_switch], [self.base_switch, ts]], dtype=np.float64)
        self.belief = np.array([0.5, 0.5], dtype=np.float64)
        self._dist_swing: multivariate_normal | None = None
        self._dist_stance: multivariate_normal | None = None

    def update_dists(
        self,
        mu_swing: npt.NDArray[np.floating],
        cov_swing: npt.NDArray[np.floating],
        mu_stance: npt.NDArray[np.floating],
        cov_stance: npt.NDArray[np.floating],
        *,
        ridge: float = 1e-6,
    ) -> None:
        d = int(np.asarray(mu_swing, dtype=np.float64).reshape(-1).shape[0])
        reg = ridge * np.eye(d, dtype=np.float64)
        cov_sw = np.asarray(cov_swing, dtype=np.float64).reshape(d, d) + reg
        cov_st = np.asarray(cov_stance, dtype=np.float64).reshape(d, d) + reg
        self._dist_swing = multivariate_normal(
            mean=np.asarray(mu_swing, dtype=np.float64).reshape(d),
            cov=cov_sw,
            allow_singular=True,
        )
        self._dist_stance = multivariate_normal(
            mean=np.asarray(mu_stance, dtype=np.float64).reshape(d),
            cov=cov_st,
            allow_singular=True,
        )

    def reset_belief(self) -> None:
        self.belief = np.array([0.5, 0.5], dtype=np.float64)

    def get_dynamic_transition_matrix(self, energy_spike: float) -> tuple[npt.NDArray[np.float64], float]:
        switch_prob = self.base_switch + (1.0 - self.base_switch) * (1.0 - np.exp(-self.gamma * energy_spike))
        switch_prob = float(np.clip(switch_prob, 0.0, 0.999))
        stay_prob = 1.0 - switch_prob
        mat = np.array([[stay_prob, switch_prob], [switch_prob, stay_prob]], dtype=np.float64)
        return mat, switch_prob

    def update(
        self,
        x: npt.NDArray[np.floating],
        *,
        energy_spike: float,
        use_energy: bool,
    ) -> tuple[float, bool, float]:
        if self._dist_swing is None or self._dist_stance is None:
            raise RuntimeError("KinGaussianHmmEnergy: call update_dists before update")
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        if use_energy:
            trans_mat, sw_prob = self.get_dynamic_transition_matrix(energy_spike)
        else:
            trans_mat = self._static_trans
            sw_prob = float(self.base_switch)
        predicted = self.belief @ trans_mat
        likelihoods = np.array(
            [self._dist_swing.pdf(xv), self._dist_stance.pdf(xv)],
            dtype=np.float64,
        )
        unnorm = predicted * likelihoods
        s = float(np.sum(unnorm))
        if s < 1e-12:
            self.belief = predicted
        else:
            self.belief = unnorm / s
        p_stance = float(self.belief[1])
        stance = p_stance > float(self.belief[0])
        return p_stance, stance, sw_prob
