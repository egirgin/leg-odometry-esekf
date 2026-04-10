"""
Two-state hidden Markov model with multivariate Gaussian emissions.

State 0 = swing, state 1 = stance. The forward pass matches the legacy ``Kinematicovariance_swingHMM`` /
``RealTimeHMM_ND`` convention: ``belief[1]`` is the filtered probability of stance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


class TwoStateGaussianHMM:
    """
    Homogeneous 2×2 transition matrix + one Gaussian per discrete state.

    Emissions are updated when the outer GMM refits (online) or once after offline fitting.
    """

    def __init__(self, trans_stay: float) -> None:
        ts = float(trans_stay)
        if not (0.0 < ts < 1.0):
            raise ValueError("trans_stay must be in (0, 1)")
        sw = 1.0 - ts
        self._trans_mat = np.array([[ts, sw], [sw, ts]], dtype=np.float64)
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
        covariance_swing = np.asarray(cov_swing, dtype=np.float64).reshape(d, d) + reg
        covariance_stance = np.asarray(cov_stance, dtype=np.float64).reshape(d, d) + reg
        self._dist_swing = multivariate_normal(
            mean=np.asarray(mu_swing, dtype=np.float64).reshape(d),
            cov=covariance_swing,
            allow_singular=True,
        )
        self._dist_stance = multivariate_normal(
            mean=np.asarray(mu_stance, dtype=np.float64).reshape(d),
            cov=covariance_stance,
            allow_singular=True,
        )

    def reset_belief(self) -> None:
        self.belief = np.array([0.5, 0.5], dtype=np.float64)

    def update(self, x: npt.NDArray[np.floating]) -> tuple[float, bool]:
        if self._dist_swing is None or self._dist_stance is None:
            raise RuntimeError("TwoStateGaussianHMM: call update_dists before update")
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        predicted = self.belief @ self._trans_mat
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
        return p_stance, stance
