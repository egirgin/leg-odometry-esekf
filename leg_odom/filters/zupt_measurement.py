"""Isotropic ZUPT measurement noise from stance probability (EKF / process loop only)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ZUPT_P_STANCE_FLOOR = 1e-9


def zupt_isotropic_meas_from_p_stance(p_stance: float) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Single mapping: ``sigma_sq = 1 / max(p_stance, ε)``, ``R_foot = sigma_sq * I₃``.

    Returns
    -------
    sigma_sq
        Isotropic variance per world foot-velocity component, (m/s)².
    R_foot
        Shape ``(3, 3)``.
    """
    pe = max(float(p_stance), ZUPT_P_STANCE_FLOOR)
    sigma_sq = 1.0 / pe
    r = np.eye(3, dtype=np.float64) * sigma_sq
    return float(sigma_sq), r
