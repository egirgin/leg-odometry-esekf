"""ZUPT measurement covariance from stance probability (shared by GMM+HMM and dual HMM)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ZUPT_P_STANCE_FLOOR = 1e-9


def zupt_R_foot_from_p_stance(p_stance: float) -> npt.NDArray[np.float64]:
    """``R = (1 / max(p_stance, ε)) I₃`` — variance per world velocity component (m/s)² scale."""
    pe = max(float(p_stance), ZUPT_P_STANCE_FLOOR)
    v = 1.0 / pe
    return np.eye(3, dtype=np.float64) * v
