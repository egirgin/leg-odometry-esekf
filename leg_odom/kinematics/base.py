"""
Abstract kinematics: foot forward kinematics and Jacobian in the **body frame** (FLU).

All concrete robots use ``leg_id`` in ``0 .. n_legs-1`` and a **3-vector** of joint angles
per leg in **radians**. Joint ordering (abad/hip/knee vs HAA/HFE/KFE) is documented per class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from leg_odom.thresholds import KINEMATICS_NUMERICAL_JACOBIAN_STEP


class BaseKinematics(ABC):
    """
    Minimal contract for leg odometry: foot position and velocity map from joint angles/rates.

    Outputs are expressed in the **robot base / body** frame (FLU: +X forward, +Y left, +Z up).
    """

    n_legs: ClassVar[int] = 4
    joints_per_leg: ClassVar[int] = 3

    @staticmethod
    def _validate_leg_and_q(leg_id: int, q: npt.NDArray[np.floating], *, n_legs: int = 4) -> npt.NDArray[np.float64]:
        """Ensure ``leg_id`` and shape ``(3,)`` joint vector; return float64 copy."""
        if leg_id < 0 or leg_id >= n_legs:
            raise ValueError(f"leg_id must be in [0, {n_legs - 1}], got {leg_id}")
        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        if qv.shape != (3,):
            raise ValueError(f"Joint vector must have shape (3,), got {qv.shape}")
        return qv

    @abstractmethod
    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Foot contact frame origin position in **body** coordinates, shape ``(3,)``.

        Parameters
        ----------
        leg_id
            Leg index; convention is robot-specific (see subclass docstrings).
        q
            Three joint angles (rad) in the order defined for that robot.
        """

    @abstractmethod
    def J_analytical(
        self, leg_id: int, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Jacobian mapping joint rates to foot **linear** velocity in the body frame:

        ``v_foot_body = J @ qdot`` with ``J`` of shape ``(3, 3)``.

        Subclasses may implement this analytically or via :meth:`jacobian_numerical`.
        """

    def jacobian_numerical(
        self,
        leg_id: int,
        q: npt.NDArray[np.floating],
        *,
        h: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Finite-difference Jacobian columns: ``(fk(q + h e_j) - fk(q)) / h``.

        Shared default for robots without a closed form in this codebase (e.g. Go2) or
        for cross-checking analytic Jacobians.
        """
        step = KINEMATICS_NUMERICAL_JACOBIAN_STEP if h is None else float(h)
        qv = self._validate_leg_and_q(leg_id, q, n_legs=self.n_legs)
        f0 = self.fk(leg_id, qv)
        f0 = np.asarray(f0, dtype=np.float64).reshape(3)
        jac = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            dq = np.zeros(3, dtype=np.float64)
            dq[j] = step
            f1 = np.asarray(self.fk(leg_id, qv + dq), dtype=np.float64).reshape(3)
            jac[:, j] = (f1 - f0) / step
        return jac
