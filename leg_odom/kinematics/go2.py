"""
Unitree Go2 quadruped — forward kinematics and numerical foot Jacobian.

Ported from ``legacy/kinematics.py`` (``Go2Kinematics``). Body frame FLU; leg layout:

- **0** — front-left (FL)
- **1** — front-right (FR)
- **2** — rear-left (RL)
- **3** — rear-right (RR)

Joint order per leg: ``[q_abad, q_hip, q_knee]`` (rad) — abduction, hip flexion, knee.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from leg_odom.kinematics.base import BaseKinematics


class Go2Kinematics(BaseKinematics):
    """Go2 leg geometry: hip offsets in body frame + serial abad–hip–knee chain."""

    def __init__(self) -> None:
        # Body-frame vectors from base origin to each hip attachment (m).
        self._hip_off: dict[int, npt.NDArray[np.float64]] = {
            0: np.array([0.247, 0.050, 0.0], dtype=np.float64),  # FL
            1: np.array([0.247, -0.050, 0.0], dtype=np.float64),  # FR
            2: np.array([-0.247, 0.050, 0.0], dtype=np.float64),  # RL
            3: np.array([-0.247, -0.050, 0.0], dtype=np.float64),  # RR
        }
        self._thigh_length = 0.210
        self._calf_length = 0.210
        self._abad_offset = 0.083

    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Foot position in body frame via rotated abad → hip → knee chain (legacy model)."""
        qv = self._validate_leg_and_q(leg_id, q)
        q_abad, q_hip, q_knee = float(qv[0]), float(qv[1]), float(qv[2])

        # Left legs (0, 2): positive abad sign; right legs (1, 3): mirror.
        side = 1.0 if leg_id in (0, 2) else -1.0

        r_abad = Rotation.from_euler("x", q_abad).as_matrix()
        r_hip = Rotation.from_euler("y", q_hip).as_matrix()
        r_knee = Rotation.from_euler("y", q_knee).as_matrix()

        # Link vectors in predecessor frames (legacy convention: thigh/calf extend −Z).
        p_ab = np.array([0.0, side * self._abad_offset, 0.0], dtype=np.float64)
        p_th = np.array([0.0, 0.0, -self._thigh_length], dtype=np.float64)
        p_cf = np.array([0.0, 0.0, -self._calf_length], dtype=np.float64)

        foot = self._hip_off[leg_id] + r_abad @ (p_ab + r_hip @ (p_th + r_knee @ p_cf))
        return foot.astype(np.float64, copy=False)

    def J_analytical(
        self, leg_id: int, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """
        Foot Jacobian ∂p/∂q.

        Go2 has no closed-form Jacobian in the legacy tree; this matches ``J_num`` there
        (forward differences in :meth:`fk`).
        """
        return self.jacobian_numerical(leg_id, q)
