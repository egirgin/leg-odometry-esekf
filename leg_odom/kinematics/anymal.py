"""
ANYmal-C style quadruped — URDF-derived homogeneous transforms + analytic foot Jacobian.

Ported from ``legacy/kinematics.py`` (``ANYmalKinematics``). Body frame FLU; leg indices:

- **0** — left front (LF)
- **1** — right front (RF)
- **2** — left hind (LH)
- **3** — right hind (RH)

Joint order per leg: ``[q_haa, q_hfe, q_kfe]`` (rad) — hip abduction/adduction, hip flexion,
knee flexion (ANYmal naming: HAA, HFE, KFE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from leg_odom.kinematics.base import BaseKinematics


def _ht_from_xyz_rpy(
    xyz: list[float] | npt.NDArray[np.floating],
    rpy: list[float] | npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """4×4 homogeneous transform: rotation from URDF extrinsic **XYZ** Euler, then translation."""
    t = np.eye(4, dtype=np.float64)
    # Uppercase 'XYZ' matches URDF extrinsic roll-pitch-yaw convention (legacy comment).
    t[:3, :3] = Rotation.from_euler("XYZ", np.asarray(rpy, dtype=np.float64)).as_matrix()
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def _ht_rot_x(angle: float, axis_sign: float) -> npt.NDArray[np.float64]:
    """Pure rotation about local X by ``axis_sign * angle`` (ANYmal URDF joint axes)."""
    return _ht_from_xyz_rpy([0.0, 0.0, 0.0], [axis_sign * angle, 0.0, 0.0])


class AnymalKinematics(BaseKinematics):
    """
    Precomputed static chain segments per leg; live joints apply X rotations with per-leg signs.
    """

    def __init__(self) -> None:
        # URDF (xyz, rpy) per leg — same numeric values as legacy.
        urdf_params: dict[int, dict[str, Any]] = {
            0: {
                "base_haa": ([0.2999, 0.104, 0.0], [2.61799387799, 0, 0]),
                "haa_axis": 1,
                "hip_fixed": ([0, 0, 0], [-2.61799387799, 0, 0]),
                "hfe_fixed": ([0.0599, 0.08381, 0.0], [0, 0, 1.57079632679]),
                "hfe_axis": 1,
                "thigh_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
                "kfe_fixed": ([0.0, 0.1003, -0.285], [0, 0, 1.57079632679]),
                "kfe_axis": 1,
                "shank_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
                "foot_fixed": ([0.08795, 0.01305, -0.33797], [0, 0, 0]),
            },
            1: {
                "base_haa": ([0.2999, -0.104, 0.0], [-2.61799387799, 0, 0]),
                "haa_axis": 1,
                "hip_fixed": ([0, 0, 0], [2.61799387799, 0, 0]),
                "hfe_fixed": ([0.0599, -0.08381, 0.0], [0, 0, -1.57079632679]),
                "hfe_axis": -1,
                "thigh_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
                "kfe_fixed": ([0.0, -0.1003, -0.285], [0, 0, -1.57079632679]),
                "kfe_axis": -1,
                "shank_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
                "foot_fixed": ([0.08795, -0.01305, -0.33797], [0, 0, 0]),
            },
            2: {
                "base_haa": ([-0.2999, 0.104, 0.0], [-2.61799387799, 0, -3.14159265359]),
                "haa_axis": -1,
                "hip_fixed": ([0, 0, 0], [-2.61799387799, 0, -3.14159265359]),
                "hfe_fixed": ([-0.0599, 0.08381, 0.0], [0, 0, 1.57079632679]),
                "hfe_axis": 1,
                "thigh_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
                "kfe_fixed": ([-0.0, 0.1003, -0.285], [0, 0, 1.57079632679]),
                "kfe_axis": 1,
                "shank_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
                "foot_fixed": ([-0.08795, 0.01305, -0.33797], [0, 0, 0]),
            },
            3: {
                "base_haa": ([-0.2999, -0.104, 0.0], [2.61799387799, 0, -3.14159265359]),
                "haa_axis": -1,
                "hip_fixed": ([0, 0, 0], [2.61799387799, 0, -3.14159265359]),
                "hfe_fixed": ([-0.0599, -0.08381, 0.0], [0, 0, -1.57079632679]),
                "hfe_axis": -1,
                "thigh_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
                "kfe_fixed": ([-0.0, -0.1003, -0.285], [0, 0, -1.57079632679]),
                "kfe_axis": -1,
                "shank_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
                "foot_fixed": ([-0.08795, -0.01305, -0.33797], [0, 0, 0]),
            },
        }

        self._t_base_haa: dict[int, npt.NDArray[np.float64]] = {}
        self._t_hip_hfe: dict[int, npt.NDArray[np.float64]] = {}
        self._t_thigh_kfe: dict[int, npt.NDArray[np.float64]] = {}
        self._t_shank_foot: dict[int, npt.NDArray[np.float64]] = {}
        self._axes: dict[int, tuple[int, int, int]] = {}

        for i in range(4):
            p = urdf_params[i]
            self._t_base_haa[i] = _ht_from_xyz_rpy(p["base_haa"][0], p["base_haa"][1])
            self._t_hip_hfe[i] = _ht_from_xyz_rpy(p["hip_fixed"][0], p["hip_fixed"][1]) @ _ht_from_xyz_rpy(
                p["hfe_fixed"][0], p["hfe_fixed"][1]
            )
            self._t_thigh_kfe[i] = _ht_from_xyz_rpy(p["thigh_fixed"][0], p["thigh_fixed"][1]) @ _ht_from_xyz_rpy(
                p["kfe_fixed"][0], p["kfe_fixed"][1]
            )
            self._t_shank_foot[i] = _ht_from_xyz_rpy(p["shank_fixed"][0], p["shank_fixed"][1]) @ _ht_from_xyz_rpy(
                p["foot_fixed"][0], p["foot_fixed"][1]
            )
            self._axes[i] = (p["haa_axis"], p["hfe_axis"], p["kfe_axis"])

    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Foot origin in body frame from full homogeneous chain (legacy ``fk``)."""
        qv = self._validate_leg_and_q(leg_id, q)
        q_haa, q_hfe, q_kfe = float(qv[0]), float(qv[1]), float(qv[2])
        ax = self._axes[leg_id]

        t_haa = _ht_rot_x(q_haa, float(ax[0]))
        t_hfe = _ht_rot_x(q_hfe, float(ax[1]))
        t_kfe = _ht_rot_x(q_kfe, float(ax[2]))

        t_foot = (
            self._t_base_haa[leg_id]
            @ t_haa
            @ self._t_hip_hfe[leg_id]
            @ t_hfe
            @ self._t_thigh_kfe[leg_id]
            @ t_kfe
            @ self._t_shank_foot[leg_id]
        )
        return np.asarray(t_foot[:3, 3], dtype=np.float64).copy()

    def J_analytical(
        self, leg_id: int, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """
        Geometric Jacobian: columns are ``ω_j × (p_ee - p_j)`` for each revolute joint axis.

        Matches legacy ``J_analytical`` (X-axis joints with per-leg sign stored in ``_axes``).
        """
        qv = self._validate_leg_and_q(leg_id, q)
        q_haa, q_hfe, q_kfe = float(qv[0]), float(qv[1]), float(qv[2])
        ax = self._axes[leg_id]

        t_haa_rot = _ht_rot_x(q_haa, float(ax[0]))
        t_hfe_rot = _ht_rot_x(q_hfe, float(ax[1]))
        t_kfe_rot = _ht_rot_x(q_kfe, float(ax[2]))

        t0 = self._t_base_haa[leg_id]
        t1 = t0 @ t_haa_rot @ self._t_hip_hfe[leg_id]
        t2 = t1 @ t_hfe_rot @ self._t_thigh_kfe[leg_id]
        t_foot = t2 @ t_kfe_rot @ self._t_shank_foot[leg_id]

        p0 = t0[:3, 3]
        p1 = t1[:3, 3]
        p2 = t2[:3, 3]
        p_foot = t_foot[:3, 3]

        # Unit joint axes in base frame (revolute about local X, signed).
        w0 = t0[:3, :3] @ np.array([ax[0], 0.0, 0.0], dtype=np.float64)
        w1 = t1[:3, :3] @ np.array([ax[1], 0.0, 0.0], dtype=np.float64)
        w2 = t2[:3, :3] @ np.array([ax[2], 0.0, 0.0], dtype=np.float64)

        jac = np.zeros((3, 3), dtype=np.float64)
        jac[:, 0] = np.cross(w0, p_foot - p0)
        jac[:, 1] = np.cross(w1, p_foot - p1)
        jac[:, 2] = np.cross(w2, p_foot - p2)
        return jac

    def leg_chain_points(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """
        Polyline of shape ``(5, 3)``: base origin, HAA frame, HFE frame, KFE frame, foot.

        Useful for visualization / debugging (legacy ``leg_chain_points``).
        """
        qv = self._validate_leg_and_q(leg_id, q)
        q_haa, q_hfe, q_kfe = float(qv[0]), float(qv[1]), float(qv[2])
        ax = self._axes[leg_id]

        t_haa = _ht_rot_x(q_haa, float(ax[0]))
        t_hfe = _ht_rot_x(q_hfe, float(ax[1]))
        t_kfe = _ht_rot_x(q_kfe, float(ax[2]))

        t0 = self._t_base_haa[leg_id]
        t1 = t0 @ t_haa @ self._t_hip_hfe[leg_id]
        t2 = t1 @ t_hfe @ self._t_thigh_kfe[leg_id]
        t_foot = t2 @ t_kfe @ self._t_shank_foot[leg_id]

        return np.vstack(
            (
                np.zeros((1, 3), dtype=np.float64),
                t0[:3, 3].reshape(1, 3),
                t1[:3, 3].reshape(1, 3),
                t2[:3, 3].reshape(1, 3),
                t_foot[:3, 3].reshape(1, 3),
            )
        )
