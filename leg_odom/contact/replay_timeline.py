"""Replay contact detectors on a merged bag timeline (no EKF)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput
from leg_odom.io.columns import FOOT_FORCE_COLS, motor_position_cols, motor_torque_cols, motor_velocity_cols

if TYPE_CHECKING:
    from leg_odom.kinematics.base import BaseKinematics

_IMU_GYRO = ("gyro_x", "gyro_y", "gyro_z")
_IMU_ACCEL = ("accel_x", "accel_y", "accel_z")


def replay_detectors_on_timeline(
    timeline: pd.DataFrame,
    kin: BaseKinematics,
    detectors: list[BaseContactDetector],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Return ``t_abs``, ``grf[leg]``, ``stance_bin[leg]``, ``p_stance[leg]`` for one sequence row-order.
    """
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())
    jpl = kin.joints_per_leg
    n_legs = kin.n_legs
    n_steps = len(timeline)
    t_abs = timeline["t_abs"].to_numpy(dtype=np.float64)
    grfs = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]
    st = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]
    ps = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]

    for d in detectors:
        d.reset()

    for k in range(n_steps):
        row = timeline.iloc[k]
        gyro = row[list(_IMU_GYRO)].to_numpy(dtype=np.float64)
        accel = row[list(_IMU_ACCEL)].to_numpy(dtype=np.float64)
        q_all = row[motor_cols].to_numpy(dtype=np.float64)
        dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        for leg in range(n_legs):
            sl = slice(leg * jpl, (leg + 1) * jpl)
            q_leg = q_all[sl]
            dq_leg = dq_all[sl]
            tau_leg = tau_all[sl]
            pb = np.asarray(kin.fk(leg, q_leg), dtype=np.float64).reshape(3)
            jac = np.asarray(kin.J_analytical(leg, q_leg), dtype=np.float64).reshape(3, jpl)
            qd = np.asarray(dq_leg, dtype=np.float64).reshape(jpl)
            v_foot_body = np.cross(gyro, pb) + jac @ qd
            grf = float(row.get(FOOT_FORCE_COLS[leg], 0.0))
            grfs[leg][k] = grf
            step = ContactDetectorStepInput(
                grf_n=grf,
                p_foot_body=pb,
                v_foot_body=v_foot_body,
                q_leg=np.asarray(q_leg, dtype=np.float64, order="C"),
                dq_leg=np.asarray(dq_leg, dtype=np.float64, order="C"),
                tau_leg=np.asarray(tau_leg, dtype=np.float64, order="C"),
                gyro_body_corrected=np.asarray(gyro, dtype=np.float64, order="C"),
                accel_body_corrected=np.asarray(accel, dtype=np.float64, order="C"),
            )
            est = detectors[leg].update(step)
            st[leg][k] = 1.0 if est.stance else 0.0
            ps[leg][k] = float(est.p_stance)
    return t_abs, grfs, st, ps
