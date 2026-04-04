"""
One-shot nominal EKF state from the first valid merged timeline row.

Used when ``ekf.initialize_nominal_from_data`` is true. All quantities are **world FLU**
(position, velocity) and body attitude as in :mod:`leg_odom.io.columns` / IMU exports.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.filters.esekf import ErrorStateEkf
from leg_odom.io.columns import IMU_BODY_QUAT_COLS

# World position (m): Tartanground IMU-style vs kinematics bag-style names.
_POS_COL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("pos_x", "pos_y", "pos_z"),
    ("p_x", "p_y", "p_z"),
)
# World velocity (m/s).
_VEL_COL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("vel_x", "vel_y", "vel_z"),
    ("v_x", "v_y", "v_z"),
)
_BIAS_ACCEL_COLS = ("bax", "bay", "baz")
_BIAS_GYRO_COLS = ("bgx", "bgy", "bgz")


def _pick_xyz_group(
    columns: Any,
    groups: tuple[tuple[str, str, str], ...],
    label: str,
) -> tuple[str, str, str]:
    colset = set(columns)
    for g in groups:
        if all(c in colset for c in g):
            return g
    raise ValueError(
        f"ekf.initialize_nominal_from_data is true but merged timeline lacks {label}. "
        f"Expected one of column triples: {list(groups)}"
    )


def _row_all_finite(row: pd.Series, names: tuple[str, ...]) -> bool:
    for c in names:
        v = row[c]
        x = float(v) if not isinstance(v, (float, np.floating)) else float(v)
        if not math.isfinite(x):
            return False
    return True


def _first_valid_row_index(df: pd.DataFrame, needed: tuple[str, ...]) -> int:
    for i in range(len(df)):
        row = df.iloc[i]
        if _row_all_finite(row, needed):
            return i
    raise ValueError(
        "ekf.initialize_nominal_from_data: no row with finite values for all required columns "
        f"{needed}"
    )


def apply_nominal_init_from_timeline(ekf: ErrorStateEkf, timeline: pd.DataFrame) -> None:
    """
    Set ``ekf`` nominal ``p``, ``v``, ``R`` from the first timeline row with finite samples.

    Optional: if ``bax,bay,baz`` and/or ``bgx,bgy,bgz`` exist and are finite on that row,
    set accel and gyro biases.

    Raises
    ------
    ValueError
        Missing columns or no finite row.
    """
    pos = _pick_xyz_group(timeline.columns, _POS_COL_GROUPS, "world position (m)")
    vel = _pick_xyz_group(timeline.columns, _VEL_COL_GROUPS, "world velocity (m/s)")
    quat_cols = IMU_BODY_QUAT_COLS
    for c in quat_cols:
        if c not in timeline.columns:
            raise ValueError(
                "ekf.initialize_nominal_from_data requires body quaternion columns "
                f"{quat_cols} on the merged timeline"
            )

    needed = pos + vel + quat_cols
    idx = _first_valid_row_index(timeline, needed)
    row = timeline.iloc[idx]

    p = np.asarray([row[pos[0]], row[pos[1]], row[pos[2]]], dtype=np.float64)
    v = np.asarray([row[vel[0]], row[vel[1]], row[vel[2]]], dtype=np.float64)
    q = np.asarray(
        [row[quat_cols[0]], row[quat_cols[1]], row[quat_cols[2]], row[quat_cols[3]]],
        dtype=np.float64,
    )
    R = Rotation.from_quat(q).as_matrix()

    ba = None
    bg = None
    if all(c in timeline.columns for c in _BIAS_ACCEL_COLS):
        if _row_all_finite(row, _BIAS_ACCEL_COLS):
            ba = np.asarray([row[c] for c in _BIAS_ACCEL_COLS], dtype=np.float64)
    if all(c in timeline.columns for c in _BIAS_GYRO_COLS):
        if _row_all_finite(row, _BIAS_GYRO_COLS):
            bg = np.asarray([row[c] for c in _BIAS_GYRO_COLS], dtype=np.float64)

    ekf.seed_nominal_state(p=p, v=v, R=R, bias_accel=ba, bias_gyro=bg)


def ekf_initialize_nominal_from_data_enabled(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    ekf = cfg.get("ekf")
    if not isinstance(ekf, Mapping):
        return False
    return bool(ekf.get("initialize_nominal_from_data", False))
