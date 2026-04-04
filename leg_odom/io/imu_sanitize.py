"""
IMU checks for **FLU body** exports: no automatic axis remaps—invalid data raises.

**Expected default**: accelerometer reports **specific force** (gravity included), ~+9.81 m/s²
along body **+Z** when the base is level.

**Tolerated variant**: **gravity-compensated** (linear) acceleration—mean magnitude is small;
then the log must include body attitude quaternions so initial orientation does not rely on
gravity in the accel channel.

Numeric thresholds live in :mod:`leg_odom.thresholds` (implementation constants, not YAML params).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from leg_odom.io.columns import IMU_ACCEL_COLS, IMU_BODY_QUAT_COLS, IMU_CORE_COLS
from leg_odom.thresholds import (
    IMU_FLU_SPECIFIC_FORCE_MAG_MAX,
    IMU_FLU_SPECIFIC_FORCE_MAG_MIN,
    IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG,
    IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD,
    IMU_GYRO_MEDIAN_NORM_DEG_S_HINT,
    IMU_VECTOR_NEAR_ZERO_NORM,
)


def _angle_to_body_plus_z_deg(v: np.ndarray) -> float:
    """
    Angle in degrees between vector *v* and body **+Z** (FLU up).

    Returns NaN if *v* is (near) zero so callers can treat that as “no direction”.
    """
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    norm_v = float(np.linalg.norm(v))
    if norm_v < IMU_VECTOR_NEAR_ZERO_NORM:
        return float("nan")
    cosang = float(np.clip(np.dot(v / norm_v, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _has_body_orientation_quaternion(dataframe: pd.DataFrame) -> bool:
    """True when all quaternion columns listed in ``IMU_BODY_QUAT_COLS`` are present."""
    return all(c in dataframe.columns for c in IMU_BODY_QUAT_COLS)


def infer_accel_gravity_compensated(dataframe: pd.DataFrame) -> bool:
    """
    Heuristic: if time-averaged |accel| is far below ~g, assume **gravity was removed**
    from the accelerometer (linear acceleration only).
    """
    mag = np.linalg.norm(dataframe[list(IMU_ACCEL_COLS)].to_numpy(dtype=float), axis=1)
    return bool(float(np.mean(mag)) < IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD)


def _assert_flu_specific_force(dataframe: pd.DataFrame, *, verbose: bool) -> None:
    """
    Require mean specific force to align with FLU +Z and have magnitude near |g|.

    Raises ``ValueError`` if the sequence does not look like FLU-specific-force at rest
    (full-sequence mean—aggressive motion can violate this; exports are expected to match).
    """
    accel = dataframe[list(IMU_ACCEL_COLS)].to_numpy(dtype=float)
    a_mean = accel.mean(axis=0)
    norm = float(np.linalg.norm(a_mean))
    tilt = _angle_to_body_plus_z_deg(a_mean)

    if not np.isfinite(tilt):
        raise ValueError(
            "IMU FLU check failed: mean acceleration has near-zero norm; cannot infer up direction."
        )

    # Sign check: FLU specific force at rest must push +Z positive (~+g), not inverted.
    if float(a_mean[2]) <= 0.0:
        raise ValueError(
            "IMU FLU check failed: mean accel_z must be positive for level specific force "
            f"(got accel_z mean {float(a_mean[2]):.4f} m/s²). Data must be FLU with gravity "
            "along body +Z when stationary, or use gravity-compensated accel with orientation quaternions."
        )

    if not (IMU_FLU_SPECIFIC_FORCE_MAG_MIN <= norm <= IMU_FLU_SPECIFIC_FORCE_MAG_MAX):
        raise ValueError(
            "IMU FLU check failed: mean |accel| is "
            f"{norm:.2f} m/s² (expected specific force magnitude roughly "
            f"{IMU_FLU_SPECIFIC_FORCE_MAG_MIN}…{IMU_FLU_SPECIFIC_FORCE_MAG_MAX} m/s² for FLU at rest). "
            "If your CSV uses gravity-compensated acceleration, include body orientation "
            f"columns {IMU_BODY_QUAT_COLS}."
        )

    if tilt > IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG:
        raise ValueError(
            "IMU FLU check failed: mean acceleration is not aligned with body +Z "
            f"(tilt {tilt:.1f}° from +Z, max allowed {IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG}°). "
            "Data must be FLU; no automatic frame correction is applied."
        )

    if verbose:
        print(
            f"[IMU] FLU specific-force check OK (mean |a|={norm:.2f} m/s², "
            f"tilt from +Z={tilt:.1f}°)."
        )


def _assert_gravity_compensated_with_orientation(dataframe: pd.DataFrame, *, verbose: bool) -> None:
    """Require quaternion attitude when linear acceleration (no gravity in meas) is detected."""
    if not _has_body_orientation_quaternion(dataframe):
        raise ValueError(
            "IMU: accelerometer appears gravity-compensated (mean |accel| small), but "
            f"missing orientation columns {IMU_BODY_QUAT_COLS}. "
            "Provide quaternions for initial attitude, or export specific force in FLU (+g on +Z when level)."
        )
    if verbose:
        print(
            "[IMU] Gravity-compensated acceleration detected; body quaternion columns present "
            "(initial orientation can use CSV attitude)."
        )


def sanitize_imu_dataframe(
    dataframe: pd.DataFrame,
    *,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Validate FLU IMU semantics, convert gyro to rad/s if clearly in deg/s.

    Returns ``(dataframe, accel_gravity_compensated)`` where
    ``accel_gravity_compensated`` is True if measurements are **linear** acceleration
    (gravity removed from the accel channel); False if **specific force** (gravity included),
    which is the default expectation for gravity-based initial alignment.
    """
    req = list(IMU_CORE_COLS)
    if not all(c in dataframe.columns for c in req):
        raise KeyError(f"CSV missing required IMU columns: {req}")

    gyro_data = dataframe[list(IMU_CORE_COLS[:3])].to_numpy(dtype=float)
    median_norm = float(np.median(np.linalg.norm(gyro_data, axis=1)))

    if median_norm > IMU_GYRO_MEDIAN_NORM_DEG_S_HINT:
        dataframe[list(IMU_CORE_COLS[:3])] *= np.pi / 180.0
        if verbose:
            print("[IMU] Converted gyro units from deg/s to rad/s.")

    accel_gravity_compensated = infer_accel_gravity_compensated(dataframe)
    if accel_gravity_compensated:
        _assert_gravity_compensated_with_orientation(dataframe, verbose=verbose)
    else:
        _assert_flu_specific_force(dataframe, verbose=verbose)

    return dataframe, accel_gravity_compensated
