"""
Shared column names for quadruped split logs (IMU CSV + kinematics ``*_bag.csv``).

Robot-specific naming (e.g. ANYmal ``motor_*``) stays here as the de facto
multi-robot convention until a second robot needs different prefixes.
"""

from __future__ import annotations

# Wall-clock columns in split CSVs (imu.csv, *_bag.csv) and EKF history CSV.
TIME_SEC_COL = "sec"
TIME_NANOSEC_COL = "nanosec"

# Minimum IMU channels expected after CSV load (before sanitization).
IMU_GYRO_COLS = ("gyro_x", "gyro_y", "gyro_z")
IMU_ACCEL_COLS = ("accel_x", "accel_y", "accel_z")
IMU_CORE_COLS = IMU_GYRO_COLS + IMU_ACCEL_COLS

# Body orientation in the log (ANYmal / Tartanground IMU CSV); required if accel is gravity-compensated.
IMU_BODY_QUAT_COLS = ("ori_qx", "ori_qy", "ori_qz", "ori_qw")

# Joint / actuator columns for 12-DoM leg layout (3 × 4 legs), radians / Nm.
def motor_position_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_q" for i in range(12))


def motor_torque_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_tau_est" for i in range(12))


def motor_velocity_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_dq" for i in range(12))


# Per-foot vertical load proxy (0-based leg index; canonical in ``leg_odom``).
FOOT_FORCE_COLS = tuple(f"foot_force_{i}" for i in range(4))
