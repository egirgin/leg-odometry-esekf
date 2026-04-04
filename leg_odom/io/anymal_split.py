"""
ANYmal hardware exports (Tartanground sim logs) use the **split IMU + bag** layout.

The generic implementation lives in ``leg_odom.io.split_imu_bag``; this module
documents the robot-specific convention and re-exports the same API.
"""

from __future__ import annotations

from leg_odom.io.split_imu_bag import (
    discover_bag_csv_path,
    load_prepared_split_sequence,
    merge_split_imu_bag,
)

__all__ = [
    "discover_bag_csv_path",
    "load_prepared_split_sequence",
    "merge_split_imu_bag",
]
