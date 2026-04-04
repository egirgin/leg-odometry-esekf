"""Structured objects returned by dataset ``__getitem__``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass
class LegOdometrySequence:
    """
    One synchronized recording: IMU-rate rows with kinematics carried on the merge.

    ``meta`` holds loader flags (e.g. ``accel_gravity_compensated``), paths, and dataset name.
    """

    frames: pd.DataFrame
    median_rate_hz: float
    position_ground_truth: pd.DataFrame
    sequence_name: str
    meta: Mapping[str, Any] = field(default_factory=dict)
