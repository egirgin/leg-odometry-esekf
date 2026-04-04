"""
Build :class:`~leg_odom.kinematics.base.BaseKinematics` from ``robot.kinematics``.

Keeps robot model selection in one place for the main EKF loop and tests.
"""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.kinematics.anymal import AnymalKinematics
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.kinematics.go2 import Go2Kinematics


def build_kinematics_backend(cfg: Mapping[str, Any]) -> BaseKinematics:
    """Return kinematics for ``cfg["robot"]["kinematics"]`` (e.g. ``anymal``, ``go2``)."""
    name = str(cfg["robot"]["kinematics"]).lower()
    if name == "anymal":
        return AnymalKinematics()
    if name == "go2":
        return Go2Kinematics()
    raise ValueError(f"Unsupported robot.kinematics {name!r}")
