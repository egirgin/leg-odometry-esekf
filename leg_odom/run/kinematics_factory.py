"""
Build :class:`~leg_odom.kinematics.base.BaseKinematics` from ``robot.kinematics``.

Keeps robot model selection in one place for the main EKF loop and tests.
"""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.kinematics.anymal import AnymalKinematics
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.kinematics.go2 import Go2Kinematics


def build_kinematics_by_name(name: str) -> BaseKinematics:
    """Construct kinematics from a robot key (e.g. ``anymal``, ``go2``)."""
    key = str(name).strip().lower()
    if key == "anymal":
        return AnymalKinematics()
    if key == "go2":
        return Go2Kinematics()
    raise ValueError(f"Unsupported robot kinematics {name!r} (expected anymal or go2)")


def build_kinematics_backend(cfg: Mapping[str, Any]) -> BaseKinematics:
    """Return kinematics for ``cfg["robot"]["kinematics"]`` (e.g. ``anymal``, ``go2``)."""
    return build_kinematics_by_name(str(cfg["robot"]["kinematics"]))
