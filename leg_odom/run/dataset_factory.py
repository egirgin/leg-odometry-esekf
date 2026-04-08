"""
Construct :class:`~leg_odom.datasets.base.BaseLegOdometryDataset` from experiment config.

Uses generic keys ``dataset.kind`` and paths under ``dataset.*`` so new layouts
(Grandtour, Go2 CSV, …) add branches here without renaming robot-specific loaders.
"""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.ocelot import OcelotDataset
from leg_odom.datasets.tartanground import TartangroundDataset


def build_leg_odometry_dataset(
    cfg: Mapping[str, Any],
    *,
    verbose: bool = False,
    sanitize_imu: bool = True,
    validate: bool = True,
    preload: bool = True,
) -> BaseLegOdometryDataset:
    """
    Instantiate the dataset backend for ``cfg["dataset"]["kind"]``.

    Parameters
    ----------
    cfg
        Experiment dict with resolved absolute ``dataset.sequence_dir`` (see
        :func:`~leg_odom.run.experiment_config.resolve_dataset_paths`).
    verbose, sanitize_imu, validate, preload
        Forwarded to concrete dataset classes (defaults match EKF / main experiments).
    """
    kind = str(cfg["dataset"]["kind"]).lower()
    sequence_root = cfg["dataset"]["sequence_dir"]
    extra = {
        "dataset_kind": kind,
        "sequence_root": str(sequence_root),
    }

    if kind == "tartanground":
        return TartangroundDataset(
            sequence_root,
            verbose=verbose,
            sanitize_imu=sanitize_imu,
            validate=validate,
            preload=preload,
            extra_meta=extra,
        )
    if kind == "ocelot":
        return OcelotDataset(
            sequence_root,
            verbose=verbose,
            sanitize_imu=sanitize_imu,
            validate=validate,
            preload=preload,
            extra_meta=extra,
        )

    raise ValueError(f"Unsupported dataset.kind {kind!r} (no factory branch)")
