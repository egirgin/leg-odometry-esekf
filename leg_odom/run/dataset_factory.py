"""
Construct :class:`~leg_odom.datasets.base.BaseLegOdometryDataset` from experiment config.

Uses generic keys ``dataset.kind`` and paths under ``dataset.*`` so new layouts
(Grandtour, Go2 CSV, …) add branches here without renaming robot-specific loaders.
"""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.ocelot import OcelotLowstateDataset
from leg_odom.datasets.tartanground import TartangroundSplitDataset


def build_leg_odometry_dataset(cfg: Mapping[str, Any]) -> BaseLegOdometryDataset:
    """
    Instantiate the dataset backend for ``cfg["dataset"]["kind"]``.

    Parameters
    ----------
    cfg
        Experiment dict with resolved absolute ``dataset.sequence_dir`` (see
        :func:`~leg_odom.run.experiment_config.resolve_dataset_paths`).
    """
    kind = str(cfg["dataset"]["kind"]).lower()
    sequence_root = cfg["dataset"]["sequence_dir"]

    if kind == "tartanground_split":
        return TartangroundSplitDataset(
            sequence_root,
            verbose=False,
            sanitize_imu=True,
            validate=True,
            preload=True,
            extra_meta={
                "dataset_kind": kind,
                "sequence_root": str(sequence_root),
            },
        )
    if kind == "ocelot":
        return OcelotLowstateDataset(
            sequence_root,
            verbose=False,
            sanitize_imu=True,
            validate=True,
            preload=True,
            extra_meta={
                "dataset_kind": kind,
                "sequence_root": str(sequence_root),
            },
        )

    raise ValueError(f"Unsupported dataset.kind {kind!r} (no factory branch)")
