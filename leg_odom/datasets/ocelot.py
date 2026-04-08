"""
Ocelot lowstate dataset layout (``dataset.kind: ocelot``).

One sequence directory contains ``lowstate.csv`` with synchronized IMU + joint + foot-load rows.
Optional ``groundtruth.csv`` and ``frames/`` are detected and exposed in metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.io.ocelot_lowstate import load_prepared_ocelot_lowstate
from leg_odom.io.validation import validate_prepared_split_dataframe


def _require_sequence_directory(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not (root / "lowstate.csv").is_file():
        raise FileNotFoundError(
            "dataset.sequence_dir must be a trajectory directory containing lowstate.csv; "
            f"missing: {root / 'lowstate.csv'}"
        )
    return root


class OcelotLowstateDataset(BaseLegOdometryDataset):
    """
    Load one Ocelot lowstate sequence from ``root``.

    Parameters mirror :class:`leg_odom.datasets.tartanground.TartangroundSplitDataset`
    to keep dataset factory behavior consistent.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        verbose: bool = False,
        sanitize_imu: bool = True,
        validate: bool = True,
        extra_meta: dict[str, Any] | None = None,
        preload: bool = True,
    ) -> None:
        self._sequence_dir = _require_sequence_directory(Path(root))
        self._verbose = verbose
        self._sanitize_imu = sanitize_imu
        self._validate = validate
        self._extra = dict(extra_meta or {})
        self._extra.setdefault("dataset", "ocelot")
        self._extra.setdefault("layout", "lowstate_csv")
        self._cache: LegOdometrySequence | None = None
        if preload:
            self._cache = self._load_sequence()

    def __len__(self) -> int:
        return 1

    def _load_sequence(self) -> LegOdometrySequence:
        frames, hz, gt, accel_gc, io_meta = load_prepared_ocelot_lowstate(
            self._sequence_dir,
            verbose=self._verbose,
            sanitize_imu=self._sanitize_imu,
        )
        if self._validate:
            validate_prepared_split_dataframe(frames)
        meta: dict[str, Any] = {
            **self._extra,
            "sequence_dir": str(self._sequence_dir),
            "accel_gravity_compensated": accel_gc,
            **io_meta,
        }
        return LegOdometrySequence(
            frames=frames,
            median_rate_hz=hz,
            position_ground_truth=gt,
            sequence_name=self._sequence_dir.name,
            meta=meta,
        )

    def __getitem__(self, index: int) -> LegOdometrySequence:
        if index != 0:
            raise IndexError(index)
        if self._cache is None:
            self._cache = self._load_sequence()
        return self._cache
