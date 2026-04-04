"""
Tartanground (sim) sequences: same split layout as ANYmal CSV exports.

This is a **data layout** (``dataset.kind: tartanground_split``), not a robot model;
kinematics are selected separately via ``robot.kinematics`` (e.g. ANYmal vs Go2).

``root`` must be the **trajectory directory** that contains ``imu.csv`` (and a bag CSV),
not a parent folder of multiple runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.io.split_imu_bag import load_prepared_split_sequence
from leg_odom.io.validation import validate_prepared_split_dataframe


def _require_sequence_directory(root: Path) -> Path:
    """
    Return ``root`` resolved if ``root/imu.csv`` exists.

    Raises
    ------
    FileNotFoundError
        If ``imu.csv`` is missing (``root`` is not a valid single-sequence directory).
    """
    root = root.expanduser().resolve()
    if not (root / "imu.csv").is_file():
        raise FileNotFoundError(
            f"dataset.sequence_dir must be a trajectory directory containing imu.csv; "
            f"missing or not a file: {root / 'imu.csv'}"
        )
    return root


class TartangroundSplitDataset(BaseLegOdometryDataset):
    """
    Load one Tartanground / ANYmal-style split CSV sequence at ``root``.

    Parameters
    ----------
    root
        Directory containing ``imu.csv`` and a discovered ``*_bag.csv``.
    verbose
        Print loader diagnostics.
    sanitize_imu
        If False, skip legacy IMU sanitization (for debugging only).
    validate
        Run :func:`leg_odom.io.validation.validate_prepared_split_dataframe` after load.
    extra_meta
        Merged into the sequence's ``meta`` dict.
    preload
        If True (default), load in ``__init__``. If False, load on first ``__getitem__``.
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
        self._extra.setdefault("dataset", "tartanground")
        self._extra.setdefault("layout", "split_imu_bag")
        self._cache: LegOdometrySequence | None = None
        if preload:
            self._cache = self._load_sequence()

    def __len__(self) -> int:
        return 1

    def _load_sequence(self) -> LegOdometrySequence:
        df, hz, gt, accel_gc = load_prepared_split_sequence(
            self._sequence_dir,
            verbose=self._verbose,
            sanitize_imu=self._sanitize_imu,
        )
        if self._validate:
            validate_prepared_split_dataframe(df)
        meta: dict[str, Any] = {
            **self._extra,
            "sequence_dir": str(self._sequence_dir),
            "accel_gravity_compensated": accel_gc,
        }
        return LegOdometrySequence(
            frames=df,
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
