"""
Shared single-directory dataset: one trajectory folder → one :class:`~leg_odom.datasets.types.LegOdometrySequence`.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.io.validation import validate_prepared_split_dataframe


class CachedSingleSequenceDataset(BaseLegOdometryDataset):
    """
    Load exactly one sequence under ``root``; optional preload cache.

    Subclasses implement marker validation and :meth:`_load_prepared`.
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
        self._sequence_dir = self._require_sequence_directory(Path(root))
        self._verbose = verbose
        self._sanitize_imu = sanitize_imu
        self._validate = validate
        self._extra = dict(extra_meta or {})
        self._cache: LegOdometrySequence | None = None
        if preload:
            self._cache = self._load_sequence()

    @abstractmethod
    def _require_sequence_directory(self, root: Path) -> Path:
        """Resolve and validate ``root`` as a single-sequence directory."""

    @abstractmethod
    def _load_prepared(
        self,
    ) -> tuple[pd.DataFrame, float, pd.DataFrame, bool, dict[str, Any]]:
        """
        Return prepared ``(frames, hz, position_gt, accel_gravity_compensated, meta_extra)``.

        ``meta_extra`` is merged into the sequence ``meta`` dict.
        """

    def __len__(self) -> int:
        return 1

    def _load_sequence(self) -> LegOdometrySequence:
        df, hz, gt, accel_gc, meta_extra = self._load_prepared()
        if self._validate:
            validate_prepared_split_dataframe(df)
        meta: dict[str, Any] = {
            **self._extra,
            "sequence_dir": str(self._sequence_dir),
            "accel_gravity_compensated": accel_gc,
            **meta_extra,
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
