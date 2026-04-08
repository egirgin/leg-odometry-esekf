"""
Tartanground (sim) sequences: ``imu.csv`` + one ``*_bag.csv`` per trajectory directory.

This is a **data product** (``dataset.kind: tartanground``), not a robot model;
kinematics are selected separately via ``robot.kinematics`` (e.g. ANYmal vs Go2).

``root`` must be the **trajectory directory** that contains ``imu.csv``,
not a parent folder of multiple runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leg_odom.datasets.single_sequence import CachedSingleSequenceDataset
from leg_odom.io.split_imu_bag import load_prepared_split_sequence


class TartangroundDataset(CachedSingleSequenceDataset):
    """
    Load one Tartanground / ANYmal-style CSV sequence at ``root``.

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
        em = dict(extra_meta or {})
        em.setdefault("dataset", "tartanground")
        em.setdefault("layout", "split_imu_bag")
        super().__init__(
            root,
            verbose=verbose,
            sanitize_imu=sanitize_imu,
            validate=validate,
            extra_meta=em,
            preload=preload,
        )

    def _require_sequence_directory(self, root: Path) -> Path:
        root = root.expanduser().resolve()
        if not (root / "imu.csv").is_file():
            raise FileNotFoundError(
                f"dataset.sequence_dir must be a trajectory directory containing imu.csv; "
                f"missing or not a file: {root / 'imu.csv'}"
            )
        return root

    def _load_prepared(self):
        df, hz, gt, accel_gc = load_prepared_split_sequence(
            self._sequence_dir,
            verbose=self._verbose,
            sanitize_imu=self._sanitize_imu,
        )
        return df, hz, gt, accel_gc, {}
