"""
Ocelot dataset layout (``dataset.kind: ocelot``).

One sequence directory contains ``lowstate.csv`` with synchronized IMU + joint + foot-load rows.
Optional ``groundtruth.csv`` and ``frames/`` are detected and exposed in metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leg_odom.datasets.single_sequence import CachedSingleSequenceDataset
from leg_odom.io.ocelot_recording import load_prepared_ocelot


class OcelotDataset(CachedSingleSequenceDataset):
    """
    Load one Ocelot trajectory from ``root``.

    Parameters mirror :class:`leg_odom.datasets.tartanground.TartangroundDataset`
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
        em = dict(extra_meta or {})
        em.setdefault("dataset", "ocelot")
        em.setdefault("layout", "ocelot_recording")
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
        if not (root / "lowstate.csv").is_file():
            raise FileNotFoundError(
                "dataset.sequence_dir must be a trajectory directory containing lowstate.csv; "
                f"missing: {root / 'lowstate.csv'}"
            )
        return root

    def _load_prepared(self):
        return load_prepared_ocelot(
            self._sequence_dir,
            verbose=self._verbose,
            sanitize_imu=self._sanitize_imu,
        )
