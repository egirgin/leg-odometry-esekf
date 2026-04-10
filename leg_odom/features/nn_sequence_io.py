"""Dataset kind dispatch for NN precompute: sequence discovery and merged frame load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leg_odom.features.discovery import discover_ocelot_sequence_dirs, discover_tartanground_sequence_dirs
from leg_odom.features.sequence_frames import load_ocelot_frames, load_tartanground_frames

DATASET_KIND_TARTANGROUND = "tartanground"
DATASET_KIND_OCELOT = "ocelot"


def discover_sequence_dirs(dataset_kind: str, root: str | Path, *, verbose: bool = False) -> list[Path]:
    """Return sorted sequence directory paths for the given data product kind."""
    k = str(dataset_kind).strip()
    if k == DATASET_KIND_TARTANGROUND:
        return discover_tartanground_sequence_dirs(root, verbose=verbose)
    if k == DATASET_KIND_OCELOT:
        return discover_ocelot_sequence_dirs(root, verbose=verbose)
    raise ValueError(
        f"Unsupported dataset.kind {dataset_kind!r} for NN discovery; "
        f"supported: {DATASET_KIND_TARTANGROUND!r}, {DATASET_KIND_OCELOT!r}"
    )


def load_training_frames(
    dataset_kind: str,
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Load merged timeline for feature extraction and stance labels."""
    k = str(dataset_kind).strip()
    if k == DATASET_KIND_TARTANGROUND:
        return load_tartanground_frames(sequence_dir, verbose=verbose, validate=validate)
    if k == DATASET_KIND_OCELOT:
        return load_ocelot_frames(sequence_dir, verbose=verbose, validate=validate)
    raise ValueError(
        f"Unsupported dataset.kind {dataset_kind!r} for NN frame load; "
        f"supported: {DATASET_KIND_TARTANGROUND!r}, {DATASET_KIND_OCELOT!r}"
    )
