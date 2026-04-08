"""
Dataset-specific discovery, frame loading, and stance labels for NN training.

``labels.method: grf_threshold`` and ``gmm_hmm`` build timelines via contact detectors + replay
(see :mod:`leg_odom.training.nn.label_timelines`); they do not use :func:`compute_stance_labels`.

Implemented ``dataset.kind`` values:
- ``tartanground``
- ``ocelot``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from leg_odom.training.nn.discovery import discover_ocelot_sequence_dirs, discover_tartanground_sequence_dirs
from leg_odom.training.nn.sequence_frames import load_ocelot_frames, load_tartanground_frames

DATASET_KIND_TARTANGROUND = "tartanground"
DATASET_KIND_OCELOT = "ocelot"

# Canonical ``labels.method`` strings (NN train YAML). ``labels.*`` is separate from experiment ``contact.*``.
LABEL_METHOD_GRF_THRESHOLD = "grf_threshold"
LABEL_METHOD_GMM_HMM = "gmm_hmm"
LABEL_METHOD_DUAL_HMM = "dual_hmm"
LABEL_METHOD_OCELOT = "ocelot"


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
    """Load merged timeline for feature extraction and labels."""
    k = str(dataset_kind).strip()
    if k == DATASET_KIND_TARTANGROUND:
        return load_tartanground_frames(sequence_dir, verbose=verbose, validate=validate)
    if k == DATASET_KIND_OCELOT:
        return load_ocelot_frames(sequence_dir, verbose=verbose, validate=validate)
    raise ValueError(
        f"Unsupported dataset.kind {dataset_kind!r} for NN frame load; "
        f"supported: {DATASET_KIND_TARTANGROUND!r}, {DATASET_KIND_OCELOT!r}"
    )


def compute_stance_labels(
    dataset_kind: str,
    leg_index: int,
    labels_cfg: Mapping[str, Any],
    *,
    frames: pd.DataFrame | None = None,
    foot_forces: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Binary stance labels ``(T,)`` float 0/1 from merged frames or raw ``foot_forces`` (one required)."""
    if (frames is None) == (foot_forces is None):
        raise ValueError("compute_stance_labels: provide exactly one of frames= or foot_forces=")
    k = str(dataset_kind).strip()
    if k in (DATASET_KIND_TARTANGROUND, DATASET_KIND_OCELOT):
        method = str(labels_cfg.get("method", "")).strip().lower()
        ds_name = k
        if method == LABEL_METHOD_GRF_THRESHOLD:
            raise ValueError(
                f"{ds_name} labels.method grf_threshold does not use compute_stance_labels; "
                "stance timelines come from GrfThresholdContactDetector replay (pass stance_by_seq_leg)."
            )
        if method == LABEL_METHOD_GMM_HMM:
            raise ValueError(
                f"{ds_name} labels.method 'gmm_hmm' does not use compute_stance_labels; "
                "provide precomputed stance timelines (train_contact_nn builds them via replay)."
            )
        if method == LABEL_METHOD_DUAL_HMM:
            raise NotImplementedError(
                "labels.method dual_hmm is not implemented yet; port leg_odom.contact.dual_hmm_fusion first."
            )
        if method == LABEL_METHOD_OCELOT:
            raise NotImplementedError(
                "labels.method ocelot is not implemented yet; port leg_odom.contact.ocelot first."
            )
        raise ValueError(
            f"{ds_name} labels.method {method!r} is not supported in compute_stance_labels."
        )
    raise ValueError(
        f"Unsupported dataset.kind {dataset_kind!r} for stance labels; "
        f"supported: {DATASET_KIND_TARTANGROUND!r}"
    )
