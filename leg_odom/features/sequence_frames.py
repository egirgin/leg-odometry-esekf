"""
Load merged timelines for NN precompute, by storage layout.

Downstream code should treat returned :class:`pandas.DataFrame` the same regardless of layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from leg_odom.io.columns import FOOT_FORCE_COLS
from leg_odom.io.ocelot_recording import load_prepared_ocelot
from leg_odom.io.split_imu_bag import load_prepared_split_sequence
from leg_odom.io.validation import validate_prepared_split_dataframe


def grf_stance_labels(
    leg_index: int,
    threshold: float,
    *,
    frames: pd.DataFrame | None = None,
    foot_forces: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Per-timestep binary stance: 1.0 if vertical foot load for ``leg_index`` >= ``threshold``, else 0.0.

    For parity with default :class:`~leg_odom.contact.grf_threshold.GrfThresholdContactDetector`
    (``use_abs=False``), this is ``column >= threshold``. Does not apply ``use_abs`` or other detector kwargs.

    Pass **either** ``frames`` (merged log with ``foot_force_*`` columns) **or**
    ``foot_forces`` as ``(T, n_legs)``. Missing / NaN force counts as 0 load.
    """
    if (frames is None) == (foot_forces is None):
        raise ValueError("grf_stance_labels: provide exactly one of frames= or foot_forces=")
    thr = float(threshold)
    li = int(leg_index)
    if foot_forces is not None:
        ff = np.asarray(foot_forces, dtype=np.float64)
        if ff.ndim != 2:
            raise ValueError(f"foot_forces must be (T, n_legs), got {ff.shape}")
        t = ff.shape[0]
        if li < 0 or li >= ff.shape[1]:
            return np.zeros(t, dtype=np.float64)
        col = np.nan_to_num(ff[:, li], nan=0.0, posinf=0.0, neginf=0.0)
        return (col >= thr).astype(np.float64)
    assert frames is not None
    name = FOOT_FORCE_COLS[li]
    if name not in frames.columns:
        s = pd.Series(0.0, index=frames.index, dtype=np.float64)
    else:
        s = pd.to_numeric(frames[name], errors="coerce").fillna(0.0)
    return (s.astype(np.float64).to_numpy() >= thr).astype(np.float64)


def load_tartanground_frames(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Load imu+bag merged timeline (same as :class:`~leg_odom.datasets.tartanground.TartangroundDataset`)."""
    df, _, _, _ = load_prepared_split_sequence(sequence_dir, verbose=verbose, sanitize_imu=True)
    if validate:
        validate_prepared_split_dataframe(df)
    return df


def load_ocelot_frames(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Load Ocelot recording merged timeline (same as :class:`~leg_odom.datasets.ocelot.OcelotDataset`)."""
    df, _, _gt, _accel_gc, _meta = load_prepared_ocelot(
        sequence_dir,
        verbose=verbose,
        sanitize_imu=True,
    )
    if validate:
        validate_prepared_split_dataframe(df)
    return df
