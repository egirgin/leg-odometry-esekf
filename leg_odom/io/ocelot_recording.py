"""
Load Ocelot trajectory layout: one primary CSV (``lowstate.csv``) per directory.

V1 contract:
- required: ``lowstate.csv`` (legacy Ocelot lowstate export schema)
- optional: ``groundtruth.csv`` (detected only, not parsed yet)
- optional: ``frames/`` with ``<sec>_<nanosec>.png`` files (detected only)

See ``OCELOT_LOWSTATE_REQUIREMENTS.md`` (repo root) for column-level requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leg_odom.io.columns import (
    FOOT_FORCE_COLS,
    IMU_CORE_COLS,
    TIME_NANOSEC_COL,
    TIME_SEC_COL,
    motor_position_cols,
    motor_torque_cols,
    motor_velocity_cols,
)
from leg_odom.io.imu_sanitize import infer_accel_gravity_compensated, sanitize_imu_dataframe
from leg_odom.io.timebase import build_timebase, estimate_median_sample_rate_hz


def _required_ocelot_columns() -> tuple[str, ...]:
    """Required columns for the primary Ocelot CSV V1. ``motor_i_ddq`` is optional."""
    return (
        TIME_SEC_COL,
        TIME_NANOSEC_COL,
        *IMU_CORE_COLS,
        *FOOT_FORCE_COLS,
        *motor_position_cols(),
        *motor_velocity_cols(),
        *motor_torque_cols(),
    )


def _assert_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _required_ocelot_columns() if c not in df.columns]
    if missing:
        raise KeyError(
            "Ocelot recording CSV missing required columns: "
            + ", ".join(missing)
            + " (motor_i_ddq is optional)"
        )


def _coerce_numeric_required(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = list(_required_ocelot_columns())
    for col in req:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    bad = [c for c in req if not np.isfinite(out[c].to_numpy(dtype=np.float64)).all()]
    if bad:
        raise ValueError(
            "Ocelot recording CSV contains non-finite/non-numeric values in required columns: "
            + ", ".join(bad)
        )
    return out


def discover_ocelot_csv_path(sequence_dir: str | Path) -> Path:
    """Return path to the primary Ocelot recording file (``lowstate.csv``) under ``sequence_dir``."""
    root = Path(sequence_dir).expanduser().resolve()
    p = root / "lowstate.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing lowstate.csv (Ocelot recording) under {root}")
    return p


def _groundtruth_path(root: Path) -> Path:
    return root / "groundtruth.csv"


def _frames_dir_path(root: Path) -> Path:
    return root / "frames"


def _count_frame_pngs(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return len(list(frames_dir.glob("*.png")))


def load_prepared_ocelot(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    sanitize_imu: bool = True,
) -> tuple[pd.DataFrame, float, pd.DataFrame, bool, dict[str, Any]]:
    """
    Load and prepare Ocelot recording frames for EKF / training.

    Returns ``(frames_df, median_hz, gt_df, accel_gravity_compensated, metadata)``.
    Ground truth is intentionally not parsed in V1 (``gt_df`` is always empty).
    """
    root = Path(sequence_dir).expanduser().resolve()
    csv_path = discover_ocelot_csv_path(root)
    gt_path = _groundtruth_path(root)
    frames_dir = _frames_dir_path(root)

    df = pd.read_csv(csv_path)
    _assert_required_columns(df)
    df = _coerce_numeric_required(df)

    df = df.sort_values([TIME_SEC_COL, TIME_NANOSEC_COL]).reset_index(drop=True)
    build_timebase(df)
    hz = estimate_median_sample_rate_hz(df["dt"])

    if sanitize_imu:
        df, accel_gc = sanitize_imu_dataframe(df, verbose=verbose)
    else:
        accel_gc = infer_accel_gravity_compensated(df)

    gt_df = pd.DataFrame()
    meta = {
        "has_groundtruth_csv": gt_path.is_file(),
        "has_frames_dir": frames_dir.is_dir(),
        "frames_png_count": _count_frame_pngs(frames_dir),
    }
    if verbose:
        print(f"[io] ocelot recording: rows={len(df)}, hz~{hz:.2f}, gt={meta['has_groundtruth_csv']}")
        if meta["has_frames_dir"]:
            print(f"[io] ocelot frames/: png_count={meta['frames_png_count']}")
    return df, hz, gt_df, accel_gc, meta
