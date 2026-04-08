"""
Load directory layout: ``imu.csv`` + at least one ``*_bag.csv`` (kinematics).

Merging follows ``legacy/data_loader.load_anymal_split_format`` (asof backward on
IMU time). Foot forces stay **0-indexed** ``foot_force_0`` … ``foot_force_3`` as in
the bag CSV (no 1-based aliases).

Both CSVs must use columns ``sec`` and ``nanosec`` for timestamps (wall time in seconds +
nanosecond remainder).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL
from leg_odom.io.ground_truth import extract_position_ground_truth
from leg_odom.io.imu_sanitize import infer_accel_gravity_compensated, sanitize_imu_dataframe
from leg_odom.io.timebase import build_timebase, estimate_median_sample_rate_hz


def discover_bag_csv_path(sequence_dir: Path) -> Path:
    """First ``*_bag.csv`` in lexicographic order (legacy behavior)."""
    candidates = sorted(sequence_dir.glob("*_bag.csv"))
    if not candidates:
        raise FileNotFoundError(f"No '*_bag.csv' under {sequence_dir}")
    return candidates[0]


def merge_split_imu_bag(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load ``imu.csv`` and one bag file, align time origins per legacy rules, asof-merge.

    Does **not** run ``build_timebase`` / IMU sanitize — use
    ``load_prepared_split_sequence`` for the full reference pipeline.
    """
    root = Path(sequence_dir).expanduser().resolve()
    imu_path = root / "imu.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(f"Missing imu.csv under {root}")

    kin_path = discover_bag_csv_path(root)
    if verbose:
        print(f"[io] split layout: {root}")
        print(f"[io] kinematics file: {kin_path.name}")

    imu_df = pd.read_csv(imu_path)
    kin_df = pd.read_csv(kin_path)

    for name, df in (("imu.csv", imu_df), ("*_bag.csv", kin_df)):
        if TIME_SEC_COL not in df.columns or TIME_NANOSEC_COL not in df.columns:
            raise KeyError(
                f"{name} must contain '{TIME_SEC_COL}' and '{TIME_NANOSEC_COL}' "
                f"(legacy 'time' / 'ros_sec'+'ros_nanosec' are not supported)."
            )

    imu_df = imu_df.copy()
    t_imu = imu_df[TIME_SEC_COL].astype(float) + imu_df[TIME_NANOSEC_COL].astype(float) * 1e-9
    imu_df["t_abs"] = t_imu - float(t_imu.iloc[0])

    kin_df = kin_df.copy()
    t_kin = kin_df[TIME_SEC_COL].astype(float) + kin_df[TIME_NANOSEC_COL].astype(float) * 1e-9
    kin_df["t_abs"] = t_kin - float(t_kin.iloc[0])

    imu_t0 = float(imu_df["t_abs"].iloc[0])
    imu_df["t_abs"] = imu_df["t_abs"] - imu_t0

    kin_t0 = float(kin_df["t_abs"].iloc[0])
    kin_df["t_abs"] = kin_df["t_abs"] - kin_t0

    imu_df = imu_df.sort_values("t_abs").reset_index(drop=True)
    kin_df = kin_df.sort_values("t_abs").reset_index(drop=True)

    drop_from_kin = [c for c in (TIME_SEC_COL, TIME_NANOSEC_COL, "time") if c in kin_df.columns]
    merged = pd.merge_asof(
        imu_df,
        kin_df.drop(columns=drop_from_kin, errors="ignore"),
        on="t_abs",
        direction="backward",
    )

    kin_only_cols = [c for c in kin_df.columns if c not in imu_df.columns and c != "t_abs"]
    fill_cols = [c for c in kin_only_cols if c in merged.columns]
    for col in fill_cols:
        merged[col] = merged[col].ffill().bfill()

    if verbose:
        print(f"[io] IMU rows={len(imu_df)}, kin rows={len(kin_df)}, merged={len(merged)}")
    return merged


def load_prepared_split_sequence(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    sanitize_imu: bool = True,
) -> tuple[pd.DataFrame, float, pd.DataFrame, bool]:
    """
    Merge split CSVs, build timebase, optional FLU IMU validation (see ``imu_sanitize``).

    Returns ``(dataframe, median_hz, position_gt, accel_gravity_compensated)``.
    """
    df = merge_split_imu_bag(sequence_dir, verbose=verbose)
    build_timebase(df)
    hz = estimate_median_sample_rate_hz(df["dt"])
    if verbose:
        print(f"[io] median sample rate ~ {hz:.2f} Hz")
    if sanitize_imu:
        df, accel_gc = sanitize_imu_dataframe(df, verbose=verbose)
    else:
        accel_gc = infer_accel_gravity_compensated(df)
    gt = extract_position_ground_truth(df)
    if verbose and not gt.empty:
        print("[io] embedded position ground truth present")
    return df, hz, gt, accel_gc
