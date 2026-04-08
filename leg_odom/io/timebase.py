"""Construct ``t_abs`` and ``dt`` on a merged log (ported from ``legacy/helpers.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL
from leg_odom.thresholds import (
    TIMEBASE_DT_CLIP_MAX_S,
    TIMEBASE_DT_CLIP_MIN_S,
    TIMEBASE_MIN_POSITIVE_DT_SAMPLES,
    TIMEBASE_RATE_FALLBACK_HZ,
)


def estimate_median_sample_rate_hz(dt_column: pd.Series) -> float:
    """Median sample rate from a positive delta-time column."""
    dt = dt_column.to_numpy(dtype=float)
    dt = dt[dt > 0.0]
    return (1.0 / float(np.median(dt))) if len(dt) > TIMEBASE_MIN_POSITIVE_DT_SAMPLES else TIMEBASE_RATE_FALLBACK_HZ


def build_timebase(dataframe: pd.DataFrame) -> None:
    """
    Ensure ``t_abs`` (seconds from first sample) and ``dt`` exist.

    Mutates ``dataframe`` in place. Requires columns ``sec`` and ``nanosec`` (epoch seconds +
    fractional part in nanoseconds), matching split ``imu.csv`` / ``*_bag.csv`` exports.

    Clamps and fallbacks use :mod:`leg_odom.thresholds`.
    """
    if TIME_SEC_COL not in dataframe.columns or TIME_NANOSEC_COL not in dataframe.columns:
        raise KeyError(
            f"Merged dataframe must contain '{TIME_SEC_COL}' and '{TIME_NANOSEC_COL}' "
            "(not time/ros_sec/timestamp_* aliases)."
        )
    s = pd.to_numeric(dataframe[TIME_SEC_COL], errors="coerce").astype(float)
    ns = pd.to_numeric(dataframe[TIME_NANOSEC_COL], errors="coerce").astype(float)
    t = s + ns * 1e-9

    dataframe["t_abs"] = t - float(t.iloc[0])

    dt = dataframe["t_abs"].diff().fillna(0.0).astype(float).to_numpy()
    dt[dt <= 0.0] = np.nan
    fallback_dt = 1.0 / TIMEBASE_RATE_FALLBACK_HZ
    median_dt = float(np.nanmedian(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else fallback_dt
    dt = np.nan_to_num(dt, nan=median_dt)
    dataframe["dt"] = np.clip(dt, TIMEBASE_DT_CLIP_MIN_S, TIMEBASE_DT_CLIP_MAX_S)
