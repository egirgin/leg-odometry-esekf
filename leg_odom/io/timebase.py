"""Construct ``t_abs`` and ``dt`` on a merged log (ported from ``legacy/helpers.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_odom.thresholds import (
    TIMEBASE_DT_CLIP_MAX_S,
    TIMEBASE_DT_CLIP_MIN_S,
    TIMEBASE_MIN_POSITIVE_DT_SAMPLES,
    TIMEBASE_RATE_FALLBACK_HZ,
    TIMEBASE_TIMESTAMP_NS_SCALE_THRESHOLD,
)


def estimate_median_sample_rate_hz(dt_column: pd.Series) -> float:
    """Median sample rate from a positive delta-time column."""
    dt = dt_column.to_numpy(dtype=float)
    dt = dt[dt > 0.0]
    return (1.0 / float(np.median(dt))) if len(dt) > TIMEBASE_MIN_POSITIVE_DT_SAMPLES else TIMEBASE_RATE_FALLBACK_HZ


def build_timebase(dataframe: pd.DataFrame) -> None:
    """
    Ensure ``t_abs`` (seconds from first sample) and ``dt`` exist.

    Mutates ``dataframe`` in place. Matches legacy logic: supports sec+nanosec
    pairs, or a single ``timestamp`` / ``time`` / ``t`` column.

    Clamps and fallbacks use :mod:`leg_odom.thresholds`.
    """
    cols = {c.lower(): c for c in dataframe.columns}

    if "timestamp_sec" in cols and "timestamp_nanosec" in cols:
        s = pd.to_numeric(dataframe[cols["timestamp_sec"]], errors="coerce").astype(float)
        ns = pd.to_numeric(dataframe[cols["timestamp_nanosec"]], errors="coerce").astype(float)
        t = s + ns * 1e-9
    elif "sec" in cols and "nanosec" in cols:
        s = pd.to_numeric(dataframe[cols["sec"]], errors="coerce").astype(float)
        ns = pd.to_numeric(dataframe[cols["nanosec"]], errors="coerce").astype(float)
        t = s + ns * 1e-9
    else:
        cand = next((cols[k] for k in ("timestamp", "time", "t") if k in cols), None)
        if cand is None:
            raise KeyError(
                "No time columns found. Need 't', 'timestamp', or a 'sec'+'nanosec' pair."
            )
        vals = pd.to_numeric(dataframe[cand], errors="coerce")
        scale = 1e9 if float(vals.max()) > TIMEBASE_TIMESTAMP_NS_SCALE_THRESHOLD else 1.0
        t = vals.astype(float) / scale

    dataframe["t_abs"] = t - float(t.iloc[0])

    dt = dataframe["t_abs"].diff().fillna(0.0).astype(float).to_numpy()
    dt[dt <= 0.0] = np.nan
    fallback_dt = 1.0 / TIMEBASE_RATE_FALLBACK_HZ
    median_dt = float(np.nanmedian(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else fallback_dt
    dt = np.nan_to_num(dt, nan=median_dt)
    dataframe["dt"] = np.clip(dt, TIMEBASE_DT_CLIP_MIN_S, TIMEBASE_DT_CLIP_MAX_S)
