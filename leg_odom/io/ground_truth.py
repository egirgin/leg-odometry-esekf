"""Embedded position ground truth from quadruped split logs."""

from __future__ import annotations

import pandas as pd


def extract_position_ground_truth(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Pull position GT if present (ANYmal / Tartanground style).

    Returns columns ``local_x``, ``local_y``, ``local_z`` and ``t_abs`` when
    available; empty DataFrame if no known GT columns.
    """
    gt_cols_imu = ("pos_x", "pos_y", "pos_z")
    gt_cols_kin = ("p_x", "p_y", "p_z")

    if all(c in dataframe.columns for c in gt_cols_imu):
        gt = dataframe[list(gt_cols_imu)].copy()
        gt = gt.rename(columns={"pos_x": "local_x", "pos_y": "local_y", "pos_z": "local_z"})
        if "t_abs" in dataframe.columns:
            gt["t_abs"] = dataframe["t_abs"].to_numpy()
        return gt

    if all(c in dataframe.columns for c in gt_cols_kin):
        gt = dataframe[list(gt_cols_kin)].copy()
        gt = gt.rename(columns={"p_x": "local_x", "p_y": "local_y", "p_z": "local_z"})
        if "t_abs" in dataframe.columns:
            gt["t_abs"] = dataframe["t_abs"].to_numpy()
        return gt

    return pd.DataFrame()
