"""
Trajectory metrics vs embedded ground truth (ported from ``legacy/analysis_eval.py``).

Uses the same overlap + interpolation policy as the legacy ``evaluate`` method:
ATE, absolute heading error, RPE (trans/rot), FPE, drift %, length error, discrete Fréchet.

Results are written as a **CSV table** (one row per sequence in batch exports).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Discrete Fréchet can recurse deeply on long paths; match legacy cap.
if sys.getrecursionlimit() < 5000:
    sys.setrecursionlimit(5000)

EVALUATION_CSV_COLUMNS: tuple[str, ...] = (
    "sequence_name",
    "skipped",
    "ate_m",
    "ahe_deg",
    "rpe_trans_pct",
    "rpe_rot_deg_per_m",
    "fpe_m",
    "drift_pct",
    "length_err",
    "frechet_m",
    "gt_length_m",
    "est_length_m",
)


def compute_path_dist(data: np.ndarray) -> np.ndarray:
    diffs = np.diff(data[:, :2], axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate(([0], np.cumsum(dists)))


def calculate_absolute_heading_error(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> float:
    if len(gt_pos) < 2 or len(matched_est_pos) < 2:
        return 0.0
    gt_vecs = gt_pos[1:] - gt_pos[:-1]
    est_vecs = matched_est_pos[1:] - matched_est_pos[:-1]
    gt_angles = np.arctan2(gt_vecs[:, 1], gt_vecs[:, 0])
    est_angles = np.arctan2(est_vecs[:, 1], est_vecs[:, 0])
    diffs = est_angles - gt_angles
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.sqrt(np.mean(diffs**2))))


def match_predictions_to_gt(
    gt_t: np.ndarray, gt_pos: np.ndarray, est_t: np.ndarray, est_pos: np.ndarray
) -> np.ndarray:
    interp_func = interp1d(est_t, est_pos, axis=0, kind="linear", fill_value="extrapolate")
    return np.asarray(interp_func(gt_t), dtype=np.float64)


def calculate_ate_rmse_synced(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> float:
    errors = np.linalg.norm(gt_pos - matched_est_pos, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def calculate_rpe_metrics_synced(
    gt_pos: np.ndarray, matched_est_pos: np.ndarray, window_m: float = 1.0
) -> tuple[float, float]:
    gt_dist = compute_path_dist(gt_pos)
    trans_errors_sq: list[float] = []
    rot_errors_sq: list[float] = []

    def get_headings(pts: np.ndarray) -> np.ndarray:
        dx = pts[1:, 0] - pts[:-1, 0]
        dy = pts[1:, 1] - pts[:-1, 1]
        return np.unwrap(np.arctan2(dy, dx))

    gt_h = get_headings(gt_pos)
    est_h = get_headings(matched_est_pos)
    n_points = len(gt_pos)

    for i in range(n_points):
        candidates = np.where(gt_dist > (gt_dist[i] + window_m))[0]
        if len(candidates) == 0:
            break
        j = int(candidates[0])
        actual_dist = gt_dist[j] - gt_dist[i]
        if actual_dist < 1e-3:
            continue

        gt_delta = gt_pos[j] - gt_pos[i]
        est_delta = matched_est_pos[j] - matched_est_pos[i]
        trans_error = float(np.linalg.norm(est_delta - gt_delta))
        trans_errors_sq.append(((trans_error / actual_dist) * 100) ** 2)

        if i < len(gt_h) and j < len(gt_h):
            gt_h_change = gt_h[j] - gt_h[i]
            est_h_change = est_h[j] - est_h[i]
            rot_err = np.abs(est_h_change - gt_h_change)
            rot_err = (rot_err + np.pi) % (2 * np.pi) - np.pi
            rot_errors_sq.append((np.degrees(np.abs(rot_err)) / actual_dist) ** 2)

    if not trans_errors_sq:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(trans_errors_sq))), float(np.sqrt(np.mean(rot_errors_sq)))


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return 0.0
    ca = np.ones((n, m), dtype=np.float64) * -1.0
    dist_matrix = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    ca[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist_matrix[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                dist_matrix[i, j],
            )
    return float(ca[n - 1, m - 1])


def calculate_shape_metrics(
    gt_resampled: np.ndarray,
    est_resampled: np.ndarray,
    gt_total_len: float,
    est_total_len: float,
) -> tuple[float, float, float]:
    ratio = est_total_len / gt_total_len if gt_total_len > 0 else 0.0
    len_error = abs(1.0 - ratio)
    fpe = float(np.linalg.norm(gt_resampled[-1] - est_resampled[-1]))
    step = max(1, len(gt_resampled) // 500)
    est_step = max(1, len(est_resampled) // 500)
    frechet_dist = discrete_frechet(gt_resampled[::step], est_resampled[::est_step])
    return len_error, fpe, frechet_dist


def resample_spatially(data: np.ndarray, step: float = 0.1) -> tuple[np.ndarray, float]:
    dists = compute_path_dist(data)
    total_dist = float(dists[-1])
    if total_dist == 0:
        return data, 0.0
    new_dists = np.arange(0, total_dist, step)
    interp_func = interp1d(dists, data[:, :2], axis=0, fill_value="extrapolate")
    resampled = np.asarray(interp_func(new_dists), dtype=np.float64)
    return resampled, total_dist


def _est_time_seconds(hist: pd.DataFrame) -> np.ndarray:
    if "timestamp_sec" in hist.columns and "timestamp_nanosec" in hist.columns:
        return (
            hist["timestamp_sec"].to_numpy(dtype=np.float64)
            + hist["timestamp_nanosec"].to_numpy(dtype=np.float64) * 1e-9
        )
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    raise ValueError("EKF history needs timestamp_sec/nanosec or t_abs for evaluation time base.")


def _gt_time_seconds(gt_df: pd.DataFrame) -> np.ndarray:
    if "ros_sec" in gt_df.columns and "ros_nanosec" in gt_df.columns:
        return gt_df["ros_sec"].to_numpy(dtype=np.float64) + gt_df["ros_nanosec"].to_numpy(
            dtype=np.float64
        ) * 1e-9
    if "t_abs" in gt_df.columns:
        return gt_df["t_abs"].to_numpy(dtype=np.float64)
    raise ValueError("Ground truth needs ros_sec/ros_nanosec or t_abs.")


def _nan_row(sequence_name: str, skipped: str) -> dict[str, Any]:
    return {
        "sequence_name": sequence_name,
        "skipped": skipped,
        "ate_m": np.nan,
        "ahe_deg": np.nan,
        "rpe_trans_pct": np.nan,
        "rpe_rot_deg_per_m": np.nan,
        "fpe_m": np.nan,
        "drift_pct": np.nan,
        "length_err": np.nan,
        "frechet_m": np.nan,
        "gt_length_m": np.nan,
        "est_length_m": np.nan,
    }


class TrajectoryEvaluator:
    """
    Trajectory metrics vs embedded GT (legacy ``AnalysisAndEvaluation.evaluate`` semantics).

    Use :meth:`evaluate` for a single recording; :meth:`write_metrics_csv` to persist one
    or more rows as a table.
    """

    @staticmethod
    def write_metrics_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write a CSV table with fixed columns (missing metrics as empty/NaN)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized: list[dict[str, Any]] = []
        for r in rows:
            row = {c: r.get(c, np.nan) for c in EVALUATION_CSV_COLUMNS}
            row["sequence_name"] = r.get("sequence_name", "")
            row["skipped"] = r.get("skipped", "")
            normalized.append(row)
        pd.DataFrame(normalized, columns=list(EVALUATION_CSV_COLUMNS)).to_csv(
            path, index=False
        )

    def evaluate(
        self,
        hist: pd.DataFrame,
        gt_df: pd.DataFrame,
        *,
        sequence_name: str = "",
        print_report: bool = True,
    ) -> dict[str, Any]:
        """
        Compute metrics; optionally print a text summary (not written to disk).

        Returns a dict suitable for :meth:`write_metrics_csv` (includes ``sequence_name``,
        ``skipped``, and scalar metrics or NaNs when skipped).
        """
        if gt_df is None or gt_df.empty:
            out = _nan_row(sequence_name, "no_ground_truth")
            if print_report:
                print("[EVAL] No ground truth available. Skipping evaluation.")
            return out

        if hist is None or hist.empty:
            out = _nan_row(sequence_name, "empty_history")
            if print_report:
                print("[EVAL] Empty EKF history. Skipping evaluation.")
            return out

        try:
            est_t = _est_time_seconds(hist)
        except ValueError as e:
            out = _nan_row(sequence_name, str(e))
            if print_report:
                print(f"[EVAL] {e}")
            return out

        est_pos = hist[["p_x", "p_y"]].to_numpy(dtype=np.float64)

        try:
            gt_t = _gt_time_seconds(gt_df)
        except ValueError as e:
            out = _nan_row(sequence_name, str(e))
            if print_report:
                print(f"[EVAL] {e}")
            return out

        if "local_x" not in gt_df.columns or "local_y" not in gt_df.columns:
            out = _nan_row(sequence_name, "missing_local_xy")
            if print_report:
                print("[EVAL] Ground truth missing local_x/local_y.")
            return out

        gt_pos = gt_df[["local_x", "local_y"]].to_numpy(dtype=np.float64)

        valid_mask = (gt_t >= est_t[0]) & (gt_t <= est_t[-1])
        gt_t_clipped = gt_t[valid_mask]
        gt_pos_clipped = gt_pos[valid_mask]

        if len(gt_t_clipped) < 2:
            out = _nan_row(sequence_name, "insufficient_overlap")
            if print_report:
                print("[EVAL] Not enough overlapping points for evaluation.")
            return out

        matched_est_pos = match_predictions_to_gt(gt_t_clipped, gt_pos_clipped, est_t, est_pos)

        ate = calculate_ate_rmse_synced(gt_pos_clipped, matched_est_pos)
        ahe_deg = calculate_absolute_heading_error(gt_pos_clipped, matched_est_pos)
        rpe_trans, rpe_rot = calculate_rpe_metrics_synced(
            gt_pos_clipped, matched_est_pos, window_m=1.0
        )

        gt_resampled, gt_len = resample_spatially(gt_pos_clipped, step=0.1)
        est_resampled, est_len = resample_spatially(matched_est_pos, step=0.1)

        len_err, fpe, frechet = calculate_shape_metrics(
            gt_resampled, est_resampled, gt_len, est_len
        )
        drift_pct = (fpe / gt_len) * 100 if gt_len > 0 else 0.0

        out = {
            "sequence_name": sequence_name,
            "skipped": "",
            "ate_m": ate,
            "ahe_deg": ahe_deg,
            "rpe_trans_pct": rpe_trans,
            "rpe_rot_deg_per_m": rpe_rot,
            "fpe_m": fpe,
            "drift_pct": drift_pct,
            "length_err": len_err,
            "frechet_m": frechet,
            "gt_length_m": gt_len,
            "est_length_m": est_len,
        }

        if print_report:
            print(
                "--- EVALUATION METRICS ---\n"
                f"ATE [m]:         {ate:.4f}\n"
                f"AHE [deg]:       {ahe_deg:.4f}\n"
                f"RPE Trans [%]:   {rpe_trans:.4f}\n"
                f"RPE Rot [deg/m]: {rpe_rot:.4f}\n"
                f"FPE [m]:         {fpe:.4f}\n"
                f"Drift [%]:       {drift_pct:.4f}\n"
                f"Length Err:      {len_err:.4f}\n"
                f"Frechet [m]:     {frechet:.4f}\n"
                f"GT Length [m]:   {gt_len:.4f}\n"
                f"Path Length [m]: {est_len:.4f}\n"
                "--------------------------\n"
            )

        return out


def evaluate_trajectory(
    hist: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    print_report: bool = True,
    sequence_name: str = "",
) -> dict[str, Any]:
    """
    Convenience wrapper: :class:`TrajectoryEvaluator` + optional single-row CSV under
    ``output_dir/evaluation_metrics.csv``.
    """
    ev = TrajectoryEvaluator()
    result = ev.evaluate(hist, gt_df, sequence_name=sequence_name, print_report=print_report)
    if output_dir is not None:
        TrajectoryEvaluator.write_metrics_csv(
            Path(output_dir) / "evaluation_metrics.csv", [result]
        )
    return result


def metrics_dict_to_lines(m: Mapping[str, Any]) -> list[str]:
    """Stable key order for tests / logging."""
    keys = (
        "ate_m",
        "ahe_deg",
        "rpe_trans_pct",
        "rpe_rot_deg_per_m",
        "fpe_m",
        "drift_pct",
        "length_err",
        "frechet_m",
        "gt_length_m",
        "est_length_m",
    )
    lines = []
    for k in keys:
        if k in m and m.get("skipped") == "":
            lines.append(f"{k}={m[k]:.6g}")
    return lines
