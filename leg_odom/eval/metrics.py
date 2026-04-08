"""
Trajectory metrics vs embedded ground truth (ported from ``legacy/analysis_eval.py``).

Uses the same overlap + interpolation policy as the legacy ``evaluate`` method:
ATE, absolute heading error, RPE (trans/rot), FPE, drift %, length error, discrete Fréchet.

Evaluation time base matches merged logs and :mod:`leg_odom.eval.analysis_plots`: **prefer**
``t_abs`` on EKF history when present; otherwise ``sec`` + ``nanosec``, so estimates align with GT
from :func:`leg_odom.io.ground_truth.extract_position_ground_truth`.

**``ate_m``** is the standard ATE: RMSE of the Euclidean position error at each synchronized
time (2D horizontal if no Z GT; 3D if ``local_z`` and ``p_z`` are used). **``ate_x_m``** …
**``ate_z_m``** are per-axis RMSEs for breakdown only.

Results are written as a **CSV table** (one row per sequence in batch exports).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL

# Discrete Fréchet can recurse deeply on long paths; match legacy cap.
if sys.getrecursionlimit() < 5000:
    sys.setrecursionlimit(5000)

EVALUATION_CSV_COLUMNS: tuple[str, ...] = (
    "sequence_name",
    "skipped",
    "ate_m",
    "ate_x_m",
    "ate_y_m",
    "ate_z_m",
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


def calculate_per_axis_rmse(
    gt_pos: np.ndarray, matched_est_pos: np.ndarray
) -> tuple[float, float, float]:
    """Per-axis RMSE; z is NaN if positions are 2D (second array width 2)."""
    d = matched_est_pos.shape[1]
    ex = gt_pos[:, 0] - matched_est_pos[:, 0]
    ey = gt_pos[:, 1] - matched_est_pos[:, 1]
    rx = float(np.sqrt(np.mean(ex**2)))
    ry = float(np.sqrt(np.mean(ey**2)))
    if d >= 3 and gt_pos.shape[1] >= 3:
        ez = gt_pos[:, 2] - matched_est_pos[:, 2]
        rz = float(np.sqrt(np.mean(ez**2)))
    else:
        rz = float("nan")
    return rx, ry, rz


def calculate_ate_norm_rmse_3d(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> float:
    """RMSE of 3D error vector length (requires 3 columns on both)."""
    if gt_pos.shape[1] < 3 or matched_est_pos.shape[1] < 3:
        return float("nan")
    err = gt_pos[:, :3] - matched_est_pos[:, :3]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


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
    """Prefer ``t_abs`` (recording time) over wall stamps so GT ``t_abs`` overlaps."""
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    if TIME_SEC_COL in hist.columns and TIME_NANOSEC_COL in hist.columns:
        return (
            hist[TIME_SEC_COL].to_numpy(dtype=np.float64)
            + hist[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9
        )
    raise ValueError(
        f"EKF history needs t_abs or {TIME_SEC_COL!r}/{TIME_NANOSEC_COL!r} for evaluation time base."
    )


def _est_time_source(hist: pd.DataFrame) -> str:
    if "t_abs" in hist.columns:
        return "t_abs"
    if TIME_SEC_COL in hist.columns and TIME_NANOSEC_COL in hist.columns:
        return f"{TIME_SEC_COL}+{TIME_NANOSEC_COL}"
    return "none"


def _gt_time_seconds(gt_df: pd.DataFrame) -> np.ndarray:
    if "t_abs" in gt_df.columns:
        return gt_df["t_abs"].to_numpy(dtype=np.float64)
    if TIME_SEC_COL in gt_df.columns and TIME_NANOSEC_COL in gt_df.columns:
        return gt_df[TIME_SEC_COL].to_numpy(dtype=np.float64) + gt_df[TIME_NANOSEC_COL].to_numpy(
            dtype=np.float64
        ) * 1e-9
    raise ValueError(
        f"Ground truth needs t_abs or {TIME_SEC_COL!r}/{TIME_NANOSEC_COL!r}."
    )


def _gt_time_source(gt_df: pd.DataFrame) -> str:
    if "t_abs" in gt_df.columns:
        return "t_abs"
    if TIME_SEC_COL in gt_df.columns and TIME_NANOSEC_COL in gt_df.columns:
        return f"{TIME_SEC_COL}+{TIME_NANOSEC_COL}"
    return "none"


def _sort_est_timeseries_for_interp(
    est_t: np.ndarray, est_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(est_t, kind="mergesort")
    t_s = est_t[order]
    p_s = est_pos[order]
    # Collapse duplicate timestamps (keep last sample) for interp1d.
    if len(t_s) < 2:
        return t_s, p_s
    keep = np.ones(len(t_s), dtype=bool)
    keep[:-1] = np.abs(np.diff(t_s)) > 1e-15
    if not np.all(keep):
        t_u: list[float] = []
        p_u: list[np.ndarray] = []
        i = 0
        while i < len(t_s):
            j = i + 1
            while j < len(t_s) and abs(t_s[j] - t_s[i]) <= 1e-15:
                j += 1
            t_u.append(float(t_s[j - 1]))
            p_u.append(np.asarray(p_s[j - 1], dtype=np.float64))
            i = j
        return np.asarray(t_u, dtype=np.float64), np.stack(p_u, axis=0)
    return t_s, p_s


def time_alignment_report(hist: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Any]:
    """
    Read-only diagnostic: time columns used, ranges, and overlap sample count.

    Does not require a full EKF run beyond having ``hist`` and ``gt_df`` CSVs loaded.
    """
    out: dict[str, Any] = {
        "est_time_source": _est_time_source(hist),
        "gt_time_source": _gt_time_source(gt_df),
        "est_t_min": None,
        "est_t_max": None,
        "gt_t_min": None,
        "gt_t_max": None,
        "overlap_duration_s": None,
        "n_gt_in_overlap": None,
        "error": None,
    }
    try:
        est_t = _est_time_seconds(hist)
    except ValueError as e:
        out["error"] = str(e)
        return out
    try:
        gt_t = _gt_time_seconds(gt_df)
    except ValueError as e:
        out["error"] = str(e)
        return out

    est_min = float(np.nanmin(est_t))
    est_max = float(np.nanmax(est_t))
    gt_min = float(np.nanmin(gt_t))
    gt_max = float(np.nanmax(gt_t))
    out["est_t_min"] = est_min
    out["est_t_max"] = est_max
    out["gt_t_min"] = gt_min
    out["gt_t_max"] = gt_max

    lo = max(est_min, gt_min)
    hi = min(est_max, gt_max)
    out["overlap_duration_s"] = max(0.0, hi - lo)
    mask = (gt_t >= est_min) & (gt_t <= est_max)
    out["n_gt_in_overlap"] = int(np.sum(mask))
    return out


def _nan_row(sequence_name: str, skipped: str) -> dict[str, Any]:
    return {
        "sequence_name": sequence_name,
        "skipped": skipped,
        "ate_m": np.nan,
        "ate_x_m": np.nan,
        "ate_y_m": np.nan,
        "ate_z_m": np.nan,
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

        use_3d = (
            "local_z" in gt_df.columns
            and "p_z" in hist.columns
            and pd.api.types.is_numeric_dtype(gt_df["local_z"])
        )
        if use_3d:
            est_pos = hist[["p_x", "p_y", "p_z"]].to_numpy(dtype=np.float64)
        else:
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

        if use_3d:
            gt_pos = gt_df[["local_x", "local_y", "local_z"]].to_numpy(dtype=np.float64)
        else:
            gt_pos = gt_df[["local_x", "local_y"]].to_numpy(dtype=np.float64)

        est_min = float(np.min(est_t))
        est_max = float(np.max(est_t))
        valid_mask = (gt_t >= est_min) & (gt_t <= est_max)
        gt_t_clipped = gt_t[valid_mask]
        gt_pos_clipped = gt_pos[valid_mask]

        if len(gt_t_clipped) < 2:
            out = _nan_row(sequence_name, "insufficient_overlap")
            if print_report:
                print("[EVAL] Not enough overlapping points for evaluation.")
            return out

        est_t_i, est_pos_i = _sort_est_timeseries_for_interp(est_t, est_pos)
        if len(est_t_i) < 2:
            out = _nan_row(sequence_name, "insufficient_est_samples")
            if print_report:
                print("[EVAL] Not enough distinct estimate times for interpolation.")
            return out

        matched_est_pos = match_predictions_to_gt(
            gt_t_clipped, gt_pos_clipped, est_t_i, est_pos_i
        )

        ate_x, ate_y, ate_z = calculate_per_axis_rmse(gt_pos_clipped, matched_est_pos)
        if use_3d and np.isfinite(ate_z):
            ate = calculate_ate_norm_rmse_3d(gt_pos_clipped, matched_est_pos)
        else:
            ate = calculate_ate_rmse_synced(gt_pos_clipped, matched_est_pos[:, :2])

        gt_xy = gt_pos_clipped[:, :2]
        matched_xy = matched_est_pos[:, :2]
        ahe_deg = calculate_absolute_heading_error(gt_xy, matched_xy)
        rpe_trans, rpe_rot = calculate_rpe_metrics_synced(gt_xy, matched_xy, window_m=1.0)

        gt_resampled, gt_len = resample_spatially(gt_xy, step=0.1)
        est_resampled, est_len = resample_spatially(matched_xy, step=0.1)

        len_err, fpe, frechet = calculate_shape_metrics(
            gt_resampled, est_resampled, gt_len, est_len
        )
        drift_pct = (fpe / gt_len) * 100 if gt_len > 0 else 0.0

        out = {
            "sequence_name": sequence_name,
            "skipped": "",
            "ate_m": ate,
            "ate_x_m": ate_x,
            "ate_y_m": ate_y,
            "ate_z_m": ate_z if use_3d else float("nan"),
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
            z_line = (
                f"ATE Z [m]:       {ate_z:.4f}\n"
                if use_3d and np.isfinite(ate_z)
                else ""
            )
            print(
                "--- EVALUATION METRICS ---\n"
                + f"ATE [m] (norm):  {ate:.4f}\n"
                + f"ATE X [m]:       {ate_x:.4f}\n"
                + f"ATE Y [m]:       {ate_y:.4f}\n"
                + z_line
                + f"AHE [deg]:       {ahe_deg:.4f}\n"
                + f"RPE Trans [%]:   {rpe_trans:.4f}\n"
                + f"RPE Rot [deg/m]: {rpe_rot:.4f}\n"
                + f"FPE [m]:         {fpe:.4f}\n"
                + f"Drift [%]:       {drift_pct:.4f}\n"
                + f"Length Err:      {len_err:.4f}\n"
                + f"Frechet [m]:     {frechet:.4f}\n"
                + f"GT Length [m]:   {gt_len:.4f}\n"
                + f"Path Length [m]: {est_len:.4f}\n"
                + "--------------------------\n"
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
        "ate_x_m",
        "ate_y_m",
        "ate_z_m",
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
            val = m[k]
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            lines.append(f"{k}={val:.6g}")
    return lines
