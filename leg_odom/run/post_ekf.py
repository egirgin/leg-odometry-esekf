"""
After EKF: plot PNGs + ``evaluation_metrics.csv`` under ``<run_dir>/<output_subdir>/``.

Uses ``dataset[0]`` (single-sequence Tartanground) for merged frames and GT when plotting.

``main.py`` calls this once and writes artifacts under ``plots/``.
Figures are written directly under ``plots/`` (no extra sequence subfolder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from leg_odom.eval.analysis_plots import EkfRunAnalysis
from leg_odom.eval.metrics import TrajectoryEvaluator
from leg_odom.io.columns import IMU_BODY_QUAT_COLS
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.ekf_process import EkfProcessSummary


def run_post_ekf_analysis_and_eval(
    run_dir: Path,
    resolved_cfg: Mapping[str, Any],
    summary: EkfProcessSummary,
    *,
    output_subdir: str = "plots",
) -> None:
    """
    When ``summary.ekf_history_csv`` is set, write
    ``<run_dir>/<output_subdir>/*.png`` and one evaluation row;
    then save ``<run_dir>/<output_subdir>/evaluation_metrics.csv``.

    Default ``output_subdir`` is ``plots`` (see ``main.py``).
    """
    run_dir = Path(run_dir)
    out_root = run_dir / output_subdir
    dataset = build_leg_odometry_dataset(resolved_cfg)
    rec0 = dataset[0]

    rows: list[dict[str, Any]] = []
    if summary.ekf_history_csv:
        hist = pd.read_csv(summary.ekf_history_csv)
        merged = rec0.frames
        gt_df = rec0.position_ground_truth
        if (
            gt_df is not None
            and not gt_df.empty
            and merged is not None
            and not merged.empty
            and len(gt_df) == len(merged)
        ):
            gt_df = gt_df.copy()
            for c in IMU_BODY_QUAT_COLS:
                if c in merged.columns and c not in gt_df.columns:
                    gt_df[c] = merged[c].to_numpy()

        row = TrajectoryEvaluator().evaluate(
            hist,
            gt_df,
            sequence_name=summary.sequence_name,
            print_report=False,
        )
        rows.append(row)

        EkfRunAnalysis(out_root).save_all(
            hist,
            merged=merged if merged is not None and not merged.empty else None,
            gt_df=gt_df if not gt_df.empty else None,
            metrics_row=row,
        )

    if rows:
        csv_path = out_root / "evaluation_metrics.csv"
        TrajectoryEvaluator.write_metrics_csv(csv_path, rows)
        print(
            f"[post_ekf] Wrote {len(rows)} evaluation row(s) and figures under {out_root} → {csv_path}"
        )
