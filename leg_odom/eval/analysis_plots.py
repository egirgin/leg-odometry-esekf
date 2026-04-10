"""
Post-run analysis plots for EKF history CSV + optional merged recording + GT.

Column conventions match :mod:`leg_odom.eval.ekf_step_log` (``leg{i}_stance``,
``leg{i}_zupt_accepted``, ``leg{i}_v_w*``, euler ``*_deg``). Merged logs use
``foot_force_0``…``foot_force_3`` and ``t_abs``.

``filtered_states`` overlays position / velocity / attitude GT on the first three panels when
``gt_df`` supplies ``local_*`` (and body quaternions for Euler GT). Scalar trajectory metrics are
not written as a PNG (CSV only).

Primary API: :class:`EkfRunAnalysis` (mirrors legacy ``AnalysisAndEvaluation`` plot side).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.io.columns import IMU_BODY_QUAT_COLS, TIME_NANOSEC_COL, TIME_SEC_COL
from leg_odom.io.ground_truth import extract_position_ground_truth


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hist_time(hist: pd.DataFrame) -> np.ndarray:
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    if TIME_SEC_COL in hist.columns and TIME_NANOSEC_COL in hist.columns:
        return (
            hist[TIME_SEC_COL].to_numpy(dtype=np.float64)
            + hist[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9
        )
    raise ValueError(f"History DataFrame needs t_abs or {TIME_SEC_COL!r}/{TIME_NANOSEC_COL!r}.")


def _interp_columns(raw_t: np.ndarray, raw_y: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    m = np.isfinite(raw_t) & np.isfinite(raw_y)
    if not np.any(m):
        return np.full_like(t_query, np.nan, dtype=float)
    return np.interp(t_query, raw_t[m], raw_y[m], left=np.nan, right=np.nan)


def _gt_time_array(gt_df: pd.DataFrame | None) -> np.ndarray | None:
    if gt_df is None or gt_df.empty:
        return None
    if TIME_SEC_COL in gt_df.columns and TIME_NANOSEC_COL in gt_df.columns:
        return (
            gt_df[TIME_SEC_COL].to_numpy(dtype=np.float64)
            + gt_df[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9
        )
    if "t_abs" in gt_df.columns:
        return gt_df["t_abs"].to_numpy(dtype=np.float64)
    return None


def _stance_accepted_mask(hist: pd.DataFrame, leg_i: int) -> pd.Series:
    st = hist[f"leg{leg_i}_stance"]
    if st.dtype == object or st.dtype == bool:
        stance_b = st.astype(bool)
    else:
        stance_b = st.astype(np.int64) != 0
    acc = hist[f"leg{leg_i}_zupt_accepted"]
    acc_b = acc.fillna(0).astype(np.float64) >= 0.5
    return stance_b & acc_b


class EkfRunAnalysis:
    """
    Matplotlib analysis bundle for one EKF run (legacy ``AnalysisAndEvaluation``-style).

    All figures are written under ``output_path`` as ``*.png``.
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig: plt.Figure, filename: str, *, tight_layout: bool = True) -> None:
        path = self.output_path / f"{filename}.png"
        if tight_layout:
            fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_states(self, hist: pd.DataFrame, gt_df: pd.DataFrame | None = None) -> None:
        if hist.empty:
            return
        t = _hist_time(hist)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        gt_t = _gt_time_array(gt_df)

        # .to_numpy(): matplotlib + newer pandas no longer accept raw Series in plot()
        axes[0].plot(t, hist["p_x"].to_numpy(dtype=np.float64), label="p_x")
        axes[0].plot(t, hist["p_y"].to_numpy(dtype=np.float64), label="p_y")
        axes[0].plot(t, hist["p_z"].to_numpy(dtype=np.float64), label="p_z")
        if (
            gt_t is not None
            and gt_df is not None
            and {"local_x", "local_y", "local_z"}.issubset(gt_df.columns)
        ):
            gx = gt_df["local_x"].to_numpy(dtype=np.float64)
            gy = gt_df["local_y"].to_numpy(dtype=np.float64)
            gz = gt_df["local_z"].to_numpy(dtype=np.float64)
            axes[0].plot(
                t,
                _interp_columns(gt_t, gx, t),
                linestyle="--",
                alpha=0.75,
                color="C0",
                label="p_x GT",
            )
            axes[0].plot(
                t,
                _interp_columns(gt_t, gy, t),
                linestyle="--",
                alpha=0.75,
                color="C1",
                label="p_y GT",
            )
            axes[0].plot(
                t,
                _interp_columns(gt_t, gz, t),
                linestyle="--",
                alpha=0.75,
                color="C2",
                label="p_z GT",
            )
        axes[0].set_ylabel("Position [m]")
        axes[0].grid(True, alpha=0.4)
        axes[0].legend(loc="upper right", fontsize="small")

        axes[1].plot(t, hist["v_x"].to_numpy(dtype=np.float64), label="v_x")
        axes[1].plot(t, hist["v_y"].to_numpy(dtype=np.float64), label="v_y")
        axes[1].plot(t, hist["v_z"].to_numpy(dtype=np.float64), label="v_z")
        if gt_t is not None and gt_df is not None and {"local_x", "local_y", "local_z"}.issubset(
            gt_df.columns
        ):
            gx = gt_df["local_x"].to_numpy(dtype=np.float64)
            gy = gt_df["local_y"].to_numpy(dtype=np.float64)
            gz = gt_df["local_z"].to_numpy(dtype=np.float64)
            gvx = np.gradient(gx, gt_t)
            gvy = np.gradient(gy, gt_t)
            gvz = np.gradient(gz, gt_t)
            axes[1].plot(
                t,
                _interp_columns(gt_t, gvx, t),
                linestyle="--",
                alpha=0.75,
                color="C0",
                label="v_x GT",
            )
            axes[1].plot(
                t,
                _interp_columns(gt_t, gvy, t),
                linestyle="--",
                alpha=0.75,
                color="C1",
                label="v_y GT",
            )
            axes[1].plot(
                t,
                _interp_columns(gt_t, gvz, t),
                linestyle="--",
                alpha=0.75,
                color="C2",
                label="v_z GT",
            )
        axes[1].set_ylabel("Velocity [m/s]")
        axes[1].grid(True, alpha=0.4)
        axes[1].legend(loc="upper right", fontsize="small")

        roll_c = "roll_deg" if "roll_deg" in hist.columns else "roll"
        pitch_c = "pitch_deg" if "pitch_deg" in hist.columns else "pitch"
        yaw_c = "yaw_deg" if "yaw_deg" in hist.columns else "yaw"
        use_deg = "roll_deg" in hist.columns
        rot_ylabel = "Rotation [deg]" if use_deg else "Rotation [rad]"
        axes[2].plot(t, hist[roll_c].to_numpy(dtype=np.float64), label="Roll")
        axes[2].plot(t, hist[pitch_c].to_numpy(dtype=np.float64), label="Pitch")
        axes[2].plot(t, hist[yaw_c].to_numpy(dtype=np.float64), label="Yaw")
        if (
            gt_t is not None
            and gt_df is not None
            and all(c in gt_df.columns for c in IMU_BODY_QUAT_COLS)
        ):
            qx = gt_df["ori_qx"].to_numpy(dtype=np.float64)
            qy = gt_df["ori_qy"].to_numpy(dtype=np.float64)
            qz = gt_df["ori_qz"].to_numpy(dtype=np.float64)
            qw = gt_df["ori_qw"].to_numpy(dtype=np.float64)
            quat = np.stack((qx, qy, qz, qw), axis=1)
            euler = Rotation.from_quat(quat).as_euler("zyx")
            for j in range(3):
                euler[:, j] = np.unwrap(euler[:, j])
            yaw_a, pitch_a, roll_a = euler[:, 0], euler[:, 1], euler[:, 2]
            if use_deg:
                yaw_a = np.rad2deg(yaw_a)
                pitch_a = np.rad2deg(pitch_a)
                roll_a = np.rad2deg(roll_a)
            axes[2].plot(
                t,
                _interp_columns(gt_t, roll_a, t),
                linestyle="--",
                alpha=0.75,
                color="C0",
                label="Roll GT",
            )
            axes[2].plot(
                t,
                _interp_columns(gt_t, pitch_a, t),
                linestyle="--",
                alpha=0.75,
                color="C1",
                label="Pitch GT",
            )
            axes[2].plot(
                t,
                _interp_columns(gt_t, yaw_a, t),
                linestyle="--",
                alpha=0.75,
                color="C2",
                label="Yaw GT",
            )
        axes[2].set_ylabel(rot_ylabel)
        axes[2].grid(True, alpha=0.4)
        axes[2].legend(loc="upper right", fontsize="small")

        ax_acc = axes[3]
        ax_acc.plot(
            t, hist["bax"].to_numpy(dtype=np.float64), label="bax", linestyle="--", color="C0"
        )
        ax_acc.plot(
            t, hist["bay"].to_numpy(dtype=np.float64), label="bay", linestyle="--", color="C1"
        )
        ax_acc.plot(
            t, hist["baz"].to_numpy(dtype=np.float64), label="baz", linestyle="--", color="C2"
        )
        ax_acc.set_ylabel("Accel bias [m/s²]")
        ax_acc.grid(True, alpha=0.4)

        ax_gyro = ax_acc.twinx()
        ax_gyro.plot(t, hist["bgx"].to_numpy(dtype=np.float64), label="bgx", color="C3")
        ax_gyro.plot(t, hist["bgy"].to_numpy(dtype=np.float64), label="bgy", color="C4")
        ax_gyro.plot(t, hist["bgz"].to_numpy(dtype=np.float64), label="bgz", color="C5")
        ax_gyro.set_ylabel("Gyro bias [rad/s]")
        ax_gyro.spines["right"].set_color("#888888")
        ax_gyro.tick_params(axis="y", colors="#555555")

        bias_lines = ax_acc.get_lines() + ax_gyro.get_lines()
        ax_acc.legend(
            bias_lines,
            [ln.get_label() for ln in bias_lines],
            loc="upper right",
            ncol=2,
            fontsize="small",
        )

        axes[-1].set_xlabel("Time [s]")
        self._save_fig(fig, "filtered_states")

    def plot_contacts_grf(self, hist: pd.DataFrame, merged: pd.DataFrame) -> None:
        if hist.empty or merged.empty:
            return
        t = _hist_time(hist)
        raw_t = merged["t_abs"].to_numpy(dtype=np.float64) if "t_abs" in merged.columns else None
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("GRF (green = stance & ZUPT accepted, red = swing / rejected)")

        for i in range(4):
            col = f"foot_force_{i}"
            if raw_t is not None and col in merged.columns:
                force = _interp_columns(raw_t, merged[col].to_numpy(dtype=np.float64), t)
            else:
                force = np.full_like(t, np.nan, dtype=float)

            accepted_mask = _stance_accepted_mask(hist, i).to_numpy()
            rejected_mask = ~accepted_mask

            axes[i].scatter(
                t[rejected_mask],
                force[rejected_mask],
                color="red",
                s=4,
                label="Swing / rejected" if i == 0 else "",
            )
            axes[i].scatter(
                t[accepted_mask],
                force[accepted_mask],
                color="green",
                s=4,
                label="Stance + accepted" if i == 0 else "",
            )
            axes[i].set_ylabel(f"Leg {i} GRF [N]")
            axes[i].grid(True, alpha=0.4)
            if i == 0:
                axes[i].legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        self._save_fig(fig, "contacts_grf")

    def plot_contacts_foot_velocity_world(self, hist: pd.DataFrame) -> None:
        if hist.empty:
            return
        t = _hist_time(hist)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Foot speed |v_w| world (green = stance & ZUPT accepted)")

        for i in range(4):
            vx = hist[f"leg{i}_v_wx"].to_numpy(dtype=np.float64)
            vy = hist[f"leg{i}_v_wy"].to_numpy(dtype=np.float64)
            vz = hist[f"leg{i}_v_wz"].to_numpy(dtype=np.float64)
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

            accepted_mask = _stance_accepted_mask(hist, i).to_numpy()
            rejected_mask = ~accepted_mask

            axes[i].scatter(
                t[rejected_mask],
                v_mag[rejected_mask],
                color="red",
                s=4,
                label="Swing / rejected" if i == 0 else "",
            )
            axes[i].scatter(
                t[accepted_mask],
                v_mag[accepted_mask],
                color="green",
                s=4,
                label="Stance + accepted" if i == 0 else "",
            )
            axes[i].set_ylabel(f"Leg {i} |v_w| [m/s]")
            axes[i].grid(True, alpha=0.4)
            if i == 0:
                axes[i].legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        self._save_fig(fig, "contacts_foot_velocity_world")

    def plot_trajectory_xy_and_z(
        self,
        hist: pd.DataFrame,
        gt_df: pd.DataFrame | None,
    ) -> None:
        if hist.empty:
            return
        t = _hist_time(hist)
        px = hist["p_x"].to_numpy(dtype=np.float64)
        py = hist["p_y"].to_numpy(dtype=np.float64)
        pz = hist["p_z"].to_numpy(dtype=np.float64)

        px0, py0 = px[0], py[0]
        ex = px - px0
        ey = py - py0

        fig, (ax_xy, ax_z) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1.0]}
        )

        ax_xy.plot(ex, ey, label="Estimated", color="C0", linewidth=2)

        min_x, max_x = float(np.min(ex)), float(np.max(ex))
        min_y, max_y = float(np.min(ey)), float(np.max(ey))

        if gt_df is not None and not gt_df.empty and {"local_x", "local_y"}.issubset(gt_df.columns):
            gx = gt_df["local_x"].to_numpy(dtype=np.float64)
            gy = gt_df["local_y"].to_numpy(dtype=np.float64)
            gx0, gy0 = gx[0], gy[0]
            gxs, gys = gx - gx0, gy - gy0
            ax_xy.plot(gxs, gys, label="Ground truth", color="black", linestyle="--", linewidth=2)
            min_x = min(min_x, float(np.min(gxs)))
            max_x = max(max_x, float(np.max(gxs)))
            min_y = min(min_y, float(np.min(gys)))
            max_y = max(max_y, float(np.max(gys)))

        ax_xy.plot(0, 0, "ko", markersize=8, label="Start")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title("Top-down trajectory")
        ax_xy.grid(True, alpha=0.4)
        ax_xy.legend(loc="best")
        ax_xy.set_aspect("equal")

        span_x = max_x - min_x
        span_y = max_y - min_y
        max_span = max(span_x, span_y, 1e-6)
        pad = max_span * 0.1
        side = max_span + pad
        mid_x = (max_x + min_x) / 2.0
        mid_y = (max_y + min_y) / 2.0
        h = side / 2.0
        ax_xy.set_xlim(mid_x - h, mid_x + h)
        ax_xy.set_ylim(mid_y - h, mid_y + h)

        ax_z.plot(t, pz, label="Est. z", color="C0", linewidth=1.5)
        if gt_df is not None and not gt_df.empty and "local_z" in gt_df.columns:
            try:
                gt_t = _gt_time_array(gt_df)
                if gt_t is None:
                    raise ValueError("no GT time column")
                gz = gt_df["local_z"].to_numpy(dtype=np.float64)
                gz_i = _interp_columns(gt_t, gz, t)
                ax_z.plot(
                    t,
                    gz_i,
                    label="GT z (interp.)",
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                )
            except Exception:
                pass
        ax_z.set_xlabel("Time [s]")
        ax_z.set_ylabel("z [m]")
        ax_z.set_title("Vertical position")
        ax_z.grid(True, alpha=0.4)
        ax_z.legend(loc="best")

        self._save_fig(fig, "trajectory_xy_z")

    def save_all(
        self,
        hist: pd.DataFrame,
        merged: pd.DataFrame | None = None,
        gt_df: pd.DataFrame | None = None,
        *,
        metrics_row: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the full PNG bundle into ``self.output_path``."""
        _ = metrics_row  # kept for API compatibility; metrics go to CSV only (no metrics PNG)
        self.plot_states(hist, gt_df=gt_df)
        if merged is not None and not merged.empty:
            self.plot_contacts_grf(hist, merged)
        self.plot_contacts_foot_velocity_world(hist)
        self.plot_trajectory_xy_and_z(hist, gt_df)


def save_analysis_bundle(
    hist: pd.DataFrame,
    merged: pd.DataFrame | None,
    gt_df: pd.DataFrame | None,
    output_dir: Path,
    *,
    metrics_row: Mapping[str, Any] | None = None,
) -> None:
    """Functional wrapper: :class:`EkfRunAnalysis` on ``output_dir``."""
    EkfRunAnalysis(output_dir).save_all(
        hist, merged=merged, gt_df=gt_df, metrics_row=metrics_row
    )


def _load_hist(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EKF history analysis plots (leg_odom).")
    p.add_argument("--ekf-csv", type=Path, required=True, help="Path to ekf_history_*.csv")
    p.add_argument("--merged-csv", type=Path, default=None)
    p.add_argument("--gt-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    hist = _load_hist(args.ekf_csv)
    merged = pd.read_csv(args.merged_csv) if args.merged_csv and args.merged_csv.is_file() else None
    gt_df: pd.DataFrame | None = None
    if args.gt_csv and args.gt_csv.is_file():
        gt_df = extract_position_ground_truth(pd.read_csv(args.gt_csv))
    elif merged is not None:
        gt_df = extract_position_ground_truth(merged)

    out = _ensure_dir(args.output_dir or args.ekf_csv.parent)
    EkfRunAnalysis(out).save_all(hist, merged=merged, gt_df=gt_df)
    print(f"[analysis_plots] Wrote figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
