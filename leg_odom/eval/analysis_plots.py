"""
Post-run analysis plots for EKF history CSV + optional merged recording + GT.

Column conventions match :mod:`leg_odom.eval.ekf_step_log` (``leg{i}_stance``,
``leg{i}_zupt_accepted``, ``leg{i}_v_w*``, euler ``*_deg``). Merged logs use
``foot_force_0``…``foot_force_3`` and ``t_abs``.

Primary API: :class:`EkfRunAnalysis` (mirrors legacy ``AnalysisAndEvaluation`` plot side).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leg_odom.io.ground_truth import extract_position_ground_truth


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hist_time(hist: pd.DataFrame) -> np.ndarray:
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    if "timestamp_sec" in hist.columns:
        ns = hist["timestamp_nanosec"].to_numpy(dtype=np.float64) if "timestamp_nanosec" in hist.columns else 0.0
        return hist["timestamp_sec"].to_numpy(dtype=np.float64) + ns * 1e-9
    raise ValueError("History DataFrame needs t_abs or timestamp_sec.")


def _interp_columns(raw_t: np.ndarray, raw_y: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    m = np.isfinite(raw_t) & np.isfinite(raw_y)
    if not np.any(m):
        return np.full_like(t_query, np.nan, dtype=float)
    return np.interp(t_query, raw_t[m], raw_y[m], left=np.nan, right=np.nan)


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

    def _save_fig(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_path / f"{filename}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def plot_states(self, hist: pd.DataFrame) -> None:
        if hist.empty:
            return
        t = _hist_time(hist)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # .to_numpy(): matplotlib + newer pandas no longer accept raw Series in plot()
        axes[0].plot(t, hist["p_x"].to_numpy(dtype=np.float64), label="p_x")
        axes[0].plot(t, hist["p_y"].to_numpy(dtype=np.float64), label="p_y")
        axes[0].plot(t, hist["p_z"].to_numpy(dtype=np.float64), label="p_z")
        axes[0].set_ylabel("Position [m]")
        axes[0].grid(True, alpha=0.4)
        axes[0].legend(loc="upper right")

        axes[1].plot(t, hist["v_x"].to_numpy(dtype=np.float64), label="v_x")
        axes[1].plot(t, hist["v_y"].to_numpy(dtype=np.float64), label="v_y")
        axes[1].plot(t, hist["v_z"].to_numpy(dtype=np.float64), label="v_z")
        axes[1].set_ylabel("Velocity [m/s]")
        axes[1].grid(True, alpha=0.4)
        axes[1].legend(loc="upper right")

        roll_c = "roll_deg" if "roll_deg" in hist.columns else "roll"
        pitch_c = "pitch_deg" if "pitch_deg" in hist.columns else "pitch"
        yaw_c = "yaw_deg" if "yaw_deg" in hist.columns else "yaw"
        axes[2].plot(t, hist[roll_c].to_numpy(dtype=np.float64), label="Roll")
        axes[2].plot(t, hist[pitch_c].to_numpy(dtype=np.float64), label="Pitch")
        axes[2].plot(t, hist[yaw_c].to_numpy(dtype=np.float64), label="Yaw")
        axes[2].set_ylabel("Rotation [deg]")
        axes[2].grid(True, alpha=0.4)
        axes[2].legend(loc="upper right")

        axes[3].plot(t, hist["bax"].to_numpy(dtype=np.float64), label="bax", linestyle="--")
        axes[3].plot(t, hist["bay"].to_numpy(dtype=np.float64), label="bay", linestyle="--")
        axes[3].plot(t, hist["baz"].to_numpy(dtype=np.float64), label="baz", linestyle="--")
        axes[3].plot(t, hist["bgx"].to_numpy(dtype=np.float64), label="bgx")
        axes[3].plot(t, hist["bgy"].to_numpy(dtype=np.float64), label="bgy")
        axes[3].plot(t, hist["bgz"].to_numpy(dtype=np.float64), label="bgz")
        axes[3].set_ylabel("IMU biases")
        axes[3].grid(True, alpha=0.4)
        axes[3].legend(loc="upper right")

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
                gt_t = (
                    gt_df["ros_sec"].to_numpy(dtype=np.float64)
                    + gt_df["ros_nanosec"].to_numpy(dtype=np.float64) * 1e-9
                    if "ros_sec" in gt_df.columns
                    else gt_df["t_abs"].to_numpy(dtype=np.float64)
                )
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
    ) -> None:
        """Write the full PNG bundle into ``self.output_path``."""
        self.plot_states(hist)
        if merged is not None and not merged.empty:
            self.plot_contacts_grf(hist, merged)
        self.plot_contacts_foot_velocity_world(hist)
        self.plot_trajectory_xy_and_z(hist, gt_df)


def save_analysis_bundle(
    hist: pd.DataFrame,
    merged: pd.DataFrame | None,
    gt_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Functional wrapper: :class:`EkfRunAnalysis` on ``output_dir``."""
    EkfRunAnalysis(output_dir).save_all(hist, merged=merged, gt_df=gt_df)


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
