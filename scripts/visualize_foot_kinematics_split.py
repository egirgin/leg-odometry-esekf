#!/usr/bin/env python3
"""
Visualize one leg's body-frame foot position, foot velocity (w.r.t. body), estimated
joint torques (``motor_*_tau_est`` for that leg), and GRF from a Tartanground split
directory (imu.csv + *_bag.csv).

Local testing only; does not change leg_odom or configs. Figures use a compact,
poster-friendly layout (colorblind-safe xyz colors, 300 dpi default when saving).

Optional second figure: by default **2×1** (‖v‖² and GRF). With ``--accel-squared``,
**3×1** adds ‖a‖² (a = dv/dt via ``numpy.gradient``). Use ``--save-norms PATH``; you may
pass only ``--save-norms`` to skip the 4×1 figure.

Example::

    cd /path/to/async_ekf_workspace
    python3 scripts/visualize_foot_kinematics_split.py --save figure.png
    python3 scripts/visualize_foot_kinematics_split.py --save figure.png --save-norms norms.png
    python3 scripts/visualize_foot_kinematics_split.py --save-norms norms.png
    python3 scripts/visualize_foot_kinematics_split.py --save-norms norms.png --accel-squared
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from leg_odom.io.columns import (
    FOOT_FORCE_COLS,
    motor_position_cols,
    motor_torque_cols,
    motor_velocity_cols,
)
from leg_odom.io.split_imu_bag import load_prepared_split_sequence
from leg_odom.kinematics.anymal import AnymalKinematics

_IMU_GYRO = ("gyro_x", "gyro_y", "gyro_z")

# Okabe–Ito–style palette: distinguish x, y, z while staying colorblind-friendly.
_COLORS_XYZ = ("#D55E00", "#0072B2", "#009E73")
_COLOR_GRF = "#4A4A4A"
_COLOR_V2 = "#6A4C93"
_COLOR_ANORM = "#1982C4"
# ANYmal leg joint order: HAA, HFE, KFE (same colors as xyz for consistency).
_TAU_LABELS = (r"$\tau_{\mathrm{HAA}}$", r"$\tau_{\mathrm{HFE}}$", r"$\tau_{\mathrm{KFE}}$")

_POSTER_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": (0, (1, 2.5)),
    "grid.linewidth": 0.7,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.9,
    "lines.linewidth": 1.35,
    "axes.grid": True,
    "axes.axisbelow": True,
}


def _slice_middle_window(df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    t = df["t_abs"].to_numpy(dtype=float)
    t_mid = 0.5 * (float(np.min(t)) + float(np.max(t)))
    half = 0.5 * float(window_s)
    mask = (df["t_abs"] >= t_mid - half) & (df["t_abs"] <= t_mid + half)
    return df.loc[mask].reset_index(drop=True)


def _compute_leg_series(
    seg: pd.DataFrame,
    kin: AnymalKinematics,
    leg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())
    jpl = kin.joints_per_leg
    n = len(seg)
    p_out = np.zeros((n, 3), dtype=np.float64)
    v_out = np.zeros((n, 3), dtype=np.float64)
    tau_out = np.zeros((n, 3), dtype=np.float64)
    grf_out = np.zeros(n, dtype=np.float64)
    force_col = FOOT_FORCE_COLS[leg]

    for k in range(n):
        row = seg.iloc[k]
        gyro = row[list(_IMU_GYRO)].to_numpy(dtype=np.float64)
        q_all = row[motor_cols].to_numpy(dtype=np.float64)
        dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        sl = slice(leg * jpl, (leg + 1) * jpl)
        q_leg = q_all[sl]
        dq_leg = dq_all[sl]
        tau_leg = tau_all[sl]
        pb = np.asarray(kin.fk(leg, q_leg), dtype=np.float64).reshape(3)
        jac = np.asarray(kin.J_analytical(leg, q_leg), dtype=np.float64).reshape(3, jpl)
        qd = np.asarray(dq_leg, dtype=np.float64).reshape(jpl)
        v_foot_body = np.cross(gyro, pb) + jac @ qd
        p_out[k, :] = pb
        v_out[k, :] = v_foot_body
        tau_out[k, :] = np.asarray(tau_leg, dtype=np.float64).reshape(3)
        grf_out[k] = float(row.get(force_col, 0.0))

    return seg["t_abs"].to_numpy(dtype=np.float64), p_out, v_out, tau_out, grf_out


def _foot_v_norm_squared(v_body: np.ndarray) -> np.ndarray:
    v_body = np.asarray(v_body, dtype=np.float64)
    return np.sum(v_body * v_body, axis=1)


def _foot_accel_norm_squared(t_abs: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """‖a‖² with a_j = ∂v_j/∂t (``numpy.gradient`` vs log time)."""
    t_abs = np.asarray(t_abs, dtype=np.float64)
    v_body = np.asarray(v_body, dtype=np.float64)
    a = np.stack([np.gradient(v_body[:, j], t_abs) for j in range(3)], axis=1)
    return np.sum(a * a, axis=1)


def _plot_foot_speed_accel_figure(
    *,
    t_plot: np.ndarray,
    v_sq: np.ndarray,
    grf: np.ndarray,
    a_norm_squared: np.ndarray | None = None,
) -> plt.Figure:
    """2×1 (‖v‖², GRF) or 3×1 if ``a_norm_squared`` is set (adds ‖a‖² row)."""
    with plt.rc_context(rc=_POSTER_RC):
        if a_norm_squared is None:
            fig, (ax0, ax1) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(5.8, 4.9),
                gridspec_kw={"hspace": 0.11},
            )
            ax0.plot(t_plot, v_sq, color=_COLOR_V2, clip_on=False)
            ax0.set_ylabel(r"$\|\mathbf{v}\|^2$ / (m$^2$s$^{-2}$)", labelpad=6)

            ax1.plot(t_plot, grf, color=_COLOR_GRF, clip_on=False)
            ax1.set_ylabel(r"GRF / N", labelpad=6)
            ax1.set_xlabel(r"Time / s", labelpad=8)
            fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.11)
        else:
            fig, (ax0, ax1, ax2) = plt.subplots(
                3,
                1,
                sharex=True,
                figsize=(5.8, 6.9),
                gridspec_kw={"hspace": 0.11},
            )
            ax0.plot(t_plot, v_sq, color=_COLOR_V2, clip_on=False)
            ax0.set_ylabel(r"$\|\mathbf{v}\|^2$ / (m$^2$s$^{-2}$)", labelpad=6)

            ax1.plot(t_plot, a_norm_squared, color=_COLOR_ANORM, clip_on=False)
            ax1.set_ylabel(r"$\|\mathbf{a}\|^2$ / (m$^2$s$^{-4}$)", labelpad=6)

            ax2.plot(t_plot, grf, color=_COLOR_GRF, clip_on=False)
            ax2.set_ylabel(r"GRF / N", labelpad=6)
            ax2.set_xlabel(r"Time / s", labelpad=8)
            fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.09)

    return fig


def _plot_foot_kinematics_figure(
    *,
    t_plot: np.ndarray,
    p_body: np.ndarray,
    v_body: np.ndarray,
    tau_leg: np.ndarray,
    grf: np.ndarray,
) -> plt.Figure:
    """Matplotlib figure: position, velocity, joint torque (est.), GRF (4×1)."""
    from matplotlib.lines import Line2D

    with plt.rc_context(rc=_POSTER_RC):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=(5.8, 8.6),
            gridspec_kw={"hspace": 0.11},
        )

        comp_labels = (r"$x$", r"$y$", r"$z$")
        for ax, arr, ylabel in (
            (ax0, p_body, r"Position / m"),
            (ax1, v_body, r"Velocity / m$\cdot$s$^{-1}$"),
        ):
            for j in range(3):
                ax.plot(
                    t_plot,
                    arr[:, j],
                    color=_COLORS_XYZ[j],
                    label=comp_labels[j],
                    clip_on=False,
                )
            ax.set_ylabel(ylabel, labelpad=6)

        for j in range(3):
            ax2.plot(
                t_plot,
                tau_leg[:, j],
                color=_COLORS_XYZ[j],
                label=_TAU_LABELS[j],
                clip_on=False,
            )
        ax2.set_ylabel(r"Torque / Nm", labelpad=6)
        ax2.legend(loc="upper right", fontsize=8, frameon=False, ncol=3, columnspacing=0.8)

        ax3.plot(t_plot, grf, color=_COLOR_GRF, clip_on=False)
        ax3.set_ylabel(r"GRF / N", labelpad=6)
        ax3.set_xlabel(r"Time / s", labelpad=8)

        legend_elems = [
            Line2D([0], [0], color=_COLORS_XYZ[j], lw=2, label=comp_labels[j]) for j in range(3)
        ]
        fig.legend(
            handles=legend_elems,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.997),
            columnspacing=1.6,
            handletextpad=0.6,
        )

        fig.subplots_adjust(left=0.15, right=0.98, top=0.91, bottom=0.06)

    return fig


def main() -> None:
    default_seq = "/home/girgine/Documents/leg-odometry/iros/data_anymal"
    p = argparse.ArgumentParser(
        description="Plot foot position, velocity, joint torques (est.), and GRF for one leg (split CSV).",
    )
    p.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path(default_seq),
        help=f"Directory with imu.csv and *_bag.csv (default: {default_seq})",
    )
    p.add_argument("--leg", type=int, default=0, choices=(0, 1, 2, 3), help="Leg index 0=LF .. 3=RH")
    p.add_argument("--window-s", type=float, default=10.0, help="Seconds of data around temporal midpoint")
    p.add_argument(
        "--no-sanitize-imu",
        action="store_true",
        help="Pass sanitize_imu=False to load_prepared_split_sequence (skip FLU checks)",
    )
    p.add_argument("--save", type=Path, default=None, help="If set, save the 4×1 kinematics figure to this path")
    p.add_argument(
        "--save-norms",
        type=Path,
        default=None,
        help="If set, save a second figure: 2×1 (‖v‖², GRF) by default; 3×1 with --accel-squared",
    )
    p.add_argument(
        "--accel-squared",
        action="store_true",
        help="On the norms figure, add a ‖a‖² row (a = dv/dt); default is 2×1 without acceleration",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution when saving (default: 300, poster-friendly)",
    )
    args = p.parse_args()

    seq_dir = args.sequence_dir.expanduser().resolve()
    df, _hz, _gt, _accel_gc = load_prepared_split_sequence(
        seq_dir,
        verbose=True,
        sanitize_imu=not args.no_sanitize_imu,
    )
    seg = _slice_middle_window(df, args.window_s)
    if seg.empty:
        raise SystemExit("Middle window slice is empty; check window-s and data length.")

    kin = AnymalKinematics()
    t, p_body, v_body, tau_leg, grf = _compute_leg_series(seg, kin, args.leg)
    t0 = float(t[0])
    t_plot = t - t0
    v_sq = _foot_v_norm_squared(v_body)

    need_main = args.save is not None or args.save_norms is None
    need_norms = args.save_norms is not None or args.save is None

    fig_main = None
    fig_norms = None
    if need_main:
        fig_main = _plot_foot_kinematics_figure(
            t_plot=t_plot,
            p_body=p_body,
            v_body=v_body,
            tau_leg=tau_leg,
            grf=grf,
        )
    if need_norms:
        a_sq = _foot_accel_norm_squared(t, v_body) if args.accel_squared else None
        fig_norms = _plot_foot_speed_accel_figure(
            t_plot=t_plot,
            v_sq=v_sq,
            grf=grf,
            a_norm_squared=a_sq,
        )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        assert fig_main is not None
        fig_main.savefig(args.save, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {args.save}")
    if args.save_norms is not None:
        args.save_norms.parent.mkdir(parents=True, exist_ok=True)
        assert fig_norms is not None
        fig_norms.savefig(args.save_norms, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {args.save_norms}")

    if args.save is None and args.save_norms is None:
        plt.show()


if __name__ == "__main__":
    main()
