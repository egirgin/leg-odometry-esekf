"""
Live matplotlib monitor for EKF runs: 2D trajectory + GT, camera placeholder, base z, velocity.

**Camera**: not wired for Tartanground / current split logs (no ego video in the dataset
pipeline). The top-right axes shows a static placeholder; see the plan section *Live visualizer
and camera (Tartanground)* in ``TARTANGROUND_EKF_REFACTOR_PLAN.md``.

Offline replay uses ``t_abs`` from the merged timeline, a configurable sliding x-window on the
z/velocity panels, optional **GT** ``p_z`` and **GT velocity** (from differentiated position),
a **unit-circle heading** panel (estimate vs GT yaw when available; GT yaw from body quaternions
when present else velocity heading, with **unwrapped** angles before time interpolation), and a bottom progress bar
for ``t_start`` … ``t_end``. Ring-buffer length, optional ``video_path``, and optional
``update_hz`` (matplotlib refresh throttling vs dataset rate) are set from experiment YAML under
``run.debug.live_visualizer`` when wired from :mod:`leg_odom.run.ekf_process`.
"""

from __future__ import annotations

import math
from collections import deque

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation

from leg_odom.io.columns import IMU_BODY_QUAT_COLS, TIME_NANOSEC_COL, TIME_SEC_COL

SLIDING_TIME_WINDOW_S = 60.0


def _viz_stride_from_rates(update_hz: float | None, dataset_hz: float) -> int:
    """
    Steps between matplotlib redraws. ``update_hz`` None → every step.

    Effective refresh rate is ``min(update_hz, dataset_hz)`` so the UI cannot outpace the
    merged timeline sampling rate.
    """
    if update_hz is None:
        return 1
    ds = float(dataset_hz)
    if not math.isfinite(ds) or ds <= 0.0:
        return 1
    uh = float(update_hz)
    if not math.isfinite(uh) or uh <= 0.0:
        return 1
    effective = min(uh, ds)
    return max(1, int(round(ds / effective)))


def _unwrap_heading_series(yaw: np.ndarray) -> np.ndarray:
    """
    Remove 2π jumps (e.g. SciPy ``zyx`` yaw in ``(-π, π]``) so :func:`numpy.interp` follows
    physical rotation instead of cutting across the circle.
    """
    y = np.asarray(yaw, dtype=np.float64)
    out = y.copy()
    m = np.isfinite(out)
    if np.any(m):
        out[m] = np.unwrap(out[m])
    return out


def _gt_time_array(gt_df: pd.DataFrame) -> np.ndarray | None:
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


def _gt_yaw_series(
    groundtruth_df: pd.DataFrame,
    gvx: np.ndarray,
    gvy: np.ndarray,
) -> np.ndarray:
    """
    Planar yaw ``ψ`` (rad), same length as ``gt_t``, aligned with EKF logging: SciPy ``zyx`` Euler,
    yaw = first component.

    Preference: ``yaw_rad`` → ``yaw_deg`` → ``yaw`` (heuristic deg vs rad) → body quaternion
    columns ``IMU_BODY_QUAT_COLS`` if all present → ``atan2(d local_y/dt, d local_x/dt)`` from
    ``np.gradient``.
    """
    if "yaw_rad" in groundtruth_df.columns:
        return groundtruth_df["yaw_rad"].to_numpy(dtype=np.float64)
    if "yaw_deg" in groundtruth_df.columns:
        return np.deg2rad(groundtruth_df["yaw_deg"].to_numpy(dtype=np.float64))
    if "yaw" in groundtruth_df.columns:
        y = groundtruth_df["yaw"].to_numpy(dtype=np.float64)
        if np.nanmax(np.abs(y)) > 4.0 * np.pi:
            return np.deg2rad(y)
        return y
    if all(c in groundtruth_df.columns for c in IMU_BODY_QUAT_COLS):
        qx = groundtruth_df["ori_qx"].to_numpy(dtype=np.float64)
        qy = groundtruth_df["ori_qy"].to_numpy(dtype=np.float64)
        qz = groundtruth_df["ori_qz"].to_numpy(dtype=np.float64)
        qw = groundtruth_df["ori_qw"].to_numpy(dtype=np.float64)
        quat = np.stack((qx, qy, qz, qw), axis=1)
        # Same convention as EKF heading in :mod:`leg_odom.run.ekf_process` (``zyx`` yaw).
        return Rotation.from_quat(quat).as_euler("zyx")[:, 0]
    return np.arctan2(gvy, gvx)


def _prepare_gt_timeseries(groundtruth_df: pd.DataFrame | None) -> tuple[np.ndarray, ...] | None:
    """
    Return ``(gt_t, gt_pz, gt_vx, gt_vy, gt_vz, gt_yaw)`` for plotting, or None if insufficient GT.

    Velocities are ``np.gradient`` of world positions w.r.t. time (embedded logs rarely ship v).
    ``gt_yaw`` uses :func:`_gt_yaw_series` (orientation columns or velocity heading), then
    :func:`_unwrap_heading_series` so ``np.interp`` in :meth:`LiveVisualizer.update` does not
    shortcut across ±π wraps.
    """
    if groundtruth_df is None or groundtruth_df.empty:
        return None
    gt_t = _gt_time_array(groundtruth_df)
    if gt_t is None or len(gt_t) < 2:
        return None
    need = ("local_x", "local_y", "local_z")
    if not all(c in groundtruth_df.columns for c in need):
        return None

    gx = groundtruth_df["local_x"].to_numpy(dtype=np.float64)
    gy = groundtruth_df["local_y"].to_numpy(dtype=np.float64)
    gz = groundtruth_df["local_z"].to_numpy(dtype=np.float64)
    gvx = np.gradient(gx, gt_t)
    gvy = np.gradient(gy, gt_t)
    gvz = np.gradient(gz, gt_t)
    gt_yaw = _unwrap_heading_series(_gt_yaw_series(groundtruth_df, gvx, gvy))
    return (gt_t, gz, gvx, gvy, gvz, gt_yaw)


class LiveVisualizer:
    """
    Figure: trajectory (+ optional GT), camera placeholder, **p_z** vs time with **heading** on a
    unit circle below (estimate vs GT yaw), **one** velocity axes with six lines (est + GT
    ``vx,vy,vz``), and a bottom progress bar for ``[t_start, t_end]``.

    Z and velocity share a **sliding** time window; yaw circle uses the same horizontal ``+x`` /
    ``+y`` convention as the trajectory panel (world XY).

    Call :meth:`update` once per step with ``t_abs=...`` and optional ``yaw_est`` (rad, EKF
    ``zyx`` yaw). :meth:`close` when done.

    ``update_hz`` with ``dataset_hz`` throttles matplotlib redraws: effective rate is
    ``min(update_hz, dataset_hz)`` (e.g. 100 Hz request on 400 Hz data redraws every fourth step).
    Omit ``update_hz`` to redraw on every step (previous default).
    """

    def __init__(
        self,
        run_name: str,
        groundtruth_df: pd.DataFrame | None = None,
        *,
        t_start: float,
        t_end: float,
        buffer_length: int = 5000,
        video_path: str | None = None,
        sliding_window_s: float = SLIDING_TIME_WINDOW_S,
        dataset_hz: float = 0.0,
        update_hz: float | None = None,
    ) -> None:
        if video_path:
            pass

        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self._t_span = max(self.t_end - self.t_start, 1e-9)
        self._window = float(sliding_window_s)

        self.buffer_length = buffer_length
        self.step_count = 0
        self._viz_stride = _viz_stride_from_rates(update_hz, float(dataset_hz))

        self._p0x: float | None = None
        self._p0y: float | None = None

        self.hist_t: deque[float] = deque(maxlen=buffer_length)
        self.hist_vel_x: deque[float] = deque(maxlen=buffer_length)
        self.hist_vel_y: deque[float] = deque(maxlen=buffer_length)
        self.hist_vel_z: deque[float] = deque(maxlen=buffer_length)
        self.hist_pz: deque[float] = deque(maxlen=buffer_length)

        gt_pack = _prepare_gt_timeseries(groundtruth_df)
        if gt_pack is not None:
            (
                self._gt_t,
                self._gt_pz,
                self._gt_vx,
                self._gt_vy,
                self._gt_vz,
                self._gt_yaw,
            ) = gt_pack
        else:
            self._gt_t = None
            self._gt_yaw = None

        plt.ion()
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass

        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title(f"Monitor: {run_name}")

        # --- Trajectory ---
        self.ax_traj = self.axs[0, 0]
        self.ax_traj.set_title("2D position (start-centered)", fontweight="bold")
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_aspect("equal")
        self.ax_traj.grid(True, linestyle="--", alpha=0.6)

        self.min_x, self.max_x = -1.0, 1.0
        self.min_y, self.max_y = -1.0, 1.0

        if groundtruth_df is not None and not groundtruth_df.empty:
            if "local_x" in groundtruth_df.columns and "local_y" in groundtruth_df.columns:
                gx = groundtruth_df["local_x"].to_numpy(dtype=np.float64)
                gy = groundtruth_df["local_y"].to_numpy(dtype=np.float64)
                gx -= gx[0]
                gy -= gy[0]
                self.min_x = min(float(np.min(gx)), self.min_x)
                self.max_x = max(float(np.max(gx)), self.max_x)
                self.min_y = min(float(np.min(gy)), self.min_y)
                self.max_y = max(float(np.max(gy)), self.max_y)
                self.ax_traj.plot(
                    gx, gy, color="#444444", linestyle="--", lw=1.0, alpha=0.5, label="GT"
                )

        self.traj_x_hist: deque[float] = deque(maxlen=5000)
        self.traj_y_hist: deque[float] = deque(maxlen=5000)
        self.line_traj, = self.ax_traj.plot([], [], color="#007ACC", lw=2.5, label="Estimated")
        self.point_head, = self.ax_traj.plot(
            [], [], marker="o", color="#D93025", markersize=6, zorder=5
        )
        self.ax_traj.legend(loc="upper right", fontsize="x-small")

        # --- Camera placeholder ---
        self.ax_video = self.axs[0, 1]
        self.ax_video.set_title("Ego camera (N/A)", fontweight="bold")
        self.ax_video.axis("off")
        self.ax_video.text(
            0.5,
            0.5,
            "No camera stream\n(Tartanground split logs)",
            ha="center",
            va="center",
            transform=self.ax_video.transAxes,
            fontsize=11,
            color="#555555",
        )

        # --- Base p_z vs time + planar heading (unit circle) ---
        pz_ss = self.axs[1, 0].get_subplotspec()
        self.axs[1, 0].remove()
        gs_pz = gridspec.GridSpecFromSubplotSpec(2, 1, pz_ss, height_ratios=[1.0, 1.0], hspace=0.4)
        self.ax_pz = self.fig.add_subplot(gs_pz[0, 0])
        self.ax_pz.set_title(
            f"Base position z (world), {self._window:.0f}s window", fontweight="bold"
        )
        self.ax_pz.set_xlabel("Time [s]")
        self.ax_pz.set_ylabel("z [m]")
        self.ax_pz.grid(True, linestyle="--", alpha=0.6)
        self.line_pz, = self.ax_pz.plot([], [], color="#6A1B9A", lw=1.5, label="p_z est.")
        if self._gt_t is not None:
            (self.line_gt_pz,) = self.ax_pz.plot(
                [],
                [],
                color="#6A1B9A",
                linestyle="--",
                lw=1.2,
                alpha=0.35,
                label="p_z GT",
            )
        else:
            self.line_gt_pz = None
        self.ax_pz.legend(loc="upper right", fontsize="x-small")

        self.ax_yaw = self.fig.add_subplot(gs_pz[1, 0])
        self.ax_yaw.set_title("Heading (world XY, unit circle)", fontweight="bold", fontsize=10)
        self.ax_yaw.set_xlabel("cos ψ → +X (trajectory)")
        self.ax_yaw.set_ylabel("sin ψ → +Y (trajectory)")
        self.ax_yaw.set_aspect("equal")
        self.ax_yaw.grid(True, linestyle="--", alpha=0.4)
        th = np.linspace(0.0, 2.0 * np.pi, 200)
        self.ax_yaw.plot(np.cos(th), np.sin(th), color="#BBBBBB", lw=0.9, zorder=0)
        self.ax_yaw.axhline(0.0, color="#DDDDDD", lw=0.6, zorder=0)
        self.ax_yaw.axvline(0.0, color="#DDDDDD", lw=0.6, zorder=0)
        self.ax_yaw.set_xlim(-1.15, 1.15)
        self.ax_yaw.set_ylim(-1.15, 1.15)
        (self.line_yaw_est,) = self.ax_yaw.plot(
            [],
            [],
            color="#007ACC",
            lw=2.2,
            solid_capstyle="round",
            label="ψ est.",
        )
        self.line_yaw_est.set_visible(False)
        if self._gt_t is not None:
            (self.line_yaw_gt,) = self.ax_yaw.plot(
                [0.0, 0.85],
                [0.0, 0.0],
                color="#444444",
                linestyle="--",
                lw=1.6,
                alpha=0.85,
                solid_capstyle="round",
                label="ψ GT",
            )
        else:
            self.line_yaw_gt = None
        self.ax_yaw.legend(loc="upper right", fontsize="x-small")

        # --- Velocity: single axes, six lines (est solid, GT dashed) ---
        vel_ss = self.axs[1, 1].get_subplotspec()
        self.axs[1, 1].remove()
        self.ax_vel = self.fig.add_subplot(vel_ss)
        self.ax_vel.set_title(
            f"Base linear velocity (world), {self._window:.0f}s window",
            fontweight="bold",
        )
        self.ax_vel.set_xlabel("Time [s]")
        self.ax_vel.set_ylabel("[m/s]")
        self.ax_vel.grid(True, linestyle="--", alpha=0.6)
        colors = ("#D32F2F", "#388E3C", "#1976D2")
        (self.line_vx,) = self.ax_vel.plot(
            [], [], color=colors[0], lw=1.5, label="vx est."
        )
        if self._gt_t is not None:
            (self.line_gt_vx,) = self.ax_vel.plot(
                [],
                [],
                color=colors[0],
                linestyle="--",
                lw=1.1,
                alpha=0.45,
                label="vx GT",
            )
        else:
            self.line_gt_vx = None
        (self.line_vy,) = self.ax_vel.plot(
            [], [], color=colors[1], lw=1.5, label="vy est."
        )
        if self._gt_t is not None:
            (self.line_gt_vy,) = self.ax_vel.plot(
                [],
                [],
                color=colors[1],
                linestyle="--",
                lw=1.1,
                alpha=0.45,
                label="vy GT",
            )
        else:
            self.line_gt_vy = None
        (self.line_vz,) = self.ax_vel.plot(
            [], [], color=colors[2], lw=1.5, label="vz est."
        )
        if self._gt_t is not None:
            (self.line_gt_vz,) = self.ax_vel.plot(
                [],
                [],
                color=colors[2],
                linestyle="--",
                lw=1.1,
                alpha=0.45,
                label="vz GT",
            )
        else:
            self.line_gt_vz = None
        self.ax_vel.legend(ncol=3, loc="upper right", fontsize="x-small")

        # --- Progress bar (figure coordinates, below subplots) ---
        self.ax_prog = self.fig.add_axes([0.12, 0.02, 0.76, 0.028])
        self.ax_prog.set_xlim(0.0, 1.0)
        self.ax_prog.set_ylim(0.0, 1.0)
        self.ax_prog.axis("off")
        self._prog_bg = Rectangle(
            (0.0, 0.1),
            1.0,
            0.8,
            facecolor="#E0E0E0",
            edgecolor="#888888",
            linewidth=0.8,
            transform=self.ax_prog.transData,
        )
        self._prog_fill = Rectangle(
            (0.0, 0.1),
            0.0,
            0.8,
            facecolor="#1976D2",
            edgecolor="none",
            transform=self.ax_prog.transData,
        )
        self.ax_prog.add_patch(self._prog_bg)
        self.ax_prog.add_patch(self._prog_fill)
        self._prog_text = self.ax_prog.text(
            0.5,
            0.5,
            "",
            ha="center",
            va="center",
            transform=self.ax_prog.transAxes,
            fontsize=8,
            color="#333333",
        )

        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.12, hspace=0.35, wspace=0.28)
        plt.show(block=False)
        plt.pause(0.1)

    def _sliding_time_limits(self, t_now: float) -> tuple[float, float]:
        t_hi = float(t_now)
        t_lo = max(self.t_start, t_hi - self._window)
        return t_lo, t_hi

    def _apply_sliding_xlims(self, t_now: float) -> tuple[float, float]:
        t_lo, t_hi = self._sliding_time_limits(t_now)
        self.ax_vel.set_xlim(t_lo, t_hi)
        self.ax_pz.set_xlim(t_lo, t_hi)
        return t_lo, t_hi

    def _gt_mask(self, t_lo: float, t_hi: float) -> np.ndarray | None:
        if self._gt_t is None:
            return None
        return (self._gt_t >= t_lo) & (self._gt_t <= t_hi)

    def _update_progress(self, t_now: float) -> None:
        frac = (t_now - self.t_start) / self._t_span
        frac = float(np.clip(frac, 0.0, 1.0))
        self._prog_fill.set_width(frac)
        self._prog_text.set_text(f"{100.0 * frac:.1f}%  t={t_now:.3f}s / {self.t_end:.3f}s")

    def update(
        self,
        px: float,
        py: float,
        pz: float,
        vx: float,
        vy: float,
        vz: float,
        *,
        t_abs: float,
        yaw_est: float | None = None,
    ) -> None:
        self.step_count += 1
        t_now = float(t_abs)
        if self._p0x is None:
            self._p0x, self._p0y = float(px), float(py)

        cx = float(px) - self._p0x
        cy = float(py) - self._p0y

        self.traj_x_hist.append(cx)
        self.traj_y_hist.append(cy)
        self.min_x = min(self.min_x, cx)
        self.max_x = max(self.max_x, cx)
        self.min_y = min(self.min_y, cy)
        self.max_y = max(self.max_y, cy)

        self.line_traj.set_data(self.traj_x_hist, self.traj_y_hist)
        self.point_head.set_data([cx], [cy])

        self.hist_t.append(t_now)
        self.hist_vel_x.append(float(vx))
        self.hist_vel_y.append(float(vy))
        self.hist_vel_z.append(float(vz))
        self.hist_pz.append(float(pz))

        t_arr = list(self.hist_t)
        self.line_vx.set_data(t_arr, list(self.hist_vel_x))
        self.line_vy.set_data(t_arr, list(self.hist_vel_y))
        self.line_vz.set_data(t_arr, list(self.hist_vel_z))
        self.line_pz.set_data(t_arr, list(self.hist_pz))

        self._update_progress(t_now)
        t_lo, t_hi = self._apply_sliding_xlims(t_now)

        gmask = self._gt_mask(t_lo, t_hi)
        if self.line_gt_pz is not None:
            if gmask is not None and np.any(gmask):
                self.line_gt_pz.set_data(self._gt_t[gmask], self._gt_pz[gmask])
                if self.line_gt_vx is not None:
                    self.line_gt_vx.set_data(self._gt_t[gmask], self._gt_vx[gmask])
                    self.line_gt_vy.set_data(self._gt_t[gmask], self._gt_vy[gmask])
                    self.line_gt_vz.set_data(self._gt_t[gmask], self._gt_vz[gmask])
            else:
                self.line_gt_pz.set_data([], [])
                if self.line_gt_vx is not None:
                    self.line_gt_vx.set_data([], [])
                    self.line_gt_vy.set_data([], [])
                    self.line_gt_vz.set_data([], [])

        # y-limits: estimated + GT visible in window
        t_est = np.array(t_arr, dtype=np.float64)
        pz_est = np.array(list(self.hist_pz), dtype=np.float64)
        win_est = (t_est >= t_lo) & (t_est <= t_hi)
        z_candidates: list[float] = []
        if np.any(win_est):
            z_candidates.extend(pz_est[win_est].tolist())
        if gmask is not None and np.any(gmask):
            z_candidates.extend(self._gt_pz[gmask].tolist())
        if z_candidates:
            zmin, zmax = min(z_candidates), max(z_candidates)
            zpad = max(0.05 * (zmax - zmin), 0.02)
            self.ax_pz.set_ylim(zmin - zpad, zmax + zpad)

        vx_e = np.array(list(self.hist_vel_x), dtype=np.float64)
        vy_e = np.array(list(self.hist_vel_y), dtype=np.float64)
        vz_e = np.array(list(self.hist_vel_z), dtype=np.float64)
        v_cand: list[float] = []
        if np.any(win_est):
            v_cand.extend(vx_e[win_est].tolist())
            v_cand.extend(vy_e[win_est].tolist())
            v_cand.extend(vz_e[win_est].tolist())
        if gmask is not None and np.any(gmask) and self._gt_t is not None:
            v_cand.extend(self._gt_vx[gmask].tolist())
            v_cand.extend(self._gt_vy[gmask].tolist())
            v_cand.extend(self._gt_vz[gmask].tolist())
        if v_cand:
            vmin, vmax = min(v_cand), max(v_cand)
            vpad = max(0.05 * (vmax - vmin), 0.2)
            self.ax_vel.set_ylim(vmin - vpad, vmax + vpad)

        if yaw_est is not None and np.isfinite(yaw_est):
            ye = float(yaw_est)
            self.line_yaw_est.set_data([0.0, np.cos(ye)], [0.0, np.sin(ye)])
            self.line_yaw_est.set_visible(True)
        else:
            self.line_yaw_est.set_data([], [])
            self.line_yaw_est.set_visible(False)

        if (
            self.line_yaw_gt is not None
            and self._gt_t is not None
            and self._gt_yaw is not None
            and len(self._gt_t) >= 2
        ):
            tg = float(np.clip(t_now, float(self._gt_t[0]), float(self._gt_t[-1])))
            yg = float(np.interp(tg, self._gt_t, self._gt_yaw))
            if np.isfinite(yg):
                self.line_yaw_gt.set_data([0.0, np.cos(yg)], [0.0, np.sin(yg)])
                self.line_yaw_gt.set_visible(True)
            else:
                self.line_yaw_gt.set_visible(False)

        refresh = self._viz_stride <= 1 or ((self.step_count - 1) % self._viz_stride == 0)
        if refresh:
            span_x = self.max_x - self.min_x
            span_y = self.max_y - self.min_y
            max_span = max(span_x, span_y)
            padding = max(max_span * 0.1, 1.0)
            mid_x = (self.max_x + self.min_x) / 2.0
            mid_y = (self.max_y + self.min_y) / 2.0
            self.ax_traj.set_xlim(mid_x - max_span / 2 - padding, mid_x + max_span / 2 + padding)
            self.ax_traj.set_ylim(mid_y - max_span / 2 - padding, mid_y + max_span / 2 + padding)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self) -> None:
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        plt.ioff()
        plt.close(self.fig)
