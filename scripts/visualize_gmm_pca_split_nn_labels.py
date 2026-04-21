#!/usr/bin/env python3
"""
Fit a 2-component full-covariance GMM on default instant features for one leg,
project to PCA PC1-PC2 with per-axis sigma scaling for a square view, and plot
viridis log-density contours with point colors from a neural contact estimator.

Contours come from the fitted GMM (density shape in PCA space), while
stance/swing labels come from neural detector replay on the same sequence.

Example::

    cd /path/to/async_ekf_workspace
    MPLBACKEND=Agg python3 scripts/visualize_gmm_pca_split_nn_labels.py \
      --checkpoint /abs/path/to/contact_gru.pt \
      --save gmm_pca_nn_labels.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from leg_odom.contact.gmm_hmm_core.fitting import order_gmm_components
from leg_odom.contact.neural import build_neural_detectors_from_cfg
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.features import (
    DEFAULT_INSTANT_FEATURE_FIELDS,
    build_timeline_features_for_leg,
    parse_instant_feature_fields,
    sliding_windows_flat,
)
from leg_odom.io.split_imu_bag import load_prepared_split_sequence
from leg_odom.kinematics.anymal import AnymalKinematics

_REFERENCE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#B8B8B8",
    "grid.alpha": 0.85,
    "axes.grid": True,
    "axes.axisbelow": True,
}

# Match common "contact / non-contact" style: stance ~= on ground -> red; swing -> blue.
_COLOR_STANCE = "#D62728"
_COLOR_SWING = "#1F77B4"
_COLORS_PHASE = (_COLOR_STANCE, _COLOR_SWING)

_CONTOUR_LEVELS = 20
_GRID_RES = 220
# Square view: half-width = max(rx, ry) * (1 + this); grid = full square so contours fill the axes.
_VIEW_PAD_FRAC = 0.12


def _fit_ordered_gmm(
    X: np.ndarray,
    spec,
    *,
    random_state: int,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, bool]:
    """Match leg_odom.contact.gmm_hmm_core.fitting.fit_gmm_ordered; also return sklearn model."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4:
        d = int(X.shape[1]) if X.ndim == 2 else 0
        return (
            GaussianMixture(),
            np.zeros((2, d), dtype=np.float64),
            np.zeros((2, d, d), dtype=np.float64) if d else np.zeros((2, 0, 0), dtype=np.float64),
            True,
        )
    d = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=int(random_state),
            max_iter=200,
        )
        try:
            gmm.fit(X)
        except (ValueError, np.linalg.LinAlgError):
            z = np.zeros((2, d), dtype=np.float64)
            return gmm, z, np.zeros((2, d, d), dtype=np.float64), True
    w = gmm.weights_
    if float(np.max(w)) >= 0.999:
        return (
            gmm,
            np.asarray(gmm.means_, dtype=np.float64),
            np.asarray(gmm.covariances_, dtype=np.float64),
            True,
        )
    mo, co = order_gmm_components(
        np.asarray(gmm.means_, dtype=np.float64),
        np.asarray(gmm.covariances_, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        spec,
        1,
    )
    return gmm, mo, co, False


def _logpdf_grid_scaled_pc(
    pca: PCA,
    gmm: GaussianMixture,
    s0: float,
    s1: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Mixture log-density on a grid in sigma-scaled PC space (map back to R^D for score_samples)."""
    zs0 = xx.ravel()
    zs1 = yy.ravel()
    z_un = np.column_stack([zs0 * s0, zs1 * s1])
    x_back = pca.inverse_transform(z_un)
    logp = gmm.score_samples(x_back)
    return logp.reshape(xx.shape)


def _nn_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    neural: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "stance_probability_threshold": float(args.stance_threshold),
    }
    if args.meta_path is not None:
        neural["meta_path"] = str(args.meta_path)
    if args.scaler_path is not None:
        neural["scaler_path"] = str(args.scaler_path)
    if args.device is not None:
        neural["device"] = str(args.device)
    return {"contact": {"detector": "neural", "neural": neural}}


def main() -> None:
    default_seq = "/home/girgine/Documents/leg-odometry/iros/data_anymal"
    ap = argparse.ArgumentParser(
        description="2-GMM contour + PCA scatter, with stance/swing colors from neural detector replay.",
    )
    ap.add_argument("--sequence-dir", type=Path, default=Path(default_seq))
    ap.add_argument("--leg", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to contact_{cnn,gru}.pt")
    ap.add_argument("--meta-path", type=Path, default=None, help="Optional meta JSON sidecar path")
    ap.add_argument("--scaler-path", type=Path, default=None, help="Optional scaler NPZ sidecar path")
    ap.add_argument("--stance-threshold", type=float, default=0.5, help="NN stance threshold in [0, 1]")
    ap.add_argument("--device", type=str, default=None, help="Optional torch device: cpu, cuda, or mps")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-sanitize-imu", action="store_true")
    ap.add_argument("--save", type=Path, default=None, help="If set, write figure to this path")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not (0.0 <= float(args.stance_threshold) <= 1.0):
        raise SystemExit("--stance-threshold must be in [0, 1].")

    seq_dir = args.sequence_dir.expanduser().resolve()
    df, hz, gt, accel_gc = load_prepared_split_sequence(
        seq_dir,
        verbose=True,
        sanitize_imu=not args.no_sanitize_imu,
    )
    rec = LegOdometrySequence(
        frames=df,
        median_rate_hz=hz,
        position_ground_truth=gt,
        sequence_name=seq_dir.name,
        meta={"sequence_dir": str(seq_dir), "accel_gravity_compensated": accel_gc},
    )

    spec = parse_instant_feature_fields(DEFAULT_INSTANT_FEATURE_FIELDS)
    kin = AnymalKinematics()
    inst = build_timeline_features_for_leg(rec.frames, kin, args.leg, spec)
    X = sliding_windows_flat(inst, 1)
    if X.shape[0] < 4:
        raise SystemExit(f"Need at least 4 rows for GMM; got {X.shape[0]}.")

    nn_cfg = _nn_cfg_from_args(args)
    detectors = build_neural_detectors_from_cfg(nn_cfg, kin_model=kin, workspace_root=_REPO_ROOT)
    _t_abs, _grf, st_list, _ps = replay_detectors_on_timeline(rec.frames, kin, detectors)
    labels = np.asarray(st_list[args.leg], dtype=np.int64).reshape(-1)
    if labels.shape[0] != X.shape[0]:
        raise SystemExit(
            f"Length mismatch: NN labels T={labels.shape[0]} but feature rows T={X.shape[0]}."
        )

    rs = int(args.random_state) + int(args.leg)
    gmm, _means, _covs, degenerate = _fit_ordered_gmm(X, spec, random_state=rs)
    if degenerate:
        raise SystemExit(
            "GMM fit degenerate (too few points, singular cov, or one component weight >= 0.999). "
            "Try another leg or sequence."
        )

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    s0 = float(np.std(Z[:, 0], ddof=0) + 1e-12)
    s1 = float(np.std(Z[:, 1], ddof=0) + 1e-12)
    Zs = np.column_stack([Z[:, 0] / s0, Z[:, 1] / s1])

    xmin_d, xmax_d = float(np.min(Zs[:, 0])), float(np.max(Zs[:, 0]))
    ymin_d, ymax_d = float(np.min(Zs[:, 1])), float(np.max(Zs[:, 1]))
    cx = 0.5 * (xmin_d + xmax_d)
    cy = 0.5 * (ymin_d + ymax_d)
    rx = 0.5 * (xmax_d - xmin_d) + 1e-9
    ry = 0.5 * (ymax_d - ymin_d) + 1e-9
    half = max(rx, ry) * (1.0 + _VIEW_PAD_FRAC)
    lo, hi = cx - half, cx + half
    gx = np.linspace(lo, hi, _GRID_RES)
    gy = np.linspace(lo, hi, _GRID_RES)
    xx, yy = np.meshgrid(gx, gy)
    logp = _logpdf_grid_scaled_pc(pca, gmm, s0, s1, xx, yy)
    lp_min = float(np.nanmin(logp))
    lp_max = float(np.nanmax(logp))
    # Include more of the tail so outer contours reach the square boundary.
    levels = np.linspace(lp_min + 0.008 * (lp_max - lp_min), lp_max, _CONTOUR_LEVELS)

    with plt.rc_context(rc=_REFERENCE_RC):
        fig, ax = plt.subplots(figsize=(10, 10))

        cs = ax.contour(
            xx,
            yy,
            logp,
            levels=levels,
            cmap="viridis",
            linewidths=0.95,
            alpha=0.95,
            zorder=1,
        )
        # Matplotlib >=3.8 uses QuadContourSet without .collections; rasterize when supported.
        cols = getattr(cs, "collections", None)
        if cols is not None:
            for c in cols:
                c.set_rasterized(True)
        elif hasattr(cs, "set_rasterized"):
            try:
                cs.set_rasterized(True)
            except (AttributeError, TypeError):
                pass

        for k in (0, 1):
            mask = labels == k
            ax.scatter(
                Zs[mask, 0],
                Zs[mask, 1],
                c=_COLORS_PHASE[k],
                s=5,
                alpha=0.38,
                linewidths=0,
                rasterized=True,
                zorder=3,
            )

        leg_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="0.2",
                markeredgewidth=0.35,
                markerfacecolor=_COLOR_STANCE,
                markersize=8,
                label="Stance (NN)",
                linestyle="none",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="0.2",
                markeredgewidth=0.35,
                markerfacecolor=_COLOR_SWING,
                markersize=8,
                label="Swing (NN)",
                linestyle="none",
            ),
        ]
        ax.legend(
            handles=leg_handles,
            loc="lower left",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            framealpha=1.0,
            handletextpad=0.45,
            borderpad=0.55,
        )

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.tick_params(
            axis="both",
            which="major",
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="inout",
        )
        secy = ax.secondary_yaxis("right", functions=(lambda x: x, lambda x: x))
        secy.tick_params(axis="y", which="major", right=True, labelright=True, direction="inout")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        # "box": keep the square limits above; "datalim" would expand past the contour grid (empty margins).
        ax.set_aspect("equal", adjustable="box")
        # Tight margins; keep small inset for axis labels + right-hand PCA 2 label.
        fig.subplots_adjust(left=0.065, right=0.94, top=0.96, bottom=0.055)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
