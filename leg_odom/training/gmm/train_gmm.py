"""
Fit a 2-component GMM on sliding-window features and save a pretrained ``.npz``.

Reads ``precomputed_instants.npz`` bundles (same as NN contact training). Run precompute first::

    python -m leg_odom.features.precompute_contact_instants \\
      --config leg_odom/features/default_precompute_config.yaml

Then::

    python -m leg_odom.training.gmm.train_gmm --help

    python -m leg_odom.training.gmm.train_gmm \\
      --precomputed-root leg_odom/features/precomputed \\
      --output leg_odom/training/gmm/weights.npz --history-length 1

Optional: ``--max-sequences N`` uses only the first N bundles after discovery order (CPU efficiency).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from leg_odom.contact.grf_stance_plot import plot_grf_contact_overview
from leg_odom.contact.gmm_hmm import (
    DEFAULT_INSTANT_FEATURE_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    fit_gmm_ordered,
    parse_instant_feature_fields,
    sliding_windows_flat,
    stance_height_meta_index,
)
from leg_odom.contact.gmm_hmm.detector import GmmHmmContactDetector
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.features.instant_spec import FULL_OFFLINE_INSTANT_FIELDS, subset_instant_columns
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.kinematics_factory import build_kinematics_backend, build_kinematics_by_name
from leg_odom.training.nn.dataset_kind import infer_dataset_kind_from_sequence_dir
from leg_odom.training.nn.precomputed_io import discover_precomputed_instants_npz, load_precomputed_sequence_npz


def save_pretrained_gmm_npz(
    path: Path,
    *,
    means: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
    feature_fields: tuple[str, ...],
    history_length: int,
    instant_dim: int,
    stance_height_feature_index: int,
    trans_stay: float,
    feature_spec_version: int,
    n_samples: int,
    random_state: int,
) -> None:
    """Write GMM parameters + metadata for :func:`~leg_odom.contact.gmm_hmm.fitting.load_pretrained_gmm_npz`."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_csv = ",".join(feature_fields)
    np.savez(
        path,
        means=np.asarray(means, dtype=np.float64),
        covariances=np.asarray(covariances, dtype=np.float64),
        history_length=np.int64(history_length),
        instant_dim=np.int64(instant_dim),
        stance_height_feature_index=np.int64(stance_height_feature_index),
        trans_stay=np.float64(trans_stay),
        feature_spec_version=np.int64(feature_spec_version),
        n_samples=np.int64(n_samples),
        random_state=np.int64(random_state),
        feature_fields_str=np.array(fields_csv),
    )


def _slice_npz_paths(paths: list[Path], max_sequences: int | None) -> list[Path]:
    n_total = len(paths)
    if max_sequences is None:
        return paths
    n_req = int(max_sequences)
    if n_req < 1:
        print("[train_gmm] error: --max-sequences must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if n_req > n_total:
        print(
            f"[train_gmm] warning: --max-sequences={n_req} > discovered {n_total}; "
            f"using all {n_total} sequences",
            file=sys.stderr,
        )
        return paths
    print(f"[train_gmm] using {n_req} of {n_total} precomputed sequences (discovery order)")
    return paths[:n_req]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit GMM contact features from precomputed_instants.npz and save .npz")
    p.add_argument(
        "--precomputed-root",
        type=str,
        required=True,
        help="Directory tree containing precomputed_instants.npz (output of precompute_contact_instants)",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N bundles after lexicographic discovery (default: all). Warns if N > discovered count.",
    )
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument(
        "--feature-fields",
        type=str,
        default=",".join(DEFAULT_INSTANT_FEATURE_FIELDS),
        help="Comma-separated names (ALLOWED_INSTANT_FEATURE_FIELDS in leg_odom/features/instant_spec.py)",
    )
    p.add_argument("--history-length", type=int, default=1)
    p.add_argument("--output", type=str, default="leg_odom/training/gmm/weights.npz", help="Output .npz path")
    p.add_argument("--trans-stay", type=float, default=0.99)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--skip-train-plot",
        action="store_true",
        help="Do not write plots/ under the output directory after training",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fields = tuple(s.strip() for s in args.feature_fields.split(",") if s.strip())
    spec = parse_instant_feature_fields(fields)
    n = int(args.history_length)
    robot = str(args.robot_kinematics)

    root = Path(args.precomputed_root).expanduser().resolve()
    all_paths = discover_precomputed_instants_npz(root, verbose=True)
    paths = _slice_npz_paths(all_paths, args.max_sequences)
    if not paths:
        raise RuntimeError("No precomputed_instants.npz paths after --max-sequences slice")

    kin = build_kinematics_by_name(robot)
    n_legs = int(kin.n_legs)
    rk = str(robot).strip().lower()

    X_blocks: list[np.ndarray] = []
    first_sequence_dir_for_plot: str | None = None

    for npz_path in tqdm(paths, desc="Precomputed npz × legs", unit="seq"):
        bundle = load_precomputed_sequence_npz(npz_path, expected_robot_kinematics=rk, n_legs=n_legs)
        if first_sequence_dir_for_plot is None:
            first_sequence_dir_for_plot = bundle.sequence_dir_stored

        for leg in range(n_legs):
            full = bundle.instants_by_leg[leg]
            inst = subset_instant_columns(full, FULL_OFFLINE_INSTANT_FIELDS, spec.fields)
            Xb = sliding_windows_flat(inst, n)
            if Xb.size:
                X_blocks.append(Xb)

    if not X_blocks:
        raise RuntimeError("No feature windows; check precomputed length vs history-length")
    X = np.vstack(X_blocks)
    mo, co, bad = fit_gmm_ordered(X, spec, n, random_state=int(args.random_state))
    if bad:
        raise RuntimeError("GMM fit failed or degenerate; try more data, N=1, or different features")

    out = Path(args.output).expanduser()
    save_pretrained_gmm_npz(
        out,
        means=mo,
        covariances=co,
        feature_fields=spec.fields,
        history_length=n,
        instant_dim=spec.instant_dim,
        stance_height_feature_index=stance_height_meta_index(spec),
        trans_stay=float(args.trans_stay),
        feature_spec_version=INSTANT_FEATURE_SPEC_VERSION,
        n_samples=int(X.shape[0]),
        random_state=int(args.random_state),
    )
    print(f"Wrote {out.resolve()}  D={mo.shape[1]}  samples={X.shape[0]}")

    if args.skip_train_plot or not first_sequence_dir_for_plot:
        return

    seq_dir = Path(str(first_sequence_dir_for_plot).strip()).expanduser().resolve()
    if not seq_dir.is_dir():
        print(f"[train_gmm] skip train plot: sequence_dir not a directory: {seq_dir}")
        return

    dataset_kind = infer_dataset_kind_from_sequence_dir(seq_dir)
    cfg = {
        "schema_version": 1,
        "run": {
            "name": "train_gmm_plot",
            "debug": {
                "enabled": False,
                "live_visualizer": {
                    "enabled": False,
                    "sliding_window_s": 10.0,
                    "buffer_length": 5000,
                    "video_path": None,
                    "hz": None,
                },
            },
        },
        "robot": {"kinematics": robot},
        "dataset": {"kind": dataset_kind, "sequence_dir": str(seq_dir)},
        "contact": {"detector": "none"},
        "ekf": {"noise_config": None, "initialize_nominal_from_data": False},
        "output": {"base_dir": ".", "include_timestamp": False},
    }
    try:
        ds = build_leg_odometry_dataset(cfg)
        kin_ds = build_kinematics_backend(cfg)
        rec = ds[0]
    except Exception as e:
        print(f"[train_gmm] skip train plot: could not load recording from {seq_dir}: {e}")
        return

    plots_dir = out.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = out.resolve()
    dets = [
        GmmHmmContactDetector(
            feature_fields=fields,
            history_length=n,
            trans_stay=float(args.trans_stay),
            mode="online",
            pretrained_path=str(out_resolved),
            fit_interval=250,
            window_size=500,
            degeneracy_max_weight=float(0.98),
            random_state=int(args.random_state),
        )
        for _ in range(kin_ds.n_legs)
    ]
    try:
        t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin_ds, dets)
        fig_path = plots_dir / f"gmm_hmm_contact_train_{rec.sequence_name}.png"
        plot_grf_contact_overview(
            t_abs,
            grfs,
            st,
            ps,
            suptitle=f"GMM+HMM (trained, online load) — {rec.sequence_name}",
            save_path=fig_path,
            show=False,
        )
        print(f"Wrote {fig_path.resolve()}")
    except Exception as e:
        print(f"[train_gmm] skip train plot: replay failed: {e}")


if __name__ == "__main__":
    main()
