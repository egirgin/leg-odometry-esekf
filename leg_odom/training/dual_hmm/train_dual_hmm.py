"""
Fit pooled 2-GMM on GRF (1D) and **instant** kinematic rows; save dual pretrained ``.npz``.

All discovered bundles under ``--precomputed-root`` (optionally capped by ``--max-sequences``) are
merged into one pooled design matrix per modality; **one** load GMM and **one** kin GMM fit.
Kin uses full-sequence instant rows (no multi-step windows). ``trans_stay`` written to the file and
used for the optional post-train HMM replay is fixed at ``0.99`` in code.

Requires ``precomputed_instants.npz`` trees (same as :mod:`leg_odom.training.gmm.train_gmm`)::

    python -m leg_odom.training.dual_hmm.train_dual_hmm \\
      --precomputed-root <root> --output leg_odom/training/dual_hmm/weights.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from leg_odom.contact.dual_hmm.detector import build_dual_hmm_detectors_from_cfg
from leg_odom.contact.dual_hmm.spec import parse_dual_kinematic_feature_fields
from leg_odom.contact.grf_stance_plot import plot_grf_contact_overview
from leg_odom.contact.gmm_hmm import INSTANT_FEATURE_SPEC_VERSION, DEFAULT_INSTANT_FEATURE_FIELDS
from leg_odom.contact.gmm_hmm_core.fitting import fit_gmm_ordered, save_pretrained_dual_hmm_npz
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.features import parse_instant_feature_fields, stance_height_meta_index
from leg_odom.features.instant_spec import FULL_OFFLINE_INSTANT_FIELDS, subset_instant_columns
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.kinematics_factory import build_kinematics_backend, build_kinematics_by_name
from leg_odom.training.nn.dataset_kind import infer_dataset_kind_from_sequence_dir
from leg_odom.training.nn.precomputed_io import discover_precomputed_instants_npz, load_precomputed_sequence_npz

_PRETRAIN_HISTORY_LENGTH = 1
_PRETRAINED_TRANS_STAY = 0.99


def _slice_npz_paths(paths: list[Path], max_sequences: int | None) -> list[Path]:
    n_total = len(paths)
    if max_sequences is None:
        return paths
    n_req = int(max_sequences)
    if n_req < 1:
        print("[train_dual_hmm] error: --max-sequences must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if n_req > n_total:
        print(
            f"[train_dual_hmm] warning: max-sequences={n_req} > discovered {n_total}; using all {n_total}",
            file=sys.stderr,
        )
        return paths
    print(f"[train_dual_hmm] using {n_req} of {n_total} precomputed sequences (discovery order)")
    return paths[:n_req]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit dual HMM GMM (load + kin) from precomputed_instants.npz")
    p.add_argument("--precomputed-root", type=str, required=True)
    p.add_argument("--max-sequences", type=int, default=None, metavar="N")
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument(
        "--feature-fields",
        type=str,
        default=",".join(DEFAULT_INSTANT_FEATURE_FIELDS),
        help="Kinematic instant fields only (no grf_n); see ALLOWED_INSTANT_FEATURE_FIELDS",
    )
    p.add_argument("--output", type=str, default="leg_odom/training/dual_hmm/weights.npz")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--skip-train-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fields = tuple(s.strip() for s in args.feature_fields.split(",") if s.strip())
    kin_spec = parse_dual_kinematic_feature_fields(fields)
    d_kin = kin_spec.instant_dim
    spec_grf = parse_instant_feature_fields(("grf_n",))
    robot = str(args.robot_kinematics)

    root = Path(args.precomputed_root).expanduser().resolve()
    paths = _slice_npz_paths(discover_precomputed_instants_npz(root, verbose=True), args.max_sequences)
    if not paths:
        raise RuntimeError("No precomputed_instants.npz paths")

    kin = build_kinematics_by_name(robot)
    n_legs = int(kin.n_legs)
    rk = robot.strip().lower()

    X_load_blocks: list[np.ndarray] = []
    X_kin_blocks: list[np.ndarray] = []
    first_sequence_dir_for_plot: str | None = None

    for npz_path in tqdm(paths, desc="Precomputed npz × legs", unit="seq"):
        bundle = load_precomputed_sequence_npz(npz_path, expected_robot_kinematics=rk, n_legs=n_legs)
        if first_sequence_dir_for_plot is None:
            first_sequence_dir_for_plot = bundle.sequence_dir_stored
        foot = bundle.foot_forces
        t_rows = int(foot.shape[0])
        for leg in range(n_legs):
            g = foot[:, leg].reshape(-1, 1)
            if g.shape[0] >= 4:
                X_load_blocks.append(g)
            full = bundle.instants_by_leg[leg]
            inst = subset_instant_columns(full, FULL_OFFLINE_INSTANT_FIELDS, kin_spec.fields)
            Xb = np.asarray(inst, dtype=np.float64)
            if Xb.size and Xb.shape[0] >= 4:
                X_kin_blocks.append(Xb)

    if not X_load_blocks:
        raise RuntimeError("No load GRF samples pooled; check precomputed foot_forces")
    if not X_kin_blocks:
        raise RuntimeError("No kinematic instant rows; check precomputed instants and feature-fields")

    Xl = np.vstack(X_load_blocks)
    Xk = np.vstack(X_kin_blocks)
    rs = int(args.random_state)

    mo_l, co_l, bad_l = fit_gmm_ordered(Xl, spec_grf, 1, random_state=rs)
    if bad_l:
        raise RuntimeError("Pooled load GMM degenerate; try more sequences or check GRF columns")
    mo_k, co_k, bad_k = fit_gmm_ordered(
        Xk, kin_spec, _PRETRAIN_HISTORY_LENGTH, random_state=rs + 1
    )
    if bad_k:
        raise RuntimeError("Pooled kinematic GMM degenerate; try different features or more data")

    out = Path(args.output).expanduser()
    save_pretrained_dual_hmm_npz(
        out,
        load_means=mo_l,
        load_covariances=co_l,
        kin_means=mo_k,
        kin_covariances=co_k,
        kin_feature_fields=kin_spec.fields,
        kin_history_length=_PRETRAIN_HISTORY_LENGTH,
        kin_instant_dim=kin_spec.instant_dim,
        stance_height_feature_index=stance_height_meta_index(kin_spec),
        trans_stay=_PRETRAINED_TRANS_STAY,
        feature_spec_version=int(INSTANT_FEATURE_SPEC_VERSION),
        n_samples_load=int(Xl.shape[0]),
        n_samples_kin=int(Xk.shape[0]),
        random_state=rs,
    )
    print(f"Wrote {out.resolve()}  D_kin={d_kin}  load_samples={Xl.shape[0]}  kin_samples={Xk.shape[0]}")

    if args.skip_train_plot or not first_sequence_dir_for_plot:
        return

    seq_dir = Path(str(first_sequence_dir_for_plot).strip()).expanduser().resolve()
    if not seq_dir.is_dir():
        print(f"[train_dual_hmm] skip plot: not a directory: {seq_dir}")
        return

    dataset_kind = infer_dataset_kind_from_sequence_dir(seq_dir)
    cfg = {
        "schema_version": 1,
        "run": {
            "name": "train_dual_hmm_plot",
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
        "contact": {
            "detector": "dual_hmm",
            "dual_hmm": {
                "mode": "online",
                "feature_fields": list(kin_spec.fields),
                "history_length": _PRETRAIN_HISTORY_LENGTH,
                "pretrained_path": str(out.resolve()),
                "trans_stay": _PRETRAINED_TRANS_STAY,
                "fit_interval": 250,
                "window_size": 500,
                "degeneracy_max_weight": 0.98,
                "random_state": rs,
                "use_energy": False,
                "verbose": False,
            },
        },
        "ekf": {"noise_config": None, "initialize_nominal_from_data": False},
        "output": {"base_dir": ".", "include_timestamp": False},
    }
    try:
        ds = build_leg_odometry_dataset(cfg)
        kin_ds = build_kinematics_backend(cfg)
        rec = ds[0]
        tr = len(rec.frames)
        cfg["contact"]["dual_hmm"]["window_size"] = int(min(500, max(50, tr)))
        dets = build_dual_hmm_detectors_from_cfg(cfg, recording=None, kin_model=kin_ds)
    except Exception as e:
        print(f"[train_dual_hmm] skip plot: {e}")
        return

    plots_dir = out.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin_ds, dets)
        fig_path = plots_dir / f"dual_hmm_contact_train_{rec.sequence_name}.png"
        plot_grf_contact_overview(
            t_abs,
            grfs,
            st,
            ps,
            suptitle=f"Dual HMM (trained, online) — {rec.sequence_name}",
            save_path=fig_path,
            show=False,
        )
        print(f"Wrote {fig_path.resolve()}")
    except Exception as e:
        print(f"[train_dual_hmm] skip plot: replay failed: {e}")


if __name__ == "__main__":
    main()
