"""
CLI: run GMM+HMM contact on a recording and plot GRF vs stance / p_stance per leg.

Does not run the EKF. Example::

    python -m leg_odom.contact.gmm_hmm.visualize \\
        --sequence-dir /path/to/seq --mode offline \\
        --robot-kinematics anymal

Online mode requires ``--pretrained-path`` (same resolution as the detector).
``--history-length`` applies to **online** only; **offline** always fits and runs with instant ``N=1``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leg_odom.contact.grf_stance_plot import plot_grf_contact_overview
from leg_odom.contact.gmm_hmm.detector import GmmHmmContactDetector
from leg_odom.contact.gmm_hmm.fitting import fit_offline_per_leg
from leg_odom.features import DEFAULT_INSTANT_FEATURE_FIELDS, parse_instant_feature_fields
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.kinematics_factory import build_kinematics_backend


def _minimal_cfg(sequence_dir: str, robot: str, dataset_kind: str) -> dict:
    return {
        "schema_version": 1,
        "run": {
            "name": "viz",
            "debug": {
                "enabled": False,
                "live_visualizer": {
                    "enabled": False,
                    "sliding_window_s": 10.0,
                    "buffer_length": 5000,
                    "video_path": None,
                },
            },
        },
        "robot": {"kinematics": robot},
        "dataset": {
            "kind": dataset_kind,
            "sequence_dir": str(Path(sequence_dir).expanduser().resolve()),
        },
        "contact": {"detector": "none"},
        "ekf": {"noise_config": None},
        "output": {"base_dir": ".", "include_timestamp": False},
    }


def build_gmm_hmm_detectors_for_replay(
    *,
    mode: str,
    rec,
    kin,
    fields: tuple[str, ...],
    history_length: int,
    pretrained_path: str | None,
    trans_stay: float,
    fit_interval: int,
    window_size: int,
    random_state: int,
    degeneracy_max_weight: float = 0.98,
) -> list[GmmHmmContactDetector]:
    """Construct per-leg detectors for offline (prefit) or online (npz) replay."""
    n = int(history_length)
    if mode == "offline":
        params = fit_offline_per_leg(
            rec,
            kin,
            feature_fields=fields,
            history_length=1,
            random_state=random_state,
        )
        return [
            GmmHmmContactDetector(
                feature_fields=fields,
                history_length=1,
                trans_stay=trans_stay,
                mode="offline",
                initial_means=m,
                initial_covariances=c,
                fit_interval=fit_interval,
                window_size=window_size,
                degeneracy_max_weight=degeneracy_max_weight,
                random_state=random_state,
            )
            for m, c in params
        ]
    return [
        GmmHmmContactDetector(
            feature_fields=fields,
            history_length=n,
            trans_stay=trans_stay,
            mode="online",
            pretrained_path=pretrained_path,
            fit_interval=fit_interval,
            window_size=window_size,
            degeneracy_max_weight=degeneracy_max_weight,
            random_state=random_state,
        )
        for _ in range(kin.n_legs)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize GMM+HMM contact vs GRF (no EKF)")
    p.add_argument("--sequence-dir", type=str, required=True)
    p.add_argument("--dataset-kind", type=str, default="tartanground")
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument("--mode", type=str, default="offline", choices=("offline", "online"))
    p.add_argument("--pretrained-path", type=str, default="weights.npz")
    p.add_argument("--feature-fields", type=str, default=",".join(DEFAULT_INSTANT_FEATURE_FIELDS))
    p.add_argument("--history-length", type=int, default=1)
    p.add_argument("--trans-stay", type=float, default=0.99)
    p.add_argument("--fit-interval", type=int, default=250)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--save", type=str, default="", help="Optional PNG path")
    args = p.parse_args()

    fields = tuple(s.strip() for s in args.feature_fields.split(",") if s.strip())
    parse_instant_feature_fields(fields)

    if args.mode == "online" and not str(args.pretrained_path).strip():
        raise SystemExit("online mode requires --pretrained-path")

    cfg = _minimal_cfg(args.sequence_dir, args.robot_kinematics, args.dataset_kind)
    ds = build_leg_odometry_dataset(cfg)
    kin = build_kinematics_backend(cfg)
    rec = ds[0]

    dets = build_gmm_hmm_detectors_for_replay(
        mode=args.mode,
        rec=rec,
        kin=kin,
        fields=fields,
        history_length=int(args.history_length),
        pretrained_path=str(args.pretrained_path) if args.pretrained_path else None,
        trans_stay=float(args.trans_stay),
        fit_interval=int(args.fit_interval),
        window_size=int(args.window_size),
        random_state=int(args.random_state),
    )
    t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin, dets)
    save_path = Path(args.save).expanduser() if str(args.save).strip() else None
    plot_grf_contact_overview(
        t_abs,
        grfs,
        st,
        ps,
        suptitle=f"GMM+HMM contact ({args.mode}) — {rec.sequence_name}",
        save_path=save_path,
        show=save_path is None,
    )
    if save_path is not None:
        print(f"Wrote {save_path.resolve()}")


if __name__ == "__main__":
    main()
