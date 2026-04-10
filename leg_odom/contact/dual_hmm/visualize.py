"""
CLI: run dual HMM contact on a recording and plot GRF vs stance / p_stance per leg (no EKF).

Example::

    python -m leg_odom.contact.dual_hmm.visualize \\
        --sequence-dir /path/to/seq --mode offline --robot-kinematics anymal

Online mode requires ``--pretrained-path`` (from ``python -m leg_odom.training.dual_hmm.train_dual_hmm``).
``--history-length`` is ignored when ``--mode offline`` (kin branch is always instant ``N=1``).
With ``--use-energy``, replay collects ``last_energy_normalized`` and passes optional energy traces to the plotter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leg_odom.contact.base import ContactDetectorStepInput
from leg_odom.contact.dual_hmm.detector import DualHmmContactDetector, build_dual_hmm_detectors_from_cfg
from leg_odom.contact.grf_stance_plot import plot_grf_contact_overview
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.features import DEFAULT_INSTANT_FEATURE_FIELDS
from leg_odom.io.columns import FOOT_FORCE_COLS, motor_position_cols, motor_torque_cols, motor_velocity_cols
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.kinematics_factory import build_kinematics_backend

_IMU_GYRO = ("gyro_x", "gyro_y", "gyro_z")
_IMU_ACCEL = ("accel_x", "accel_y", "accel_z")


def _replay_collect_energy(
    timeline: pd.DataFrame,
    kin,
    detectors: list[DualHmmContactDetector],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Same row-order replay as ``replay_detectors_on_timeline``, plus per-leg normalized energy traces."""
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())
    jpl = kin.joints_per_leg
    n_legs = kin.n_legs
    n_steps = len(timeline)
    t_abs = timeline["t_abs"].to_numpy(dtype=np.float64)
    grfs = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]
    st = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]
    ps = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]
    en = [np.zeros(n_steps, dtype=np.float64) for _ in range(n_legs)]

    for d in detectors:
        d.reset()

    for k in range(n_steps):
        row = timeline.iloc[k]
        gyro = row[list(_IMU_GYRO)].to_numpy(dtype=np.float64)
        accel = row[list(_IMU_ACCEL)].to_numpy(dtype=np.float64)
        q_all = row[motor_cols].to_numpy(dtype=np.float64)
        dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        for leg in range(n_legs):
            sl = slice(leg * jpl, (leg + 1) * jpl)
            q_leg = q_all[sl]
            dq_leg = dq_all[sl]
            tau_leg = tau_all[sl]
            pb = np.asarray(kin.fk(leg, q_leg), dtype=np.float64).reshape(3)
            jac = np.asarray(kin.J_analytical(leg, q_leg), dtype=np.float64).reshape(3, jpl)
            qd = np.asarray(dq_leg, dtype=np.float64).reshape(jpl)
            v_foot_body = np.cross(gyro, pb) + jac @ qd
            grf = float(row.get(FOOT_FORCE_COLS[leg], 0.0))
            grfs[leg][k] = grf
            step = ContactDetectorStepInput(
                grf_n=grf,
                p_foot_body=pb,
                v_foot_body=v_foot_body,
                q_leg=np.asarray(q_leg, dtype=np.float64, order="C"),
                dq_leg=np.asarray(dq_leg, dtype=np.float64, order="C"),
                tau_leg=np.asarray(tau_leg, dtype=np.float64, order="C"),
                gyro_body_corrected=np.asarray(gyro, dtype=np.float64, order="C"),
                accel_body_corrected=np.asarray(accel, dtype=np.float64, order="C"),
            )
            est = detectors[leg].update(step)
            st[leg][k] = 1.0 if est.stance else 0.0
            ps[leg][k] = float(est.p_stance)
            en[leg][k] = float(detectors[leg].last_energy_normalized)
    return t_abs, grfs, st, ps, en


def _minimal_cfg(sequence_dir: str, robot: str, dataset_kind: str) -> dict:
    return {
        "schema_version": 1,
        "run": {
            "name": "dual_viz",
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
        "dataset": {
            "kind": dataset_kind,
            "sequence_dir": str(Path(sequence_dir).expanduser().resolve()),
        },
        "contact": {"detector": "none"},
        "ekf": {"noise_config": None},
        "output": {"base_dir": ".", "include_timestamp": False},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize dual HMM contact vs GRF (no EKF)")
    p.add_argument("--sequence-dir", type=str, required=True)
    p.add_argument("--dataset-kind", type=str, default="tartanground")
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument("--mode", type=str, default="offline", choices=("offline", "online"))
    p.add_argument("--pretrained-path", type=str, default="")
    p.add_argument("--feature-fields", type=str, default=",".join(DEFAULT_INSTANT_FEATURE_FIELDS))
    p.add_argument("--history-length", type=int, default=1)
    p.add_argument("--trans-stay", type=float, default=0.99)
    p.add_argument("--fit-interval", type=int, default=250)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--degeneracy-max-weight", type=float, default=0.98)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--use-energy", action="store_true")
    p.add_argument("--save", type=str, default="", help="Optional PNG path")
    args = p.parse_args()

    fields = tuple(s.strip() for s in args.feature_fields.split(",") if s.strip())
    if args.mode == "online" and not str(args.pretrained_path).strip():
        raise SystemExit("online mode requires --pretrained-path")

    base = _minimal_cfg(args.sequence_dir, args.robot_kinematics, args.dataset_kind)
    hl = 1 if str(args.mode) == "offline" else int(args.history_length)
    dual_block: dict = {
        "mode": str(args.mode),
        "feature_fields": list(fields),
        "history_length": hl,
        "trans_stay": float(args.trans_stay),
        "fit_interval": int(args.fit_interval),
        "window_size": int(args.window_size),
        "degeneracy_max_weight": float(args.degeneracy_max_weight),
        "random_state": int(args.random_state),
        "use_energy": bool(args.use_energy),
        "verbose": False,
    }
    if str(args.pretrained_path).strip():
        dual_block["pretrained_path"] = str(args.pretrained_path).strip()
    base["contact"] = {"detector": "dual_hmm", "dual_hmm": dual_block}

    ds = build_leg_odometry_dataset(base)
    kin = build_kinematics_backend(base)
    rec = ds[0]
    dets = build_dual_hmm_detectors_from_cfg(
        base,
        recording=rec if args.mode == "offline" else None,
        kin_model=kin,
    )
    save_path = Path(args.save).expanduser() if str(args.save).strip() else None
    energy_kw: dict = {}
    if args.use_energy:
        t_abs, grfs, st, ps, energy_per_leg = _replay_collect_energy(rec.frames, kin, dets)
        energy_kw["energy_per_leg"] = energy_per_leg
    else:
        t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin, dets)
    plot_grf_contact_overview(
        t_abs,
        grfs,
        st,
        ps,
        suptitle=f"Dual HMM ({args.mode}) — {rec.sequence_name}",
        save_path=save_path,
        show=save_path is None,
        **energy_kw,
    )
    if save_path is not None:
        print(f"Wrote {save_path.resolve()}")


if __name__ == "__main__":
    main()
