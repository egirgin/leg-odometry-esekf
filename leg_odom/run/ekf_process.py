"""
Main EKF **process loop** orchestration: dataset → timeline iteration → filter hooks.

This module is the integration point for ``main.py``. Robot- and dataset-specific names
are kept in config (``robot.kinematics``, ``dataset.kind``); here we use neutral terms
``recording`` (:class:`~leg_odom.datasets.types.LegOdometrySequence`) and ``kin_model``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.contact.base import ContactDetectorStepInput
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.eval.ekf_step_log import (
    EkfStepLogWriter,
    build_ekf_step_log_row,
    empty_zupt_info,
    sanitize_sequence_slug,
)
from leg_odom.filters.esekf import ErrorStateEkf, build_error_state_ekf
from leg_odom.filters.zupt_measurement import zupt_isotropic_meas_from_p_stance
from leg_odom.io.columns import (
    FOOT_FORCE_COLS,
    IMU_ACCEL_COLS,
    IMU_BODY_QUAT_COLS,
    IMU_GYRO_COLS,
    motor_position_cols,
    motor_torque_cols,
    motor_velocity_cols,
)
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.run.contact_factory import ContactStack, build_contact_stack
from leg_odom.run.ekf_nominal_init import (
    apply_nominal_init_from_timeline,
    ekf_initialize_nominal_from_data_enabled,
)
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.experiment_config import (
    live_visualizer_buffer_length,
    live_visualizer_sliding_window_s,
    live_visualizer_update_hz,
    live_visualizer_video_path,
)
from leg_odom.run.kinematics_factory import build_kinematics_backend


@dataclass
class EkfProcessSummary:
    """Aggregated result after running the EKF on the single dataset recording."""

    robot_kinematics: str
    dataset_kind: str
    contact_detector: str
    sequence_name: str = ""
    median_rate_hz: float = 0.0
    ekf_history_csv: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "robot_kinematics": self.robot_kinematics,
            "dataset_kind": self.dataset_kind,
            "contact_detector": self.contact_detector,
            "sequence_name": self.sequence_name,
            "median_rate_hz": self.median_rate_hz,
        }
        if self.ekf_history_csv:
            d["ekf_history_csv"] = self.ekf_history_csv
        return d


def run_ekf_on_recording(
    recording: LegOdometrySequence,
    *,
    kin_model: BaseKinematics,
    filter_state: ErrorStateEkf,
    contact_stack: ContactStack,
    experiment_cfg: Mapping[str, Any] | None = None,
    history_csv_path: Path | None = None,
    debug: bool = False,
    live_visualizer: bool = False,
) -> tuple[str, float, str | None]:
    """
    Execute one recording: IMU prediction each step; optional GRF ZUPT; optional CSV history.

    When ``history_csv_path`` is set, writes one row per timestep (see
    :mod:`leg_odom.eval.ekf_step_log`) with nominal state, biases, position-error ``P`` diagonal,
    contact,
    ZUPT NIS, and per-foot world velocities.

    ``debug`` enables a short end-of-recording print; ``live_visualizer`` alone opens the
    matplotlib replay (normally both come from :mod:`leg_odom.run.experiment_config` with
    debug effective **and** ``run.debug.live_visualizer.enabled``).

    ``contact_stack`` comes from :func:`~leg_odom.run.contact_factory.build_contact_stack`
    (includes ``detector_id`` for logging). When ``experiment_cfg["ekf"]["initialize_nominal_from_data"]``
    is true, nominal ``p``, ``v``, ``R`` (and optional biases) are set from the merged timeline
    before the loop (see :mod:`leg_odom.run.ekf_nominal_init`).

    Returns ``(sequence_name, median_rate_hz, ekf_history_csv_path_or_none)``.
    """
    filter_state.reset()

    cfg_map = experiment_cfg if isinstance(experiment_cfg, Mapping) else {}
    foot_dets = contact_stack.per_foot
    if foot_dets is not None:
        for d in foot_dets:
            d.reset()

    timeline = recording.frames
    if ekf_initialize_nominal_from_data_enabled(cfg_map):
        apply_nominal_init_from_timeline(filter_state, timeline)
    filter_state.set_contact_pipeline(contact_stack.detector_id)

    gyro_cols = list(IMU_GYRO_COLS)
    accel_cols = list(IMU_ACCEL_COLS)
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())

    accel_gc = bool(recording.meta.get("accel_gravity_compensated", False))

    n_legs = kin_model.n_legs
    log_writer: EkfStepLogWriter | None = None
    hist_resolved: str | None = None
    if history_csv_path is not None:
        hp = Path(history_csv_path)
        log_writer = EkfStepLogWriter(hp, n_legs=n_legs)
        hist_resolved = str(hp.resolve())

    n_timesteps = len(timeline)
    t_abs_series = timeline["t_abs"].to_numpy(dtype=np.float64)
    t_start_viz = float(t_abs_series[0])
    t_end_viz = float(t_abs_series[n_timesteps - 1]) if n_timesteps > 0 else t_start_viz + 1.0

    live_viz = None
    if live_visualizer:
        try:
            from leg_odom.eval.live_visualizer import LiveVisualizer

            gt_pdf = recording.position_ground_truth
            gt_for_viz = None
            if gt_pdf is not None and not gt_pdf.empty:
                gt_for_viz = gt_pdf.copy()
                # ``extract_position_ground_truth`` keeps only position + time; copy body quats
                # from the merged timeline so heading uses ``zyx`` yaw (not noisy velocity atan2).
                if len(gt_for_viz) == len(recording.frames):
                    for c in IMU_BODY_QUAT_COLS:
                        if c in recording.frames.columns and c not in gt_for_viz.columns:
                            gt_for_viz[c] = recording.frames[c].to_numpy()
            cfg_map = experiment_cfg if isinstance(experiment_cfg, Mapping) else {}
            slide_s = live_visualizer_sliding_window_s(cfg_map)
            live_viz = LiveVisualizer(
                recording.sequence_name,
                groundtruth_df=gt_for_viz,
                t_start=t_start_viz,
                t_end=t_end_viz,
                buffer_length=live_visualizer_buffer_length(cfg_map),
                video_path=live_visualizer_video_path(cfg_map),
                sliding_window_s=slide_s,
                dataset_hz=float(recording.median_rate_hz),
                update_hz=live_visualizer_update_hz(cfg_map),
            )
        except Exception as e:
            print(f"[ekf] live visualizer disabled: {e}", flush=True)

    try:
        for k in range(len(timeline)):
            row = timeline.iloc[k]
            dt_s = float(row["dt"])
            gyro = row[gyro_cols].to_numpy(dtype=np.float64)
            accel = row[accel_cols].to_numpy(dtype=np.float64)
            filter_state.imu_predict(dt_s, gyro, accel, accel_gravity_compensated=accel_gc)

            gyro_corr = gyro - filter_state.bias_gyro
            accel_corr = accel - filter_state.bias_accel

            q_all = row[motor_cols].to_numpy(dtype=np.float64)
            dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
            tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)

            stance = [False] * n_legs
            contact_scores = [0.0] * n_legs
            contact_vars = [float("nan")] * n_legs
            foot_kin: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            stance_legs: list[dict[str, Any]] = []

            for leg_index in range(n_legs):
                sl = slice(
                    leg_index * kin_model.joints_per_leg,
                    (leg_index + 1) * kin_model.joints_per_leg,
                )
                q_leg = q_all[sl]
                dq_leg = dq_all[sl]
                tau_leg = tau_all[sl]
                p_b = kin_model.fk(leg_index, q_leg)
                jacobian = kin_model.J_analytical(leg_index, q_leg)
                pb = np.asarray(p_b, dtype=np.float64).reshape(3)
                jac = np.asarray(jacobian, dtype=np.float64)
                nj = jac.size // 3
                jac = jac.reshape(3, nj)
                qd = np.asarray(dq_leg, dtype=np.float64).reshape(nj)
                v_foot_body = np.cross(gyro_corr, pb) + jac @ qd
                foot_kin.append((p_b, jacobian, dq_leg))

                if foot_dets is not None:
                    grf = float(row.get(FOOT_FORCE_COLS[leg_index], 0.0))
                    step_in = ContactDetectorStepInput(
                        grf_n=grf,
                        p_foot_body=np.asarray(p_b, dtype=np.float64, order="C"),
                        v_foot_body=np.asarray(v_foot_body, dtype=np.float64, order="C"),
                        q_leg=np.asarray(q_leg, dtype=np.float64, order="C"),
                        dq_leg=np.asarray(dq_leg, dtype=np.float64, order="C"),
                        tau_leg=np.asarray(tau_leg, dtype=np.float64, order="C"),
                        gyro_body_corrected=np.asarray(gyro_corr, dtype=np.float64, order="C"),
                        accel_body_corrected=np.asarray(accel_corr, dtype=np.float64, order="C"),
                    )
                    det = foot_dets[leg_index]
                    est = det.update(step_in)
                    stance[leg_index] = bool(est.stance)
                    contact_scores[leg_index] = float(est.p_stance)
                    sigma_sq, r_foot = zupt_isotropic_meas_from_p_stance(float(est.p_stance))
                    contact_vars[leg_index] = float(sigma_sq)
                    if est.stance:
                        r_foot = np.asarray(r_foot, dtype=np.float64).reshape(3, 3)
                        if np.all(np.isfinite(r_foot)):
                            stance_legs.append(
                                {
                                    "leg_id": leg_index,
                                    "p_foot_body": p_b,
                                    "J": jacobian,
                                    "qdot": dq_leg,
                                    "R_foot": r_foot.copy(),
                                    "qscore": float(est.p_stance),
                                }
                            )

            zupt_info: dict[str, Any] = empty_zupt_info()
            if stance_legs:
                zupt_info = filter_state.update_zupt(stance_legs, gyro)

            if log_writer is not None:
                log_writer.write_row(
                    build_ekf_step_log_row(
                        row,
                        filter_state,
                        gyro_raw=gyro,
                        foot_kin=foot_kin,
                        stance=stance,
                        contact_score=contact_scores,
                        contact_zupt_var=contact_vars,
                        zupt_info=zupt_info,
                        n_legs=n_legs,
                    )
                )

            if live_viz is not None:
                yaw_est = float(
                    Rotation.from_matrix(np.asarray(filter_state.R, dtype=np.float64)).as_euler(
                        "zyx"
                    )[0]
                )
                live_viz.update(
                    float(filter_state.p[0]),
                    float(filter_state.p[1]),
                    float(filter_state.p[2]),
                    float(filter_state.v[0]),
                    float(filter_state.v[1]),
                    float(filter_state.v[2]),
                    t_abs=float(row["t_abs"]),
                    yaw_est=yaw_est,
                )
    finally:
        if log_writer is not None:
            log_writer.close()
        if live_viz is not None:
            live_viz.close()

    if debug:
        n_steps = len(timeline)
        print(
            f"[ekf] recording {recording.sequence_name!r}: steps={n_steps}, "
            f"hz~{recording.median_rate_hz:.1f}, detector={contact_stack.detector_id!r}",
            flush=True,
        )

    return (
        recording.sequence_name,
        float(recording.median_rate_hz),
        hist_resolved,
    )


def run_ekf_pipeline(
    resolved_cfg: Mapping[str, Any],
    *,
    run_dir: Path | None = None,
    debug: bool = False,
    live_visualizer: bool = False,
    workspace_root: Path | None = None,
) -> EkfProcessSummary:
    """
    Load dataset from resolved experiment config and run the ESEKF on the single recording.

    Parameters
    ----------
    resolved_cfg
        Experiment mapping with absolute ``dataset.sequence_dir`` (e.g. from
        ``experiment_resolved.yaml``).
    run_dir
        If set, writes ``ekf_process_summary.json`` and
        ``ekf_history_<sequence_slug>.csv`` under ``run_dir``.
    workspace_root
        Repo root for resolving relative ``ekf.noise_config`` when paths are not absolute.
    """
    dataset = build_leg_odometry_dataset(resolved_cfg)
    kin_model = build_kinematics_backend(resolved_cfg)
    recording = dataset[0]
    contact_stack = build_contact_stack(
        resolved_cfg,
        recording=recording,
        kin_model=kin_model,
        workspace_root=workspace_root,
    )

    summary = EkfProcessSummary(
        robot_kinematics=str(resolved_cfg["robot"]["kinematics"]).lower(),
        dataset_kind=str(resolved_cfg["dataset"]["kind"]).lower(),
        contact_detector=contact_stack.detector_id,
    )

    ekf = build_error_state_ekf(resolved_cfg, workspace_root=workspace_root)
    hist_path = None
    if run_dir is not None:
        slug = sanitize_sequence_slug(recording.sequence_name)
        hist_path = Path(run_dir) / f"ekf_history_{slug}.csv"
    seq_name, hz, hist = run_ekf_on_recording(
        recording,
        kin_model=kin_model,
        filter_state=ekf,
        contact_stack=contact_stack,
        experiment_cfg=resolved_cfg,
        history_csv_path=hist_path,
        debug=debug,
        live_visualizer=live_visualizer,
    )
    summary.sequence_name = seq_name
    summary.median_rate_hz = hz
    summary.ekf_history_csv = hist

    if run_dir is not None:
        out = Path(run_dir) / "ekf_process_summary.json"
        out.write_text(json.dumps(summary.to_json_dict(), indent=2), encoding="utf-8")

    return summary
