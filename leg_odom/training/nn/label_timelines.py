"""
Per-sequence label timelines for NN training via **contact detectors** + replay.

- ``labels.method: grf_threshold``: :func:`~leg_odom.contact.grf_threshold.build_grf_threshold_detectors_from_cfg`
  + :func:`~leg_odom.contact.replay_timeline.replay_detectors_on_timeline` (same OOP path as EKF).
- ``gmm_hmm``: offline GMM+HMM + replay.

``model.window_size`` is the NN temporal input only; GMM pseudo-labeler uses ``history_length: 1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from leg_odom.contact.grf_threshold import build_grf_threshold_detectors_from_cfg
from leg_odom.contact.gmm_hmm.detector import build_gmm_hmm_detectors_from_cfg
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.training.nn.dataset_kind import infer_dataset_kind_from_sequence_dir
from leg_odom.training.nn.precomputed_io import load_precomputed_sequence_npz


def _load_recording_for_labels(sequence_dir: str | Path, *, validate_frames: bool):
    seq_dir = Path(sequence_dir).expanduser().resolve()
    dataset_kind = infer_dataset_kind_from_sequence_dir(seq_dir)
    ds = build_leg_odometry_dataset(
        {"dataset": {"kind": dataset_kind, "sequence_dir": str(seq_dir)}},
        validate=validate_frames,
    )
    return ds[0]


def _contact_cfg_for_nn_grf_labels(grf_threshold_block: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``labels.grf_threshold`` into experiment-shaped ``contact`` config (detector factory)."""
    return {"contact": {"detector": "grf_threshold", "grf_threshold": dict(grf_threshold_block)}}


def stance_timeline_grf_threshold(
    *,
    sequence_dir: str | Path,
    grf_threshold_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
) -> dict[int, npt.NDArray[np.float64]]:
    """
    :class:`~leg_odom.contact.grf_threshold.GrfThresholdContactDetector` replay → per-leg stance ``(T,)`` 0/1.

    ``grf_threshold_cfg`` uses the same keys as experiment ``contact.grf_threshold`` (at least
    ``force_threshold``).
    """
    cfg = _contact_cfg_for_nn_grf_labels(grf_threshold_cfg)
    recording = _load_recording_for_labels(sequence_dir, validate_frames=validate_frames)
    detectors = build_grf_threshold_detectors_from_cfg(cfg)
    n_legs = int(kin.n_legs)
    if len(detectors) != n_legs:
        raise RuntimeError(
            f"GRF threshold factory returned {len(detectors)} detectors but kinematics has n_legs={n_legs}"
        )
    _t_abs, _grf, st_list, _ps = replay_detectors_on_timeline(recording.frames, kin, detectors)
    out: dict[int, npt.NDArray[np.float64]] = {}
    for leg in range(n_legs):
        out[leg] = np.asarray(st_list[leg], dtype=np.float64).reshape(-1)
    return out


def precompute_grf_threshold_stance_by_seq(
    npz_paths: Sequence[Path | str],
    foot_forces_by_seq: Mapping[Path, npt.NDArray[np.float64]],
    labels_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    *,
    expected_robot_kinematics: str,
    validate_frames: bool,
    show_progress: bool = True,
) -> dict[Path, dict[int, npt.NDArray[np.float64]]]:
    """For each precomputed npz: replay ``GrfThresholdContactDetector`` on ``sequence_dir_stored``."""
    grf_block = labels_cfg.get("grf_threshold")
    if not isinstance(grf_block, Mapping):
        raise ValueError(
            "labels.grf_threshold mapping required when labels.method is grf_threshold (same keys as contact.grf_threshold)"
        )
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in npz_paths:
        key = Path(p).expanduser().resolve()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)

    n_legs = int(kin.n_legs)
    out: dict[Path, dict[int, npt.NDArray[np.float64]]] = {}

    ordered_iter = tqdm(ordered, desc="GRF threshold contact labels", unit="seq") if show_progress else ordered
    for key in ordered_iter:
        bundle = load_precomputed_sequence_npz(
            key,
            expected_robot_kinematics=expected_robot_kinematics,
            n_legs=n_legs,
        )
        t_expect = int(foot_forces_by_seq[key].shape[0])
        if int(bundle.foot_forces.shape[0]) != t_expect:
            raise RuntimeError(
                f"{key}: foot_forces length mismatch bundle {bundle.foot_forces.shape[0]} "
                f"vs loaded split {t_expect}"
            )
        seq_dir = Path(bundle.sequence_dir_stored).expanduser().resolve()
        by_leg = stance_timeline_grf_threshold(
            sequence_dir=seq_dir,
            grf_threshold_cfg=grf_block,
            kin=kin,
            validate_frames=validate_frames,
        )
        for leg in range(n_legs):
            y = by_leg[leg]
            if int(y.shape[0]) != t_expect:
                raise RuntimeError(
                    f"{key} leg {leg}: GRF stance length {y.shape[0]} != precomputed T={t_expect} "
                    f"(check sequence_dir {seq_dir} vs preprocess)."
                )
        out[key] = by_leg
    return out


def _contact_cfg_for_nn_gmm_labels(gmm_hmm_block: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``labels.gmm_hmm`` into experiment-shaped ``contact`` config (offline, N=1)."""
    block = dict(gmm_hmm_block)
    if block.get("pretrained_path"):
        raise ValueError(
            "labels.gmm_hmm.pretrained_path is not supported for NN training labels "
            "(offline per-sequence fit only)."
        )
    mode = str(block.get("mode", "offline")).lower()
    if mode != "offline":
        raise ValueError(f"labels.gmm_hmm.mode must be 'offline' for NN labels, got {mode!r}")
    hl = int(block.get("history_length", 1))
    if hl != 1:
        raise ValueError(
            f"labels.gmm_hmm.history_length must be 1 for NN labels (instant GMM emissions), got {hl}"
        )
    block["mode"] = "offline"
    block["history_length"] = 1
    return {"contact": {"detector": "gmm", "gmm": block}}


def stance_timeline_gmm_hmm(
    *,
    sequence_dir: str | Path,
    gmm_hmm_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
) -> dict[int, npt.NDArray[np.float64]]:
    """
    One offline GMM+HMM fit on **this** recording, then replay → per-leg binary stance ``(T,)`` float 0/1.

    Parameters
    ----------
        Trajectory folder containing either split CSVs (imu + bag) or Ocelot ``lowstate.csv``.
    gmm_hmm_cfg
        Contents of ``labels.gmm_hmm`` (validated: ``mode: offline``, ``history_length: 1``).
    kin
        Kinematics matching the precomputed bundle.
    validate_frames
        Passed to :func:`~leg_odom.run.dataset_factory.build_leg_odometry_dataset` via inferred ``dataset.kind``.
    """
    cfg = _contact_cfg_for_nn_gmm_labels(gmm_hmm_cfg)
    recording = _load_recording_for_labels(sequence_dir, validate_frames=validate_frames)
    detectors = build_gmm_hmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin)
    _t_abs, _grf, st_list, _ps = replay_detectors_on_timeline(recording.frames, kin, detectors)
    n_legs = int(kin.n_legs)
    out: dict[int, npt.NDArray[np.float64]] = {}
    for leg in range(n_legs):
        out[leg] = np.asarray(st_list[leg], dtype=np.float64).reshape(-1)
    return out


def precompute_gmm_hmm_stance_by_seq(
    npz_paths: Sequence[Path | str],
    foot_forces_by_seq: Mapping[Path, npt.NDArray[np.float64]],
    labels_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    *,
    expected_robot_kinematics: str,
    validate_frames: bool,
    show_progress: bool = True,
) -> dict[Path, dict[int, npt.NDArray[np.float64]]]:
    """
    For each distinct precomputed npz path: offline GMM+HMM on ``sequence_dir_stored`` only.

    Asserts stance length matches ``foot_forces`` rows for that bundle.
    """
    gmm_block = labels_cfg.get("gmm_hmm")
    if not isinstance(gmm_block, Mapping):
        raise ValueError("labels.gmm_hmm mapping required for gmm_hmm label method")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in npz_paths:
        key = Path(p).expanduser().resolve()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)

    n_legs = int(kin.n_legs)
    out: dict[Path, dict[int, npt.NDArray[np.float64]]] = {}

    ordered_iter = tqdm(ordered, desc="GMM+HMM pseudo-labels", unit="seq") if show_progress else ordered
    for key in ordered_iter:
        bundle = load_precomputed_sequence_npz(
            key,
            expected_robot_kinematics=expected_robot_kinematics,
            n_legs=n_legs,
        )
        t_expect = int(foot_forces_by_seq[key].shape[0])
        if int(bundle.foot_forces.shape[0]) != t_expect:
            raise RuntimeError(
                f"{key}: foot_forces length mismatch bundle {bundle.foot_forces.shape[0]} "
                f"vs loaded split {t_expect}"
            )
        seq_dir = Path(bundle.sequence_dir_stored).expanduser().resolve()
        by_leg = stance_timeline_gmm_hmm(
            sequence_dir=seq_dir,
            gmm_hmm_cfg=gmm_block,
            kin=kin,
            validate_frames=validate_frames,
        )
        for leg in range(n_legs):
            y = by_leg[leg]
            if int(y.shape[0]) != t_expect:
                raise RuntimeError(
                    f"{key} leg {leg}: GMM stance length {y.shape[0]} != precomputed T={t_expect} "
                    f"(check sequence_dir {seq_dir} vs preprocess)."
                )
        out[key] = by_leg
    return out
