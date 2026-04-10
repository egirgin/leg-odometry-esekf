"""
Contact-detector replay → per-leg stance timelines for NN precompute.

- ``labels.method: grf_threshold``: GRF threshold detectors + replay (same path as EKF).
- ``gmm_hmm``: offline GMM+HMM + replay (YAML ``history_length`` coerced to instant ``N=1``).
- ``dual_hmm``: offline dual HMM + replay (``mode: offline``; kin branch coerced to ``N=1``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from leg_odom.contact.grf_threshold import build_grf_threshold_detectors_from_cfg
from leg_odom.contact.dual_hmm.detector import build_dual_hmm_detectors_from_cfg
from leg_odom.contact.gmm_hmm.detector import build_gmm_hmm_detectors_from_cfg
from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.run.dataset_factory import build_leg_odometry_dataset


def _load_recording_for_labels(
    sequence_dir: str | Path,
    dataset_kind: str,
    *,
    validate_frames: bool,
):
    seq_dir = Path(sequence_dir).expanduser().resolve()
    ds = build_leg_odometry_dataset(
        {"dataset": {"kind": str(dataset_kind).strip(), "sequence_dir": str(seq_dir)}},
        validate=validate_frames,
    )
    return ds[0]


def _contact_cfg_for_grf_labels(grf_threshold_block: Mapping[str, Any]) -> dict[str, Any]:
    return {"contact": {"detector": "grf_threshold", "grf_threshold": dict(grf_threshold_block)}}


def stance_timeline_grf_threshold(
    *,
    sequence_dir: str | Path,
    dataset_kind: str,
    grf_threshold_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
) -> dict[int, npt.NDArray[np.float64]]:
    """GrfThresholdContactDetector replay → per-leg stance ``(T,)`` 0/1."""
    cfg = _contact_cfg_for_grf_labels(grf_threshold_cfg)
    recording = _load_recording_for_labels(
        sequence_dir, dataset_kind, validate_frames=validate_frames
    )
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


def _contact_cfg_for_gmm_labels(gmm_hmm_block: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(gmm_hmm_block)
    if block.get("pretrained_path"):
        raise ValueError("labels.gmm_hmm.pretrained_path is not supported for precompute labels")
    mode = str(block.get("mode", "offline")).lower()
    if mode != "offline":
        raise ValueError(f"labels.gmm_hmm.mode must be 'offline' for precompute labels, got {mode!r}")
    block["mode"] = "offline"
    block["history_length"] = 1
    return {"contact": {"detector": "gmm", "gmm": block}}


def _contact_cfg_for_dual_labels(dual_block: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(dual_block)
    if block.get("pretrained_path"):
        raise ValueError("labels.dual_hmm.pretrained_path is not supported for precompute labels (offline fit only)")
    mode = str(block.get("mode", "offline")).lower()
    if mode != "offline":
        raise ValueError(f"labels.dual_hmm.mode must be offline for precompute labels, got {mode!r}")
    block["mode"] = "offline"
    block["history_length"] = 1
    return {"contact": {"detector": "dual_hmm", "dual_hmm": block}}


def stance_timeline_dual_hmm(
    *,
    sequence_dir: str | Path,
    dataset_kind: str,
    dual_hmm_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
) -> dict[int, npt.NDArray[np.float64]]:
    """Offline dual HMM on this recording, then replay → per-leg binary stance ``(T,)`` float 0/1."""
    cfg = _contact_cfg_for_dual_labels(dual_hmm_cfg)
    recording = _load_recording_for_labels(
        sequence_dir, dataset_kind, validate_frames=validate_frames
    )
    detectors = build_dual_hmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin)
    _t_abs, _grf, st_list, _ps = replay_detectors_on_timeline(recording.frames, kin, detectors)
    n_legs = int(kin.n_legs)
    out: dict[int, npt.NDArray[np.float64]] = {}
    for leg in range(n_legs):
        out[leg] = np.asarray(st_list[leg], dtype=np.float64).reshape(-1)
    return out


def stance_timeline_gmm_hmm(
    *,
    sequence_dir: str | Path,
    dataset_kind: str,
    gmm_hmm_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
) -> dict[int, npt.NDArray[np.float64]]:
    """Offline GMM+HMM on this recording, then replay → per-leg binary stance ``(T,)`` float 0/1."""
    cfg = _contact_cfg_for_gmm_labels(gmm_hmm_cfg)
    recording = _load_recording_for_labels(
        sequence_dir, dataset_kind, validate_frames=validate_frames
    )
    detectors = build_gmm_hmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin)
    _t_abs, _grf, st_list, _ps = replay_detectors_on_timeline(recording.frames, kin, detectors)
    n_legs = int(kin.n_legs)
    out: dict[int, npt.NDArray[np.float64]] = {}
    for leg in range(n_legs):
        out[leg] = np.asarray(st_list[leg], dtype=np.float64).reshape(-1)
    return out


def stance_by_leg_from_labels_cfg(
    *,
    sequence_dir: str | Path,
    dataset_kind: str,
    labels_cfg: Mapping[str, Any],
    kin: BaseKinematics,
    validate_frames: bool,
    t_expect: int,
) -> dict[int, npt.NDArray[np.float64]]:
    """
    Build per-leg stance for one sequence; assert each leg length matches ``t_expect``.
    """
    method = str(labels_cfg.get("method", "")).strip().lower()
    if method == "grf_threshold":
        grf = labels_cfg.get("grf_threshold")
        if not isinstance(grf, Mapping):
            raise ValueError("labels.grf_threshold mapping required when labels.method is grf_threshold")
        by_leg = stance_timeline_grf_threshold(
            sequence_dir=sequence_dir,
            dataset_kind=dataset_kind,
            grf_threshold_cfg=grf,
            kin=kin,
            validate_frames=validate_frames,
        )
    elif method == "gmm_hmm":
        gmm = labels_cfg.get("gmm_hmm")
        if not isinstance(gmm, Mapping):
            raise ValueError("labels.gmm_hmm mapping required when labels.method is gmm_hmm")
        by_leg = stance_timeline_gmm_hmm(
            sequence_dir=sequence_dir,
            dataset_kind=dataset_kind,
            gmm_hmm_cfg=gmm,
            kin=kin,
            validate_frames=validate_frames,
        )
    elif method == "dual_hmm":
        dual = labels_cfg.get("dual_hmm")
        if not isinstance(dual, Mapping):
            raise ValueError("labels.dual_hmm mapping required when labels.method is dual_hmm")
        by_leg = stance_timeline_dual_hmm(
            sequence_dir=sequence_dir,
            dataset_kind=dataset_kind,
            dual_hmm_cfg=dual,
            kin=kin,
            validate_frames=validate_frames,
        )
    else:
        raise ValueError(f"Unsupported labels.method for precompute: {method!r}")
    n_legs = int(kin.n_legs)
    for leg in range(n_legs):
        y = by_leg[leg]
        if int(y.shape[0]) != int(t_expect):
            raise RuntimeError(
                f"leg {leg}: stance length {y.shape[0]} != expected T={t_expect} for {sequence_dir}"
            )
    return by_leg
