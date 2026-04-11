"""
Build per-foot contact detectors from ``contact.detector`` and nested blocks.

Mirrors :mod:`leg_odom.run.kinematics_factory` / :mod:`leg_odom.run.dataset_factory`
so the EKF loop does not branch on detector string ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from leg_odom.contact.base import BaseContactDetector
from leg_odom.contact.dual_hmm.detector import build_dual_hmm_detectors_from_cfg
from leg_odom.contact.gmm_hmm.detector import build_gmm_hmm_detectors_from_cfg
from leg_odom.contact.grf_threshold import build_grf_threshold_detectors_from_cfg
from leg_odom.contact.ocelot import build_ocelot_detectors_from_cfg
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.kinematics.base import BaseKinematics


@dataclass(frozen=True)
class ContactStack:
    """
    Contact pipeline id (same normalized string as ``contact.detector`` in YAML) plus optional
    per-foot detectors for ZUPT.
    """

    detector_id: str
    per_foot: list[BaseContactDetector] | None


def _contact_detector_id(cfg: Mapping[str, Any]) -> str:
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        return "none"
    return str(block.get("detector", "none")).lower()


def build_contact_stack(
    cfg: Mapping[str, Any],
    *,
    recording: LegOdometrySequence | None = None,
    kin_model: BaseKinematics | None = None,
    workspace_root: Path | None = None,
) -> ContactStack:
    """
    Return :class:`ContactStack` with normalized ``detector_id`` and optional per-foot detectors.

    ``recording`` and ``kin_model`` are required when ``contact.detector`` is ``gmm`` and
    ``contact.gmm.mode`` is ``offline`` (whole-sequence GMM fit before the EKF loop;
    ``contact.gmm.history_length`` must be ``1`` for offline).

    ``kin_model`` is required for ``contact.detector`` ``neural``; ``recording`` is unused.
    Relative ``contact.neural.*`` paths resolve against ``workspace_root`` (repo root when unset:
    :func:`~leg_odom.contact.neural.build_neural_detectors_from_cfg` falls back to ``cwd``).

    ``contact.detector`` ``dual_hmm``: ``kin_model`` always required; ``recording`` required when
    ``contact.dual_hmm.mode`` is ``offline``. Online mode needs ``contact.dual_hmm.pretrained_path``
    (see :mod:`leg_odom.training.dual_hmm.train_dual_hmm`).

    ``contact.detector`` ``ocelot``: ``recording`` required; ``kin_model`` unused. See
    :mod:`leg_odom.contact.ocelot`.
    """
    det = _contact_detector_id(cfg)
    if det == "grf_threshold":
        return ContactStack(detector_id=det, per_foot=build_grf_threshold_detectors_from_cfg(cfg))
    if det == "gmm":
        return ContactStack(detector_id=det, per_foot=build_gmm_hmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin_model))
    if det == "dual_hmm":
        return ContactStack(
            detector_id=det,
            per_foot=build_dual_hmm_detectors_from_cfg(
                cfg,
                recording=recording,
                kin_model=kin_model,
                workspace_root=workspace_root,
            ),
        )
    if det == "neural":
        from leg_odom.contact.neural import build_neural_detectors_from_cfg

        if kin_model is None:
            raise ValueError("contact.detector neural requires kin_model in build_contact_stack")
        return ContactStack(
            detector_id=det,
            per_foot=build_neural_detectors_from_cfg(cfg, kin_model=kin_model, workspace_root=workspace_root),
        )
    if det == "ocelot":
        return ContactStack(
            detector_id=det,
            per_foot=build_ocelot_detectors_from_cfg(cfg, recording=recording, kin_model=kin_model),
        )
    return ContactStack(detector_id=det, per_foot=None)
