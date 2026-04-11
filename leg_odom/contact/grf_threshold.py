"""
GRF-based contact detector: one scalar load feature per foot.

Uses ``foot_force_*`` style signals (vertical load proxy, Newtons in Tartanground exports).
``p_stance`` increases with load; ``stance`` is true when load is at or above ``force_threshold``.
ZUPT measurement covariance is formed from ``p_stance`` in :mod:`leg_odom.filters.zupt_measurement`.

Visualize contact vs GRF on a recording (no EKF)::

    python -m leg_odom.contact.grf_threshold \\
        --sequence-dir /path/to/seq --save output/grf_contact.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate


def _p_stance_from_g(g: float, force_threshold: float) -> float:
    """Map load to [0, 1]: 0 at no load, 1 at or above ``force_threshold``."""
    thr = max(float(force_threshold), 1e-12)
    return float(min(1.0, max(0.0, g / thr)))


class GrfThresholdContactDetector(BaseContactDetector):
    """
    Per-foot detector: ``feature_dim == 1`` (GRF scalar), ``history_length == 1``.

    ``stance`` is ``g >= force_threshold``; ``p_stance`` is a linear ramp in ``g`` up to 1 at the
    threshold and above.
    """

    def __init__(
        self,
        *,
        force_threshold: float,
        use_abs: bool = False,
    ) -> None:
        if force_threshold < 0:
            raise ValueError("force_threshold must be non-negative")
        self._thr = float(force_threshold)
        self._use_abs = bool(use_abs)

    @property
    def feature_dim(self) -> int:
        return 1

    @property
    def history_length(self) -> int:
        return 1

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        grf = float(step.grf_n)
        g = abs(grf) if self._use_abs else grf
        stance = g >= self._thr
        p_stance = _p_stance_from_g(g, self._thr)
        return ContactEstimate(stance=stance, p_stance=p_stance)

    def reset(self) -> None:
        pass


def make_quadruped_grf_threshold_detectors(
    *,
    force_threshold: float = 5.0,
    use_abs: bool = False,
) -> list[GrfThresholdContactDetector]:
    """Four independent instances (same hyperparameters) for legs ``0..3``."""
    return [
        GrfThresholdContactDetector(force_threshold=force_threshold, use_abs=use_abs)
        for _ in range(4)
    ]


def build_grf_threshold_detectors_from_cfg(cfg: Mapping[str, Any]) -> list[GrfThresholdContactDetector]:
    """Optional ``contact.grf_threshold`` mapping overrides factory defaults."""
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        return make_quadruped_grf_threshold_detectors()
    g = block.get("grf_threshold")
    if not isinstance(g, Mapping):
        return make_quadruped_grf_threshold_detectors()
    allowed = frozenset({"force_threshold", "use_abs"})
    kw = {k: g[k] for k in allowed if k in g}
    return make_quadruped_grf_threshold_detectors(**kw)


def _minimal_cfg(sequence_dir: str, robot: str, dataset_kind: str) -> dict:
    return {
        "schema_version": 1,
        "run": {
            "name": "grf_viz",
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


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GRF vs GRF-threshold contact (no EKF)")
    p.add_argument("--sequence-dir", type=str, required=True)
    p.add_argument("--dataset-kind", type=str, default="tartanground")
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument("--force-threshold", type=float, default=5.0)
    p.add_argument("--use-abs", action="store_true")
    p.add_argument("--save", type=str, default="", help="PNG path (if empty, show interactively)")
    args = p.parse_args()

    from leg_odom.contact.grf_stance_plot import plot_grf_contact_overview
    from leg_odom.contact.replay_timeline import replay_detectors_on_timeline
    from leg_odom.run.dataset_factory import build_leg_odometry_dataset
    from leg_odom.run.kinematics_factory import build_kinematics_backend

    cfg = _minimal_cfg(args.sequence_dir, args.robot_kinematics, args.dataset_kind)
    ds = build_leg_odometry_dataset(cfg)
    kin = build_kinematics_backend(cfg)
    rec = ds[0]

    det_kw: dict[str, Any] = {
        "force_threshold": float(args.force_threshold),
        "use_abs": bool(args.use_abs),
    }
    dets = make_quadruped_grf_threshold_detectors(**det_kw)

    t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin, dets)
    sp = Path(args.save) if str(args.save).strip() else None
    plot_grf_contact_overview(
        t_abs,
        grfs,
        st,
        ps,
        suptitle=rec.sequence_name,
        save_path=sp,
        show=sp is None,
    )


if __name__ == "__main__":
    main()
