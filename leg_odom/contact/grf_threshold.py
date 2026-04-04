"""
Hard GRF threshold contact detector: one scalar load feature per foot.

Uses ``foot_force_*`` style signals (vertical load proxy, Newtons in Tartanground exports).
Stance if load ≥ ``force_threshold``. ZUPT measurement noise ``R_foot`` is **constant**
(constructor / YAML only), not a function of GRF.

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

from leg_odom.contact.base import (
    BaseContactDetector,
    ContactDetectorStepInput,
    ContactEstimate,
    zupt_isotropic_R_foot,
)


class GrfThresholdContactDetector(BaseContactDetector):
    """
    Per-foot detector: ``feature_dim == 1`` (GRF scalar), ``history_length == 1``.

    While in stance, ``zupt_meas_var`` and :attr:`~BaseContactDetector.last_zupt_R_foot`
    use a fixed isotropic variance per velocity axis: ``zupt_meas_var`` if provided, else
    ``var_at_threshold`` (kept for backward-compatible YAML).
    """

    def __init__(
        self,
        *,
        force_threshold: float,
        var_at_threshold: float,
        var_min: float = 0.0,
        saturation_force: float = 1.0,
        zupt_meas_var: float | None = None,
        use_abs: bool = False,
    ) -> None:
        super().__init__()
        if force_threshold < 0:
            raise ValueError("force_threshold must be non-negative")
        sigma_sq = float(zupt_meas_var) if zupt_meas_var is not None else float(var_at_threshold)
        if sigma_sq <= 0:
            raise ValueError("zupt_meas_var / var_at_threshold must be positive")
        self._thr = float(force_threshold)
        self._zupt_sigma_sq = sigma_sq
        self._use_abs = bool(use_abs)
        _ = (var_min, saturation_force)  # accepted for YAML backward compatibility; unused

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
        r = zupt_isotropic_R_foot(self._zupt_sigma_sq)
        self._last_zupt_R_foot = r
        if not stance:
            return ContactEstimate(stance=False, p_stance=0.0, zupt_meas_var=float("nan"))
        return ContactEstimate(stance=True, p_stance=1.0, zupt_meas_var=self._zupt_sigma_sq)

    def reset(self) -> None:
        self._last_zupt_R_foot = np.full((3, 3), np.nan, dtype=np.float64)


def make_quadruped_grf_threshold_detectors(
    *,
    force_threshold: float = 5.0,
    var_at_threshold: float = 0.15**2,
    var_min: float = 0.02**2,
    saturation_force: float = 350.0,
    zupt_meas_var: float | None = None,
    use_abs: bool = False,
) -> list[GrfThresholdContactDetector]:
    """Four independent instances (same hyperparameters) for legs ``0..3``."""
    kw = {
        "force_threshold": force_threshold,
        "var_at_threshold": var_at_threshold,
        "var_min": var_min,
        "saturation_force": saturation_force,
        "use_abs": use_abs,
    }
    if zupt_meas_var is not None:
        kw["zupt_meas_var"] = zupt_meas_var
    return [GrfThresholdContactDetector(**kw) for _ in range(4)]


def build_grf_threshold_detectors_from_cfg(cfg: Mapping[str, Any]) -> list[GrfThresholdContactDetector]:
    """Optional ``contact.grf_threshold`` mapping overrides factory defaults."""
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        return make_quadruped_grf_threshold_detectors()
    g = block.get("grf_threshold")
    if not isinstance(g, Mapping):
        return make_quadruped_grf_threshold_detectors()
    allowed = frozenset(
        {
            "force_threshold",
            "var_at_threshold",
            "var_min",
            "saturation_force",
            "zupt_meas_var",
            "use_abs",
        }
    )
    kw = {k: g[k] for k in allowed if k in g}
    return make_quadruped_grf_threshold_detectors(**kw)


def _minimal_cfg(sequence_dir: str, robot: str, dataset_kind: str) -> dict:
    return {
        "schema_version": 1,
        "run": {
            "name": "grf_viz",
            "debug": {
                "enabled": False,
                "generate_analysis_plots": False,
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
    p.add_argument("--dataset-kind", type=str, default="tartanground_split")
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument("--force-threshold", type=float, default=5.0)
    p.add_argument("--var-at-threshold", type=float, default=0.15**2)
    p.add_argument("--zupt-meas-var", type=float, default=None)
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
        "var_at_threshold": float(args.var_at_threshold),
        "use_abs": bool(args.use_abs),
    }
    if args.zupt_meas_var is not None:
        det_kw["zupt_meas_var"] = float(args.zupt_meas_var)
    dets = [GrfThresholdContactDetector(**det_kw) for _ in range(kin.n_legs)]

    t_abs, grfs, st, ps = replay_detectors_on_timeline(rec.frames, kin, dets)
    save_path = Path(args.save).expanduser() if str(args.save).strip() else None
    plot_grf_contact_overview(
        t_abs,
        grfs,
        st,
        ps,
        suptitle=f"GRF threshold contact — {rec.sequence_name}",
        save_path=save_path,
        show=save_path is None,
    )
    if save_path is not None:
        print(f"Wrote {save_path.resolve()}")


if __name__ == "__main__":
    main()
