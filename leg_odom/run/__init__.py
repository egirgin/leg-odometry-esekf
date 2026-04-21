"""Run orchestration: experiment YAML, output directories."""

from __future__ import annotations

from typing import Any

from leg_odom.run.experiment_config import (
    EXPERIMENT_SCHEMA_VERSION,
    debug_effective_from_cli,
    live_visualizer_effective,
    live_visualizer_sliding_window_s,
    live_visualizer_update_hz,
    load_experiment_yaml,
    merge_experiment_defaults,
    resolve_contact_neural_paths,
    resolve_ekf_noise_config_path,
    validate_experiment_dict,
)

# ``ekf_process`` / ``output_layout`` pull matplotlib (eval); load lazily so
# ``from leg_odom.run.experiment_config`` / ``contact_factory`` stay lightweight.


def __getattr__(name: str) -> Any:
    if name == "run_ekf_pipeline":
        from leg_odom.run.ekf_process import run_ekf_pipeline

        return run_ekf_pipeline
    if name == "prepare_run_output_dir":
        from leg_odom.run.output_layout import prepare_run_output_dir

        return prepare_run_output_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "debug_effective_from_cli",
    "live_visualizer_effective",
    "live_visualizer_sliding_window_s",
    "live_visualizer_update_hz",
    "load_experiment_yaml",
    "merge_experiment_defaults",
    "prepare_run_output_dir",
    "resolve_contact_neural_paths",
    "resolve_ekf_noise_config_path",
    "run_ekf_pipeline",
    "validate_experiment_dict",
]
