#!/usr/bin/env python3
"""
Entry point for the refactored Tartanground / ANYmal leg-odometry workspace.

Experiment definition: YAML under ``config/`` (see ``config/default_experiment.yaml``).
The repo root is the directory containing this file (used to resolve relative paths).

Examples::

    conda activate leg-odometry
    cd /path/to/async_ekf_workspace

    python main.py --config config/default_experiment.yaml

    # ``ekf_history_*.csv`` is written next to ``ekf_process_summary.json`` under the run directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from leg_odom.run.ekf_process import run_ekf_pipeline
from leg_odom.run.experiment_config import (
    debug_effective_from_cli,
    live_visualizer_effective,
    load_experiment_yaml,
)
from leg_odom.run.output_layout import prepare_run_output_dir

_REPO_ROOT = Path(__file__).resolve().parent


def _touch_subpackages() -> None:
    import leg_odom.contact.base  # noqa: F401
    import leg_odom.datasets.base  # noqa: F401
    import leg_odom.kinematics.base  # noqa: F401


def _run_experiment(*, config_path: Path, workspace_root: Path) -> int:
    workspace = workspace_root.resolve()

    cfg_path = config_path.expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (workspace / cfg_path).resolve()

    cfg = load_experiment_yaml(cfg_path)
    debug_on = debug_effective_from_cli(cfg, cli_debug=False)
    live_viz_on = live_visualizer_effective(cfg, cli_debug=False)

    if debug_on:
        print(f"[debug] workspace_root={workspace}", file=sys.stderr)
        print(f"[debug] config={cfg_path}", file=sys.stderr)

    run_dir, resolved_cfg = prepare_run_output_dir(
        cfg,
        workspace_root=workspace,
        source_config_path=cfg_path,
        validate_paths=True,
    )
    print(f"[run] output directory: {run_dir}")
    print(f"[run] resolved config: {run_dir / 'experiment_resolved.yaml'}")
    _touch_subpackages()

    summary = run_ekf_pipeline(
        resolved_cfg,
        run_dir=run_dir,
        debug=debug_on,
        live_visualizer=live_viz_on,
        workspace_root=workspace,
    )
    print(
        f"[ekf] pipeline: 1 recording → "
        f"{run_dir / 'ekf_process_summary.json'}"
    )
    if summary.ekf_history_csv:
        print(f"[ekf] history csv: {summary.ekf_history_csv}")
    from leg_odom.run.post_ekf import run_post_ekf_analysis_and_eval

    run_post_ekf_analysis_and_eval(
        run_dir, resolved_cfg, summary, output_subdir="plots"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tartanground / ANYmal leg odometry (leg_odom). "
            "Requires --config (experiment YAML); run.name and dataset.sequence_dir must be set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML (robot, dataset, contact, output). Creates a timestamped run directory.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return _run_experiment(config_path=args.config, workspace_root=_REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
