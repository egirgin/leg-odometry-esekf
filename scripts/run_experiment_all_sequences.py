#!/usr/bin/env python3
"""
Batch-smoke the repo-root ``main.py`` experiment pipeline on every Tartanground split sequence
under a data root (each directory with ``imu.csv`` + exactly one ``*_bag.csv``).

For each sequence, ``dataset.sequence_dir`` is set to that path and the same steps as
``main._run_experiment`` run (prepare output dir → EKF → optional post-EKF).

Not for production sweeps; use for local validation only.

Example::

    conda activate leg-odometry
    cd /path/to/async_ekf_workspace
    python scripts/run_experiment_all_sequences.py \\
      --data-root /home/you/data_anymal \\
      --config config/default_experiment.yaml

    # List sequences only
    python scripts/run_experiment_all_sequences.py --data-root /path/to/tree --dry-run
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from leg_odom.run.ekf_process import run_ekf_pipeline
from leg_odom.run.experiment_config import (
    debug_effective_from_cli,
    live_visualizer_effective,
    load_experiment_yaml,
)
from leg_odom.run.output_layout import prepare_run_output_dir
from leg_odom.features.discovery import discover_tartanground_sequence_dirs


def _touch_subpackages() -> None:
    import leg_odom.contact.base  # noqa: F401
    import leg_odom.datasets.base  # noqa: F401
    import leg_odom.kinematics.base  # noqa: F401


def _run_one_sequence(
    *,
    cfg: dict,
    workspace_root: Path,
    source_config_path: Path,
    seq_index: int,
    seq_total: int,
) -> int:
    """Mirror ``main._run_experiment`` for an already-merged config dict. Returns 0 on success."""
    debug_on = debug_effective_from_cli(cfg, cli_debug=False)
    live_viz_on = live_visualizer_effective(cfg, cli_debug=False)

    seq_dir = Path(cfg["dataset"]["sequence_dir"]).expanduser().resolve()
    print(f"\n[{seq_index + 1}/{seq_total}] sequence_dir={seq_dir}", file=sys.stderr)

    if debug_on:
        print(f"[debug] workspace_root={workspace_root}", file=sys.stderr)
        print(f"[debug] base_config={source_config_path}", file=sys.stderr)

    run_dir, resolved_cfg = prepare_run_output_dir(
        cfg,
        workspace_root=workspace_root,
        source_config_path=source_config_path,
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
        workspace_root=workspace_root,
    )
    print(
        f"[ekf] pipeline: 1 recording → {run_dir / 'ekf_process_summary.json'}",
        file=sys.stderr,
    )
    if summary.ekf_history_csv:
        print(f"[ekf] history csv: {summary.ekf_history_csv}", file=sys.stderr)

    from leg_odom.run.post_ekf import run_post_ekf_analysis_and_eval

    run_post_ekf_analysis_and_eval(
        run_dir, resolved_cfg, summary, output_subdir="plots"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run main.py-style EKF for each sequence under --data-root (test helper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory tree to search for imu.csv + one *_bag.csv per sequence folder",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "config" / "default_experiment.yaml",
        help="Base experiment YAML (merged once; sequence_dir overridden per sequence)",
    )
    p.add_argument(
        "--workspace-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repo root (path resolution for noise config, neural checkpoint, output.base_dir)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and print sequence paths; do not run EKF",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="If > 0, run at most this many sequences (order: discover_tartanground_sequence_dirs sort)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failing sequence (default: run all and report failures)",
    )
    args = p.parse_args(argv)

    workspace = args.workspace_root.expanduser().resolve()
    cfg_path = args.config.expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (workspace / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 2

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        print(f"error: --data-root is not a directory: {data_root}", file=sys.stderr)
        return 2

    try:
        sequences = discover_tartanground_sequence_dirs(data_root, verbose=True)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.max_sequences and args.max_sequences > 0:
        sequences = sequences[: int(args.max_sequences)]

    print(f"[batch] data_root={data_root}", file=sys.stderr)
    print(f"[batch] found {len(sequences)} sequence(s)", file=sys.stderr)
    if args.dry_run:
        for i, s in enumerate(sequences):
            print(f"  {i + 1:4d}  {s}")
        return 0

    base_cfg = load_experiment_yaml(cfg_path)
    failures: list[tuple[Path, str]] = []

    for i, seq_path in enumerate(sequences):
        cfg = copy.deepcopy(base_cfg)
        cfg["dataset"] = dict(cfg["dataset"])
        cfg["dataset"]["sequence_dir"] = str(seq_path.resolve())

        try:
            _run_one_sequence(
                cfg=cfg,
                workspace_root=workspace,
                source_config_path=cfg_path,
                seq_index=i,
                seq_total=len(sequences),
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((seq_path, msg))
            print(f"[batch] FAILED {seq_path}: {msg}", file=sys.stderr)
            if args.fail_fast:
                return 1

    if failures:
        print(f"\n[batch] completed with {len(failures)}/{len(sequences)} failure(s)", file=sys.stderr)
        for sp, msg in failures:
            print(f"  - {sp}\n    {msg}", file=sys.stderr)
        return 1

    print(f"\n[batch] all {len(sequences)} sequence(s) OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
