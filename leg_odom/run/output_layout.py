"""
Create timestamped run directories and persist experiment configuration snapshots.
"""

from __future__ import annotations

import copy
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from leg_odom.run.experiment_config import validate_experiment_dict


def _path_resolution_diff_lines(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Human-readable YAML comment lines for keys changed by path resolution only."""
    lines: list[str] = []
    ds_b = before.get("dataset")
    ds_a = after.get("dataset")
    if isinstance(ds_b, dict) and isinstance(ds_a, dict):
        vb, va = ds_b.get("sequence_dir"), ds_a.get("sequence_dir")
        if vb is not None and va is not None and str(vb) != str(va):
            lines.append(f"#   dataset.sequence_dir: input/validated {vb!r} → saved {va!r}")
    ekf_b = before.get("ekf")
    ekf_a = after.get("ekf")
    if isinstance(ekf_b, dict) and isinstance(ekf_a, dict):
        nb, na = ekf_b.get("noise_config"), ekf_a.get("noise_config")
        if nb not in (None, "") and na not in (None, "") and str(nb) != str(na):
            lines.append(f"#   ekf.noise_config: input/validated {nb!r} → saved {na!r}")
    cb, ca = before.get("contact"), after.get("contact")
    if isinstance(cb, dict) and isinstance(ca, dict):
        nb, na = cb.get("neural"), ca.get("neural")
        if isinstance(nb, dict) and isinstance(na, dict):
            bck, ack = nb.get("checkpoint"), na.get("checkpoint")
            if bck is not None and ack is not None and str(bck) != str(ack):
                lines.append(f"#   contact.neural.checkpoint: input/validated {bck!r} → saved {ack!r}")
    return lines


def prepare_run_output_dir(
    cfg: Mapping[str, Any],
    *,
    workspace_root: Path,
    source_config_path: Path | None = None,
    validate_paths: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """
    Validate experiment dict, create the run directory, and write a single YAML snapshot.

    Run directory layout::

        output.base_dir / output_{run.name} / dataset.kind / {env_name} / {traj_leaf}

    where ``env_name`` is ``Path(sequence_dir).parent.name`` and ``traj_leaf`` is the
    trajectory folder name, optionally suffixed with a timestamp when
    ``output.include_timestamp`` is true.

    Writes ``experiment_resolved.yaml``: validated config with absolute paths. A short comment
    header lists ``dataset.*`` / ``ekf.noise_config`` values that changed during path resolution
    relative to the dict **after** :func:`validate_experiment_dict` and **before** resolving paths.

    Parameters
    ----------
    source_config_path
        If set, recorded in the header as ``# input_yaml: ...`` (not a second file).
    """
    cfg_dict = copy.deepcopy(dict(cfg))
    validate_experiment_dict(
        cfg_dict,
        strict_paths=validate_paths,
        workspace_root=workspace_root,
    )

    from leg_odom.run.experiment_config import (
        resolve_contact_neural_paths,
        resolve_dataset_paths,
        resolve_ekf_noise_config_path,
    )

    pre_resolve = copy.deepcopy(cfg_dict)
    resolved_cfg = resolve_dataset_paths(cfg_dict, workspace_root)
    resolve_ekf_noise_config_path(resolved_cfg, workspace_root)
    resolve_contact_neural_paths(resolved_cfg, workspace_root)

    run_name = str(resolved_cfg["run"]["name"])
    if any(c in run_name for c in "/\\"):
        raise ValueError(f"run.name must not contain path separators, got {run_name!r}")

    base = Path(str(resolved_cfg["output"]["base_dir"])).expanduser()
    if not base.is_absolute():
        base = (workspace_root / base).resolve()

    seq_path = Path(resolved_cfg["dataset"]["sequence_dir"]).resolve()
    dataset_kind_segment = str(resolved_cfg["dataset"]["kind"]).strip().lower()
    traj_name = seq_path.name or "sequence"
    env_name = seq_path.parent.name
    if not env_name or env_name == seq_path.anchor:
        env_name = "_"
        warnings.warn(
            "dataset.sequence_dir has no usable parent folder name; using env_name='_' "
            "in the output path.",
            UserWarning,
            stacklevel=2,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resolved_cfg["output"]["include_timestamp"]:
        traj_leaf = f"{traj_name}_{ts}"
    else:
        traj_leaf = traj_name

    run_dir = base / f"output_{run_name}" / dataset_kind_segment / env_name / traj_leaf
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = run_dir / "experiment_resolved.yaml"
    diff_lines = _path_resolution_diff_lines(pre_resolve, resolved_cfg)
    with resolved_path.open("w", encoding="utf-8") as f:
        f.write(
            "# Validated experiment (saved with absolute paths for reproducibility).\n"
            "# Below: body is the exact dict used for this run.\n"
            "# Header notes fields that differ from values *before* path resolution "
            "(relative → absolute, canonicalization):\n"
        )
        if source_config_path is not None:
            try:
                src = source_config_path.expanduser().resolve()
            except OSError:
                src = source_config_path
            f.write(f"# input_yaml: {src}\n")
        if diff_lines:
            for line in diff_lines:
                f.write(line + "\n")
        else:
            f.write(
                "#   (no path-resolution rewrites — dataset.sequence_dir, "
                "ekf.noise_config matched pre-resolve strings)\n"
            )
        f.write("\n")
        yaml.safe_dump(
            resolved_cfg,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return run_dir, resolved_cfg
