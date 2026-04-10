"""
Precompute per-sequence ``precomputed_instants.npz`` for NN contact training:
full kinematic instants, raw GRF, and per-leg stance (contact detector replay).

Run once after preparing source sequence trees::

    python -m leg_odom.features.precompute_contact_instants \\
      --config leg_odom/features/default_precompute_config.yaml

Edit the YAML for ``dataset_root``, ``output_root``, ``dataset_kind``, ``robot``, and ``labels``.

NN training discovers ``precomputed_instants.npz`` only under ``dataset.precomputed_root`` (no CSV tree at train time);
see :mod:`leg_odom.training.nn.precomputed_io`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from leg_odom.features.contact_label_timelines import stance_by_leg_from_labels_cfg
from leg_odom.features.instant_spec import (
    FULL_OFFLINE_INSTANT_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    NN_PRECOMPUTE_FORMAT_VERSION,
    build_timeline_features_for_leg,
    parse_instant_feature_fields,
)
from leg_odom.features.nn_sequence_io import discover_sequence_dirs, load_training_frames
from leg_odom.features.precompute_config import load_precompute_config
from leg_odom.io.columns import FOOT_FORCE_COLS
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.run.kinematics_factory import build_kinematics_by_name
from leg_odom.training.nn.precomputed_io import PRECOMPUTED_INSTANTS_FILENAME, precomputed_npz_relpath

MANIFEST_NAME = "precompute_manifest.json"


def sequence_uid_for_dir(sequence_dir: Path) -> np.int64:
    """
    Stable int64 id for a sequence: SHA-256 of resolved path + format/spec versions (first 8 bytes, signed).
    """
    s = (
        f"{Path(sequence_dir).expanduser().resolve()}|"
        f"{NN_PRECOMPUTE_FORMAT_VERSION}|{INSTANT_FEATURE_SPEC_VERSION}"
    )
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return np.int64(int.from_bytes(digest[:8], "big", signed=True))


def foot_forces_from_frames(frames, n_legs: int) -> npt.NDArray[np.float64]:
    """``(T, n_legs)`` raw vertical load columns; NaN → 0."""
    import pandas as pd

    t = len(frames)
    out = np.zeros((t, int(n_legs)), dtype=np.float64)
    for i in range(min(int(n_legs), len(FOOT_FORCE_COLS))):
        col = FOOT_FORCE_COLS[i]
        if col in frames.columns:
            s = pd.to_numeric(frames[col], errors="coerce").fillna(0.0)
            out[:, i] = s.to_numpy(dtype=np.float64)
    return out


def _optional_source_mtimes(sequence_dir: Path, dataset_kind: str) -> dict[str, float | None]:
    d = Path(sequence_dir).expanduser().resolve()
    kind = str(dataset_kind).strip().lower()
    if kind == "tartanground":
        imu = d / "imu.csv"
        bags = sorted(d.glob("*_bag.csv"))
        out = {"imu_csv_mtime": None, "bag_csv_mtime": None}
        if imu.is_file():
            out["imu_csv_mtime"] = os.path.getmtime(imu)
        if len(bags) == 1 and bags[0].is_file():
            out["bag_csv_mtime"] = os.path.getmtime(bags[0])
        return out
    if kind == "ocelot":
        lowstate = d / "lowstate.csv"
        gt = d / "groundtruth.csv"
        out = {"lowstate_csv_mtime": None, "groundtruth_csv_mtime": None}
        if lowstate.is_file():
            out["lowstate_csv_mtime"] = os.path.getmtime(lowstate)
        if gt.is_file():
            out["groundtruth_csv_mtime"] = os.path.getmtime(gt)
        return out
    return {"source_mtime": None}


def write_sequence_npz(
    *,
    sequence_dir: Path,
    dataset_root: Path,
    output_root: Path,
    kin: BaseKinematics,
    full_spec,
    overwrite: bool,
    dataset_kind: str,
    labels_cfg: dict,
) -> Path:
    out_rel = precomputed_npz_relpath(dataset_root, sequence_dir)
    out_dir = Path(output_root).expanduser().resolve() / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / PRECOMPUTED_INSTANTS_FILENAME
    if npz_path.is_file() and not overwrite:
        return npz_path

    frames = load_training_frames(
        dataset_kind,
        sequence_dir,
        verbose=False,
        validate=True,
    )
    t_rows = len(frames)
    n_legs = kin.n_legs
    uid = sequence_uid_for_dir(sequence_dir)
    foot = foot_forces_from_frames(frames, n_legs)

    stance_by_leg = stance_by_leg_from_labels_cfg(
        sequence_dir=sequence_dir,
        dataset_kind=dataset_kind,
        labels_cfg=labels_cfg,
        kin=kin,
        validate_frames=True,
        t_expect=t_rows,
    )

    save_kw: dict[str, npt.NDArray[np.float64] | np.ndarray] = {
        "foot_forces": foot,
        "sequence_uid": np.array(uid, dtype=np.int64),
        "instant_feature_spec_version": np.array(int(INSTANT_FEATURE_SPEC_VERSION), dtype=np.int64),
        "nn_precompute_format_version": np.array(int(NN_PRECOMPUTE_FORMAT_VERSION), dtype=np.int64),
    }
    for leg in range(n_legs):
        inst = build_timeline_features_for_leg(frames, kin, leg, full_spec)
        save_kw[f"instants_leg{leg}"] = inst.astype(np.float64, copy=False)
        save_kw[f"stance_leg{leg}"] = stance_by_leg[leg].astype(np.float64, copy=False)

    method = str(labels_cfg.get("method", "")).strip()
    save_kw["contact_label_method"] = np.array(method, dtype=np.str_)
    save_kw["contact_labels_config_json"] = np.array(json.dumps(labels_cfg, sort_keys=True), dtype=np.str_)

    field_joined = "\n".join(FULL_OFFLINE_INSTANT_FIELDS)
    seq_resolved = str(Path(sequence_dir).expanduser().resolve())
    save_kw["sequence_dir"] = np.array(seq_resolved, dtype=np.str_)
    save_kw["robot_kinematics"] = np.array(type(kin).__name__, dtype=np.str_)
    save_kw["field_names"] = np.array(field_joined, dtype=np.str_)
    save_kw["source_mtimes_json"] = np.array(
        json.dumps(_optional_source_mtimes(sequence_dir, dataset_kind)),
        dtype=np.str_,
    )

    np.savez_compressed(npz_path, **save_kw)
    return npz_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute precomputed_instants.npz per sequence (single YAML config)")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML precompute config (dataset_root, output_root, dataset_kind, robot, labels, overwrite, ...)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_precompute_config(config_path)
    dataset_root = Path(str(cfg["dataset_root"])).expanduser().resolve()
    output_root = Path(str(cfg["output_root"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    max_sequences = cfg.get("max_sequences")
    if max_sequences is not None:
        n_chk = int(max_sequences)
        if n_chk < 1 or n_chk > 240:
            raise SystemExit("max_sequences in config must be between 1 and 240 (inclusive)")

    dataset_kind = str(cfg["dataset_kind"]).strip().lower()
    verbose = bool(cfg.get("verbose", True))
    sequences = discover_sequence_dirs(dataset_kind, dataset_root, verbose=verbose)
    n_total = len(sequences)
    if max_sequences is not None:
        n_req = int(max_sequences)
        if n_req > n_total:
            print(
                f"[precompute_contact_instants] warning: max_sequences={n_req} > discovered {n_total}; "
                f"processing all {n_total} sequences"
            )
            n_use = n_total
        else:
            sequences = sequences[:n_req]
            n_use = n_req
        print(
            f"[precompute_contact_instants] max_sequences requested={n_req} processing={n_use} "
            f"(of {n_total} discovered); sample sequence_dir for EKF/testing: {sequences[0].resolve()}"
        )
    robot = str(cfg["robot"]).strip().lower()
    kin = build_kinematics_by_name(robot)
    full_spec = parse_instant_feature_fields(FULL_OFFLINE_INSTANT_FIELDS)
    labels_cfg = dict(cfg["labels"])

    manifest: dict[str, object] = {
        "config_path": str(config_path.resolve()),
        "config": {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "dataset_kind": dataset_kind,
            "robot": robot,
            "labels": labels_cfg,
            "overwrite": bool(cfg["overwrite"]),
            "max_sequences": max_sequences,
            "verbose": verbose,
        },
        "n_sequences": len(sequences),
        "n_sequences_discovered": n_total,
        "instant_feature_spec_version": int(INSTANT_FEATURE_SPEC_VERSION),
        "nn_precompute_format_version": int(NN_PRECOMPUTE_FORMAT_VERSION),
        "sequence_uids": {},
        "npz_paths": {},
    }

    for seq_dir in tqdm(sequences, desc="Precompute sequences", unit="seq", disable=not verbose):
        npz_path = write_sequence_npz(
            sequence_dir=seq_dir,
            dataset_root=dataset_root,
            output_root=output_root,
            kin=kin,
            full_spec=full_spec,
            overwrite=bool(cfg["overwrite"]),
            dataset_kind=dataset_kind,
            labels_cfg=labels_cfg,
        )
        key = str(seq_dir.resolve())
        manifest["sequence_uids"][key] = int(sequence_uid_for_dir(seq_dir))
        manifest["npz_paths"][key] = str(npz_path.resolve())

    manifest_path = output_root / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[precompute_contact_instants] wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
