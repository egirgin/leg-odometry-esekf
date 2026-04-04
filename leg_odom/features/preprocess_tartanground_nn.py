"""
Precompute per-sequence ``precomputed_instants.npz`` for NN contact training (full kinematic instants + raw GRF).

Run once after preparing Tartanground split trees::

    python -m leg_odom.features.preprocess_tartanground_nn \\
      --dataset-root /path/to/processed \\
      --output-root /path/to/precomputed_nn \\
      --robot anymal

    python -m leg_odom.features.preprocess_tartanground_nn 
      --dataset-root /home/girgine/Documents/leg-odometry/tartanground/processed 
      --output-root ./leg_odom/features/precomputed_tartanground
      --robot anymal --max-sequences 10

Optional test subset: ``--max-sequences N`` with ``1 <= N <= 240`` processes only the first N sequences in
discovery order and prints a sample resolved ``sequence_dir`` for EKF testing. Omit the flag to process all.

NN training discovers ``precomputed_instants.npz`` only under ``dataset.precomputed_root`` (no CSV tree at train time);
see :mod:`leg_odom.training.nn.precomputed_io`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from leg_odom.training.nn.precomputed_io import PRECOMPUTED_INSTANTS_FILENAME, precomputed_npz_relpath

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from leg_odom.features.instant_spec import (
    FULL_OFFLINE_INSTANT_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    NN_PRECOMPUTE_FORMAT_VERSION,
    build_timeline_features_for_leg,
    parse_instant_feature_fields,
)
from leg_odom.io.columns import FOOT_FORCE_COLS
from leg_odom.kinematics.anymal import AnymalKinematics
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.kinematics.go2 import Go2Kinematics
from leg_odom.training.nn.discovery import discover_split_sequence_dirs
from leg_odom.training.nn.tartanground_io import load_validated_frames

MANIFEST_NAME = "preprocess_manifest.json"


def _build_kinematics(robot: str) -> BaseKinematics:
    name = str(robot).lower()
    if name == "anymal":
        return AnymalKinematics()
    if name == "go2":
        return Go2Kinematics()
    raise ValueError(f"Unsupported robot kinematics {robot!r} (use anymal or go2)")


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


def _optional_source_mtimes(sequence_dir: Path) -> dict[str, float | None]:
    d = Path(sequence_dir).expanduser().resolve()
    imu = d / "imu.csv"
    bags = sorted(d.glob("*_bag.csv"))
    out: dict[str, float | None] = {"imu_csv_mtime": None, "bag_csv_mtime": None}
    if imu.is_file():
        out["imu_csv_mtime"] = os.path.getmtime(imu)
    if len(bags) == 1 and bags[0].is_file():
        out["bag_csv_mtime"] = os.path.getmtime(bags[0])
    return out


def write_sequence_npz(
    *,
    sequence_dir: Path,
    dataset_root: Path,
    output_root: Path,
    kin: BaseKinematics,
    full_spec,
    overwrite: bool,
    validate_frames: bool = True,
) -> Path:
    out_rel = precomputed_npz_relpath(dataset_root, sequence_dir)
    out_dir = Path(output_root).expanduser().resolve() / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / PRECOMPUTED_INSTANTS_FILENAME
    if npz_path.is_file() and not overwrite:
        return npz_path

    frames = load_validated_frames(sequence_dir, verbose=False, validate=validate_frames)
    t_rows = len(frames)
    n_legs = kin.n_legs
    uid = sequence_uid_for_dir(sequence_dir)
    foot = foot_forces_from_frames(frames, n_legs)

    save_kw: dict[str, npt.NDArray[np.float64] | np.ndarray] = {
        "foot_forces": foot,
        "sequence_uid": np.array(uid, dtype=np.int64),
        "instant_feature_spec_version": np.array(int(INSTANT_FEATURE_SPEC_VERSION), dtype=np.int64),
        "nn_precompute_format_version": np.array(int(NN_PRECOMPUTE_FORMAT_VERSION), dtype=np.int64),
    }
    for leg in range(n_legs):
        inst = build_timeline_features_for_leg(frames, kin, leg, full_spec)
        save_kw[f"instants_leg{leg}"] = inst.astype(np.float64, copy=False)

    field_joined = "\n".join(FULL_OFFLINE_INSTANT_FIELDS)
    seq_resolved = str(Path(sequence_dir).expanduser().resolve())
    save_kw["sequence_dir"] = np.array(seq_resolved, dtype=np.str_)
    save_kw["robot_kinematics"] = np.array(type(kin).__name__, dtype=np.str_)
    save_kw["field_names"] = np.array(field_joined, dtype=np.str_)
    save_kw["source_mtimes_json"] = np.array(json.dumps(_optional_source_mtimes(sequence_dir)), dtype=np.str_)

    np.savez_compressed(npz_path, **save_kw)
    return npz_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute precomputed_instants.npz per Tartanground split sequence")
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Processed Tartanground split tree (imu.csv + *_bag.csv); training does not read this path",
    )
    p.add_argument("--output-root", type=str, required=True, help="Root directory for mirrored precomputed .npz tree")
    p.add_argument("--robot", type=str, required=True, help="Kinematics model: anymal or go2")
    p.add_argument("--overwrite", action="store_true", help="Replace existing precomputed_instants.npz")
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validate_prepared_split_dataframe when loading frames (not recommended)",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Test mode: process only the first N sequences after discovery order (1..240). "
        "Default: all sequences. Prints one sample sequence_dir when set.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.max_sequences is not None:
        n_chk = int(args.max_sequences)
        if n_chk < 1 or n_chk > 240:
            raise SystemExit("--max-sequences must be between 1 and 240 (inclusive)")

    sequences = discover_split_sequence_dirs(dataset_root, verbose=True)
    n_total = len(sequences)
    if args.max_sequences is not None:
        n_req = int(args.max_sequences)
        if n_req > n_total:
            print(
                f"[preprocess_tartanground_nn] warning: --max-sequences={n_req} > discovered {n_total}; "
                f"processing all {n_total} sequences"
            )
            n_use = n_total
        else:
            sequences = sequences[:n_req]
            n_use = n_req
        print(
            f"[preprocess_tartanground_nn] --max-sequences requested={n_req} processing={n_use} "
            f"(of {n_total} discovered); sample sequence_dir for EKF/testing: {sequences[0].resolve()}"
        )
    kin = _build_kinematics(args.robot)
    full_spec = parse_instant_feature_fields(FULL_OFFLINE_INSTANT_FIELDS)

    manifest: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "robot": str(args.robot),
        "n_sequences": len(sequences),
        "n_sequences_discovered": n_total,
        "max_sequences_cap": args.max_sequences,
        "instant_feature_spec_version": int(INSTANT_FEATURE_SPEC_VERSION),
        "nn_precompute_format_version": int(NN_PRECOMPUTE_FORMAT_VERSION),
        "sequence_uids": {},
        "npz_paths": {},
    }

    validate_frames = not bool(args.no_validate)
    for seq_dir in tqdm(sequences, desc="Preprocess sequences", unit="seq"):
        npz_path = write_sequence_npz(
            sequence_dir=seq_dir,
            dataset_root=dataset_root,
            output_root=output_root,
            kin=kin,
            full_spec=full_spec,
            overwrite=bool(args.overwrite),
            validate_frames=validate_frames,
        )
        key = str(seq_dir.resolve())
        manifest["sequence_uids"][key] = int(sequence_uid_for_dir(seq_dir))
        manifest["npz_paths"][key] = str(npz_path.resolve())

    manifest_path = output_root / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[preprocess_tartanground_nn] wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
