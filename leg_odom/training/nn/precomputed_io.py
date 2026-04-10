"""
Resolve and load precomputed ``precomputed_instants.npz`` bundles for NN contact training.

Training discovers ``precomputed_instants.npz`` under ``dataset.precomputed_root`` only (no CSV tree at train time).
If a bundle is missing or invalid, errors include instructions to run::

    python -m leg_odom.features.precompute_contact_instants --config <precompute.yaml>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

import hashlib

from leg_odom.features.instant_spec import (
    FULL_OFFLINE_INSTANT_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    NN_PRECOMPUTE_FORMAT_VERSION,
)

PRECOMPUTED_INSTANTS_FILENAME = "precomputed_instants.npz"


def precomputed_npz_relpath(dataset_root: Path, sequence_dir: Path) -> Path:
    """
    Mirror ``sequence_dir`` relative to ``dataset_root`` under the precomputed tree.

    If ``sequence_dir`` is not under ``dataset_root``, use ``_external/<16 hex>/``.
    """
    dr = Path(dataset_root).expanduser().resolve()
    sd = Path(sequence_dir).expanduser().resolve()
    try:
        rel = sd.relative_to(dr)
    except ValueError:
        h = hashlib.sha256(str(sd).encode("utf-8")).hexdigest()[:16]
        rel = Path("_external") / h
    return rel


_PREPROCESS_HINT = (
    "Run precompute:\n"
    "  python -m leg_odom.features.precompute_contact_instants \\\n"
    "    --config leg_odom/features/default_precompute_config.yaml\n"
    "(Edit YAML for dataset_root, output_root, labels, etc.)"
)


def discover_precomputed_instants_npz(precomputed_root: str | Path, *, verbose: bool = False) -> list[Path]:
    """
    Find all ``precomputed_instants.npz`` files under ``precomputed_root`` (resolved paths, sorted lexicographically).

    Used by NN training as the sequence list for train/val/test splits.
    """
    root_p = Path(precomputed_root).expanduser().resolve()
    if not root_p.is_dir():
        raise NotADirectoryError(f"precomputed_root is not a directory: {root_p}")

    seen: set[Path] = set()
    for p in root_p.rglob(PRECOMPUTED_INSTANTS_FILENAME):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)

    if not seen:
        raise FileNotFoundError(
            f"No {PRECOMPUTED_INSTANTS_FILENAME!r} files under {root_p}. {_PREPROCESS_HINT}"
        )
    out = sorted(seen, key=lambda x: str(x))
    if verbose:
        print(
            f"[leg_odom.training.nn.precomputed_io] discovered {len(out)} {PRECOMPUTED_INSTANTS_FILENAME} under {root_p}"
        )
    return out


class PrecomputedNnLoadError(FileNotFoundError):
    """Missing or invalid NN precompute bundle."""


def _scalar_int64(z: Mapping[str, Any], key: str) -> int:
    if key not in z:
        raise PrecomputedNnLoadError(f"npz missing {key!r}; {_PREPROCESS_HINT}")
    a = z[key]
    return int(np.asarray(a).reshape(-1)[0])


def _str_field(z: Mapping[str, Any], key: str) -> str:
    if key not in z:
        raise PrecomputedNnLoadError(f"npz missing {key!r}; {_PREPROCESS_HINT}")
    v = z[key]
    if hasattr(v, "dtype") and v.dtype.kind in "SU":
        return str(np.asarray(v).item())
    return str(v)


@dataclass(frozen=True, slots=True)
class PrecomputedSequenceBundle:
    """Loaded arrays + metadata for one sequence."""

    npz_path: Path
    sequence_uid: np.int64
    foot_forces: npt.NDArray[np.float64]
    instants_by_leg: dict[int, npt.NDArray[np.float64]]
    stance_by_leg: dict[int, npt.NDArray[np.float64]]
    field_names: tuple[str, ...]
    sequence_dir_stored: str
    robot_kinematics_stored: str
    contact_label_method: str
    contact_labels_config: dict[str, Any]


def load_precomputed_sequence_npz(
    npz_path: Path,
    *,
    expected_robot_kinematics: str,
    n_legs: int,
) -> PrecomputedSequenceBundle:
    """
    Load and validate one ``precomputed_instants.npz``.

    Raises
    ------
    PrecomputedNnLoadError
        Missing file, wrong format version, or robot mismatch.
    """
    p = Path(npz_path).expanduser().resolve()
    if not p.is_file():
        raise PrecomputedNnLoadError(f"Precomputed bundle not found: {p}\n{_PREPROCESS_HINT}")

    with np.load(p, allow_pickle=False) as z:
        fmt = _scalar_int64(z, "nn_precompute_format_version")
        if fmt != int(NN_PRECOMPUTE_FORMAT_VERSION):
            raise PrecomputedNnLoadError(
                f"{p}: nn_precompute_format_version={fmt}, expected {NN_PRECOMPUTE_FORMAT_VERSION}. "
                f"Re-run preprocess with current code.\n{_PREPROCESS_HINT}"
            )
        inst_v = _scalar_int64(z, "instant_feature_spec_version")
        if inst_v != int(INSTANT_FEATURE_SPEC_VERSION):
            raise PrecomputedNnLoadError(
                f"{p}: instant_feature_spec_version={inst_v}, expected {INSTANT_FEATURE_SPEC_VERSION}. "
                f"Re-run preprocess.\n{_PREPROCESS_HINT}"
            )
        field_raw = _str_field(z, "field_names")
        field_names = tuple(str(x).strip() for x in field_raw.split("\n") if str(x).strip())
        if field_names != FULL_OFFLINE_INSTANT_FIELDS:
            raise PrecomputedNnLoadError(
                f"{p}: field_names do not match FULL_OFFLINE_INSTANT_FIELDS; re-run preprocess.\n"
                f"{_PREPROCESS_HINT}"
            )
        rob = _str_field(z, "robot_kinematics")
        exp = str(expected_robot_kinematics).strip().lower()
        # Stored value is class name e.g. AnymalKinematics; match by substring.
        if exp == "anymal" and "anymal" not in rob.lower():
            raise PrecomputedNnLoadError(
                f"{p}: robot_kinematics={rob!r} incompatible with expected anymal; {_PREPROCESS_HINT}"
            )
        if exp == "go2" and "go2" not in rob.lower():
            raise PrecomputedNnLoadError(
                f"{p}: robot_kinematics={rob!r} incompatible with expected go2; {_PREPROCESS_HINT}"
            )

        uid = np.int64(np.asarray(z["sequence_uid"]).reshape(-1)[0])
        foot = np.array(z["foot_forces"], dtype=np.float64, copy=True)
        if foot.ndim != 2:
            raise PrecomputedNnLoadError(f"{p}: foot_forces must be 2-D; {_PREPROCESS_HINT}")
        instants_by_leg: dict[int, npt.NDArray[np.float64]] = {}
        for leg in range(int(n_legs)):
            key = f"instants_leg{leg}"
            if key not in z:
                raise PrecomputedNnLoadError(f"{p}: missing {key!r}; {_PREPROCESS_HINT}")
            arr = np.array(z[key], dtype=np.float64, copy=True)
            if arr.ndim != 2 or arr.shape[1] != len(FULL_OFFLINE_INSTANT_FIELDS):
                raise PrecomputedNnLoadError(
                    f"{p}: {key} shape {arr.shape} invalid (expected D={len(FULL_OFFLINE_INSTANT_FIELDS)}); "
                    f"{_PREPROCESS_HINT}"
                )
            if int(arr.shape[0]) != int(foot.shape[0]):
                raise PrecomputedNnLoadError(
                    f"{p}: {key} T={arr.shape[0]} != foot_forces T={foot.shape[0]}; {_PREPROCESS_HINT}"
                )
            instants_by_leg[leg] = arr
        seq_dir_stored = _str_field(z, "sequence_dir")
        contact_method = _str_field(z, "contact_label_method")
        cfg_raw = _str_field(z, "contact_labels_config_json")
        try:
            contact_labels_config = json.loads(cfg_raw)
        except json.JSONDecodeError as e:
            raise PrecomputedNnLoadError(f"{p}: invalid contact_labels_config_json: {e}\n{_PREPROCESS_HINT}") from e
        if not isinstance(contact_labels_config, dict):
            raise PrecomputedNnLoadError(f"{p}: contact_labels_config_json must decode to an object\n{_PREPROCESS_HINT}")

        t_rows = int(foot.shape[0])
        stance_by_leg: dict[int, npt.NDArray[np.float64]] = {}
        for leg in range(int(n_legs)):
            sk = f"stance_leg{leg}"
            if sk not in z:
                raise PrecomputedNnLoadError(f"{p}: missing {sk!r}; {_PREPROCESS_HINT}")
            st = np.array(z[sk], dtype=np.float64, copy=True).reshape(-1)
            if int(st.shape[0]) != t_rows:
                raise PrecomputedNnLoadError(
                    f"{p}: {sk} length {st.shape[0]} != foot_forces T={t_rows}; {_PREPROCESS_HINT}"
                )
            stance_by_leg[leg] = st

    return PrecomputedSequenceBundle(
        npz_path=p,
        sequence_uid=uid,
        foot_forces=foot,
        instants_by_leg=instants_by_leg,
        stance_by_leg=stance_by_leg,
        field_names=field_names,
        sequence_dir_stored=seq_dir_stored,
        robot_kinematics_stored=rob,
        contact_label_method=contact_method,
        contact_labels_config=contact_labels_config,
    )
