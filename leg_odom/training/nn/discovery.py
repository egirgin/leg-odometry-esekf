"""
Discover Tartanground split sequence directories under a dataset root.

Each valid sequence directory must contain **exactly** ``imu.csv`` and **exactly one**
``*_bag.csv`` (no merge ambiguity).
"""

from __future__ import annotations

from pathlib import Path


def _bag_csv_paths(sequence_dir: Path) -> list[Path]:
    return sorted(sequence_dir.glob("*_bag.csv"))


def is_valid_tartanground_sequence_dir(sequence_dir: Path) -> tuple[bool, str]:
    """
    Return ``(True, "")`` if ``sequence_dir`` is a usable split layout, else ``(False, reason)``.
    """
    d = sequence_dir.expanduser().resolve()
    imu = d / "imu.csv"
    if not imu.is_file():
        return False, "missing imu.csv"
    bags = _bag_csv_paths(d)
    if len(bags) == 0:
        return False, "no *_bag.csv"
    if len(bags) > 1:
        return False, f"multiple *_bag.csv ({len(bags)}); expected exactly one"
    return True, ""


def discover_split_sequence_dirs(root: str | Path, *, verbose: bool = False) -> list[Path]:
    """
    Find all directories under ``root`` that contain ``imu.csv`` and exactly one ``*_bag.csv``.

    Uses ``rglob("imu.csv")`` and validates each parent directory. Duplicate parents are
    visited once. Returned paths are sorted lexicographically by ``str(path)``.
    """
    root_p = Path(root).expanduser().resolve()
    if not root_p.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root_p}")

    seen: set[Path] = set()
    valid: list[Path] = []
    for imu_path in root_p.rglob("imu.csv"):
        parent = imu_path.parent.resolve()
        if parent in seen:
            continue
        seen.add(parent)
        ok, reason = is_valid_tartanground_sequence_dir(parent)
        if ok:
            valid.append(parent)
        elif verbose and reason:
            print(f"[leg_odom.training.nn.discovery] skip {parent}: {reason}")

    if not valid:
        raise FileNotFoundError(
            f"No valid Tartanground split sequences under {root_p}: "
            "need imu.csv + exactly one *_bag.csv per sequence directory."
        )
    return sorted(valid, key=lambda p: str(p))
