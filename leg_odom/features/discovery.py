"""
Discover sequence directories for NN precompute (processed CSV trees).

Supported layouts:
- Tartanground: ``imu.csv`` + exactly one ``*_bag.csv`` per trajectory directory.
- Ocelot: ``lowstate.csv`` (primary recording file) per trajectory directory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _bag_csv_paths(sequence_dir: Path) -> list[Path]:
    return sorted(sequence_dir.glob("*_bag.csv"))


def _as_root(root: str | Path) -> Path:
    root_p = Path(root).expanduser().resolve()
    if not root_p.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root_p}")
    return root_p


def is_valid_tartanground_sequence_dir(sequence_dir: Path) -> tuple[bool, str]:
    """
    Return ``(True, "")`` if ``sequence_dir`` is a usable imu+bag layout, else ``(False, reason)``.
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


def is_valid_ocelot_sequence_dir(sequence_dir: Path) -> tuple[bool, str]:
    """
    Return ``(True, "")`` if ``sequence_dir`` has ``lowstate.csv``, else ``(False, reason)``.
    """
    d = sequence_dir.expanduser().resolve()
    if not (d / "lowstate.csv").is_file():
        return False, "missing lowstate.csv"
    return True, ""


def _discover_under_marker(
    root: str | Path,
    *,
    marker_rglob: str,
    validate: Callable[[Path], tuple[bool, str]],
    empty_message: str,
    verbose: bool = False,
) -> list[Path]:
    root_p = _as_root(root)
    seen: set[Path] = set()
    valid: list[Path] = []
    for path in root_p.rglob(marker_rglob):
        parent = path.parent.resolve()
        if parent in seen:
            continue
        seen.add(parent)
        ok, reason = validate(parent)
        if ok:
            valid.append(parent)
        elif verbose and reason:
            print(f"[leg_odom.features.discovery] skip {parent}: {reason}")

    if not valid:
        raise FileNotFoundError(empty_message.format(root=root_p))
    return sorted(valid, key=lambda p: str(p))


def discover_tartanground_sequence_dirs(root: str | Path, *, verbose: bool = False) -> list[Path]:
    """
    Find directories under ``root`` that contain ``imu.csv`` and exactly one ``*_bag.csv``.

    Uses ``rglob("imu.csv")``. Returned paths are sorted lexicographically by ``str(path)``.
    """
    return _discover_under_marker(
        root,
        marker_rglob="imu.csv",
        validate=is_valid_tartanground_sequence_dir,
        empty_message=(
            "No valid Tartanground sequences under {root}: "
            "need imu.csv + exactly one *_bag.csv per sequence directory."
        ),
        verbose=verbose,
    )


def discover_ocelot_sequence_dirs(root: str | Path, *, verbose: bool = False) -> list[Path]:
    """Find directories under ``root`` that contain ``lowstate.csv`` (Ocelot recording)."""
    return _discover_under_marker(
        root,
        marker_rglob="lowstate.csv",
        validate=is_valid_ocelot_sequence_dir,
        empty_message="No valid Ocelot sequences under {root}: need lowstate.csv per sequence directory.",
        verbose=verbose,
    )
