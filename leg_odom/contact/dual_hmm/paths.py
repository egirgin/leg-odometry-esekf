"""Resolve pretrained dual HMM ``.npz`` paths (package-relative vs absolute)."""

from __future__ import annotations

from pathlib import Path

import leg_odom


def resolve_pretrained_dual_hmm_path(p: str | Path) -> Path:
    """If ``p`` is relative, resolve under ``leg_odom/training/dual_hmm/``."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    pkg = Path(leg_odom.__file__).resolve().parent
    return (pkg / "training" / "dual_hmm" / path).resolve()
