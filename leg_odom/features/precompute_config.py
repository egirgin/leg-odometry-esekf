"""Load and validate YAML for :mod:`leg_odom.features.precompute_contact_instants`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from leg_odom.features.nn_labels_config import validate_nn_labels_config

_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_PRECOMPUTE_CONFIG_PATH = _CONFIG_DIR / "default_precompute_config.yaml"


def default_precompute_config_path() -> Path:
    return DEFAULT_PRECOMPUTE_CONFIG_PATH


def load_precompute_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).expanduser().resolve() if path is not None else DEFAULT_PRECOMPUTE_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Precompute config not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError("Precompute config must be a YAML mapping at top level")
    cfg: dict[str, Any] = dict(raw)
    _validate_precompute_config(cfg)
    return cfg


def _require_section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    v = cfg.get(key)
    if not isinstance(v, Mapping):
        raise ValueError(f"Precompute config: missing or invalid section {key!r}")
    return dict(v)


def _validate_precompute_config(cfg: Mapping[str, Any]) -> None:
    for k in ("dataset_root", "output_root", "dataset_kind", "robot"):
        if k not in cfg or not str(cfg.get(k, "")).strip():
            raise ValueError(f"precompute config: {k!r} is required")
    dk = str(cfg["dataset_kind"]).strip().lower()
    if dk not in ("tartanground", "ocelot"):
        raise ValueError("dataset_kind must be tartanground or ocelot")
    rob = str(cfg["robot"]).strip().lower()
    if rob not in ("anymal", "go2"):
        raise ValueError("robot must be anymal or go2")

    if "overwrite" not in cfg:
        raise ValueError("overwrite is required (bool)")
    if not isinstance(cfg["overwrite"], bool):
        raise TypeError("overwrite must be a boolean")

    lb = _require_section(cfg, "labels")
    validate_nn_labels_config(lb)

    if "verbose" in cfg and not isinstance(cfg["verbose"], bool):
        raise TypeError("verbose must be a boolean")

    if "max_sequences" in cfg and cfg["max_sequences"] is not None:
        n = int(cfg["max_sequences"])
        if n < 1 or n > 240:
            raise ValueError("max_sequences must be between 1 and 240 (inclusive) or null/omitted")
