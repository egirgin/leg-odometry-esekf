"""Load and validate YAML for :mod:`leg_odom.training.ssl_nn.train_ssl_nn`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_SSL_TRAIN_CONFIG_PATH = _CONFIG_DIR / "default_ssl_config.yaml"


def default_ssl_train_config_path() -> Path:
    return DEFAULT_SSL_TRAIN_CONFIG_PATH


def load_ssl_train_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).expanduser().resolve() if path is not None else DEFAULT_SSL_TRAIN_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"SSL train config not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError("SSL train config must be a YAML mapping at top level")
    cfg: dict[str, Any] = dict(raw)
    _validate_ssl_train_config(cfg)
    _apply_default_ssl_output_dir(cfg)
    return cfg


def _apply_default_ssl_output_dir(cfg: dict[str, Any]) -> None:
    arch = str(cfg.get("architecture", "")).strip().lower()
    if arch not in ("cnn", "gru"):
        return
    out = cfg.get("output")
    if not isinstance(out, Mapping):
        return
    out_d = dict(out)
    raw = out_d.get("dir")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        out_d["dir"] = f"leg_odom/training/ssl_nn/pretrained_{arch}_ssl"
    cfg["output"] = out_d


def _require_section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    v = cfg.get(key)
    if not isinstance(v, Mapping):
        raise ValueError(f"SSL train config: missing or invalid section {key!r}")
    return dict(v)


def _validate_ssl_train_config(cfg: Mapping[str, Any]) -> None:
    ds = _require_section(cfg, "dataset")
    if "kind" not in ds or not str(ds["kind"]).strip():
        raise ValueError("dataset.kind is required")
    if "precomputed_root" not in ds or not str(ds["precomputed_root"]).strip():
        raise ValueError("dataset.precomputed_root is required")

    arch = cfg.get("architecture")
    if arch not in ("cnn", "gru"):
        raise ValueError("architecture must be 'cnn' or 'gru'")

    tr = _require_section(cfg, "training")
    for k in (
        "epochs",
        "batch_size",
        "learning_rate",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
        "num_workers",
    ):
        if k not in tr:
            raise ValueError(f"training.{k} is required in SSL train config")

    md = _require_section(cfg, "model")
    if "window_size" not in md:
        raise ValueError("model.window_size is required")

    feat = _require_section(cfg, "features")
    fields = feat.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("features.fields must be a non-empty list of feature names")

    ssl_cfg = _require_section(cfg, "ssl")
    method = str(ssl_cfg.get("method", "")).strip().lower()
    if method not in ("contrastive",):
        raise ValueError("ssl.method must be 'contrastive' for the initial scaffold")
    for k in ("projection_dim", "temperature"):
        if k not in ssl_cfg:
            raise ValueError(f"ssl.{k} is required")

    aug = _require_section(ssl_cfg, "augmentation")
    for k in ("gaussian_noise_std", "feature_dropout_prob", "scale_jitter_std"):
        if k not in aug:
            raise ValueError(f"ssl.augmentation.{k} is required")

    _require_section(cfg, "output")
    _require_section(cfg, "robot")
    dl = _require_section(cfg, "data_loading")
    if "verbose" not in dl or not isinstance(dl["verbose"], bool):
        raise ValueError("data_loading.verbose is required (bool)")

    viz = _require_section(cfg, "visualization")
    if "enabled" not in viz or not isinstance(viz["enabled"], bool):
        raise ValueError("visualization.enabled is required (bool)")
    for k in ("num_train_sections", "num_test_sections", "dpi"):
        if k not in viz:
            raise ValueError(f"visualization.{k} is required")
        v = viz[k]
        if not isinstance(v, int) or int(v) < 1:
            raise ValueError(f"visualization.{k} must be a positive integer")
