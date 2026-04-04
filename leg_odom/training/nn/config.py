"""
Load and validate YAML for :mod:`leg_odom.training.nn.train_contact_nn`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG_PATH = _CONFIG_DIR / "default_train_config.yaml"


def default_train_config_path() -> Path:
    return DEFAULT_TRAIN_CONFIG_PATH


def load_nn_train_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load training config from YAML. Required sections are validated.

    Parameters
    ----------
    path
        YAML file; default: ``default_train_config.yaml`` beside this module.
    """
    p = Path(path).expanduser().resolve() if path is not None else DEFAULT_TRAIN_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"NN train config not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError("NN train config must be a YAML mapping at top level")
    cfg: dict[str, Any] = dict(raw)
    _validate_nn_train_config(cfg)
    _apply_default_nn_output_dir(cfg)
    return cfg


def _apply_default_nn_output_dir(cfg: dict[str, Any]) -> None:
    """If ``output.dir`` is missing, null, or blank, set ``leg_odom/training/nn/pretrained_{architecture}``."""
    arch = str(cfg.get("architecture", "")).strip().lower()
    if arch not in ("cnn", "gru"):
        return
    out = cfg.get("output")
    if not isinstance(out, Mapping):
        return
    out_d = dict(out)
    raw = out_d.get("dir")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        out_d["dir"] = f"leg_odom/training/nn/pretrained_{arch}"
    cfg["output"] = out_d


def _require_section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    v = cfg.get(key)
    if not isinstance(v, Mapping):
        raise ValueError(f"NN train config: missing or invalid section {key!r}")
    return dict(v)


def _validate_labels_section(lb: Mapping[str, Any]) -> None:
    """Canonical ``labels.method`` only; GMM pseudo-labels are offline per-sequence, ``history_length: 1``."""
    method = str(lb.get("method", "")).strip().lower()
    if method == "grf_threshold":
        g = lb.get("grf_threshold")
        if not isinstance(g, Mapping):
            raise ValueError(
                "labels.grf_threshold must be a mapping when labels.method is grf_threshold "
                "(same keys as contact.grf_threshold; at least force_threshold)"
            )
        if "force_threshold" not in g:
            raise ValueError("labels.grf_threshold.force_threshold is required when labels.method is grf_threshold")
        float(g["force_threshold"])
        return
    if method == "gmm_hmm":
        g = lb.get("gmm_hmm")
        if not isinstance(g, Mapping):
            raise ValueError("labels.gmm_hmm must be a mapping when labels.method is gmm_hmm")
        gm = dict(g)
        if gm.get("pretrained_path"):
            raise ValueError(
                "labels.gmm_hmm.pretrained_path is not allowed for NN training (offline per-sequence fit only)"
            )
        mode = str(gm.get("mode", "offline")).lower()
        if mode != "offline":
            raise ValueError(f"labels.gmm_hmm.mode must be offline for NN labels, got {mode!r}")
        hl = int(gm.get("history_length", 1))
        if hl != 1:
            raise ValueError(
                f"labels.gmm_hmm.history_length must be 1 (instant GMM emissions; NN window_size is separate), got {hl}"
            )
        return
    if method == "dual_hmm":
        raise NotImplementedError(
            "labels.method dual_hmm is not implemented yet; port leg_odom.contact.dual_hmm_fusion first."
        )
    if method == "ocelot":
        raise NotImplementedError(
            "labels.method ocelot is not implemented yet; port leg_odom.contact.ocelot first."
        )
    raise ValueError(
        f"Unknown labels.method {method!r}; use grf_threshold, gmm_hmm, dual_hmm, or ocelot"
    )


def _validate_nn_train_config(cfg: Mapping[str, Any]) -> None:
    ds = _require_section(cfg, "dataset")
    if "kind" not in ds or not str(ds["kind"]).strip():
        raise ValueError("dataset.kind is required (e.g. tartanground_split)")
    if "precomputed_root" not in ds or not str(ds["precomputed_root"]).strip():
        raise ValueError(
            "dataset.precomputed_root is required (tree of precomputed_instants.npz from "
            "python -m leg_odom.features.preprocess_tartanground_nn)"
        )

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
            raise ValueError(f"training.{k} is required in NN train config")

    md = _require_section(cfg, "model")
    if "window_size" not in md:
        raise ValueError("model.window_size is required")

    lb = _require_section(cfg, "labels")
    if "method" not in lb:
        raise ValueError("labels.method is required")
    _validate_labels_section(lb)

    feat = _require_section(cfg, "features")
    fields = feat.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("features.fields must be a non-empty list of feature names")

    _require_section(cfg, "output")
    _require_section(cfg, "robot")
    _require_section(cfg, "data_loading")
    _require_section(cfg, "visualization")
