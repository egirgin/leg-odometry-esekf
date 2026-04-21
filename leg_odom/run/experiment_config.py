"""
Load and validate experiment YAML (robot, dataset, contact, EKF noise path, debug/output layout).

This is **user-facing experiment configuration**, distinct from :mod:`leg_odom.thresholds`
(implementation constants inside the code).

**How this file is laid out (top to bottom)**

1. Defaults + merge — fill omitted YAML keys; :func:`load_experiment_yaml` chains in file-level checks.
2. Load — parse YAML, require ``run.name`` and ``dataset.sequence_dir`` in the **file** (``dataset.kind`` must be valid; defaults apply if omitted), then merge.
3. Validate — schema/enums/types; optional ``strict_paths`` touches disk (sequence dir, noise YAML).
4. Resolve paths — canonicalize absolute ``dataset.sequence_dir`` and relative ``ekf.noise_config`` for snapshots.
5. Debug accessors — small readers for ``main.py`` / EKF (effective flags vs YAML-only toggles).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

# --- Schema constants (allowed enum values; keep in sync with factories / docs) ---------------

EXPERIMENT_SCHEMA_VERSION = 1

ALLOWED_KINEMATICS = frozenset({"anymal", "go2"})
ALLOWED_DATASET_KINDS = frozenset({"tartanground", "ocelot"})
ALLOWED_CONTACT_DETECTORS = frozenset(
    {"none", "stub", "gmm", "neural", "dual_hmm", "ocelot", "grf_threshold"}
)

# --- Defaults + deep merge --------------------------------------------------------------------
# _deep_merge + default_experiment_dict + merge_experiment_defaults. In-memory only; for disk use
# load_experiment_yaml (which enforces explicit run.name and dataset.sequence_dir in the file).


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_experiment_dict() -> dict[str, Any]:
    """Defaults used when keys are omitted from YAML."""
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run": {
            "name": "unnamed_run",
            "debug": {
                "enabled": False,
                "live_visualizer": {
                    "enabled": False,
                    "sliding_window_s": 10.0,
                    "video_path": None,
                    "hz": None,
                },
            },
        },
        "robot": {"kinematics": "anymal"},
        "dataset": {
            "kind": "tartanground",
            # Placeholder absolute path for programmatic merge-only configs; YAML must set explicitly.
            "sequence_dir": str(Path.home() / "data_anymal"),
        },
        "contact": {
            "detector": "none",
            "ocelot": {
                "use_fsm": True,
                "use_glrt": True,
                "fsm_gmm_mode": "offline",
                "force_on": 25.0,
                "force_off": 15.0,
                "window_size": 500,
                "fit_interval": 250,
                "noise_std_dev": 0.45,
                "rate_hz": None,
                "random_state": 42,
            },
        },
        "ekf": {"noise_config": None, "initialize_nominal_from_data": False},
        "output": {
            "base_dir": "output_leg_odom",
            "include_timestamp": True,
        },
    }


def merge_experiment_defaults(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge ``loaded`` YAML over :func:`default_experiment_dict`.

    Omitted nested keys inherit defaults. ``run.debug`` must be a mapping in the
    file (or merged result): boolean ``debug`` or legacy output flags are not accepted.
    """
    if not isinstance(loaded, Mapping):
        raise TypeError("Experiment YAML root must be a mapping")
    return _deep_merge(default_experiment_dict(), dict(loaded))


# --- Load from YAML ---------------------------------------------------------------------------
# Parse file → require keys in raw document → merge_experiment_defaults.


def _validate_yaml_file_has_run_and_dataset(raw: Mapping[str, Any] | None) -> None:
    """
    Require explicit ``run.name`` and ``dataset.sequence_dir`` in the file
    (not only merge defaults). :func:`merge_experiment_defaults` does not enforce this.
    """
    if raw is None or not isinstance(raw, Mapping):
        raise ValueError(
            "Experiment YAML must be a mapping with run.name and dataset.sequence_dir"
        )
    run = raw.get("run")
    if not isinstance(run, Mapping) or "name" not in run:
        raise ValueError("Experiment YAML must set run.name explicitly")
    name = run["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Experiment YAML run.name must be a non-empty string")

    ds = raw.get("dataset")
    if not isinstance(ds, Mapping) or "sequence_dir" not in ds:
        raise ValueError("Experiment YAML must set dataset.sequence_dir explicitly")


def load_experiment_yaml(path: str | Path) -> dict[str, Any]:
    """Parse YAML from disk, require explicit ``run.name`` and ``dataset.sequence_dir``, then merge defaults."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate_yaml_file_has_run_and_dataset(raw if isinstance(raw, Mapping) else None)
    return merge_experiment_defaults(raw)


# --- Validate merged config ---------------------------------------------------------------------
# Raises on bad types/enums. strict_paths=True additionally checks dataset layout and noise file.


def validate_experiment_dict(
    cfg: Mapping[str, Any],
    *,
    strict_paths: bool = False,
    workspace_root: Path | None = None,
) -> None:
    """
    Raise ``ValueError`` if enum fields or types are invalid.

    Parameters
    ----------
    strict_paths
        If True, require resolved ``dataset`` paths to exist (IMU + bag layout).
    workspace_root
        Required when ``strict_paths`` is True; used to resolve relative ``ekf.noise_config``.
    """
    # schema + high-level enums
    ver = int(cfg.get("schema_version", -1))
    if ver != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {ver!r}; expected {EXPERIMENT_SCHEMA_VERSION}"
        )

    kin = str(cfg["robot"]["kinematics"]).lower()
    if kin not in ALLOWED_KINEMATICS:
        raise ValueError(f"robot.kinematics must be one of {sorted(ALLOWED_KINEMATICS)}, got {kin!r}")

    dkind = str(cfg["dataset"]["kind"]).lower()
    if dkind not in ALLOWED_DATASET_KINDS:
        raise ValueError(
            f"dataset.kind must be one of {sorted(ALLOWED_DATASET_KINDS)}, got {dkind!r}"
        )

    ds = cfg["dataset"]
    if not isinstance(ds, Mapping):
        raise TypeError("dataset must be a mapping")
    if "data_root" in ds:
        raise ValueError(
            "dataset.data_root is no longer supported; set dataset.sequence_dir to the "
            "absolute path of the trajectory directory (folder containing imu.csv)."
        )
    seq_raw = ds.get("sequence_dir")
    if not isinstance(seq_raw, str) or not str(seq_raw).strip():
        raise ValueError("dataset.sequence_dir must be a non-empty string")
    seq_expanded = Path(str(seq_raw)).expanduser()
    if not seq_expanded.is_absolute():
        raise ValueError(
            f"dataset.sequence_dir must be an absolute path (tilde expansion is allowed); got {seq_raw!r}"
        )

    det = str(cfg["contact"]["detector"]).lower()
    if det not in ALLOWED_CONTACT_DETECTORS:
        raise ValueError(
            f"contact.detector must be one of {sorted(ALLOWED_CONTACT_DETECTORS)}, got {det!r}"
        )
    if det == "neural":
        _validate_contact_neural_block(cfg)
    if det == "ocelot":
        _validate_contact_ocelot_block(cfg)

    # EKF sidecar (optional path string)
    ekf = cfg.get("ekf")
    if not isinstance(ekf, Mapping):
        raise TypeError("ekf must be a mapping")
    nc = ekf.get("noise_config")
    if nc is not None and nc != "" and not isinstance(nc, str):
        raise TypeError("ekf.noise_config must be a string path or null/omitted")
    if isinstance(nc, str) and not str(nc).strip():
        raise ValueError("ekf.noise_config must be a non-empty path when set")

    init_nom = ekf.get("initialize_nominal_from_data", False)
    if not isinstance(init_nom, bool):
        raise TypeError("ekf.initialize_nominal_from_data must be a boolean")

    # run.* + nested run.debug (independent flags; see accessors section below)
    run_name = cfg["run"]["name"]
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run.name must be a non-empty string")

    dbg = cfg["run"].get("debug")
    if not isinstance(dbg, Mapping):
        raise TypeError("run.debug must be a mapping with enabled and live_visualizer")
    if not isinstance(dbg.get("enabled"), bool):
        raise TypeError("run.debug.enabled must be a boolean")

    lv = dbg.get("live_visualizer")
    if not isinstance(lv, Mapping):
        raise TypeError("run.debug.live_visualizer must be a mapping")
    if not isinstance(lv.get("enabled"), bool):
        raise TypeError("run.debug.live_visualizer.enabled must be a boolean")
    w_lv = float(lv.get("sliding_window_s", 60.0))
    if not math.isfinite(w_lv) or w_lv <= 0:
        raise ValueError(
            "run.debug.live_visualizer.sliding_window_s must be a finite positive number"
        )

    vp = lv.get("video_path")
    if vp is not None and vp != "" and not isinstance(vp, str):
        raise TypeError("run.debug.live_visualizer.video_path must be a string or null")

    hz_lv = lv.get("hz")
    if hz_lv is not None and hz_lv != "":
        if isinstance(hz_lv, bool):
            raise TypeError("run.debug.live_visualizer.hz must be a number or null, not boolean")
        hz_f = float(hz_lv)
        if not math.isfinite(hz_f) or hz_f <= 0.0:
            raise ValueError(
                "run.debug.live_visualizer.hz must be null/omitted or a finite positive number"
            )

    out_base = cfg["output"]["base_dir"]
    if not isinstance(out_base, str) or not str(out_base).strip():
        raise ValueError("output.base_dir must be a non-empty string")

    if not isinstance(cfg["output"]["include_timestamp"], bool):
        raise TypeError("output.include_timestamp must be boolean")

    # Optional disk checks (sequence_dir, imu.csv, bag CSV, noise_config file)
    if strict_paths:
        if workspace_root is None:
            raise ValueError("workspace_root is required when strict_paths=True")
        _validate_dataset_paths(cfg)
        _validate_noise_config_file(cfg, workspace_root)
        if str(cfg["contact"]["detector"]).lower() == "neural":
            _validate_neural_checkpoint_paths(cfg, workspace_root)


# --- strict_paths helpers (used only from validate_experiment_dict) -----------------------------


def _neural_checkpoint_sidecar_paths(checkpoint: Path) -> tuple[Path, Path]:
    """``contact_gru.pt`` → ``contact_gru_meta.json`` / ``contact_gru_scaler.npz`` (same as trainer)."""
    parent = checkpoint.parent
    stem = checkpoint.stem
    return parent / f"{stem}_meta.json", parent / f"{stem}_scaler.npz"


def _validate_contact_ocelot_block(cfg: Mapping[str, Any]) -> None:
    c = cfg.get("contact")
    if not isinstance(c, Mapping):
        return
    oc = c.get("ocelot")
    if not isinstance(oc, Mapping):
        raise ValueError("contact.detector is ocelot but contact.ocelot must be a mapping")
    ufs = oc.get("use_fsm", True)
    ugl = oc.get("use_glrt", True)
    if not isinstance(ufs, bool) or not isinstance(ugl, bool):
        raise TypeError("contact.ocelot.use_fsm and use_glrt must be booleans")
    if not ufs and not ugl:
        raise ValueError("contact.ocelot: at least one of use_fsm or use_glrt must be true")
    mode = str(oc.get("fsm_gmm_mode", "offline")).lower().strip()
    if mode not in ("offline", "online"):
        raise ValueError("contact.ocelot.fsm_gmm_mode must be offline or online")
    for key in ("force_on", "force_off"):
        if key not in oc:
            continue
        v = oc[key]
        if isinstance(v, (list, tuple)):
            raise TypeError(
                f"contact.ocelot.{key} must be a single number (same threshold for all legs), not a list"
            )
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"contact.ocelot.{key} must be a number")
        if not math.isfinite(float(v)):
            raise ValueError(f"contact.ocelot.{key} must be finite")
    if mode == "online":
        ws = oc.get("window_size", 500)
        fi = oc.get("fit_interval", 250)
        if isinstance(ws, bool) or not isinstance(ws, (int, float)):
            raise TypeError("contact.ocelot.window_size must be a number")
        if isinstance(fi, bool) or not isinstance(fi, (int, float)):
            raise TypeError("contact.ocelot.fit_interval must be a number")
        if int(ws) < 1:
            raise ValueError("contact.ocelot.window_size must be >= 1 when fsm_gmm_mode is online")
        if int(fi) < 1:
            raise ValueError("contact.ocelot.fit_interval must be >= 1 when fsm_gmm_mode is online")
    if "noise_std_dev" in oc:
        ns = oc["noise_std_dev"]
        if isinstance(ns, bool):
            raise TypeError("contact.ocelot.noise_std_dev must be a number")
        nf = float(ns)
        if not math.isfinite(nf) or nf <= 0:
            raise ValueError("contact.ocelot.noise_std_dev must be a finite positive number")
    rh = oc.get("rate_hz", None)
    if rh is not None and rh != "":
        if isinstance(rh, bool):
            raise TypeError("contact.ocelot.rate_hz must be a number or null")
        rf = float(rh)
        if not math.isfinite(rf) or rf <= 0:
            raise ValueError("contact.ocelot.rate_hz must be null or a finite positive number")


def _validate_contact_neural_block(cfg: Mapping[str, Any]) -> None:
    c = cfg.get("contact")
    if not isinstance(c, Mapping):
        return
    nn = c.get("neural")
    if not isinstance(nn, Mapping):
        raise ValueError("contact.detector is neural but contact.neural must be a mapping")
    ck = nn.get("checkpoint")
    if not isinstance(ck, str) or not str(ck).strip():
        raise ValueError("contact.neural.checkpoint must be a non-empty string")
    for key in ("meta_path", "scaler_path"):
        v = nn.get(key)
        if v is None or v == "":
            continue
        if not isinstance(v, str):
            raise TypeError(f"contact.neural.{key} must be a string path or omitted")
    if "stance_probability_threshold" in nn:
        t = nn["stance_probability_threshold"]
        if isinstance(t, bool):
            raise TypeError("contact.neural.stance_probability_threshold must be a number")
        tf = float(t)
        if not (0.0 <= tf <= 1.0):
            raise ValueError("contact.neural.stance_probability_threshold must be in [0, 1]")
    if "device" in nn and nn["device"] is not None and str(nn["device"]).strip():
        dv = nn["device"]
        if not isinstance(dv, str):
            raise TypeError("contact.neural.device must be a string or null")
        ds = str(dv).strip().lower()
        if ds not in ("cpu", "cuda", "mps"):
            raise ValueError(f"contact.neural.device must be cpu, cuda, mps, or null; got {dv!r}")


def _resolve_config_path(raw: str, workspace_root: Path) -> Path:
    p = Path(str(raw).strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (workspace_root / p).resolve()


def _validate_neural_checkpoint_paths(cfg: Mapping[str, Any], workspace_root: Path) -> None:
    c = cfg.get("contact")
    if not isinstance(c, Mapping):
        return
    nn = c.get("neural")
    if not isinstance(nn, Mapping):
        return
    ck_raw = nn.get("checkpoint")
    if not isinstance(ck_raw, str):
        return
    ck = _resolve_config_path(ck_raw, workspace_root)
    if not ck.is_file():
        raise ValueError(f"contact.neural.checkpoint must be an existing file, got {ck}")

    mp_raw, sp_raw = nn.get("meta_path"), nn.get("scaler_path")
    if isinstance(mp_raw, str) and str(mp_raw).strip():
        mp = _resolve_config_path(mp_raw, workspace_root)
        if not mp.is_file():
            raise ValueError(f"contact.neural.meta_path must be an existing file, got {mp}")
    else:
        mp, _sp = _neural_checkpoint_sidecar_paths(ck)
        if not mp.is_file():
            raise ValueError(f"neural meta JSON missing next to checkpoint: {mp}")
    if isinstance(sp_raw, str) and str(sp_raw).strip():
        sp = _resolve_config_path(sp_raw, workspace_root)
        if not sp.is_file():
            raise ValueError(f"contact.neural.scaler_path must be an existing file, got {sp}")
    else:
        _mp, sp = _neural_checkpoint_sidecar_paths(ck)
        if not sp.is_file():
            raise ValueError(f"neural scaler npz missing next to checkpoint: {sp}")


def _validate_noise_config_file(cfg: Mapping[str, Any], workspace_root: Path) -> None:
    ekf = cfg.get("ekf")
    if not isinstance(ekf, Mapping):
        return
    nc = ekf.get("noise_config")
    if nc is None or nc == "":
        return
    p = Path(str(nc)).expanduser()
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise ValueError(f"ekf.noise_config must point to an existing file, got {p}")


def _validate_dataset_paths(cfg: Mapping[str, Any]) -> None:
    seq = Path(str(cfg["dataset"]["sequence_dir"])).expanduser().resolve()
    kind = str(cfg["dataset"]["kind"]).lower()
    if not seq.is_dir():
        raise ValueError(f"dataset: sequence_dir is not a directory: {seq}")
    if kind == "tartanground":
        from leg_odom.io.split_imu_bag import discover_bag_csv_path

        imu = seq / "imu.csv"
        if not imu.is_file():
            raise ValueError(f"dataset: missing imu.csv under {seq}")
        try:
            discover_bag_csv_path(seq)
        except FileNotFoundError as e:
            raise ValueError(f"dataset: {e}") from e
        return
    if kind == "ocelot":
        lowstate = seq / "lowstate.csv"
        if not lowstate.is_file():
            raise ValueError(f"dataset: missing lowstate.csv under {seq}")
        return
    raise ValueError(f"dataset: unsupported kind for strict path validation: {kind!r}")


# --- Path resolution for experiment_resolved.yaml -----------------------------------------------
# Does not validate existence; output_layout runs validate then resolve for the saved snapshot.


def resolve_dataset_paths(cfg: dict[str, Any], _workspace_root: Path) -> dict[str, Any]:
    """
    Return a copy of ``cfg`` with ``dataset.sequence_dir`` canonicalized (``resolve()``),
    ``dataset.data_root`` removed if present, and no ``data_root`` in the saved dict.
    """
    out = copy.deepcopy(cfg)
    ds = out["dataset"]
    ds.pop("data_root", None)
    seq = Path(str(ds["sequence_dir"])).expanduser()
    if not seq.is_absolute():
        raise ValueError("dataset.sequence_dir must be absolute before resolve_dataset_paths")
    ds["sequence_dir"] = str(seq.resolve())
    return out


def resolve_ekf_noise_config_path(cfg: dict[str, Any], workspace_root: Path) -> None:
    """Make ``ekf.noise_config`` an absolute path string (mutates ``cfg``)."""
    ekf = cfg.get("ekf")
    if not isinstance(ekf, Mapping):
        return
    nc = ekf.get("noise_config")
    if nc is None or nc == "":
        return
    p = Path(str(nc)).expanduser()
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    else:
        p = p.resolve()
    ekf["noise_config"] = str(p)


def resolve_contact_neural_paths(cfg: dict[str, Any], workspace_root: Path) -> None:
    """Make ``contact.neural.checkpoint`` (and optional ``meta_path`` / ``scaler_path``) absolute (mutates ``cfg``)."""
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        return
    if str(block.get("detector", "")).lower() != "neural":
        return
    nn = block.get("neural")
    if not isinstance(nn, Mapping):
        return

    def _abs(key: str) -> None:
        v = nn.get(key)
        if not isinstance(v, str) or not str(v).strip():
            return
        p = Path(str(v).strip()).expanduser()
        if not p.is_absolute():
            nn[key] = str((workspace_root / p).resolve())
        else:
            nn[key] = str(p.resolve())

    for k in ("checkpoint", "meta_path", "scaler_path"):
        _abs(k)


# --- Debug / analysis / live visualizer (read merged cfg) --------------------------------------
#
#   debug_enabled            → YAML run.debug.enabled only.
#   debug_effective_from_cli → above OR programmatic cli_debug (tests / future CLI).
#   live_visualizer_effective → effective debug AND live_visualizer.enabled (matplotlib loop).


def debug_enabled(cfg: Mapping[str, Any]) -> bool:
    """True if ``run.debug.enabled`` in the merged experiment dict."""
    d = cfg.get("run", {}).get("debug")
    if not isinstance(d, Mapping):
        return False
    return bool(d.get("enabled", False))


def debug_effective_from_cli(cfg: Mapping[str, Any], *, cli_debug: bool) -> bool:
    """True if ``cli_debug`` is set (programmatic) or ``run.debug.enabled`` is true in YAML."""
    return bool(cli_debug) or debug_enabled(cfg)


def live_visualizer_yaml_enabled(cfg: Mapping[str, Any]) -> bool:
    """Reads ``run.debug.live_visualizer.enabled`` (meaningful only when debug is effective)."""
    lv = cfg.get("run", {}).get("debug", {}).get("live_visualizer")
    if not isinstance(lv, Mapping):
        return False
    return bool(lv.get("enabled", False))


def live_visualizer_effective(cfg: Mapping[str, Any], *, cli_debug: bool) -> bool:
    """Live matplotlib monitor: effective debug on **and** ``live_visualizer.enabled`` in YAML."""
    if not debug_effective_from_cli(cfg, cli_debug=cli_debug):
        return False
    return live_visualizer_yaml_enabled(cfg)


def live_visualizer_sliding_window_s(cfg: Mapping[str, Any]) -> float:
    """Sliding time-window width [s] (``run.debug.live_visualizer.sliding_window_s``)."""
    lv = cfg.get("run", {}).get("debug", {}).get("live_visualizer") or {}
    return float(lv.get("sliding_window_s", 60.0))


def live_visualizer_video_path(cfg: Mapping[str, Any]) -> str | None:
    """Optional ego video path for future wiring (``run.debug.live_visualizer.video_path``)."""
    lv = cfg.get("run", {}).get("debug", {}).get("live_visualizer") or {}
    vp = lv.get("video_path")
    if vp is None or vp == "":
        return None
    return str(vp)


def live_visualizer_update_hz(cfg: Mapping[str, Any]) -> float | None:
    """
    Target matplotlib refresh rate [Hz] (``run.debug.live_visualizer.hz``).

    ``None`` means redraw every EKF step. When set, redraw stride uses
    ``min(hz, dataset_hz)`` against the recording's median IMU rate (see
    :class:`~leg_odom.eval.live_visualizer.LiveVisualizer`).
    """
    lv = cfg.get("run", {}).get("debug", {}).get("live_visualizer") or {}
    raw = lv.get("hz")
    if raw is None or raw == "":
        return None
    return float(raw)
