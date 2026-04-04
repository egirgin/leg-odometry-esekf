"""
Neural contact detectors (CNN / GRU): load ``train_contact_nn`` artifacts and run online.

Uses the same instant layout as training (:func:`~leg_odom.features.instant_spec.instant_vector_from_step`)
and the same left padding as :class:`~leg_odom.training.nn.data.SlidingWindowDatasetGru` /
:class:`~leg_odom.training.nn.data.SlidingWindowDatasetCnn`. ZUPT covariance follows GMM+HMM:
``R = (1 / max(p_stance, ε)) I₃``.
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate
from leg_odom.contact.gmm_hmm.detector import ZUPT_P_STANCE_FLOOR, zupt_R_foot_from_p_stance
from leg_odom.features.instant_spec import (
    INSTANT_FEATURE_SPEC_VERSION,
    InstantFeatureSpec,
    instant_vector_from_step,
    parse_instant_feature_fields,
)

try:
    import torch
except ImportError as e:  # pragma: no cover
    torch = None  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "Neural contact detection requires PyTorch. Install with the repo's requirements-nn.txt "
            f"(original import error: {_TORCH_IMPORT_ERROR})"
        ) from _TORCH_IMPORT_ERROR


def _pick_device(name: str | None) -> "torch.device":
    _require_torch()
    assert torch is not None
    if name is not None and str(name).strip():
        n = str(name).strip().lower()
        if n in ("cpu", "cuda", "mps"):
            return torch.device(n)
        raise ValueError(f"contact.neural.device must be cpu, cuda, mps, or null; got {name!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_sidecar_paths(checkpoint: Path) -> tuple[Path, Path]:
    stem = checkpoint.stem
    parent = checkpoint.parent
    return parent / f"{stem}_meta.json", parent / f"{stem}_scaler.npz"


def _scale_row(x: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], scale: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    s = np.where(np.asarray(scale, dtype=np.float64) == 0.0, 1.0, np.asarray(scale, dtype=np.float64))
    return (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / s


def _build_padded_window_rows(
    k: int,
    window: int,
    first_x: npt.NDArray[np.float64],
    buf: deque[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Match ``SlidingWindowDataset*``: padded index ``i`` is ``x0`` for ``i < window-1`` else
    ``x[i - (window - 1)]``; timestep ``k`` uses rows ``padded[k : k + window]``.
    """
    w = int(window)
    w1 = w - 1
    start = max(0, k - w1)
    out = np.zeros((w, first_x.shape[0]), dtype=np.float64)
    for o in range(w):
        idx_p = k + o
        if idx_p < w1:
            out[o] = first_x
        else:
            fi = idx_p - w1
            j = fi - start
            out[o] = buf[j]
    return out


class NeuralSharedRuntime:
    """One loaded model + scaler + spec; shared by all per-leg detectors."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        meta_path: Path | None,
        scaler_path: Path | None,
        device: "torch.device",
    ) -> None:
        _require_torch()
        assert torch is not None
        from leg_odom.training.nn.models import ContactCNN, ContactGRU

        ckpt = Path(checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"neural checkpoint not found: {ckpt}")
        mp = Path(meta_path).expanduser().resolve() if meta_path is not None else _default_sidecar_paths(ckpt)[0]
        sp = Path(scaler_path).expanduser().resolve() if scaler_path is not None else _default_sidecar_paths(ckpt)[1]
        if not mp.is_file():
            raise FileNotFoundError(f"neural meta JSON not found: {mp}")
        if not sp.is_file():
            raise FileNotFoundError(f"neural scaler npz not found: {sp}")

        with open(mp, encoding="utf-8") as f:
            meta: dict[str, Any] = json.load(f)
        arch = str(meta.get("architecture", "")).lower()
        if arch not in ("cnn", "gru"):
            raise ValueError(f"meta architecture must be cnn or gru, got {arch!r}")
        fields = meta.get("feature_fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("meta.feature_fields must be a non-empty list")
        feature_fields = tuple(str(x) for x in fields)
        self._spec: InstantFeatureSpec = parse_instant_feature_fields(feature_fields)

        hist = int(meta.get("history_length", 0))
        if hist < 1:
            raise ValueError(f"meta.history_length must be >= 1, got {hist}")
        self._window = hist

        d_meta = int(meta.get("instant_dim", 0))
        if d_meta != self._spec.instant_dim:
            raise ValueError(f"meta.instant_dim {d_meta} != spec.instant_dim {self._spec.instant_dim}")

        ver = meta.get("instant_feature_spec_version")
        if ver is not None and int(ver) != int(INSTANT_FEATURE_SPEC_VERSION):
            warnings.warn(
                f"Checkpoint meta instant_feature_spec_version={ver} != current {INSTANT_FEATURE_SPEC_VERSION}; "
                "feature semantics may differ.",
                stacklevel=1,
            )

        z = np.load(sp, allow_pickle=False)
        mean = np.asarray(z["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(z["scale"], dtype=np.float64).reshape(-1)
        if mean.shape[0] != self._spec.instant_dim or scale.shape[0] != self._spec.instant_dim:
            raise ValueError(
                f"scaler length {mean.shape[0]} != instant_dim {self._spec.instant_dim}"
            )
        self._mean = mean
        self._scale = scale

        try:
            bundle = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            bundle = torch.load(ckpt, map_location=device)
        if not isinstance(bundle, dict) or "state_dict" not in bundle:
            raise ValueError(f"checkpoint must be a dict with 'state_dict' key: {ckpt}")
        d_in = self._spec.instant_dim
        if arch == "cnn":
            self._model = ContactCNN(d_in, window_size=self._window).to(device)
        else:
            self._model = ContactGRU(d_in).to(device)
        self._model.load_state_dict(bundle["state_dict"], strict=True)
        self._model.eval()
        self._arch: Literal["cnn", "gru"] = arch  # type: ignore[assignment]
        self._device = device
        self._robot_kinematics = str(meta.get("robot_kinematics", "")).lower()

    @property
    def robot_kinematics(self) -> str:
        return self._robot_kinematics

    @property
    def spec(self) -> InstantFeatureSpec:
        return self._spec

    @property
    def window_size(self) -> int:
        return self._window

    @property
    def architecture(self) -> str:
        return self._arch

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        return self._mean

    @property
    def scale(self) -> npt.NDArray[np.float64]:
        return self._scale

    @property
    def device(self) -> "torch.device":
        return self._device

    def forward_window(self, rows_f64: npt.NDArray[np.float64]) -> float:
        """Return stance probability in (0, 1) from a ``(window, d)`` float array."""
        _require_torch()
        assert torch is not None
        x = torch.tensor(rows_f64, dtype=torch.float32, device=self._device).unsqueeze(0)
        if self._arch == "cnn":
            x = x.transpose(1, 2)
        with torch.no_grad():
            logit = self._model(x)
            p = torch.sigmoid(logit).item()
        return float(p)


class NeuralContactDetector(BaseContactDetector):
    """Per-foot sliding window over scaled instants; shares :class:`NeuralSharedRuntime`."""

    def __init__(self, runtime: NeuralSharedRuntime, *, stance_probability_threshold: float = 0.5) -> None:
        super().__init__()
        self._rt = runtime
        self._thr = float(stance_probability_threshold)
        self._buf: deque[npt.NDArray[np.float64]] = deque(maxlen=runtime.window_size)
        self._k = -1
        self._first: npt.NDArray[np.float64] | None = None

    @property
    def feature_dim(self) -> int:
        return int(self._rt.window_size * self._rt.spec.instant_dim)

    @property
    def history_length(self) -> int:
        return int(self._rt.window_size)

    def reset(self) -> None:
        self._buf.clear()
        self._k = -1
        self._first = None

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        raw = instant_vector_from_step(step, self._rt.spec)
        x = _scale_row(raw, self._rt.mean, self._rt.scale)
        self._k += 1
        k = self._k
        if self._first is None:
            self._first = np.asarray(x, dtype=np.float64, order="C").copy()
        self._buf.append(np.asarray(x, dtype=np.float64, order="C"))
        assert self._first is not None
        win = _build_padded_window_rows(k, self._rt.window_size, self._first, self._buf)
        p_stance = self._rt.forward_window(win)
        stance = bool(p_stance >= self._thr)
        pe = max(float(p_stance), ZUPT_P_STANCE_FLOOR)
        zupt_var = 1.0 / pe
        r = zupt_R_foot_from_p_stance(float(p_stance))
        self._last_zupt_R_foot = np.asarray(r, dtype=np.float64, order="C")
        return ContactEstimate(stance=stance, p_stance=float(p_stance), zupt_meas_var=float(zupt_var))


def _resolve_nn_path(p: Path, *, workspace_root: Path | None) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    root = workspace_root if workspace_root is not None else Path.cwd()
    return (root / p).resolve()


def build_neural_detectors_from_cfg(
    cfg: Mapping[str, Any],
    *,
    kin_model: Any,
    workspace_root: Path | None = None,
) -> list[NeuralContactDetector]:
    """
    Build one :class:`NeuralContactDetector` per leg (shared weights and scaler).

    Expects ``contact.neural.checkpoint`` (path to ``contact_{cnn,gru}.pt``). Optional:
    ``meta_path``, ``scaler_path``, ``stance_probability_threshold``, ``device``.
    """
    block = cfg.get("contact")
    if not isinstance(block, Mapping):
        raise ValueError("contact config missing")
    nn_cfg = block.get("neural")
    if not isinstance(nn_cfg, Mapping):
        raise ValueError("contact.detector is neural but contact.neural mapping is missing")
    ck = nn_cfg.get("checkpoint")
    if not isinstance(ck, str) or not str(ck).strip():
        raise ValueError("contact.neural.checkpoint must be a non-empty string")

    mp = nn_cfg.get("meta_path")
    sp = nn_cfg.get("scaler_path")
    meta_path: Path | None = None
    scaler_path: Path | None = None
    if isinstance(mp, str) and str(mp).strip():
        meta_path = _resolve_nn_path(Path(str(mp).strip()), workspace_root=workspace_root)
    if isinstance(sp, str) and str(sp).strip():
        scaler_path = _resolve_nn_path(Path(str(sp).strip()), workspace_root=workspace_root)

    thr = float(nn_cfg.get("stance_probability_threshold", 0.5))
    if not (0.0 <= thr <= 1.0):
        raise ValueError("contact.neural.stance_probability_threshold must be in [0, 1]")

    dev_kw = nn_cfg.get("device")
    device = _pick_device(str(dev_kw) if dev_kw is not None and str(dev_kw).strip() else None)

    ck_path = _resolve_nn_path(Path(str(ck).strip()), workspace_root=workspace_root)

    runtime = NeuralSharedRuntime(
        ck_path,
        meta_path=meta_path,
        scaler_path=scaler_path,
        device=device,
    )

    robot_cfg = str(cfg.get("robot", {}).get("kinematics", "")).lower() if isinstance(cfg.get("robot"), Mapping) else ""
    mr = runtime.robot_kinematics
    if robot_cfg and mr and robot_cfg != mr:
        warnings.warn(
            f"Experiment robot.kinematics={robot_cfg!r} != checkpoint robot_kinematics={mr!r}",
            stacklevel=1,
        )

    n_legs = int(getattr(kin_model, "n_legs"))
    return [NeuralContactDetector(runtime, stance_probability_threshold=thr) for _ in range(n_legs)]
