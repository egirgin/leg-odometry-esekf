"""
2-component Gaussian mixture fitting and pretrained loading.

Sklearn fits unconstrained mixture components; we **relabel** them so row 0 = stance and
row 1 = swing using either GRF mean (higher ⇒ stance) or height feature (lower ⇒ stance).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from leg_odom.features import (
    DEFAULT_INSTANT_FEATURE_FIELDS,
    InstantFeatureSpec,
    build_timeline_features_for_leg,
    parse_instant_feature_fields,
    sliding_windows_flat,
)
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.kinematics.base import BaseKinematics


def flat_ordering_component_index(spec: InstantFeatureSpec, history_length: int) -> int:
    """Column in the flattened ``N * d`` emission used for stance vs swing labeling."""
    return (int(history_length) - 1) * spec.instant_dim + spec.ordering_component_index()


def order_gmm_components(
    means: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    spec: InstantFeatureSpec,
    history_length: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Permute sklearn rows → stance row 0, swing row 1 (for HMM emission wiring)."""
    m = np.asarray(means, dtype=np.float64)
    c = np.asarray(covariances, dtype=np.float64)
    j = flat_ordering_component_index(spec, history_length)
    col = m[:, j]
    if spec.higher_mean_is_stance():
        stance_idx = int(np.argmax(col))
    else:
        stance_idx = int(np.argmin(col))
    swing_idx = 1 - stance_idx
    mo = np.stack([m[stance_idx], m[swing_idx]], axis=0)
    co = np.stack([c[stance_idx], c[swing_idx]], axis=0)
    _ = weights
    return mo, co


def fit_gmm_ordered(
    X: npt.NDArray[np.float64],
    spec: InstantFeatureSpec,
    history_length: int,
    *,
    random_state: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    """
    Fit full-covariance 2-GMM on rows of ``X``.

    Returns ``(means_stance_swing (2,D), covs (2,D,D), degenerate)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4:
        d = int(X.shape[1]) if X.ndim == 2 else 0
        return (
            np.zeros((2, d), dtype=np.float64),
            np.zeros((2, d, d), dtype=np.float64) if d else np.zeros((2, 0, 0), dtype=np.float64),
            True,
        )
    d = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=int(random_state),
            max_iter=200,
        )
        try:
            gmm.fit(X)
        except (ValueError, np.linalg.LinAlgError):
            z = np.zeros((2, d), dtype=np.float64)
            return z, np.zeros((2, d, d), dtype=np.float64), True
    w = gmm.weights_
    if float(np.max(w)) >= 0.999:
        return (
            np.asarray(gmm.means_, dtype=np.float64),
            np.asarray(gmm.covariances_, dtype=np.float64),
            True,
        )
    mo, co = order_gmm_components(
        np.asarray(gmm.means_, dtype=np.float64),
        np.asarray(gmm.covariances_, dtype=np.float64),
        np.asarray(gmm.weights_, dtype=np.float64),
        spec,
        history_length,
    )
    return mo, co, False


def load_pretrained_gmm_npz(
    path: Path,
    *,
    expected_feature_dim: int,
    expected_history_length: int | None = None,
    expected_instant_dim: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load ``means`` / ``covariances`` and validate shapes against the live detector."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pretrained GMM file not found: {path}")
    data = np.load(path, allow_pickle=True)
    means = np.asarray(data["means"], dtype=np.float64)
    covs = np.asarray(data["covariances"], dtype=np.float64)
    if means.shape != (2, expected_feature_dim):
        raise ValueError(
            f"Pretrained GMM means have shape {means.shape}, expected (2, {expected_feature_dim}). "
            "Match training history_length and feature_fields to the detector."
        )
    if covs.shape != (2, expected_feature_dim, expected_feature_dim):
        raise ValueError(
            f"Pretrained GMM covariances have shape {covs.shape}, expected "
            f"(2, {expected_feature_dim}, {expected_feature_dim})."
        )
    if expected_history_length is not None and "history_length" in data.files:
        fh = int(np.asarray(data["history_length"]).reshape(()))
        if fh != expected_history_length:
            raise ValueError(f"Pretrained history_length {fh} != detector {expected_history_length}")
    if expected_instant_dim is not None and "instant_dim" in data.files:
        di = int(np.asarray(data["instant_dim"]).reshape(()))
        if di != expected_instant_dim:
            raise ValueError(f"Pretrained instant_dim {di} != detector {expected_instant_dim}")
    return means, covs


def fit_offline_per_leg(
    recording: LegOdometrySequence,
    kin_model: BaseKinematics,
    *,
    feature_fields: tuple[str, ...] | None,
    history_length: int,
    random_state: int = 42,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Whole-sequence GMM per leg; used by ``mode: offline`` before the EKF loop."""
    spec = parse_instant_feature_fields(feature_fields or DEFAULT_INSTANT_FEATURE_FIELDS)
    n = int(history_length)
    out: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for leg in range(kin_model.n_legs):
        inst = build_timeline_features_for_leg(recording.frames, kin_model, leg, spec)
        X = sliding_windows_flat(inst, n)
        if X.shape[0] < 4:
            raise ValueError(
                f"Leg {leg}: need at least {n + 3} frames for offline GMM (history_length={n}); "
                f"got {len(inst)} rows."
            )
        mo, co, bad = fit_gmm_ordered(X, spec, n, random_state=random_state + leg)
        if bad:
            raise RuntimeError(
                f"Leg {leg}: offline GMM degenerate. Try different features, N=1, or use online mode "
                "with a pretrained fallback .npz."
            )
        out.append((mo, co))
    return out
