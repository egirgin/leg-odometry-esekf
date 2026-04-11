"""Lightweight window augmentations for SSL view generation."""

from __future__ import annotations

import torch


def augment_window(
    window: torch.Tensor,
    *,
    gaussian_noise_std: float,
    feature_dropout_prob: float,
    scale_jitter_std: float,
) -> torch.Tensor:
    """Apply simple stochastic augmentations to one window tensor."""
    x = window.clone()

    if gaussian_noise_std > 0.0:
        x = x + torch.randn_like(x) * float(gaussian_noise_std)

    if feature_dropout_prob > 0.0:
        keep = (torch.rand_like(x) >= float(feature_dropout_prob)).to(dtype=x.dtype)
        x = x * keep

    if scale_jitter_std > 0.0:
        scale = 1.0 + torch.randn(1, dtype=x.dtype, device=x.device) * float(scale_jitter_std)
        x = x * scale

    return x
