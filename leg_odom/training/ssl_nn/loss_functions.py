"""Self-supervised losses for training placeholders."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Simple NT-Xent contrastive loss for paired views in one batch."""
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError(f"Expected z1/z2 with identical shape (B, D), got {tuple(z1.shape)} vs {tuple(z2.shape)}")

    if z1.shape[0] < 2:
        return (z1.sum() + z2.sum()) * 0.0

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    logits = reps @ reps.T / float(temperature)

    n = z1.shape[0]
    mask = torch.eye(2 * n, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, float("-inf"))

    targets = torch.arange(n, device=logits.device)
    targets = torch.cat([targets + n, targets], dim=0)
    return F.cross_entropy(logits, targets)
