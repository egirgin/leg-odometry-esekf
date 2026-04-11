"""SSL model wrappers that reuse supervised contact backbones."""

from __future__ import annotations

import torch
import torch.nn as nn

from leg_odom.training.nn.models import ContactCNN, ContactGRU


class ProjectionHead(nn.Module):
    """Tiny projection MLP for contrastive training."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hid = max(8, int(out_dim))
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hid),
            nn.ReLU(),
            nn.Linear(hid, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(architecture: str, d_in: int, window_size: int) -> nn.Module:
    arch = str(architecture).strip().lower()
    if arch == "cnn":
        return ContactCNN(in_channels=int(d_in), window_size=int(window_size))
    if arch == "gru":
        return ContactGRU(in_channels=int(d_in))
    raise ValueError(f"Unsupported architecture {architecture!r}")
