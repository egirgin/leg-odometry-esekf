"""
CNN and GRU contact estimators (ported from ``legacy/pretrained_models``).

``ContactCNN`` expects ``(batch, in_channels, seq_len)``.
``ContactGRU`` expects ``(batch, seq_len, in_channels)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContactCNN(nn.Module):
    """1D CNN for sliding-window contact classification (logits)."""

    def __init__(self, in_channels: int, window_size: int = 150):
        super().__init__()
        self._window_size = int(window_size)

        self.conv1a = nn.Conv1d(in_channels, 64, kernel_size=3, padding="same")
        self.conv1b = nn.Conv1d(64, 64, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_conv1 = nn.Dropout(0.5)

        self.conv2a = nn.Conv1d(64, 128, kernel_size=3, padding="same")
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_conv2 = nn.Dropout(0.5)

        # Two stride-2 pools → length floor(floor(L/2)/2)
        flattened_length = (self._window_size // 2) // 2
        self.flat_size = 128 * flattened_length

        self.fc1 = nn.Linear(self.flat_size, 2048)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 512)
        self.drop_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1a(x))
        x = self.drop_conv1(torch.relu(self.conv1b(x)))
        x = self.pool1(x)

        x = torch.relu(self.conv2a(x))
        x = self.drop_conv2(torch.relu(self.conv2b(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.drop_fc1(torch.relu(self.fc1(x)))
        x = self.drop_fc2(torch.relu(self.fc2(x)))
        return self.fc3(x)


class ContactGRU(nn.Module):
    """GRU + MLP; last timestep hidden → logit."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        final_step_out = out[:, -1, :]
        return self.mlp(final_step_out)
