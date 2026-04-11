"""SSL window datasets built from precomputed instants."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset

from leg_odom.training.ssl_nn.augmentations import augment_window


class SslSlidingWindowDataset(Dataset):
    """Return two augmented views for each time-aligned sliding window."""

    def __init__(
        self,
        features: npt.NDArray[np.float64],
        window_size: int,
        *,
        for_cnn: bool,
        gaussian_noise_std: float,
        feature_dropout_prob: float,
        scale_jitter_std: float,
        sequence_uid: int,
    ):
        self.sequence_uid = int(sequence_uid)
        self.for_cnn = bool(for_cnn)
        self.window_size = int(window_size)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.feature_dropout_prob = float(feature_dropout_prob)
        self.scale_jitter_std = float(scale_jitter_std)

        self.features = torch.tensor(features, dtype=torch.float32)
        pad_size = self.window_size - 1
        self.padded_features = torch.cat((self.features[0].repeat(pad_size, 1), self.features), dim=0)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def _format_window(self, window: torch.Tensor) -> torch.Tensor:
        # Match supervised NN shape conventions for runtime parity.
        return window.T if self.for_cnn else window

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.padded_features[idx : idx + self.window_size]
        window = self._format_window(window)
        x1 = augment_window(
            window,
            gaussian_noise_std=self.gaussian_noise_std,
            feature_dropout_prob=self.feature_dropout_prob,
            scale_jitter_std=self.scale_jitter_std,
        )
        x2 = augment_window(
            window,
            gaussian_noise_std=self.gaussian_noise_std,
            feature_dropout_prob=self.feature_dropout_prob,
            scale_jitter_std=self.scale_jitter_std,
        )
        return x1, x2


def build_ssl_window_dataset(
    precomputed_instants_npz_paths: list[Path],
    n_legs: int,
    scaler: StandardScaler,
    window_size: int,
    *,
    for_cnn: bool,
    instants_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]],
    sequence_uid_by_seq: Mapping[Path, int],
    gaussian_noise_std: float,
    feature_dropout_prob: float,
    scale_jitter_std: float,
) -> ConcatDataset:
    parts: list[Dataset] = []
    for sd in precomputed_instants_npz_paths:
        key = Path(sd).expanduser().resolve()
        seq_uid = int(sequence_uid_by_seq[key])
        for leg in range(int(n_legs)):
            inst = instants_by_seq_leg[key][leg]
            if inst.shape[0] == 0:
                continue
            scaled = scaler.transform(inst.astype(np.float64, copy=False))
            parts.append(
                SslSlidingWindowDataset(
                    scaled,
                    window_size,
                    for_cnn=for_cnn,
                    gaussian_noise_std=gaussian_noise_std,
                    feature_dropout_prob=feature_dropout_prob,
                    scale_jitter_std=scale_jitter_std,
                    sequence_uid=seq_uid,
                )
            )
    if not parts:
        raise RuntimeError("No SSL sliding-window samples after processing sequences.")
    return ConcatDataset(parts)
