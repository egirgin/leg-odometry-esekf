"""
Training tensors: stance labels from precomputed per-sequence timelines (``grf_threshold`` / ``gmm_hmm`` via detector replay),
subset kinematic instants, sliding windows.

Loads ``precomputed_instants.npz`` per sequence (see :mod:`leg_odom.training.nn.precomputed_io`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from leg_odom.features.instant_spec import FULL_OFFLINE_INSTANT_FIELDS, subset_instant_columns
from leg_odom.training.nn.io_labels import compute_stance_labels
from leg_odom.training.nn.precomputed_io import load_precomputed_sequence_npz
from leg_odom.training.nn.sequence_frames import grf_stance_labels, load_tartanground_frames

__all__ = [
    "grf_stance_labels",
    "load_tartanground_frames",
    "load_precomputed_subset_by_npz_paths",
    "collect_train_instant_matrix",
    "build_sliding_window_datasets",
    "SlidingWindowDatasetCnn",
    "SlidingWindowDatasetGru",
    "concat_dataset_part_lengths_and_uids",
    "global_index_to_sequence_uid",
]


def load_precomputed_subset_by_npz_paths(
    path_groups: Iterable[Iterable[Path | str]],
    expected_robot_kinematics: str,
    n_legs: int,
    subset_fields: tuple[str, ...],
    *,
    show_progress: bool = True,
) -> tuple[
    dict[Path, dict[int, npt.NDArray[np.float64]]],
    dict[Path, npt.NDArray[np.float64]],
    dict[Path, int],
]:
    """
    For each distinct ``precomputed_instants.npz`` path (dedup order across ``path_groups``), load the bundle,
    subset instant columns to ``subset_fields``, and retain only subsets plus ``foot_forces``.
    """
    seen: set[Path] = set()
    ordered: list[Path] = []
    for group in path_groups:
        for p in group:
            key = Path(p).expanduser().resolve()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)

    instants_by_seq_leg: dict[Path, dict[int, npt.NDArray[np.float64]]] = {k: {} for k in ordered}
    foot_by_seq: dict[Path, npt.NDArray[np.float64]] = {}
    uid_by_seq: dict[Path, int] = {}

    for key in tqdm(ordered, desc="Load precomputed npz", unit="seq", disable=not show_progress):
        bundle = load_precomputed_sequence_npz(
            key,
            expected_robot_kinematics=expected_robot_kinematics,
            n_legs=n_legs,
        )
        foot_by_seq[key] = np.asarray(bundle.foot_forces, dtype=np.float64, order="C")
        uid_by_seq[key] = int(bundle.sequence_uid)
        for leg in range(int(n_legs)):
            full = bundle.instants_by_leg[int(leg)]
            instants_by_seq_leg[key][int(leg)] = subset_instant_columns(
                full, FULL_OFFLINE_INSTANT_FIELDS, subset_fields
            )
    return instants_by_seq_leg, foot_by_seq, uid_by_seq


def collect_train_instant_matrix(
    train_paths: list[Path],
    n_legs: int,
    instants_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]],
) -> npt.NDArray[np.float64]:
    """Stack ``(T, d)`` instant rows from train splits × all legs for scaler fitting."""
    blocks: list[npt.NDArray[np.float64]] = []
    for sd in train_paths:
        key = Path(sd).expanduser().resolve()
        for leg in range(int(n_legs)):
            inst = instants_by_seq_leg[key][leg]
            if inst.size:
                blocks.append(inst)
    if not blocks:
        raise RuntimeError("No training instant features; check precomputed_instants.npz paths.")
    return np.vstack(blocks)


def build_sliding_window_datasets(
    precomputed_instants_npz_paths: list[Path],
    n_legs: int,
    scaler: StandardScaler,
    window_size: int,
    dataset_kind: str,
    labels_cfg: Mapping[str, Any],
    *,
    for_cnn: bool,
    foot_forces_by_seq: Mapping[Path, npt.NDArray[np.float64]],
    instants_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]],
    sequence_uid_by_seq: Mapping[Path, int],
    stance_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]] | None = None,
) -> ConcatDataset:
    """
    One concatenated ``Dataset`` over all sequences × legs after scaling instants.

    Label at index ``i`` is stance at timestep ``i`` (end of window). Each part stores
    ``sequence_uid`` for the parent sequence (see :func:`global_index_to_sequence_uid`).

    If ``stance_by_seq_leg`` is set (GRF threshold or GMM+HMM replay timelines), use it as ``y``;
    otherwise labels come from :func:`~leg_odom.training.nn.io_labels.compute_stance_labels` (extensions only).
    """
    parts: list[Dataset] = []
    for sd in precomputed_instants_npz_paths:
        key = Path(sd).expanduser().resolve()
        foot = foot_forces_by_seq[key]
        seq_uid = int(sequence_uid_by_seq[key])
        for leg in range(int(n_legs)):
            inst = instants_by_seq_leg[key][leg]
            if inst.shape[0] == 0:
                continue
            if stance_by_seq_leg is not None:
                y = np.asarray(stance_by_seq_leg[key][int(leg)], dtype=np.float64).reshape(-1)
            else:
                y = compute_stance_labels(
                    dataset_kind,
                    int(leg),
                    labels_cfg,
                    foot_forces=foot,
                )
            if y.shape[0] != inst.shape[0]:
                raise RuntimeError(
                    f"Label length {y.shape[0]} != features {inst.shape[0]} for {key} leg {leg}"
                )
            scaled = scaler.transform(inst.astype(np.float64, copy=False))
            if for_cnn:
                parts.append(SlidingWindowDatasetCnn(scaled, y, window_size, sequence_uid=seq_uid))
            else:
                parts.append(SlidingWindowDatasetGru(scaled, y, window_size, sequence_uid=seq_uid))
    if not parts:
        raise RuntimeError("No sliding-window samples after processing sequences.")
    return ConcatDataset(parts)


def concat_dataset_part_lengths_and_uids(ds: ConcatDataset) -> tuple[list[int], list[int]]:
    """Lengths and ``sequence_uid`` for each part of a :class:`ConcatDataset` built by this module."""
    parts = ds.datasets
    lengths = [len(p) for p in parts]
    uids = [int(getattr(p, "sequence_uid", -1)) for p in parts]
    return lengths, uids


def global_index_to_sequence_uid(global_idx: int, lengths: Sequence[int], uids: Sequence[int]) -> int:
    """Map a flat index from :class:`ConcatDataset` to the corresponding ``sequence_uid``."""
    if global_idx < 0:
        raise IndexError(global_idx)
    acc = 0
    for L, u in zip(lengths, uids):
        if global_idx < acc + L:
            return int(u)
        acc += L
    raise IndexError(global_idx)


class SlidingWindowDatasetCnn(Dataset):
    """``__getitem__`` returns ``(window_CxL, label)`` for Conv1d."""

    def __init__(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        window_size: int,
        *,
        sequence_uid: int = 0,
    ):
        self.sequence_uid = int(sequence_uid)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = int(window_size)
        pad_size = self.window_size - 1
        self.padded_features = torch.cat((self.features[0].repeat(pad_size, 1), self.features), dim=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.padded_features[idx : idx + self.window_size]
        label = self.labels[idx]
        # TODO: Keep CNN windows as (C, L) because nn.Conv1d expects (B, C, L).
        # Revisit if we later standardize all model inputs to one canonical layout.
        return window.T, label


class SlidingWindowDatasetGru(Dataset):
    """``__getitem__`` returns ``(window_LxC, label)`` for GRU ``batch_first``."""

    def __init__(
        self,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        window_size: int,
        *,
        sequence_uid: int = 0,
    ):
        self.sequence_uid = int(sequence_uid)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = int(window_size)
        pad_size = self.window_size - 1
        self.padded_features = torch.cat((self.features[0].repeat(pad_size, 1), self.features), dim=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.padded_features[idx : idx + self.window_size]
        label = self.labels[idx]
        # TODO: Keep GRU windows as (L, C) because nn.GRU(batch_first=True) expects (B, L, C).
        # Revisit if we later standardize all model inputs to one canonical layout.
        return window, label
