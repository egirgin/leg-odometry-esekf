"""Helpers for dataset kind resolution in training tools."""

from __future__ import annotations

from pathlib import Path

from leg_odom.training.nn.discovery import is_valid_tartanground_sequence_dir


def infer_dataset_kind_from_sequence_dir(sequence_dir: str | Path) -> str:
    """
    Infer ``dataset.kind`` from files present in one sequence directory.

    Priority:
    1) ``lowstate.csv`` -> ``ocelot``
    2) Tartanground layout (``imu.csv`` + exactly one ``*_bag.csv``) -> ``tartanground``
    """
    seq = Path(sequence_dir).expanduser().resolve()
    if (seq / "lowstate.csv").is_file():
        return "ocelot"
    ok, _ = is_valid_tartanground_sequence_dir(seq)
    if ok:
        return "tartanground"
    raise FileNotFoundError(
        f"Could not infer dataset.kind from {seq}; expected lowstate.csv or imu.csv + exactly one *_bag.csv."
    )
