"""
PyTorch-style access to leg-odometry **sequences** (whole trajectories).

``__getitem__`` returns a :class:`LegOdometrySequence` with a merged dataframe
at IMU rate unless a subclass documents otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from leg_odom.datasets.types import LegOdometrySequence


class BaseLegOdometryDataset(ABC):
    """One index = one contiguous log (directory or file group)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of sequences exposed by this dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> LegOdometrySequence:
        """Return the prepared sequence at ``index``."""
