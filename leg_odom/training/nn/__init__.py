"""Neural contact training (CNN/GRU); dataset routing via ``dataset.kind`` in train YAML."""

from leg_odom.training.nn.discovery import discover_split_sequence_dirs, is_valid_tartanground_sequence_dir
from leg_odom.training.nn.io_labels import discover_sequence_dirs

__all__ = [
    "discover_sequence_dirs",
    "discover_split_sequence_dirs",
    "is_valid_tartanground_sequence_dir",
]
