"""Neural contact training (CNN/GRU); dataset routing via ``dataset.kind`` in train YAML."""

from leg_odom.training.nn.discovery import (
    discover_ocelot_sequence_dirs,
    discover_tartanground_sequence_dirs,
    is_valid_ocelot_sequence_dir,
    is_valid_tartanground_sequence_dir,
)
from leg_odom.training.nn.io_labels import discover_sequence_dirs

__all__ = [
    "discover_sequence_dirs",
    "discover_ocelot_sequence_dirs",
    "discover_tartanground_sequence_dirs",
    "is_valid_ocelot_sequence_dir",
    "is_valid_tartanground_sequence_dir",
]
