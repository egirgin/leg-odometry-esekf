"""
Leg odometry package (refactor target).

This tree is **independent** of ``legacy/``: do not import from ``legacy`` here.
The frozen reference implementation lives only under ``legacy/`` for comparison
and porting; see ARCHITECTURE.md.

Implementation numeric guards (IMU FLU checks, ``dt`` clamps, etc.) live in
``leg_odom.thresholds`` — separate from experiment hyperparameters (YAML / ``parameters.py``).
"""

__all__: list[str] = []
