"""
Shared instant feature layout for contact detectors, NN precompute, and GMM+HMM.

Canonical definitions live in :mod:`leg_odom.features.instant_spec`.
"""

from leg_odom.features.instant_spec import (
    ALLOWED_INSTANT_FEATURE_FIELDS,
    DEFAULT_INSTANT_FEATURE_FIELDS,
    FULL_OFFLINE_INSTANT_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    NN_PRECOMPUTE_FORMAT_VERSION,
    InstantFeatureSpec,
    N_JOINT_SCALAR_INDICES,
    build_timeline_features_for_leg,
    flatten_history_window,
    instant_vector_from_step,
    is_allowed_instant_field,
    parse_instant_feature_fields,
    sliding_windows_flat,
    stance_height_meta_index,
    subset_instant_columns,
)

__all__ = [
    "ALLOWED_INSTANT_FEATURE_FIELDS",
    "DEFAULT_INSTANT_FEATURE_FIELDS",
    "FULL_OFFLINE_INSTANT_FIELDS",
    "INSTANT_FEATURE_SPEC_VERSION",
    "NN_PRECOMPUTE_FORMAT_VERSION",
    "InstantFeatureSpec",
    "N_JOINT_SCALAR_INDICES",
    "build_timeline_features_for_leg",
    "flatten_history_window",
    "instant_vector_from_step",
    "is_allowed_instant_field",
    "parse_instant_feature_fields",
    "sliding_windows_flat",
    "stance_height_meta_index",
    "subset_instant_columns",
]
