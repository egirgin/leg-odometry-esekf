"""
GMM + single HMM contact stack (not dual HMM).

Public imports stay stable for ``from leg_odom.contact.gmm_hmm import ...``.
Instant feature layout is defined in :mod:`leg_odom.features`.
"""

from leg_odom.contact.gmm_hmm.detector import (
    GmmHmmContactDetector,
    build_gmm_hmm_detectors_from_cfg,
    zupt_R_foot_from_p_stance,
)
from leg_odom.contact.gmm_hmm.fitting import (
    fit_gmm_ordered,
    fit_offline_per_leg,
    load_pretrained_gmm_npz,
)
from leg_odom.contact.gmm_hmm.hmm_gaussian import TwoStateGaussianHMM
from leg_odom.features import (
    ALLOWED_INSTANT_FEATURE_FIELDS,
    DEFAULT_INSTANT_FEATURE_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    InstantFeatureSpec,
    N_JOINT_SCALAR_INDICES,
    build_timeline_features_for_leg,
    flatten_history_window,
    instant_vector_from_step,
    is_allowed_instant_field,
    parse_instant_feature_fields,
    sliding_windows_flat,
    stance_height_meta_index,
)

__all__ = [
    "ALLOWED_INSTANT_FEATURE_FIELDS",
    "DEFAULT_INSTANT_FEATURE_FIELDS",
    "INSTANT_FEATURE_SPEC_VERSION",
    "N_JOINT_SCALAR_INDICES",
    "InstantFeatureSpec",
    "GmmHmmContactDetector",
    "TwoStateGaussianHMM",
    "build_gmm_hmm_detectors_from_cfg",
    "build_timeline_features_for_leg",
    "fit_gmm_ordered",
    "fit_offline_per_leg",
    "flatten_history_window",
    "instant_vector_from_step",
    "is_allowed_instant_field",
    "load_pretrained_gmm_npz",
    "parse_instant_feature_fields",
    "sliding_windows_flat",
    "stance_height_meta_index",
    "zupt_R_foot_from_p_stance",
]
