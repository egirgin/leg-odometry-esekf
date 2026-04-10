"""Dual HMM contact detector (GRF load + kinematic GMM/HMM, optional energy)."""

from leg_odom.contact.dual_hmm.detector import DualHmmContactDetector, build_dual_hmm_detectors_from_cfg
from leg_odom.contact.dual_hmm.spec import parse_dual_kinematic_feature_fields

__all__ = [
    "DualHmmContactDetector",
    "build_dual_hmm_detectors_from_cfg",
    "parse_dual_kinematic_feature_fields",
]
