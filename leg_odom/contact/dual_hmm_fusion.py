"""
Dual HMM fusion (GRF + kinematics), optional energy-modulated kin transitions.

Implementation: :mod:`leg_odom.contact.dual_hmm`.
"""

from leg_odom.contact.dual_hmm import (
    DualHmmContactDetector,
    build_dual_hmm_detectors_from_cfg,
    parse_dual_kinematic_feature_fields,
)

__all__ = [
    "DualHmmContactDetector",
    "build_dual_hmm_detectors_from_cfg",
    "parse_dual_kinematic_feature_fields",
]
