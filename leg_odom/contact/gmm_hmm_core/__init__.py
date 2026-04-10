"""
Shared HMM + GMM fitting utilities for single-modality GMM+HMM and dual HMM contact detectors.
"""

from leg_odom.contact.gmm_hmm_core.fitting import (
    fit_gmm_ordered,
    fit_offline_dual_per_leg,
    fit_offline_load_grf_per_leg,
    fit_offline_per_leg,
    flat_ordering_component_index,
    load_pretrained_dual_hmm_npz,
    load_pretrained_gmm_npz,
    order_gmm_components,
    save_pretrained_dual_hmm_npz,
)
from leg_odom.contact.gmm_hmm_core.hmm_gaussian import TwoStateGaussianHMM
from leg_odom.contact.gmm_hmm_core.zupt import ZUPT_P_STANCE_FLOOR, zupt_R_foot_from_p_stance

__all__ = [
    "ZUPT_P_STANCE_FLOOR",
    "TwoStateGaussianHMM",
    "fit_gmm_ordered",
    "fit_offline_dual_per_leg",
    "fit_offline_load_grf_per_leg",
    "fit_offline_per_leg",
    "flat_ordering_component_index",
    "load_pretrained_dual_hmm_npz",
    "load_pretrained_gmm_npz",
    "order_gmm_components",
    "save_pretrained_dual_hmm_npz",
    "zupt_R_foot_from_p_stance",
]
