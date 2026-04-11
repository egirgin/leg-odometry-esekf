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

__all__ = [
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
]
