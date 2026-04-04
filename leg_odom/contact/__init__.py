"""Contact inference: ABCs and concrete detectors (GMM+HMM, neural, Ocelot, dual)."""

from __future__ import annotations

from leg_odom.contact.base import (
    BaseContactDetector,
    ContactDetectorStepInput,
    ContactEstimate,
    zupt_isotropic_R_foot,
)
from leg_odom.contact.grf_threshold import (
    GrfThresholdContactDetector,
    build_grf_threshold_detectors_from_cfg,
    make_quadruped_grf_threshold_detectors,
)

# GMM+HMM is loaded lazily so `leg_odom.contact.base` (and `leg_odom.features.instant_spec`)
# can import without pulling in `gmm_hmm` → `leg_odom.features` while that package is still
# initializing (circular import).
_GMM_HMM_EXPORTS = frozenset({"GmmHmmContactDetector", "build_gmm_hmm_detectors_from_cfg"})


def __getattr__(name: str):
    if name in _GMM_HMM_EXPORTS:
        from leg_odom.contact import gmm_hmm as _gmm_hmm

        return getattr(_gmm_hmm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "BaseContactDetector",
    "ContactDetectorStepInput",
    "ContactEstimate",
    "GmmHmmContactDetector",
    "GrfThresholdContactDetector",
    "build_gmm_hmm_detectors_from_cfg",
    "build_grf_threshold_detectors_from_cfg",
    "make_quadruped_grf_threshold_detectors",
    "zupt_isotropic_R_foot",
]
