"""Instant feature spec for dual HMM kinematic branch (no ``grf_n``)."""

from __future__ import annotations

from typing import Sequence

from leg_odom.features import InstantFeatureSpec, parse_instant_feature_fields


def parse_dual_kinematic_feature_fields(names: Sequence[str]) -> InstantFeatureSpec:
    """
    Kinematic-only fields for the dual HMM kin GMM/HMM.

    ``grf_n`` is forbidden (load uses a separate 1D GRF path). Other rules match
    :func:`~leg_odom.features.parse_instant_feature_fields` (e.g. include ``p_foot_body_z``
    when using multiple fields).
    """
    fields = tuple(str(x).strip() for x in names)
    if not fields:
        raise ValueError("dual_hmm.feature_fields (kinematic) is empty")
    if any(f == "grf_n" for f in fields):
        raise ValueError("dual_hmm kinematic feature_fields must not include grf_n; load uses foot forces separately")
    return parse_instant_feature_fields(fields)
