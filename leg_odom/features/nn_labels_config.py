"""Validate ``labels`` mapping for NN precompute (GRF threshold / GMM+HMM pseudo-labels)."""

from __future__ import annotations

from typing import Any, Mapping


def validate_nn_labels_config(lb: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` / ``NotImplementedError`` if ``labels`` block is invalid."""
    if "method" not in lb:
        raise ValueError("labels.method is required")
    method = str(lb.get("method", "")).strip().lower()
    if method == "grf_threshold":
        g = lb.get("grf_threshold")
        if not isinstance(g, Mapping):
            raise ValueError(
                "labels.grf_threshold must be a mapping when labels.method is grf_threshold "
                "(same keys as contact.grf_threshold; at least force_threshold)"
            )
        if "force_threshold" not in g:
            raise ValueError("labels.grf_threshold.force_threshold is required when labels.method is grf_threshold")
        float(g["force_threshold"])
        return
    if method == "gmm_hmm":
        g = lb.get("gmm_hmm")
        if not isinstance(g, Mapping):
            raise ValueError("labels.gmm_hmm must be a mapping when labels.method is gmm_hmm")
        gm = dict(g)
        if gm.get("pretrained_path"):
            raise ValueError(
                "labels.gmm_hmm.pretrained_path is not allowed for precompute labels (offline per-sequence fit only)"
            )
        mode = str(gm.get("mode", "offline")).lower()
        if mode != "offline":
            raise ValueError(f"labels.gmm_hmm.mode must be offline for precompute labels, got {mode!r}")
        hl = int(gm.get("history_length", 1))
        if hl != 1:
            raise ValueError(
                f"labels.gmm_hmm.history_length must be 1 (instant GMM emissions), got {hl}"
            )
        return
    if method == "dual_hmm":
        raise NotImplementedError(
            "labels.method dual_hmm is not implemented yet; port leg_odom.contact.dual_hmm_fusion first."
        )
    if method == "ocelot":
        raise NotImplementedError(
            "labels.method ocelot is not implemented yet; port leg_odom.contact.ocelot first."
        )
    raise ValueError(
        f"Unknown labels.method {method!r}; use grf_threshold, gmm_hmm, dual_hmm, or ocelot"
    )
