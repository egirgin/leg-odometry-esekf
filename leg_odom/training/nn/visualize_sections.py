"""
Random test-sequence window plots: GRF, GT stance, model probability at window end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from leg_odom.training.nn.io_labels import compute_stance_labels


def plot_random_test_sections(
    *,
    test_paths: list[Path],
    model: nn.Module,
    device: torch.device,
    dataset_kind: str,
    labels_cfg: Mapping[str, Any],
    scaler: StandardScaler,
    window_size: int,
    n_legs: int,
    for_cnn: bool,
    num_sections: int,
    dpi: int,
    save_path: Path,
    rng_seed: int,
    foot_forces_by_seq: Mapping[Path, np.ndarray],
    instants_by_seq_leg: Mapping[Path, Mapping[int, np.ndarray]],
    sequence_uid_by_seq: Mapping[Path, int] | None = None,
    stance_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]] | None = None,
) -> Path:
    """
    Sample random (sequence, leg in ``0..n_legs-1``, end index) from test paths; save one figure with subplots.

    Each subplot: GRF over the window (primary), GT stance and predicted prob (secondary).
    Uses precomputed ``foot_forces`` and subset instant matrices (same as training).
    """
    import matplotlib.pyplot as plt

    if not test_paths or num_sections <= 0:
        return save_path

    gt_note = "pseudo-label stance" if stance_by_seq_leg is not None else "GT stance"

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(rng_seed))
    w = int(window_size)
    n = int(num_sections)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    model.eval()

    plotted = 0
    attempts = 0
    max_attempts = max(50, n * 20)

    while plotted < n and attempts < max_attempts:
        attempts += 1
        seq_path = Path(rng.choice(np.array(test_paths, dtype=object)))
        leg = int(rng.integers(0, int(n_legs)))
        try:
            resolved = Path(seq_path).expanduser().resolve()
            foot_all = np.asarray(foot_forces_by_seq[resolved], dtype=np.float64)
            inst = np.asarray(instants_by_seq_leg[resolved][leg], dtype=np.float64)
        except (OSError, ValueError, KeyError):
            continue
        t_rows = foot_all.shape[0]
        if t_rows < w or foot_all.shape[1] <= leg:
            continue
        if inst.shape[0] != t_rows:
            continue
        idx = int(rng.integers(w - 1, t_rows))
        scaled = scaler.transform(inst.astype(np.float64, copy=False))
        if stance_by_seq_leg is not None:
            y_full = np.asarray(stance_by_seq_leg[resolved][leg], dtype=np.float64).reshape(-1)
        else:
            m = str(labels_cfg.get("method", "")).strip().lower()
            if m in ("grf_threshold", "gmm_hmm"):
                raise ValueError(
                    f"plot_random_test_sections requires stance_by_seq_leg when labels.method is {m!r} "
                    "(detector replay timelines)."
                )
            y_full = compute_stance_labels(dataset_kind, leg, labels_cfg, foot_forces=foot_all)
        grf = np.nan_to_num(foot_all[:, leg], nan=0.0, posinf=0.0, neginf=0.0)
        i0 = idx - w + 1
        grf_w = grf[i0 : idx + 1].astype(np.float64)
        gt_w = y_full[i0 : idx + 1]

        feat_t = torch.tensor(scaled, dtype=torch.float32)
        pad = w - 1
        padded = torch.cat((feat_t[0].repeat(pad, 1), feat_t), dim=0)
        win = padded[idx : idx + w]
        if for_cnn:
            x = win.T.unsqueeze(0).to(device)
        else:
            x = win.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                logit = model(x)
                p_hat = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
        except Exception:
            continue

        r, c = divmod(plotted, ncols)
        ax1 = axes[r][c]
        tw = np.arange(w, dtype=np.float64)
        ax1.plot(tw, grf_w, color="black", alpha=0.45, label="GRF (N)")
        ax1.set_xlabel("window step")
        ax1.set_ylabel("GRF", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax2 = ax1.twinx()
        ax2.plot(
            tw,
            gt_w,
            color="green",
            linewidth=2.0,
            label="pseudo-label" if stance_by_seq_leg is not None else "GT stance",
        )
        ax2.axhline(p_hat, color="purple", linestyle="--", linewidth=2.0, label=f"pred={p_hat:.2f}")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("prob", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
        uid_note = ""
        if sequence_uid_by_seq is not None:
            uid_note = f" uid={sequence_uid_by_seq.get(resolved, '?')}"
        bundle_label = f"{resolved.parent.name}/{resolved.name}"
        ax1.set_title(f"{bundle_label} leg{leg} end@{idx}{uid_note}", fontsize=8)
        ax1.grid(True, alpha=0.25)
        plotted += 1

    for j in range(plotted, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Random test windows: GRF + {gt_note} + predicted stance prob (window end)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=int(dpi))
    plt.close(fig)
    return save_path
