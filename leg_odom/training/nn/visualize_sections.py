"""
Random train/test sequence window plots: GRF, precomputed stance, sliding predicted stance prob.

Each panel uses only full-length history (no first-row padding): the segment ends at ``idx`` with
``idx >= 2 * window_size - 2`` so every timestep in the plot has a real ``window_size``-sample input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def plot_random_train_test_sections(
    *,
    train_paths: list[Path],
    test_paths: list[Path],
    model: nn.Module,
    device: torch.device,
    scaler: StandardScaler,
    window_size: int,
    n_legs: int,
    for_cnn: bool,
    num_train_sections: int,
    num_test_sections: int,
    dpi: int,
    save_path: Path,
    rng_seed: int,
    foot_forces_by_seq: Mapping[Path, np.ndarray],
    instants_by_seq_leg: Mapping[Path, Mapping[int, np.ndarray]],
    stance_by_seq_leg: Mapping[Path, Mapping[int, npt.NDArray[np.float64]]],
    sequence_uid_by_seq: Mapping[Path, int] | None = None,
) -> Path:
    """
    Sample random windows from train then test bundles; save one figure (2 columns).

    Rows: first row(s) train panels, then test. Subplot titles include ``[train]`` / ``[test]``.
    Purple curve: model probability at each step along the segment (sliding full-history windows).
    """
    import matplotlib.pyplot as plt

    n_tr = int(num_train_sections)
    n_te = int(num_test_sections)
    n = n_tr + n_te
    if n <= 0:
        return save_path
    if n_tr > 0 and not train_paths:
        return save_path
    if n_te > 0 and not test_paths:
        return save_path

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(rng_seed))
    w = int(window_size)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    model.eval()

    plotted = 0
    attempts = 0
    max_attempts = max(50, n * 20)

    phases: list[tuple[str, list[Path], int]] = []
    if n_tr > 0:
        phases.append(("train", train_paths, n_tr))
    if n_te > 0:
        phases.append(("test", test_paths, n_te))

    for split_name, path_list, need in phases:
        got = 0
        while got < need and plotted < n and attempts < max_attempts:
            attempts += 1
            seq_path = Path(rng.choice(np.array(path_list, dtype=object)))
            leg = int(rng.integers(0, int(n_legs)))
            try:
                resolved = Path(seq_path).expanduser().resolve()
                foot_all = np.asarray(foot_forces_by_seq[resolved], dtype=np.float64)
                inst = np.asarray(instants_by_seq_leg[resolved][leg], dtype=np.float64)
                y_full = np.asarray(stance_by_seq_leg[resolved][leg], dtype=np.float64).reshape(-1)
            except (OSError, ValueError, KeyError):
                continue
            t_rows = foot_all.shape[0]
            min_rows = 2 * w - 1
            if t_rows < min_rows or foot_all.shape[1] <= leg:
                continue
            if inst.shape[0] != t_rows or y_full.shape[0] != t_rows:
                continue
            idx = int(rng.integers(2 * w - 2, t_rows))
            scaled = scaler.transform(inst.astype(np.float64, copy=False))
            grf = np.nan_to_num(foot_all[:, leg], nan=0.0, posinf=0.0, neginf=0.0)
            i0 = idx - w + 1
            grf_w = grf[i0 : idx + 1].astype(np.float64)
            gt_w = y_full[i0 : idx + 1]

            feat_t = torch.tensor(scaled, dtype=torch.float32)
            wins = [feat_t[e - w + 1 : e + 1] for e in range(i0, idx + 1)]
            batched = torch.stack(wins, dim=0)
            if for_cnn:
                x = batched.permute(0, 2, 1).to(device)
            else:
                x = batched.to(device)
            try:
                with torch.no_grad():
                    logit = model(x)
                    pred_curve = torch.sigmoid(logit).cpu().numpy().reshape(-1).astype(np.float64)
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
            ax2.plot(tw, gt_w, color="green", linewidth=2.0, label="precomputed stance")
            ax2.plot(
                tw,
                pred_curve,
                color="purple",
                linewidth=2.0,
                linestyle="--",
                label="pred (sliding)",
                alpha=0.9,
            )
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
            tag = "[train]" if split_name == "train" else "[test]"
            ax1.set_title(f"{tag} {bundle_label} leg{leg} end@{idx} full_hist{uid_note}", fontsize=8)
            ax1.grid(True, alpha=0.25)
            plotted += 1
            got += 1

    for j in range(plotted, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Random train (top rows) then test: GRF, precomputed stance, sliding pred prob "
        f"(full {w}-sample history per step, no padding)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=int(dpi))
    plt.close(fig)
    return save_path
