"""Shared matplotlib layout: GRF vs stance probability / binary stance per leg."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def plot_grf_contact_overview(
    t_abs: npt.NDArray[np.floating],
    grfs: list[npt.NDArray[np.floating]],
    st: list[npt.NDArray[np.floating]],
    ps: list[npt.NDArray[np.floating]],
    *,
    suptitle: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    One subplot per leg: GRF (left axis), ``p_stance`` and stance fill (right axis).

    If ``save_path`` is set, writes PNG and does not show unless ``show`` is True.
    """
    import matplotlib.pyplot as plt

    n_legs = len(grfs)
    fig, axes = plt.subplots(n_legs, 1, figsize=(12, 2.5 * n_legs), sharex=True)
    if n_legs == 1:
        axes = [axes]
    ta = np.asarray(t_abs, dtype=np.float64)
    for leg in range(n_legs):
        ax = axes[leg]
        ax.plot(ta, np.asarray(grfs[leg], dtype=np.float64), color="C0", lw=0.8, label="GRF (N)")
        ax.set_ylabel(f"leg {leg}")
        ax2 = ax.twinx()
        ax2.plot(ta, np.asarray(ps[leg], dtype=np.float64), color="C2", alpha=0.7, lw=0.8, label="p_stance")
        ax2.fill_between(
            ta,
            0.0,
            np.asarray(st[leg], dtype=np.float64),
            color="C3",
            alpha=0.15,
            step="pre",
            label="stance",
        )
        ax2.set_ylim(-0.05, 1.05)
        if leg == 0:
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
    axes[-1].set_xlabel("t_abs (s)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    if save_path is not None:
        outp = Path(save_path).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
