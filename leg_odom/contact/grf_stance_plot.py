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
    energy_per_leg: list[npt.NDArray[np.floating]] | None = None,
) -> None:
    """
    One subplot per leg: GRF (left axis), ``p_stance`` and stance fill (right axis).

    If ``energy_per_leg`` is set (length ``n_legs``, same time length as ``t_abs``), uses two rows
    per leg: GRF / stance row then normalized energy for that leg.

    If ``save_path`` is set, writes PNG and does not show unless ``show`` is True.
    """
    import matplotlib.pyplot as plt

    n_legs = len(grfs)
    ta = np.asarray(t_abs, dtype=np.float64)
    n_rows = 2 * n_legs if energy_per_leg is not None else n_legs
    fig_h = 2.5 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for leg in range(n_legs):
        row_grf = 2 * leg if energy_per_leg is not None else leg
        ax = axes[row_grf]
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

        if energy_per_leg is not None:
            en = np.asarray(energy_per_leg[leg], dtype=np.float64)
            ax_e = axes[row_grf + 1]
            ax_e.plot(ta, en, color="C1", lw=0.8, label="energy (norm.)")
            ax_e.set_ylabel("energy")
            ax_e.set_ylim(-0.05, 1.05)
            if leg == 0:
                ax_e.legend(loc="upper right")

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
