"""
Per-timestep ESEKF + contact bookkeeping for evaluation (CSV export).

Mirrors the intent of ``legacy/helpers.create_log_entry`` with a fixed column set suitable
for downstream metrics: ``sec``/``nanosec``/``t_abs``, nominal state, bias estimates,
**position** error-state diagonal variances only (``dp`` block of :class:`~leg_odom.filters.esekf.ErrorStateEkf`),
per-leg contact / ZUPT measurement variance, batch NIS, and foot velocities in world.

**Error-state diagonal (position only):** ``P_dp_x``, ``P_dp_y``, ``P_dp_z`` — variances for the
first three error states (position error in world).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.filters.esekf import ErrorStateEkf
from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL

_NLEGS = 4

_P_DIAG_NAMES = ("P_dp_x", "P_dp_y", "P_dp_z")


def empty_zupt_info() -> dict[str, Any]:
    return {
        "per_foot": [],
        "accepted": 0,
        "nis": float("nan"),
        "dof": 0,
        "nis_lo": float("nan"),
        "nis_hi": float("nan"),
    }


def _zupt_by_leg(per_foot: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(d["leg_id"]): d for d in per_foot if "leg_id" in d}


def build_ekf_step_log_row(
    timeline_row: pd.Series,
    ekf: ErrorStateEkf,
    *,
    gyro_raw: npt.NDArray[np.floating],
    foot_kin: list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    stance: list[bool],
    contact_score: list[float],
    contact_zupt_var: list[float],
    zupt_info: Mapping[str, Any],
    n_legs: int = _NLEGS,
) -> dict[str, Any]:
    """
    One dict aligned with :data:`EKF_STEP_LOG_COLUMNS`.

    Parameters
    ----------
    foot_kin
        Per leg ``(p_foot_body, J, qdot)`` in body frame / rad/s, length ``n_legs`` (expected 4).
    stance
        Contact detector stance booleans per leg.
    contact_score
        Detector confidence in ``[0, 1]`` (e.g. ``p_stance``).
    contact_zupt_var
        Isotropic ZUPT variance (m/s)² per leg from :func:`~leg_odom.filters.zupt_measurement.zupt_isotropic_meas_from_p_stance`
        applied to ``p_stance`` (always finite for valid ``p_stance``).
    zupt_info
        Return value of :meth:`~leg_odom.filters.esekf.ErrorStateEkf.update_zupt` for this step
        (or :func:`empty_zupt_info`).
    """
    n = len(foot_kin)
    if n != n_legs:
        raise ValueError(f"foot_kin length {n} != n_legs {n_legs}")
    if len(stance) != n or len(contact_score) != n or len(contact_zupt_var) != n:
        raise ValueError("stance, contact_score, contact_zupt_var must match foot_kin length")

    eul = Rotation.from_matrix(ekf.R).as_euler("zyx", degrees=True)

    row: dict[str, Any] = {
        TIME_SEC_COL: float(timeline_row.get(TIME_SEC_COL, float("nan"))),
        TIME_NANOSEC_COL: float(timeline_row.get(TIME_NANOSEC_COL, float("nan"))),
        "t_abs": float(timeline_row["t_abs"]),
        "p_x": float(ekf.p[0]),
        "p_y": float(ekf.p[1]),
        "p_z": float(ekf.p[2]),
        "v_x": float(ekf.v[0]),
        "v_y": float(ekf.v[1]),
        "v_z": float(ekf.v[2]),
        "roll_deg": float(eul[2]),
        "pitch_deg": float(eul[1]),
        "yaw_deg": float(eul[0]),
        "bgx": float(ekf.bias_gyro[0]),
        "bgy": float(ekf.bias_gyro[1]),
        "bgz": float(ekf.bias_gyro[2]),
        "bax": float(ekf.bias_accel[0]),
        "bay": float(ekf.bias_accel[1]),
        "baz": float(ekf.bias_accel[2]),
        "zupt_n_feet_accepted": int(zupt_info.get("accepted", 0)),
        "zupt_nis": float(zupt_info.get("nis", float("nan"))),
        "zupt_nis_dof": int(zupt_info.get("dof", 0)),
        "zupt_nis_lo": float(zupt_info.get("nis_lo", float("nan"))),
        "zupt_nis_hi": float(zupt_info.get("nis_hi", float("nan"))),
    }

    d = np.diag(ekf.P).astype(np.float64, copy=False)
    for i, name in enumerate(_P_DIAG_NAMES):
        row[name] = float(d[i]) if i < d.size else float("nan")

    zupt_legs = _zupt_by_leg(list(zupt_info.get("per_foot", [])))
    g = np.asarray(gyro_raw, dtype=np.float64).reshape(3)

    for i in range(n_legs):
        row[f"leg{i}_stance"] = int(bool(stance[i]))
        row[f"leg{i}_contact_score"] = float(contact_score[i])
        row[f"leg{i}_zupt_meas_var"] = (
            float(contact_zupt_var[i]) if np.isfinite(contact_zupt_var[i]) else float("nan")
        )
        if i in zupt_legs:
            zd = zupt_legs[i]
            row[f"leg{i}_zupt_mahal"] = float(zd.get("mahal", float("nan")))
            acc = zd.get("accepted", None)
            row[f"leg{i}_zupt_accepted"] = (
                float(int(bool(acc))) if isinstance(acc, (bool, np.bool_)) else float("nan")
            )
            row[f"leg{i}_zupt_innov_vx"] = float(zd.get("v_pred_x", float("nan")))
            row[f"leg{i}_zupt_innov_vy"] = float(zd.get("v_pred_y", float("nan")))
            row[f"leg{i}_zupt_innov_vz"] = float(zd.get("v_pred_z", float("nan")))
        else:
            row[f"leg{i}_zupt_mahal"] = float("nan")
            row[f"leg{i}_zupt_accepted"] = float("nan")
            row[f"leg{i}_zupt_innov_vx"] = float("nan")
            row[f"leg{i}_zupt_innov_vy"] = float("nan")
            row[f"leg{i}_zupt_innov_vz"] = float("nan")

        pb, jj, qd = foot_kin[i]
        vw = ekf.foot_velocity_world(g, pb, jj, qd)
        row[f"leg{i}_v_wx"] = float(vw[0])
        row[f"leg{i}_v_wy"] = float(vw[1])
        row[f"leg{i}_v_wz"] = float(vw[2])

    return row


def ekf_step_log_columns(n_legs: int = _NLEGS) -> tuple[str, ...]:
    """Stable CSV column order."""
    base = (
        TIME_SEC_COL,
        TIME_NANOSEC_COL,
        "t_abs",
        "p_x",
        "p_y",
        "p_z",
        "v_x",
        "v_y",
        "v_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "bgx",
        "bgy",
        "bgz",
        "bax",
        "bay",
        "baz",
        *_P_DIAG_NAMES,
        "zupt_n_feet_accepted",
        "zupt_nis",
        "zupt_nis_dof",
        "zupt_nis_lo",
        "zupt_nis_hi",
    )
    per = []
    for i in range(n_legs):
        per.extend(
            [
                f"leg{i}_stance",
                f"leg{i}_contact_score",
                f"leg{i}_zupt_meas_var",
                f"leg{i}_zupt_mahal",
                f"leg{i}_zupt_accepted",
                f"leg{i}_zupt_innov_vx",
                f"leg{i}_zupt_innov_vy",
                f"leg{i}_zupt_innov_vz",
                f"leg{i}_v_wx",
                f"leg{i}_v_wy",
                f"leg{i}_v_wz",
            ]
        )
    return base + tuple(per)


EKF_STEP_LOG_COLUMNS = ekf_step_log_columns()


def _csv_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    if isinstance(x, (np.floating, np.integer)):
        return str(float(x)) if isinstance(x, np.floating) else str(int(x))
    return str(x)


class EkfStepLogWriter:
    """Stream rows to CSV without holding the full run in memory."""

    def __init__(self, path: Path, *, n_legs: int = _NLEGS) -> None:
        self.path = Path(path)
        self._fieldnames = ekf_step_log_columns(n_legs)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self._fieldnames, extrasaction="ignore")
        self._w.writeheader()

    def write_row(self, row: Mapping[str, Any]) -> None:
        self._w.writerow({k: _csv_cell(row.get(k, "")) for k in self._fieldnames})

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> EkfStepLogWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_ekf_step_log_csv(path: Path, rows: list[dict[str, Any]], *, n_legs: int = _NLEGS) -> None:
    """Write all rows at once (small recordings / tests)."""
    path = Path(path)
    cols = ekf_step_log_columns(n_legs)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_cell(r.get(k, "")) for k in cols})


def sanitize_sequence_slug(sequence_name: str) -> str:
    """Filesystem-safe fragment for ``ekf_history_<slug>.csv``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in sequence_name.strip())[:200] or "recording"
