"""
Scalar instant field layout for :class:`~leg_odom.contact.base.ContactDetectorStepInput`.

Shared by GMM+HMM, neural contact training, offline preprocess, and EKF step inputs.
Emission dimension for sliding history is ``history_length * instant_dim`` (time-major flatten of the last N instants).
Pretraining uses raw log kinematics + IMU (no EKF biases); online uses the same field names with live values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from leg_odom.contact.base import ContactDetectorStepInput
from leg_odom.io.columns import (
    FOOT_FORCE_COLS,
    IMU_ACCEL_COLS,
    IMU_GYRO_COLS,
    motor_position_cols,
    motor_torque_cols,
    motor_velocity_cols,
)
from leg_odom.kinematics.base import BaseKinematics

# ---------------------------------------------------------------------------
# Versioning (bump INSTANT_FEATURE_SPEC_VERSION when layout semantics change).
# ---------------------------------------------------------------------------
INSTANT_FEATURE_SPEC_VERSION = 3

# Bump when the on-disk NN precompute bundle layout or keys change (paths, keys, dtypes).
NN_PRECOMPUTE_FORMAT_VERSION = 2

# Fixed scalar keys (each maps to one float from ContactDetectorStepInput).
_EXACT_SCALAR_FIELDS = frozenset(
    {
        "grf_n",
        "p_foot_body_x",
        "p_foot_body_y",
        "p_foot_body_z",
        "v_foot_body_x",
        "v_foot_body_y",
        "v_foot_body_z",
        "est_tau_hip",
        "est_tau_thigh",
        "est_tau_calf",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
    }
)

_JOINT_FIELD_RE = re.compile(r"^(q_leg|dq_leg|tau_leg)_(\d+)$")
# Quadruped × 3 joints → motor indices 0..11 in flattened leg order.
N_JOINT_SCALAR_INDICES = 12


def _joint_field_names() -> frozenset[str]:
    out: set[str] = set()
    for i in range(N_JOINT_SCALAR_INDICES):
        out.add(f"q_leg_{i}")
        out.add(f"dq_leg_{i}")
        out.add(f"tau_leg_{i}")
    return frozenset(out)


# Every name a user may list in ``feature_fields`` (covers all ContactDetectorStepInput channels).
ALLOWED_INSTANT_FEATURE_FIELDS: frozenset[str] = _EXACT_SCALAR_FIELDS | _joint_field_names()


def _full_offline_instant_field_tuple() -> tuple[str, ...]:
    """
    Canonical column order for offline ``(T, D_full)`` instants in ``precomputed_instants.npz``.

    All names in :data:`ALLOWED_INSTANT_FEATURE_FIELDS`; order is stable across runs
    (sorted exact scalars, then per-joint ``q_leg_*``, ``dq_leg_*``, ``tau_leg_*`` for indices 0..11).
    """
    exact = tuple(sorted(_EXACT_SCALAR_FIELDS))
    joint_block: list[str] = []
    for i in range(N_JOINT_SCALAR_INDICES):
        joint_block.extend((f"q_leg_{i}", f"dq_leg_{i}", f"tau_leg_{i}"))
    return exact + tuple(joint_block)


# Full instant layout written by preprocess; training YAML ``features.fields`` must be a subset (same names).
FULL_OFFLINE_INSTANT_FIELDS: tuple[str, ...] = _full_offline_instant_field_tuple()


def is_allowed_instant_field(name: str) -> bool:
    return name in ALLOWED_INSTANT_FEATURE_FIELDS


# Default 5D body kinematic stack (no duplicated IMU per foot in default).
DEFAULT_INSTANT_FEATURE_FIELDS: tuple[str, ...] = (
    "est_tau_calf",
    "v_foot_body_x",
    "v_foot_body_y",
    "v_foot_body_z",
    "p_foot_body_z",
)


@dataclass(frozen=True, slots=True)
class InstantFeatureSpec:
    """Validated ordered field list for one instant feature vector."""

    fields: tuple[str, ...]
    stance_height_instant_index: int | None
    use_higher_grf_mean_for_stance: bool

    @property
    def instant_dim(self) -> int:
        return len(self.fields)

    def ordering_component_index(self) -> int:
        if self.use_higher_grf_mean_for_stance:
            return 0
        if self.stance_height_instant_index is None:
            raise ValueError(
                "Feature spec needs p_foot_body_z for stance/swing ordering, or use grf_n only."
            )
        return int(self.stance_height_instant_index)

    def higher_mean_is_stance(self) -> bool:
        return bool(self.use_higher_grf_mean_for_stance)


def _stance_height_index_in_fields(fields: tuple[str, ...]) -> int | None:
    for i, f in enumerate(fields):
        if f == "p_foot_body_z":
            return i
    return None


def parse_instant_feature_fields(names: Sequence[str]) -> InstantFeatureSpec:
    """
    Return a validated ordered feature column list (:class:`InstantFeatureSpec`).
    Names must be in :data:`ALLOWED_INSTANT_FEATURE_FIELDS`.
    If no stance height index is found, use the grf_n field.
    If no stance height index is found and the grf_n field is not used, raise an error.
    """
    if not names:
        raise ValueError("instant feature fields list is empty")
    fields = tuple(str(x).strip() for x in names)
    for f in fields:
        if f not in ALLOWED_INSTANT_FEATURE_FIELDS:
            raise ValueError(
                f"Unknown instant feature field {f!r}; allowed keys cover ContactDetectorStepInput "
                f"(see ALLOWED_INSTANT_FEATURE_FIELDS in leg_odom.features.instant_spec)."
            )
    zi = _stance_height_index_in_fields(fields)
    use_grf = len(fields) == 1 and fields[0] == "grf_n"
    if zi is None and not use_grf:
        raise ValueError(
            "Multi-field instant features must include p_foot_body_z for stance/swing ordering, "
            "or use a single grf_n field."
        )
    return InstantFeatureSpec(
        fields=fields,
        stance_height_instant_index=zi,
        use_higher_grf_mean_for_stance=use_grf,
    )


def stance_height_meta_index(spec: InstantFeatureSpec) -> int:
    """Persist in .npz; -1 when GRF-only."""
    return int(spec.stance_height_instant_index) if spec.stance_height_instant_index is not None else -1


def instant_vector_from_step(step: ContactDetectorStepInput, spec: InstantFeatureSpec) -> npt.NDArray[np.float64]:
    """Map one ``ContactDetectorStepInput`` to ``(instant_dim,)`` in feature order."""
    out = np.zeros(spec.instant_dim, dtype=np.float64)
    tau = np.asarray(step.tau_leg, dtype=np.float64).reshape(-1)
    vb = np.asarray(step.v_foot_body, dtype=np.float64).reshape(3)
    pb = np.asarray(step.p_foot_body, dtype=np.float64).reshape(3)
    gy = np.asarray(step.gyro_body_corrected, dtype=np.float64).reshape(3)
    ac = np.asarray(step.accel_body_corrected, dtype=np.float64).reshape(3)
    q = np.asarray(step.q_leg, dtype=np.float64).reshape(-1)
    dq = np.asarray(step.dq_leg, dtype=np.float64).reshape(-1)

    for i, name in enumerate(spec.fields):
        if name == "grf_n":
            out[i] = float(step.grf_n)
        elif name == "p_foot_body_x":
            out[i] = float(pb[0])
        elif name == "p_foot_body_y":
            out[i] = float(pb[1])
        elif name == "p_foot_body_z":
            out[i] = float(pb[2])
        elif name == "v_foot_body_x":
            out[i] = float(vb[0])
        elif name == "v_foot_body_y":
            out[i] = float(vb[1])
        elif name == "v_foot_body_z":
            out[i] = float(vb[2])
        elif name == "est_tau_hip":
            out[i] = float(tau[0]) if tau.size > 0 else 0.0
        elif name == "est_tau_thigh":
            out[i] = float(tau[1]) if tau.size > 1 else 0.0
        elif name == "est_tau_calf":
            out[i] = float(tau[2]) if tau.size > 2 else (float(tau[-1]) if tau.size else 0.0)
        elif name == "gyro_x":
            out[i] = float(gy[0])
        elif name == "gyro_y":
            out[i] = float(gy[1])
        elif name == "gyro_z":
            out[i] = float(gy[2])
        elif name == "accel_x":
            out[i] = float(ac[0])
        elif name == "accel_y":
            out[i] = float(ac[1])
        elif name == "accel_z":
            out[i] = float(ac[2])
        else:
            m = _JOINT_FIELD_RE.match(name)
            if not m:
                raise AssertionError(name)
            kind, idx_s = m.group(1), int(m.group(2))
            if kind == "q_leg":
                out[i] = float(q[idx_s]) if idx_s < q.size else 0.0
            elif kind == "dq_leg":
                out[i] = float(dq[idx_s]) if idx_s < dq.size else 0.0
            else:
                out[i] = float(tau[idx_s]) if idx_s < tau.size else 0.0
    return out


def flatten_history_window(rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``(N, d)`` oldest-first → ``(N * d,)`` row-major."""
    r = np.asarray(rows, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"Expected (N, d) window, got shape {r.shape}")
    return r.reshape(-1, order="C")


def sliding_windows_flat(
    instants: npt.NDArray[np.float64],
    history_length: int,
) -> npt.NDArray[np.float64]:
    """``(T, d)`` → ``(T - N + 1, N * d)``; first ``N-1`` rows dropped (no padding)."""
    x = np.asarray(instants, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected (T, d), got {x.shape}")
    t, d = x.shape
    n = int(history_length)
    if n < 1:
        raise ValueError("history_length must be >= 1")
    if t < n:
        return np.zeros((0, n * d), dtype=np.float64)
    out = np.empty((t - n + 1, n * d), dtype=np.float64)
    for k in range(n - 1, t):
        out[k - (n - 1), :] = flatten_history_window(x[k - n + 1 : k + 1, :])
    return out


def subset_instant_columns(
    instants: npt.NDArray[np.float64],
    full_field_names: Sequence[str],
    subset_fields: Sequence[str],
) -> npt.NDArray[np.float64]:
    """
    Select a column subset from full offline instants ``(T, D_full)``.

    ``full_field_names`` must match the column order of ``instants`` (e.g.
    :data:`FULL_OFFLINE_INSTANT_FIELDS`). Raises if any ``subset_fields`` name is missing.
    """
    x = np.asarray(instants, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected instants (T, D), got shape {x.shape}")
    name_to_i = {str(n): i for i, n in enumerate(full_field_names)}
    idx: list[int] = []
    for f in subset_fields:
        key = str(f).strip()
        if key not in name_to_i:
            raise KeyError(
                f"subset field {key!r} not in full_field_names "
                f"(expected columns from FULL_OFFLINE_INSTANT_FIELDS)."
            )
        idx.append(name_to_i[key])
    return np.ascontiguousarray(x[:, idx], dtype=np.float64)


def build_timeline_features_for_leg(
    frames: pd.DataFrame,
    kin_model: BaseKinematics,
    leg_index: int,
    spec: InstantFeatureSpec,
) -> npt.NDArray[np.float64]:
    """
    Offline ``(T, instant_dim)`` for one leg: same FK / foot velocity as the EKF contact step.

    Uses **raw** gyro and accel from the merged frame (no EKF bias subtraction), matching the
    decision to omit filter biases from pretraining.
    """
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())
    gyro_cols = list(IMU_GYRO_COLS)
    accel_cols = list(IMU_ACCEL_COLS)

    n_legs = kin_model.n_legs
    jpl = kin_model.joints_per_leg
    if leg_index < 0 or leg_index >= n_legs:
        raise ValueError(f"leg_index must be in [0, {n_legs - 1}]")

    t_rows = len(frames)
    out = np.zeros((t_rows, spec.instant_dim), dtype=np.float64)

    for k in range(t_rows):
        row = frames.iloc[k]
        gyro = row[gyro_cols].to_numpy(dtype=np.float64)
        accel = row[accel_cols].to_numpy(dtype=np.float64)
        q_all = row[motor_cols].to_numpy(dtype=np.float64)
        dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        sl = slice(leg_index * jpl, (leg_index + 1) * jpl)
        q_leg = q_all[sl]
        dq_leg = dq_all[sl]
        tau_leg = tau_all[sl]
        p_fb = kin_model.fk(leg_index, q_leg)
        pb = np.asarray(p_fb, dtype=np.float64).reshape(3)
        jac = np.asarray(kin_model.J_analytical(leg_index, q_leg), dtype=np.float64).reshape(3, jpl)
        qd = np.asarray(dq_leg, dtype=np.float64).reshape(jpl)
        v_foot_body = np.cross(gyro, pb) + jac @ qd
        grf = float(row.get(FOOT_FORCE_COLS[leg_index], 0.0))
        step = ContactDetectorStepInput(
            grf_n=grf,
            p_foot_body=pb,
            v_foot_body=v_foot_body,
            q_leg=np.asarray(q_leg, dtype=np.float64, order="C"),
            dq_leg=np.asarray(dq_leg, dtype=np.float64, order="C"),
            tau_leg=np.asarray(tau_leg, dtype=np.float64, order="C"),
            gyro_body_corrected=np.asarray(gyro, dtype=np.float64, order="C"),
            accel_body_corrected=np.asarray(accel, dtype=np.float64, order="C"),
        )
        out[k, :] = instant_vector_from_step(step, spec)
    return out
