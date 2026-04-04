"""
Error-state extended Kalman filter: IMU propagation + ZUPT (stacked foot velocities).

Ported from ``legacy/estimator.py`` with the same 15-D error-state layout. **World frame is
FLU** (+X forward, +Y left, +Z up).

**Gravity convention (single path, no sign knob)**  
Accelerometer inputs are expected to be validated as **FLU specific force** at load time
(see :mod:`leg_odom.io.imu_sanitize`): at rest, mean specific force points **up** along
body +Z (~+9.81 m/s²). In world frame we use a fixed vector ``g_w = [0, 0, G]`` with
**positive** scalar ``G = 9.81``. Kinematic acceleration is

    a_world = R @ f_body - g_w

so at rest ``R = I``, ``f_body ≈ [0,0,G]`` gives ``a_world ≈ 0``. If the log is
**gravity-compensated** (linear acceleration only), the caller passes
``accel_gravity_compensated=True`` and the ``- g_w`` term is **omitted** so gravity is not
double-removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
import numpy.typing as npt
import scipy.stats as st
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Default noise / initial covariance (aligned with legacy/parameters.py)
# ---------------------------------------------------------------------------

_DEFAULT_P0_DIAG = np.array(
    [
        0.01**2,
        0.01**2,
        0.01**2,
        0.5**2,
        0.5**2,
        0.5**2,
        np.deg2rad(5.0) ** 2,
        np.deg2rad(5.0) ** 2,
        np.deg2rad(5.0) ** 2,
        0.5**2,
        0.5**2,
        0.5**2,
        np.deg2rad(1.0) ** 2,
        np.deg2rad(1.0) ** 2,
        np.deg2rad(1.0) ** 2,
    ],
    dtype=np.float64,
)

_DEFAULT_IMU_NOISE: dict[str, float] = {
    "accel_std": 0.5,
    "gyro_std": float(np.deg2rad(1.0)),
    "accel_bias_std": 0.01,
    "gyro_bias_std": float(np.deg2rad(0.01)),
}

# World-frame gravity vector in FLU: magnitude G > 0 along +Z (see module docstring).
GRAVITY_WORLD_FLU = np.array([0.0, 0.0, 9.81], dtype=np.float64)


class ErrorStateEkf:
    """
    15-D error state: position, velocity, orientation (so(3)), accel bias, gyro bias.

    Nominal state: ``p``, ``v``, ``R`` (body-to-world), ``bias_accel``, ``bias_gyro``.
    """

    def __init__(
        self,
        *,
        P0: npt.NDArray[np.floating] | None = None,
        imu_noise: dict[str, float] | None = None,
    ) -> None:
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.bias_accel = np.zeros(3, dtype=np.float64)
        self.bias_gyro = np.zeros(3, dtype=np.float64)

        if P0 is None:
            self.P = np.diag(_DEFAULT_P0_DIAG).copy()
        else:
            self.P = np.asarray(P0, dtype=np.float64).reshape(15, 15).copy()

        self.imu_noise = dict(_DEFAULT_IMU_NOISE)
        if imu_noise:
            self.imu_noise.update(imu_noise)

        ast = float(self.imu_noise["accel_std"])
        gst = float(self.imu_noise["gyro_std"])
        ab = float(self.imu_noise["accel_bias_std"])
        gb = float(self.imu_noise["gyro_bias_std"])
        self.Q = np.asarray(
            np.diag([ast**2] * 3 + [gst**2] * 3 + [ab**2] * 3 + [gb**2] * 3),
            dtype=np.float64,
        )

        self._contact_mode = "none"

        self._p0 = self.p.copy()
        self._v0 = self.v.copy()
        self._R0 = self.R.copy()
        self._ba0 = self.bias_accel.copy()
        self._bg0 = self.bias_gyro.copy()
        self._P0 = self.P.copy()

    # -----------------------------------------------------------------------
    # Lifecycle / bookkeeping (not part of the EKF math)
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Restore nominal state and covariance to values captured at construction."""
        self.p = self._p0.copy()
        self.v = self._v0.copy()
        self.R = self._R0.copy()
        self.bias_accel = self._ba0.copy()
        self.bias_gyro = self._bg0.copy()
        self.P = self._P0.copy()

    def seed_nominal_state(
        self,
        *,
        p: npt.NDArray[np.floating],
        v: npt.NDArray[np.floating],
        R: npt.NDArray[np.floating],
        bias_accel: npt.NDArray[np.floating] | None = None,
        bias_gyro: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """
        Set nominal ``p``, ``v``, ``R`` (body-to-world) and optionally biases; refresh the
        baseline used by :meth:`reset`. Does not modify ``P``.
        """
        self.p = np.asarray(p, dtype=np.float64).reshape(3).copy()
        self.v = np.asarray(v, dtype=np.float64).reshape(3).copy()
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
        if bias_accel is not None:
            self.bias_accel = np.asarray(bias_accel, dtype=np.float64).reshape(3).copy()
        if bias_gyro is not None:
            self.bias_gyro = np.asarray(bias_gyro, dtype=np.float64).reshape(3).copy()
        self._p0 = self.p.copy()
        self._v0 = self.v.copy()
        self._R0 = self.R.copy()
        self._ba0 = self.bias_accel.copy()
        self._bg0 = self.bias_gyro.copy()

    def set_contact_pipeline(self, detector_id: str) -> None:
        """Record configured contact detector name (for logging until ZUPT is wired)."""
        self._contact_mode = str(detector_id).lower()

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def skew(v: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """
        Skew-symmetric matrix [v]_× such that [v]_× @ w = v × w.

        Used in error-state Jacobians for products involving body angular rate and
        specific force.
        """
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )

    # -----------------------------------------------------------------------
    # Predict: nominal INS + discrete error-state covariance
    # -----------------------------------------------------------------------

    def predict(
        self,
        accel_raw: npt.NDArray[np.floating],
        gyro_raw: npt.NDArray[np.floating],
        dt: float,
        *,
        accel_gravity_compensated: bool = False,
    ) -> None:
        """
        Time-update: integrate IMU into the **nominal** state and propagate ``P``.

        **Bias**: subtract estimated ``bias_gyro`` / ``bias_accel`` from raw gyro and accel.

        **Specific force vs linear accel**  
        - ``accel_gravity_compensated=False`` (default): ``accel_raw`` is **specific force**
          (gravity in the measurement). World kinematic acceleration is
          ``R @ f_body - g_world`` with fixed ``g_world = GRAVITY_WORLD_FLU`` (see module
          docstring).
        - ``accel_gravity_compensated=True``: ``accel_raw`` is already **linear**
          acceleration in the body frame; **do not** subtract ``g_world`` (gravity was
          removed in the dataset / IMU pipeline).

        **Covariance**: first-order discretization ``Phi ≈ I + F dt``, process noise
        ``Qd = G Q G^T dt`` with ``G`` mapping gyro/accel/bias-walk noise into the 15-D
        error state (same structure as legacy ``estimator.py``).
        """
        dt = float(dt)
        gyro_corrected = np.asarray(gyro_raw, dtype=np.float64).reshape(3) - self.bias_gyro
        accel_body_corrected = np.asarray(accel_raw, dtype=np.float64).reshape(3) - self.bias_accel

        # Specific force rotated to world; subtract fixed FLU gravity when meas includes g.
        accel_world = self.R @ accel_body_corrected
        if not accel_gravity_compensated:
            accel_world = accel_world - GRAVITY_WORLD_FLU

        self.p += self.v * dt + 0.5 * accel_world * dt**2
        self.v += accel_world * dt

        delta_R = Rotation.from_rotvec(gyro_corrected * dt).as_matrix()
        self.R = self.R @ np.asarray(delta_R, dtype=np.float64)

        # Error-state continuous-time F (same blocks as legacy).
        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -self.R @ self.skew(accel_body_corrected)
        F[3:6, 9:12] = -self.R
        F[6:9, 6:9] = -self.skew(gyro_corrected)
        F[6:9, 12:15] = -np.eye(3)

        discrete_STM = np.eye(15, dtype=np.float64) + F * dt

        G = np.zeros((15, 12), dtype=np.float64)
        G[3:6, 0:3] = -self.R
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        Qd = G @ self.Q @ G.T * dt
        self.P = discrete_STM @ self.P @ discrete_STM.T + Qd

    def imu_predict(
        self,
        dt_s: float,
        gyro_rad_s: npt.NDArray[np.floating],
        accel_m_s2: npt.NDArray[np.floating],
        *,
        accel_gravity_compensated: bool = False,
    ) -> None:
        """
        Single IMU step for the process loop: calls :meth:`predict` with the same
        ``accel_gravity_compensated`` flag as :attr:`LegOdometrySequence.meta` from
        :func:`leg_odom.io.imu_sanitize.sanitize_imu_dataframe`.
        """
        self.predict(accel_m_s2, gyro_rad_s, dt_s, accel_gravity_compensated=accel_gravity_compensated)

    def zupt_update_if_stance(
        self,
        *,
        leg_index: int,
        in_stance: bool,
        foot_jacobian_body: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Reserved for per-foot wiring; use :meth:`update_zupt` with a full stance list."""
        _ = (leg_index, in_stance, foot_jacobian_body)

    def foot_velocity_world(
        self,
        gyro_raw: npt.NDArray[np.floating],
        p_foot_body: npt.NDArray[np.floating],
        J: npt.NDArray[np.floating],
        qdot: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        """
        World-frame foot velocity from current nominal state (same kinematic chain as ZUPT).

        ``v_foot = v + R @ ( (ω - b_g) × p_foot_body + J q̇ )`` with measured gyro ``ω``.
        """
        w = np.asarray(gyro_raw, dtype=np.float64).reshape(3)
        pb = np.asarray(p_foot_body, dtype=np.float64).reshape(3)
        jac = np.asarray(J, dtype=np.float64)
        nj = jac.size // 3
        jac = jac.reshape(3, nj)
        qd = np.asarray(qdot, dtype=np.float64).reshape(nj)
        v_rel_body = np.cross(w - self.bias_gyro, pb) + jac @ qd
        return (self.v + self.R @ v_rel_body).astype(np.float64, copy=False)

    # -----------------------------------------------------------------------
    # Update: stacked ZUPT (zero foot velocity in world)
    # -----------------------------------------------------------------------

    def update_zupt(
        self,
        stance_legs: list[dict[str, Any]],
        gyro_raw: npt.NDArray[np.floating],
    ) -> dict[str, Any]:
        """
        Measurement update: each stance foot contributes ``v_foot_world ≈ 0``.

        **Predicted foot velocity** (world): from :meth:`foot_velocity_world` (same kinematics
        as the step-input ``v_foot_body`` definition, then rotated and shifted by nominal
        ``v``, ``R``). **Innovation**: ``0 - v_foot_world`` (measurement is zero velocity).

        **Gating**: per foot, Mahalanobis distance vs χ²(3) at 95%; only feet that pass
        are stacked for a single Joseph-form Kalman update.

        **Stance dict keys** (per leg): ``leg_id``, ``p_foot_body`` (3,), ``J`` (3, n_j),
        ``qdot`` (n_j,), ``R_foot`` (3, 3) measurement noise covariance; optional ``qscore``.

        **Returns** diagnostics: ``per_foot``, batch ``nis``, ``dof``, ``accepted`` (count
        of feet that passed gating and entered the joint update).
        """
        empty = {
            "per_foot": [],
            "accepted": 0,
            "nis": np.nan,
            "dof": 0,
            "nis_lo": np.nan,
            "nis_hi": np.nan,
        }
        if not stance_legs:
            return empty

        num_stance_legs = len(stance_legs)
        H = np.zeros((3 * num_stance_legs, 15), dtype=np.float64)
        measurement_cov_R = np.zeros((3 * num_stance_legs, 3 * num_stance_legs), dtype=np.float64)
        innov = np.zeros(3 * num_stance_legs, dtype=np.float64)
        per_foot_info: list[dict[str, Any]] = []

        gyro_raw = np.asarray(gyro_raw, dtype=np.float64).reshape(3)
        nis_threshold = st.chi2.ppf(0.95, df=3)

        for i, leg_data in enumerate(stance_legs):
            leg_id = leg_data["leg_id"]
            foot_position_body = np.asarray(leg_data["p_foot_body"], dtype=np.float64).reshape(3)
            jacobian = np.asarray(leg_data["J"], dtype=np.float64)
            qdot = np.asarray(leg_data["qdot"], dtype=np.float64)
            leg_covariance = np.asarray(leg_data["R_foot"], dtype=np.float64).reshape(3, 3)

            v_foot_pred = self.foot_velocity_world(
                gyro_raw, foot_position_body, jacobian, qdot
            )
            v_foot_rel_body = self.R.T @ (v_foot_pred - self.v)

            idx = slice(3 * i, 3 * (i + 1))
            innov[idx] = -v_foot_pred

            Hi = np.zeros((3, 15), dtype=np.float64)
            Hi[:, 3:6] = np.eye(3)
            Hi[:, 6:9] = -self.R @ self.skew(v_foot_rel_body)
            Hi[:, 12:15] = self.R @ self.skew(foot_position_body)
            H[idx, :] = Hi
            measurement_cov_R[np.ix_(range(3 * i, 3 * i + 3), range(3 * i, 3 * i + 3))] = (
                leg_covariance
            )
            per_foot_info.append(
                {
                    "leg_id": leg_id,
                    "qscore": float(leg_data.get("qscore", 1.0)),
                    "speed_world": float(np.linalg.norm(v_foot_pred)),
                    "v_pred_x": float(v_foot_pred[0]),
                    "v_pred_y": float(v_foot_pred[1]),
                    "v_pred_z": float(v_foot_pred[2]),
                }
            )

        accepted_legs_info: list[int] = []
        for i in range(num_stance_legs):
            idx = slice(3 * i, 3 * (i + 1))
            innov_i = innov[idx]
            Hi = H[idx, :]
            cov_i = measurement_cov_R[idx, idx]

            S_i = Hi @ self.P @ Hi.T + cov_i
            nis_i = np.nan
            try:
                cS_i, lower = cho_factor(S_i, lower=True, check_finite=False)
                nis_i = float(innov_i.T @ cho_solve((cS_i, lower), innov_i, check_finite=False))
                if nis_i < nis_threshold:
                    accepted_legs_info.append(i)
            except np.linalg.LinAlgError:
                S_reg = 0.5 * (S_i + S_i.T) + 1e-9 * np.eye(3, dtype=np.float64)
                nis_i = float(innov_i.T @ np.linalg.pinv(S_reg) @ innov_i)
                if nis_i < nis_threshold:
                    accepted_legs_info.append(i)

            per_foot_info[i]["mahal"] = float(nis_i)
            per_foot_info[i]["accepted"] = bool(nis_i < nis_threshold)

        if not accepted_legs_info:
            return {
                "per_foot": per_foot_info,
                "accepted": 0,
                "nis": np.nan,
                "dof": 0,
                "nis_lo": np.nan,
                "nis_hi": np.nan,
            }

        idx_arr = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in accepted_legs_info])
        H_acc = H[idx_arr, :]
        innov_acc = innov[idx_arr]
        R_acc = measurement_cov_R[np.ix_(idx_arr, idx_arr)]

        S = H_acc @ self.P @ H_acc.T + R_acc
        try:
            cS, lower = cho_factor(S, lower=True, check_finite=False)
            K = cho_solve((cS, lower), H_acc @ self.P).T
            nis = float(innov_acc.T @ cho_solve((cS, lower), innov_acc, check_finite=False))
        except (np.linalg.LinAlgError, ValueError):
            S_reg = 0.5 * (S + S.T) + 1e-9 * np.eye(S.shape[0], dtype=np.float64)
            S_inv = np.linalg.pinv(S_reg)
            K = self.P @ H_acc.T @ S_inv
            nis = float(innov_acc.T @ S_inv @ innov_acc)

        dx = K @ innov_acc
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ Rotation.from_rotvec(dx[6:9]).as_matrix()
        self.bias_accel += dx[9:12]
        self.bias_gyro += dx[12:15]

        I = np.eye(15, dtype=np.float64)
        self.P = (I - K @ H_acc) @ self.P @ (I - K @ H_acc).T + K @ R_acc @ K.T

        dof = int(len(idx_arr))
        nis_lo = float(st.chi2.ppf(0.05, dof))
        nis_hi = float(st.chi2.ppf(0.95, dof))

        return {
            "per_foot": per_foot_info,
            "accepted": len(accepted_legs_info),
            "nis": nis,
            "dof": dof,
            "nis_lo": nis_lo,
            "nis_hi": nis_hi,
        }


def _resolve_noise_config_path(raw: str, workspace_root: Path | None) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        if workspace_root is not None:
            p = (Path(workspace_root) / p).resolve()
        else:
            p = p.resolve()
    else:
        p = p.resolve()
    return p


def _apply_ekf_noise_mapping(
    block: Mapping[str, Any],
    noise: dict[str, float],
    p0: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Merge ``imu_noise`` / ``P0_diagonal`` from a mapping into current noise + P0."""
    imu_block = block.get("imu_noise")
    if isinstance(imu_block, Mapping):
        for k in noise:
            if k in imu_block:
                noise[k] = float(imu_block[k])
    p0d = block.get("P0_diagonal")
    if isinstance(p0d, (list, tuple)) and len(p0d) == 15:
        p0 = np.diag(np.asarray(p0d, dtype=np.float64))
    return noise, p0


def build_error_state_ekf(
    resolved_cfg: Mapping[str, Any] | None = None,
    *,
    workspace_root: Path | None = None,
) -> ErrorStateEkf:
    """
    Build an ESEKF from optional experiment ``ekf`` overrides.

    Resolution order:

    1. Code defaults (:data:`_DEFAULT_IMU_NOISE`, :data:`_DEFAULT_P0_DIAG`).
    2. If ``ekf.noise_config`` is set, YAML file at that path (``imu_noise``,
       ``P0_diagonal``) — relative paths resolved against ``workspace_root`` when given.
    3. Any inline keys on ``cfg["ekf"]`` (e.g. tests): ``imu_noise``, ``P0_diagonal``
       (override file / defaults).
    """
    block: Mapping[str, Any] = {}
    if resolved_cfg is not None:
        raw = resolved_cfg.get("ekf")
        if isinstance(raw, Mapping):
            block = raw

    noise = dict(_DEFAULT_IMU_NOISE)
    p0 = np.diag(_DEFAULT_P0_DIAG)

    nc = block.get("noise_config")
    if nc is not None and str(nc).strip():
        path = _resolve_noise_config_path(str(nc), workspace_root)
        if not path.is_file():
            raise FileNotFoundError(f"ekf.noise_config: not a file: {path}")
        file_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(file_data, Mapping):
            noise, p0 = _apply_ekf_noise_mapping(file_data, noise, p0)

    noise, p0 = _apply_ekf_noise_mapping(block, noise, p0)

    return ErrorStateEkf(P0=p0, imu_noise=noise)
