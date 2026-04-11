"""
Abstract contact detector and output type.

Contract: :class:`ContactDetectorStepInput` carries per-foot GRF and kinematics for every
detector; implementations read only the fields they need. Jacobians stay in the EKF /
kinematics path only, not in the step input.

ZUPT measurement covariance :math:`R` is formed from :attr:`ContactEstimate.p_stance` in the
EKF process loop (:mod:`leg_odom.filters.zupt_measurement`), not in detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ContactDetectorStepInput:
    """
    Standardized per-foot, per-timestep input for :meth:`BaseContactDetector.update`.

    **Fields**

    - ``grf_n``: Scalar load proxy for this foot (N); use ``0.0`` when the log has no sample.
    - ``p_foot_body``: Foot origin position in **body** frame, shape ``(3,)``.
    - ``v_foot_body``: Foot linear velocity **relative to the body**, expressed in the body
      frame, shape ``(3,)``. Same kinematics as inside ZUPT:
      ``(ω_meas - b_gyro) × p_foot_body + J q̇`` with **measured** gyro ``ω_meas`` and the filter's
      current nominal ``b_gyro`` (computed in the process loop after ``imu_predict``).
    - ``q_leg``, ``dq_leg``: Joint positions (rad) and rates (rad/s), shape ``(n_j,)`` each.
    - ``tau_leg``: Estimated joint torques (Nm), same slice as ``q_leg`` / ``dq_leg`` (e.g.
      ``motor_*_tau_est`` columns parallel to ``motor_*_q`` / ``motor_*_dq``).
    - ``gyro_body_corrected``: ``ω_meas - b_gyro`` (rad/s), shape ``(3,)``. Duplicate per foot
      if convenient; same value for all feet on one timestep.
    - ``accel_body_corrected``: ``f_meas - b_accel`` (m/s²). **IMU convention**
      must match :meth:`~leg_odom.filters.esekf.ErrorStateEkf.predict`: if the log is **specific
      force** (default), corrected accel still contains gravity in the body frame; if the
      recording is gravity-compensated (``accel_gravity_compensated`` in sequence meta), gravity
      is already absent in ``f_meas`` and thus in this field. The pipeline fills this using the
      same flag as the EKF predict step.

    **Not included:** body-frame Jacobian ``J`` (EKF / FK only).

    **Consumers (non-exhaustive)**

    - :class:`~leg_odom.contact.grf_threshold.GrfThresholdContactDetector`: ``grf_n`` only
      (other fields reserved for richer detectors).
    """

    grf_n: float
    p_foot_body: npt.NDArray[np.float64]
    v_foot_body: npt.NDArray[np.float64]
    q_leg: npt.NDArray[np.float64]
    dq_leg: npt.NDArray[np.float64]
    tau_leg: npt.NDArray[np.float64]
    gyro_body_corrected: npt.NDArray[np.float64]
    accel_body_corrected: npt.NDArray[np.float64]


class ContactEstimate(NamedTuple):
    """Stance belief from a contact detector; ZUPT :math:`R` is derived in the EKF loop from ``p_stance``."""

    stance: bool
    p_stance: float


class BaseContactDetector(ABC):
    """Shared interface for all stance estimators (GMM+HMM, CNN/GRU, Ocelot, …)."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Feature vector length D (legacy windowed detectors; scalar pipelines may use 1)."""

    @property
    @abstractmethod
    def history_length(self) -> int:
        """Window size N (including current row at index -1 when N > 1)."""

    @abstractmethod
    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        """Consume latest step; return stance flag and stance probability."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state at sequence boundaries."""
