"""
Implementation thresholds: numeric guards, heuristics, and clamps inside ``leg_odom``.

These are **not** user-facing experiment hyperparameters (noise matrices, detector flags,
run paths). Those stay in ``parameters.py`` / future YAML. Values here define when
the **code** treats data as valid, converts units, or clamps derived quantities.

Each constant lists **where it is read** (module + function).

--------------------------------------------------------------------
IMU — :mod:`leg_odom.io.imu_sanitize`
--------------------------------------------------------------------
"""

# --- infer_accel_gravity_compensated, sanitize_imu_dataframe (gravity-compensated path) ---
IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD = 3.0
"""
If the time-averaged accelerometer magnitude is below this (m/s²), the signal is
treated as **linear acceleration** (gravity removed from the accel channel).

**Used by:** :func:`leg_odom.io.imu_sanitize.infer_accel_gravity_compensated`,
:func:`leg_odom.io.imu_sanitize.sanitize_imu_dataframe` (via ``infer_*``),
:func:`leg_odom.io.split_imu_bag.load_prepared_split_sequence` when ``sanitize_imu=False``.
"""

# --- sanitize_imu_dataframe (gyro deg/s auto-detect) ---
IMU_GYRO_MEDIAN_NORM_DEG_S_HINT = 10.0
"""
Median gyro vector norm above this suggests **deg/s**; below suggests **rad/s**.

**Used by:** :func:`leg_odom.io.imu_sanitize.sanitize_imu_dataframe`.
"""

# --- _assert_flu_specific_force (specific-force / FLU path) ---
IMU_FLU_SPECIFIC_FORCE_MAG_MIN = 7.0
IMU_FLU_SPECIFIC_FORCE_MAG_MAX = 12.0
"""
Bounds on **mean** accelerometer magnitude (m/s²) for “looks like specific force at rest”
(~9.81 m/s² on +Z when level in FLU).

**Used by:** ``leg_odom.io.imu_sanitize._assert_flu_specific_force``.
"""

IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG = 35.0
"""
Maximum angle (degrees) between **mean** accel vector and body **+Z** for the default
specific-force FLU check.

**Used by:** ``leg_odom.io.imu_sanitize._assert_flu_specific_force``.
"""

# --- _angle_to_body_plus_z_deg ---
IMU_VECTOR_NEAR_ZERO_NORM = 1e-9
"""
Norm below this ⇒ direction of mean accel is undefined (avoid divide-by-zero).

**Used by:** ``leg_odom.io.imu_sanitize._angle_to_body_plus_z_deg``.
"""

# --------------------------------------------------------------------
# Timebase — :mod:`leg_odom.io.timebase`
# --------------------------------------------------------------------

TIMEBASE_RATE_FALLBACK_HZ = 400.0
"""
Fallback sample rate (Hz) when ``dt`` has too few positive samples to estimate median.

**Used by:** :func:`leg_odom.io.timebase.estimate_median_sample_rate_hz`,
``leg_odom.io.timebase.build_timebase`` (implicit via median ``dt`` fallback).
"""

TIMEBASE_MIN_POSITIVE_DT_SAMPLES = 10
"""
Minimum count of positive ``dt`` values required to trust median-based rate estimate.

**Used by:** :func:`leg_odom.io.timebase.estimate_median_sample_rate_hz`.
"""

TIMEBASE_TIMESTAMP_NS_SCALE_THRESHOLD = 1e12
"""
If max raw time value exceeds this, values are treated as **nanoseconds** and scaled by ``1e-9``.

**Used by:** :func:`leg_odom.io.timebase.build_timebase`.
"""

TIMEBASE_DT_CLIP_MIN_S = 1e-4
TIMEBASE_DT_CLIP_MAX_S = 0.2
"""
Clamp reconstructed per-step ``dt`` to this interval (seconds) after filling NaNs.

**Used by:** :func:`leg_odom.io.timebase.build_timebase`.
"""

# --------------------------------------------------------------------
# Kinematics — :mod:`leg_odom.kinematics.base`, :mod:`leg_odom.kinematics.go2`
# --------------------------------------------------------------------

KINEMATICS_NUMERICAL_JACOBIAN_STEP = 1e-6
"""
Forward-difference step (rad) for ∂(foot position)/∂q when no closed-form Jacobian exists.

**Used by:** :meth:`leg_odom.kinematics.base.BaseKinematics.jacobian_numerical`,
:meth:`leg_odom.kinematics.go2.Go2Kinematics.J_analytical` (delegates to numerical).
"""
