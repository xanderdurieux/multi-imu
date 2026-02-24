"""Complementary orientation filter (AHRS-based tilt backend).

This filter keeps the same external behaviour as before (gyro propagation plus
gravity-based drift correction with gating), but uses the `ahrs` Python package
to compute the accelerometer-based tilt estimate.

Note: The `ahrs.filters.Complementary` implementation is batch-oriented (no
per-sample update), so we use its `am_estimation()` helper and apply the
complementary fusion step-by-step to support variable `dt` and gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ahrs.filters import Complementary as _AhrsComplementary

from .quaternion import euler_from_quat, quat_from_euler, quat_identity, quat_normalize


@dataclass
class ComplementaryFilterConfig:
    """Configuration parameters for the complementary orientation filter."""

    # Gyro vs. accelerometer trust. Roughly, beta is the fraction of the tilt
    # error we correct per sample (0 → no correction, 1 → jump fully to accel tilt).
    accel_correction_gain: float = 0.02

    # Approximate gravity magnitude used for gating (m/s^2).
    gravity: float = 9.81

    # Allowable deviation of |a| from g, expressed in units of g.
    # Example: 0.2 → accept accelerometer if |a| ∈ [0.8 g, 1.2 g].
    accel_norm_tolerance_g: float = 0.2

    # Optional maximum accelerometer variance over a small window (not used here,
    # but could be added for additional gating).
    use_accel_gating: bool = True


class ComplementaryOrientationFilter:
    """Complementary filter combining gyro integration and accelerometer tilt."""

    def __init__(self, config: Optional[ComplementaryFilterConfig] = None):
        self.config = config or ComplementaryFilterConfig()
        self.q_bw = quat_identity()
        # Internal roll/pitch/yaw state (radians), in roll-pitch-yaw order.
        self._rpy = np.zeros(3, dtype=float)
        self._tilt = _AhrsComplementary(frequency=100.0, gain=0.9)
        self._initialized = False

    def reset(self, initial_quaternion: Optional[np.ndarray] = None) -> None:
        """Reset the filter to ``initial_quaternion`` or identity."""
        if initial_quaternion is None:
            self.q_bw = quat_identity()
            self._rpy[:] = 0.0
        else:
            q = quat_normalize(np.asarray(initial_quaternion, dtype=float))
            self.q_bw = q
            yaw, pitch, roll = euler_from_quat(q)
            self._rpy[:] = [roll, pitch, yaw]
        self._initialized = False

    def _accel_is_trustworthy(self, acc_body: np.ndarray) -> bool:
        if not self.config.use_accel_gating:
            return True
        g = self.config.gravity
        norm = float(np.linalg.norm(acc_body))
        if norm == 0.0:
            return False
        ratio = norm / g
        tol = self.config.accel_norm_tolerance_g
        return (1.0 - tol) <= ratio <= (1.0 + tol)

    def step(
        self,
        dt: float,
        gyro_body_rad: Iterable[float],
        acc_body_ms2: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Advance the filter by one time step.

        Parameters
        ----------
        dt:
            Time step in seconds.
        gyro_body_rad:
            Angular velocity vector in rad/s in sensor/body frame.
        acc_body_ms2:
            Optional accelerometer vector in m/s^2.
        """
        omega = np.asarray(gyro_body_rad, dtype=float).reshape(3)

        if omega.shape != (3,):
            raise ValueError("gyro_body_rad must be a 3-vector.")

        # 1) Initialize roll/pitch from accelerometer tilt if possible.
        if not self._initialized and acc_body_ms2 is not None:
            acc = np.asarray(acc_body_ms2, dtype=float).reshape(3)
            if self._accel_is_trustworthy(acc):
                roll, pitch, yaw0 = self._tilt.am_estimation(acc)  # yaw0 == 0 for IMU-only
                self._rpy[:] = [float(roll), float(pitch), float(yaw0)]
                self._initialized = True

        # 2) Predict angles using gyro integration (simple Euler approximation).
        roll_pred = float(self._rpy[0] + omega[0] * dt)
        pitch_pred = float(self._rpy[1] + omega[1] * dt)
        yaw_pred = float(self._rpy[2] + omega[2] * dt)

        # 3) Low-frequency correction from accelerometer tilt.
        if acc_body_ms2 is not None:
            acc = np.asarray(acc_body_ms2, dtype=float).reshape(3)
            if self._accel_is_trustworthy(acc):
                roll_meas, pitch_meas, _ = self._tilt.am_estimation(acc)
                alpha = float(1.0 - self.config.accel_correction_gain)
                roll = alpha * roll_pred + (1.0 - alpha) * float(roll_meas)
                pitch = alpha * pitch_pred + (1.0 - alpha) * float(pitch_meas)
                yaw = yaw_pred  # yaw not observable without magnetometer
                self._rpy[:] = [roll, pitch, yaw]
                self._initialized = True
            else:
                # High dynamics: do not correct from accelerometer.
                self._rpy[:] = [roll_pred, pitch_pred, yaw_pred]
        else:
            self._rpy[:] = [roll_pred, pitch_pred, yaw_pred]

        # 4) Convert roll/pitch/yaw -> body→world quaternion (yaw, pitch, roll order).
        self.q_bw = quat_from_euler(yaw=self._rpy[2], pitch=self._rpy[1], roll=self._rpy[0])
        return self.q_bw

