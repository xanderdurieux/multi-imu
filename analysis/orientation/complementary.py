"""Quaternion-based complementary orientation filter.

Algorithm
---------
Each time step:

1. **Gyro propagation** — integrate angular velocity in quaternion space:
   ``q_gyro = q_prev ⊗ Δq(ω, dt)``.

2. **Accelerometer correction** — when the accelerometer norm is within the
   gating tolerance:

   a. Compute the expected specific-force direction in the body frame from
      the gyro-propagated quaternion: ``g_exp = R_bw^T [0, 0, 1]``.
   b. Compute the measured specific-force direction: ``g_meas = acc / ‖acc‖``.
   c. Find the smallest rotation from ``g_exp`` to ``g_meas`` and apply a
      fraction ``accel_correction_gain`` of it to ``q_gyro``.

This approach works for **any initial orientation**, including sensors
mounted upside-down or at large roll/pitch angles, because the correction is
always relative to the current estimate rather than absolute.

World-frame convention: ENU (+Z = up).  The specific force at rest is
approximately ``[0, 0, +9.81]`` in the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .quaternion import (
    quat_from_axis_angle,
    quat_from_gyro,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_conjugate,
    tilt_quat_from_acc,
)


@dataclass
class ComplementaryFilterConfig:
    """Configuration parameters for the complementary orientation filter."""

    # Fraction of the gravity-alignment error corrected per step.
    # Range (0, 1): 0 → gyro only; 1 → jump fully to accelerometer direction.
    accel_correction_gain: float = 0.02

    # Approximate gravity magnitude used for gating (m/s²).
    gravity: float = 9.81

    # Allowable deviation of ‖acc‖ from g, expressed in units of g.
    # 0.2 → accept if ‖acc‖ ∈ [0.8 g, 1.2 g].
    accel_norm_tolerance_g: float = 0.2

    use_accel_gating: bool = True


class ComplementaryOrientationFilter:
    """Complementary filter combining gyro integration and accelerometer correction.

    Unlike a simple Euler-angle complementary filter, this implementation
    works entirely in quaternion space and therefore handles sensors mounted
    at arbitrary angles without gimbal-lock or incorrect absolute-tilt issues.
    """

    def __init__(self, config: Optional[ComplementaryFilterConfig] = None):
        self.config = config or ComplementaryFilterConfig()
        self.q_bw = quat_identity()
        self._initialized = False

    def reset(self, initial_quaternion: Optional[np.ndarray] = None) -> None:
        """Reset the filter to *initial_quaternion* or identity.

        When *initial_quaternion* is provided (e.g. from ``calibration.json``),
        the filter is considered already initialized so the first
        accelerometer-based tilt estimate does not overwrite the calibrated
        starting orientation.
        """
        if initial_quaternion is None:
            self.q_bw = quat_identity()
            self._initialized = False
        else:
            self.q_bw = quat_normalize(np.asarray(initial_quaternion, dtype=float))
            self._initialized = True

    def _accel_is_trustworthy(self, acc_body: np.ndarray) -> bool:
        if not self.config.use_accel_gating:
            return True
        norm = float(np.linalg.norm(acc_body))
        if norm == 0.0:
            return False
        ratio = norm / self.config.gravity
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
            Angular velocity in rad/s in the body/sensor frame.
        acc_body_ms2:
            Optional accelerometer reading in m/s².  When ``None`` or outside
            the gating tolerance, only gyro integration is performed.

        Returns
        -------
        np.ndarray
            Updated body→world unit quaternion ``[w, x, y, z]``.
        """
        omega = np.asarray(gyro_body_rad, dtype=float).reshape(3)

        # 1. Gyro propagation in quaternion space.
        dq = quat_from_gyro(omega, dt)
        q_gyro = quat_normalize(quat_multiply(self.q_bw, dq))

        if acc_body_ms2 is not None:
            acc = np.asarray(acc_body_ms2, dtype=float).reshape(3)

            if not self._initialized:
                # First valid acc sample: bootstrap tilt from accelerometer.
                self.q_bw = tilt_quat_from_acc(acc, g=self.config.gravity)
                self._initialized = True
                return self.q_bw

            if self._accel_is_trustworthy(acc):
                # 2. Gravity-alignment correction.
                #
                # Expected specific-force direction in body frame:
                #   g_exp = R_bw^T @ [0, 0, 1]
                # where R_bw is the body→world rotation implied by q_gyro.
                # Rotating world +Z (up) to body frame gives the direction the
                # accelerometer should point when acc ≈ [0, 0, g] in world.
                g_exp = quat_rotate(quat_conjugate(q_gyro), np.array([0.0, 0.0, 1.0]))
                g_exp_norm = np.linalg.norm(g_exp)
                if g_exp_norm < 1e-10:
                    self.q_bw = q_gyro
                    return self.q_bw
                g_exp = g_exp / g_exp_norm

                acc_norm = float(np.linalg.norm(acc))
                g_meas = acc / acc_norm

                # Angle between expected and measured directions.
                cos_a = float(np.clip(np.dot(g_exp, g_meas), -1.0, 1.0))
                angle = float(np.arccos(cos_a))

                if angle > 1e-8:
                    axis = np.cross(g_exp, g_meas)
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm > 1e-8:
                        axis = axis / axis_norm
                        # Apply a fraction of the total error.
                        corr_q = quat_from_axis_angle(
                            axis, angle * self.config.accel_correction_gain
                        )
                        # Correction is in the world frame (pre-multiply).
                        self.q_bw = quat_normalize(quat_multiply(corr_q, q_gyro))
                        return self.q_bw

        self.q_bw = q_gyro
        if not self._initialized:
            self._initialized = True
        return self.q_bw
