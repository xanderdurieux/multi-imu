"""Madgwick AHRS filter for IMU orientation estimation.

Reference
---------
Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).
Estimation of IMU and MARG orientation using a gradient descent algorithm.
IEEE International Conference on Rehabilitation Robotics.
"""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_identity,
    quat_normalize,
    quat_multiply,
    tilt_quat_from_acc,
)


class MadgwickFilter:
    """Madgwick AHRS orientation filter.

    Integrates gyroscope measurements and corrects drift using accelerometer
    (and optionally magnetometer) readings via gradient descent.

    Parameters
    ----------
    beta:
        Filter gain. Higher values weight the accelerometer correction more.
        0.1 is a good default for cycling data.
    sample_rate_hz:
        Expected sampling rate in Hz (used to compute dt when not supplied).
    """

    def __init__(self, beta: float = 0.1, sample_rate_hz: float = 100.0) -> None:
        self.beta = beta
        self.sample_rate_hz = sample_rate_hz
        self._dt = 1.0 / sample_rate_hz
        self._q: np.ndarray = quat_identity()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, q: np.ndarray | None = None) -> None:
        """Reset internal quaternion state."""
        self._q = quat_normalize(q) if q is not None else quat_identity()

    def initialize_from_acc(self, acc: np.ndarray) -> None:
        """Set initial orientation from the first accelerometer sample."""
        self._q = tilt_quat_from_acc(acc)

    def current_quaternion(self) -> np.ndarray:
        """Copy of the current orientation quaternion [w, x, y, z]."""
        return self._q.copy()

    def update(
        self,
        acc: np.ndarray,
        gyro_rad: np.ndarray,
        mag: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Run one filter step and return the updated quaternion [w, x, y, z].

        Parameters
        ----------
        acc:
            Accelerometer reading [ax, ay, az] in any consistent units (will be
            normalized internally).
        gyro_rad:
            Gyroscope reading [gx, gy, gz] in rad/s.
        mag:
            Optional magnetometer reading [mx, my, mz].  When provided the
            full MARG update is used; otherwise accelerometer-only correction.
        dt:
            Time step in seconds.  Defaults to ``1 / sample_rate_hz``.

        Returns
        -------
        np.ndarray
            Updated unit quaternion [w, x, y, z].
        """
        if dt is None:
            dt = self._dt

        q = self._q
        w, x, y, z = q

        acc = np.asarray(acc, dtype=float)
        gyro_rad = np.asarray(gyro_rad, dtype=float)

        # -- Gyroscope rate derivative -----------------------------------------
        # q_dot_gyro = 0.5 * q ⊗ [0, gx, gy, gz]
        gx, gy, gz = gyro_rad
        q_dot = 0.5 * np.array([
            -x * gx - y * gy - z * gz,
             w * gx + y * gz - z * gy,
             w * gy - x * gz + z * gx,
             w * gz + x * gy - y * gx,
        ])

        # -- Gradient descent correction ----------------------------------------
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-6:
            a = acc / acc_norm  # normalized accelerometer

            if mag is not None:
                mag = np.asarray(mag, dtype=float)
                mag_norm = np.linalg.norm(mag)
                if mag_norm > 1e-6:
                    q_dot -= self.beta * self._gradient_marg(q, a, mag / mag_norm)
                else:
                    q_dot -= self.beta * self._gradient_imu(q, a)
            else:
                q_dot -= self.beta * self._gradient_imu(q, a)

        # -- Integrate ------------------------------------------------------------
        q_new = q + q_dot * dt
        self._q = quat_normalize(q_new)
        return self._q.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_imu(q: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Gradient of the objective function f_g w.r.t. q.

        f_g = q* ⊗ [0,0,0,1] ⊗ q - [0, ax, ay, az]
        (expected gravity direction in body frame vs measured acc)
        """
        w, x, y, z = q
        ax, ay, az = a

        # Objective function f_g = q* ⊗ [0,0,1] ⊗ q − a  (world gravity [0,0,1])
        # Analytical gradient of ||f_g||^2 w.r.t. q.
        f1 = 2.0 * (x * z - w * y) - ax
        f2 = 2.0 * (w * x + y * z) - ay
        f3 = 2.0 * (0.5 - x * x - y * y) - az

        # Jacobian J^T (4x3 transposed) times f (3x1).
        gw = 2.0 * (-y * f1 + x * f2)
        gx = 2.0 * ( z * f1 + w * f2 - 2.0 * x * f3)
        gy = 2.0 * (-w * f1 + z * f2 - 2.0 * y * f3)
        gz = 2.0 * ( x * f1 + y * f2)

        grad = np.array([gw, gx, gy, gz])
        n = np.linalg.norm(grad)
        if n > 1e-12:
            grad /= n
        return grad

    @staticmethod
    def _gradient_marg(q: np.ndarray, a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Gradient for MARG (acc + mag) correction."""
        w, x, y, z = q
        ax, ay, az = a
        mx, my, mz = m

        # Rotate magnetometer into world frame using current q.
        # h = q ⊗ [0, mx, my, mz] ⊗ q*
        # Only keep x and z components to get b (reference direction in world XZ plane).
        hx = 2.0 * (mx * (0.5 - y * y - z * z) + my * (x * y - w * z) + mz * (x * z + w * y))
        hy = 2.0 * (mx * (x * y + w * z) + my * (0.5 - x * x - z * z) + mz * (y * z - w * x))
        hz = 2.0 * (mx * (x * z - w * y) + my * (y * z + w * x) + mz * (0.5 - x * x - y * y))
        bx = np.sqrt(hx * hx + hy * hy)
        bz = hz

        # Gravity objective (same as IMU case).
        f1 = 2.0 * (x * z - w * y) - ax
        f2 = 2.0 * (w * x + y * z) - ay
        f3 = 2.0 * (0.5 - x * x - y * y) - az

        # Magnetometer objective.
        f4 = 2.0 * bx * (0.5 - y * y - z * z) + 2.0 * bz * (x * z - w * y) - mx
        f5 = 2.0 * bx * (x * y - w * z) + 2.0 * bz * (w * x + y * z) - my
        f6 = 2.0 * bx * (w * y + x * z) + 2.0 * bz * (0.5 - x * x - y * y) - mz

        # Jacobian^T * f for gravity part.
        gw = -2.0 * y * f1 + 2.0 * x * f2
        gx =  2.0 * z * f1 + 2.0 * w * f2 - 4.0 * x * f3
        gy = -2.0 * w * f1 + 2.0 * z * f2 - 4.0 * y * f3
        gz =  2.0 * x * f1 + 2.0 * y * f2

        # Jacobian^T * f for magnetometer part.
        gw += -2.0 * bz * y * f4 + (-2.0 * bx * z + 2.0 * bz * x) * f5 + 2.0 * bx * y * f6
        gx +=  2.0 * bz * z * f4 + ( 2.0 * bx * y + 2.0 * bz * w) * f5 + (2.0 * bx * z - 2.0 * bz * w) * f6
        gy += (-4.0 * bx * y - 2.0 * bz * w) * f4 + (2.0 * bx * x + 2.0 * bz * z) * f5 + (2.0 * bx * w - 4.0 * bz * y) * f6
        gz += (-4.0 * bx * z + 2.0 * bz * x) * f4 + (-2.0 * bx * w + 2.0 * bz * y) * f5 + 2.0 * bx * x * f6

        grad = np.array([gw, gx, gy, gz])
        n = np.linalg.norm(grad)
        if n > 1e-12:
            grad /= n
        return grad
