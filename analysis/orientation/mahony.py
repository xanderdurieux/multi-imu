"""Mahony helpers for estimate sensor orientation for calibrated section data."""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_identity,
    quat_normalize,
    tilt_quat_from_acc,
)


class MahonyFilter:
    """Stateful Mahony orientation filter."""

    def __init__(self, Kp: float = 2.0, Ki: float = 0.005, sample_rate_hz: float = 100.0) -> None:
        """Return init."""
        self.Kp = Kp
        self.Ki = Ki
        self._dt = 1.0 / sample_rate_hz
        self._q: np.ndarray = quat_identity()
        self._integral: np.ndarray = np.zeros(3)

    def reset(self, q: np.ndarray | None = None) -> None:
        """Return reset."""
        self._q = quat_normalize(q) if q is not None else quat_identity()
        self._integral = np.zeros(3)

    def initialize_from_acc(self, acc: np.ndarray) -> None:
        """Return initialize from acc."""
        self._q = tilt_quat_from_acc(acc)
        self._integral = np.zeros(3)

    def current_quaternion(self) -> np.ndarray:
        """Return current quaternion."""
        return self._q.copy()

    def update(
        self,
        acc: np.ndarray,
        gyro_rad: np.ndarray,
        mag: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Return update."""
        if dt is None:
            dt = self._dt

        acc = np.asarray(acc, dtype=float)
        gyro_rad = np.asarray(gyro_rad, dtype=float)
        w, x, y, z = self._q

        error = np.zeros(3)

        # --- Accelerometer correction -----------------------------------------
        an = np.linalg.norm(acc)
        if an > 1e-6:
            a = acc / an
            # Estimated gravity direction in body frame from current quaternion:
            # v = R(q)ᵀ · [0, 0, 1]
            v = np.array([
                2.0 * (x * z - w * y),
                2.0 * (w * x + y * z),
                w * w - x * x - y * y + z * z,
            ])
            error += np.cross(a, v)

        # --- Magnetometer correction (MARG) ------------------------------------
        if mag is not None:
            mag = np.asarray(mag, dtype=float)
            mn = np.linalg.norm(mag)
            if mn > 1e-6:
                m = mag / mn
                # Rotate m to world frame, project onto XZ plane to get reference b.
                hx = 2.0 * (m[0] * (0.5 - y*y - z*z) + m[1] * (x*y - w*z) + m[2] * (x*z + w*y))
                hy = 2.0 * (m[0] * (x*y + w*z) + m[1] * (0.5 - x*x - z*z) + m[2] * (y*z - w*x))
                hz = 2.0 * (m[0] * (x*z - w*y) + m[1] * (y*z + w*x) + m[2] * (0.5 - x*x - y*y))
                bx = float(np.sqrt(hx * hx + hy * hy))
                bz = float(hz)
                # Reference direction in body frame: R(q)ᵀ · [bx, 0, bz]
                w_ref = np.array([
                    2.0 * (bx * (0.5 - y*y - z*z) + bz * (x*z - w*y)),
                    2.0 * (bx * (x*y - w*z) + bz * (w*x + y*z)),
                    2.0 * (bx * (x*z + w*y) + bz * (0.5 - x*x - y*y)),
                ])
                error += np.cross(m, w_ref)

        # --- PI gyroscope correction -------------------------------------------
        self._integral += self.Ki * error * dt
        g_corr = gyro_rad + self.Kp * error + self._integral
        gx, gy, gz = g_corr

        # --- Quaternion integration --------------------------------------------
        w, x, y, z = self._q
        q_dot = 0.5 * np.array([
            -x * gx - y * gy - z * gz,
             w * gx + y * gz - z * gy,
             w * gy - x * gz + z * gx,
             w * gz + x * gy - y * gx,
        ])
        self._q = quat_normalize(self._q + q_dot * dt)
        return self._q.copy()
