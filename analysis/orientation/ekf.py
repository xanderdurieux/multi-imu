"""Extended Kalman Filter for IMU orientation estimation.

The state is the unit quaternion q = [w, x, y, z] representing the rotation
from sensor/body frame to world frame (same convention as the rest of this
package).

Process model
-------------
Quaternion kinematics driven by the gyroscope:

    q_{k+1} ≈ (I + dt/2 · Ω) · q_k

where Ω is the 4×4 skew-symmetric gyroscope rate matrix.

Measurement model
-----------------
IMU mode   : normalised accelerometer reading predicted from gravity in body frame.
MARG mode  : same as IMU, plus normalised magnetometer reading predicted from a
             reference field computed in world frame (XZ-plane projection, same
             approach as the Madgwick MARG filter).
"""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_identity,
    quat_normalize,
    tilt_quat_from_acc,
)


class EKFFilter:
    """Quaternion-based EKF orientation filter.

    Parameters
    ----------
    sigma_gyro:
        Standard deviation of gyroscope noise (rad/s).  Controls how fast the
        covariance grows during the prediction step.
    sigma_acc:
        Standard deviation of accelerometer measurement noise (normalised
        units).  Smaller values trust the accelerometer more.
    sigma_mag:
        Standard deviation of magnetometer measurement noise (normalised
        units).  Only used in MARG mode.
    sample_rate_hz:
        Expected sampling rate in Hz (used as default dt).
    """

    def __init__(
        self,
        sigma_gyro: float = 0.01,
        sigma_acc: float = 0.1,
        sigma_mag: float = 0.1,
        sample_rate_hz: float = 100.0,
    ) -> None:
        self.sigma_gyro = sigma_gyro
        self.sigma_acc = sigma_acc
        self.sigma_mag = sigma_mag
        self.sample_rate_hz = sample_rate_hz
        self._dt = 1.0 / sample_rate_hz
        self._q: np.ndarray = quat_identity()
        self._P: np.ndarray = np.eye(4) * 1e-3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, q: np.ndarray | None = None) -> None:
        """Reset quaternion state and covariance."""
        self._q = quat_normalize(q) if q is not None else quat_identity()
        self._P = np.eye(4) * 1e-3

    def initialize_from_acc(self, acc: np.ndarray) -> None:
        """Set initial orientation from the first accelerometer sample."""
        self._q = tilt_quat_from_acc(acc)
        self._P = np.eye(4) * 1e-3

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
        """Run one EKF step and return the updated quaternion [w, x, y, z].

        Parameters
        ----------
        acc:
            Accelerometer reading [ax, ay, az] in any consistent units
            (normalised internally).
        gyro_rad:
            Gyroscope reading [gx, gy, gz] in rad/s.
        mag:
            Optional magnetometer reading [mx, my, mz].  When provided the
            full MARG measurement update is performed.
        dt:
            Time step in seconds.  Defaults to ``1 / sample_rate_hz``.
        """
        if dt is None:
            dt = self._dt

        gyro_rad = np.asarray(gyro_rad, dtype=float)
        acc = np.asarray(acc, dtype=float)

        # -- Predict ----------------------------------------------------------
        q, P = self._predict(gyro_rad, dt)

        # -- Update: accelerometer --------------------------------------------
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-6:
            q, P = self._update_acc(q, P, acc / acc_norm)

        # -- Update: magnetometer (MARG) --------------------------------------
        if mag is not None:
            mag = np.asarray(mag, dtype=float)
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 1e-6:
                q, P = self._update_mag(q, P, mag / mag_norm)

        self._q = q
        self._P = P
        return self._q.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(
        self, gyro_rad: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """EKF prediction step using gyroscope kinematics."""
        gx, gy, gz = gyro_rad
        # 4×4 quaternion rate matrix
        Omega = np.array(
            [
                [0.0, -gx, -gy, -gz],
                [gx,  0.0,  gz, -gy],
                [gy, -gz,  0.0,  gx],
                [gz,  gy,  -gx, 0.0],
            ]
        )
        F = np.eye(4) + 0.5 * dt * Omega

        q_pred = quat_normalize(F @ self._q)

        Q = (self.sigma_gyro * dt) ** 2 * np.eye(4)
        P_pred = F @ self._P @ F.T + Q

        return q_pred, P_pred

    def _update_acc(
        self, q: np.ndarray, P: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """EKF measurement update for the normalised accelerometer."""
        w, x, y, z = q

        # Predicted gravity direction in body frame: R(q)^T · [0, 0, 1]
        h = np.array(
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ]
        )

        # Jacobian of h w.r.t. q (3×4)
        H = 2.0 * np.array(
            [
                [-y,  z, -w,  x],
                [ x,  w,  z,  y],
                [ 0, -2*x, -2*y, 0],
            ]
        )

        R_noise = self.sigma_acc ** 2 * np.eye(3)
        return self._kalman_update(q, P, a - h, H, R_noise)

    def _update_mag(
        self, q: np.ndarray, P: np.ndarray, m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """EKF measurement update for the normalised magnetometer."""
        w, x, y, z = q

        # Rotate magnetometer into world frame and project onto XZ plane
        # (same approach as Madgwick MARG to obtain the field reference).
        mx, my, mz = m
        hx = 2.0 * (mx * (0.5 - y*y - z*z) + my * (x*y - w*z) + mz * (x*z + w*y))
        hy = 2.0 * (mx * (x*y + w*z) + my * (0.5 - x*x - z*z) + mz * (y*z - w*x))
        hz = 2.0 * (mx * (x*z - w*y) + my * (y*z + w*x) + mz * (0.5 - x*x - y*y))
        bx = float(np.sqrt(hx * hx + hy * hy))
        bz = float(hz)

        # Predicted magnetometer reading in body frame: R(q)^T · [bx, 0, bz]
        h_m = np.array(
            [
                (1.0 - 2.0*(y*y + z*z)) * bx + 2.0*(x*z - w*y) * bz,
                2.0*(x*y - w*z) * bx + 2.0*(y*z + w*x) * bz,
                2.0*(x*z + w*y) * bx + (1.0 - 2.0*(x*x + y*y)) * bz,
            ]
        )

        # Jacobian of h_m w.r.t. q (3×4)
        H_m = np.array(
            [
                [
                    -2.0*bz*y,
                     2.0*bz*z,
                    -4.0*bx*y - 2.0*bz*w,
                    -4.0*bx*z + 2.0*bz*x,
                ],
                [
                    -2.0*bx*z + 2.0*bz*x,
                     2.0*bx*y + 2.0*bz*w,
                     2.0*bx*x + 2.0*bz*z,
                    -2.0*bx*w + 2.0*bz*y,
                ],
                [
                     2.0*bx*y,
                     2.0*bx*z - 4.0*bz*x,
                     2.0*bx*w - 4.0*bz*y,
                     2.0*bx*x,
                ],
            ]
        )

        R_noise = self.sigma_mag ** 2 * np.eye(3)
        return self._kalman_update(q, P, m - h_m, H_m, R_noise)

    @staticmethod
    def _kalman_update(
        q: np.ndarray,
        P: np.ndarray,
        innov: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generic Kalman gain + state/covariance update."""
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        q_new = quat_normalize(q + K @ innov)
        P_new = (np.eye(4) - K @ H) @ P
        return q_new, P_new
