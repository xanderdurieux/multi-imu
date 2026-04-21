"""Extended Kalman Filter for IMU/MARG orientation estimation.

State: unit quaternion q = [w, x, y, z] (body→world).
Process model: quaternion kinematics driven by the gyroscope.
Measurement model: normalised accelerometer (gravity) + normalised magnetometer.
"""

from __future__ import annotations

import numpy as np

from common.quaternion import quat_identity, quat_normalize, tilt_quat_from_acc


class EKFFilter:
    """Quaternion EKF orientation filter.

    Parameters
    ----------
    sigma_gyro:
        Gyroscope noise standard deviation (rad/s).
    sigma_acc:
        Accelerometer measurement noise std (normalised units).
    sigma_mag:
        Magnetometer measurement noise std (normalised units).
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
        self._dt = 1.0 / sample_rate_hz
        self._q = quat_identity()
        self._P = np.eye(4) * 1e-3

    def reset(self, q: np.ndarray | None = None) -> None:
        self._q = quat_normalize(q) if q is not None else quat_identity()
        self._P = np.eye(4) * 1e-3

    def initialize_from_acc(self, acc: np.ndarray) -> None:
        self._q = tilt_quat_from_acc(acc)
        self._P = np.eye(4) * 1e-3

    def current_quaternion(self) -> np.ndarray:
        return self._q.copy()

    def update(
        self,
        acc: np.ndarray,
        gyro_rad: np.ndarray,
        mag: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        dt = dt or self._dt
        q, P = self._predict(gyro_rad, dt)
        an = np.linalg.norm(acc)
        if an > 1e-6:
            q, P = self._update_acc(q, P, acc / an)
        if mag is not None:
            mn = np.linalg.norm(mag)
            if mn > 1e-6:
                q, P = self._update_mag(q, P, mag / mn)
        self._q, self._P = q, P
        return self._q.copy()

    def _predict(self, gyro_rad: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        gx, gy, gz = gyro_rad
        Omega = np.array([
            [ 0, -gx, -gy, -gz],
            [gx,   0,  gz, -gy],
            [gy, -gz,   0,  gx],
            [gz,  gy, -gx,   0],
        ])
        F = np.eye(4) + 0.5 * dt * Omega
        q = quat_normalize(F @ self._q)
        Q = (self.sigma_gyro * dt) ** 2 * np.eye(4)
        return q, F @ self._P @ F.T + Q

    def _update_acc(self, q: np.ndarray, P: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, x, y, z = q
        h = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        H = 2.0 * np.array([[-y, z, -w, x], [x, w, z, y], [0, -2*x, -2*y, 0]])
        return self._kalman(q, P, a - h, H, self.sigma_acc**2 * np.eye(3))

    def _update_mag(self, q: np.ndarray, P: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, x, y, z = q
        mx, my, mz = m
        hx = 2*(mx*(0.5-y*y-z*z) + my*(x*y-w*z) + mz*(x*z+w*y))
        hy = 2*(mx*(x*y+w*z) + my*(0.5-x*x-z*z) + mz*(y*z-w*x))
        hz = 2*(mx*(x*z-w*y) + my*(y*z+w*x) + mz*(0.5-x*x-y*y))
        bx, bz = float(np.sqrt(hx*hx + hy*hy)), float(hz)
        h_m = np.array([
            (1-2*(y*y+z*z))*bx + 2*(x*z-w*y)*bz,
            2*(x*y-w*z)*bx + 2*(y*z+w*x)*bz,
            2*(x*z+w*y)*bx + (1-2*(x*x+y*y))*bz,
        ])
        H_m = np.array([
            [-2*bz*y,  2*bz*z, -4*bx*y-2*bz*w, -4*bx*z+2*bz*x],
            [-2*bx*z+2*bz*x,  2*bx*y+2*bz*w,  2*bx*x+2*bz*z, -2*bx*w+2*bz*y],
            [ 2*bx*y,  2*bx*z-4*bz*x,  2*bx*w-4*bz*y,  2*bx*x],
        ])
        return self._kalman(q, P, m - h_m, H_m, self.sigma_mag**2 * np.eye(3))

    @staticmethod
    def _kalman(q, P, innov, H, R):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        return quat_normalize(q + K @ innov), (np.eye(4) - K @ H) @ P
