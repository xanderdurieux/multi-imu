"""Madgwick MARG orientation filter.

Reference
---------
Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).
Estimation of IMU and MARG orientation using a gradient descent algorithm.
IEEE International Conference on Rehabilitation Robotics.
"""

from __future__ import annotations

import numpy as np

from common.quaternion import quat_identity, quat_normalize, tilt_quat_from_acc


class MadgwickFilter:
    """Madgwick MARG orientation filter.

    Parameters
    ----------
    beta:
        Filter gain. Higher values weight the accelerometer correction more.
    sample_rate_hz:
        Expected sampling rate in Hz (used as default dt).
    """

    def __init__(self, beta: float = 0.1, sample_rate_hz: float = 100.0) -> None:
        self.beta = beta
        self._dt = 1.0 / sample_rate_hz
        self._q = quat_identity()

    def reset(self, q: np.ndarray | None = None) -> None:
        self._q = quat_normalize(q) if q is not None else quat_identity()

    def initialize_from_acc(self, acc: np.ndarray) -> None:
        self._q = tilt_quat_from_acc(acc)

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
        q = self._q
        w, x, y, z = q
        gx, gy, gz = gyro_rad

        q_dot = 0.5 * np.array([
            -x*gx - y*gy - z*gz,
             w*gx + y*gz - z*gy,
             w*gy - x*gz + z*gx,
             w*gz + x*gy - y*gx,
        ])

        an = np.linalg.norm(acc)
        if an > 1e-6:
            a = acc / an
            if mag is not None:
                mn = np.linalg.norm(mag)
                if mn > 1e-6:
                    q_dot -= self.beta * _gradient_marg(q, a, mag / mn)
                else:
                    q_dot -= self.beta * _gradient_imu(q, a)
            else:
                q_dot -= self.beta * _gradient_imu(q, a)

        self._q = quat_normalize(q + q_dot * dt)
        return self._q.copy()


def _gradient_imu(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ax, ay, az = a
    f1 = 2*(x*z - w*y) - ax
    f2 = 2*(w*x + y*z) - ay
    f3 = 2*(0.5 - x*x - y*y) - az
    grad = np.array([2*(-y*f1 + x*f2), 2*(z*f1 + w*f2 - 2*x*f3),
                     2*(-w*f1 + z*f2 - 2*y*f3), 2*(x*f1 + y*f2)])
    n = np.linalg.norm(grad)
    return grad / n if n > 1e-12 else grad


def _gradient_marg(q: np.ndarray, a: np.ndarray, m: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ax, ay, az = a
    mx, my, mz = m
    hx = 2*(mx*(0.5-y*y-z*z) + my*(x*y-w*z) + mz*(x*z+w*y))
    hy = 2*(mx*(x*y+w*z) + my*(0.5-x*x-z*z) + mz*(y*z-w*x))
    hz = 2*(mx*(x*z-w*y) + my*(y*z+w*x) + mz*(0.5-x*x-y*y))
    bx, bz = np.sqrt(hx*hx + hy*hy), hz
    f1 = 2*(x*z - w*y) - ax
    f2 = 2*(w*x + y*z) - ay
    f3 = 2*(0.5 - x*x - y*y) - az
    f4 = 2*bx*(0.5-y*y-z*z) + 2*bz*(x*z-w*y) - mx
    f5 = 2*bx*(x*y-w*z) + 2*bz*(w*x+y*z) - my
    f6 = 2*bx*(w*y+x*z) + 2*bz*(0.5-x*x-y*y) - mz
    gw = -2*y*f1 + 2*x*f2 - 2*bz*y*f4 + (-2*bx*z+2*bz*x)*f5 + 2*bx*y*f6
    gx =  2*z*f1 + 2*w*f2 - 4*x*f3 + 2*bz*z*f4 + (2*bx*y+2*bz*w)*f5 + (2*bx*z-2*bz*w)*f6
    gy = -2*w*f1 + 2*z*f2 - 4*y*f3 + (-4*bx*y-2*bz*w)*f4 + (2*bx*x+2*bz*z)*f5 + (2*bx*w-4*bz*y)*f6
    gz =  2*x*f1 + 2*y*f2 + (-4*bx*z+2*bz*x)*f4 + (-2*bx*w+2*bz*y)*f5 + 2*bx*x*f6
    grad = np.array([gw, gx, gy, gz])
    n = np.linalg.norm(grad)
    return grad / n if n > 1e-12 else grad
