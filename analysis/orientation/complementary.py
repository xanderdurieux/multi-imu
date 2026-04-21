"""Complementary filter for IMU orientation estimation.

Blends gyroscope integration (fast, drift-prone) with accelerometer-derived
tilt correction (slow, noise-prone) via quaternion SLERP.
"""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_identity,
    quat_normalize,
    quat_multiply,
    quat_from_gyro,
    quat_slerp,
    tilt_quat_from_acc,
)


class ComplementaryFilter:
    """Complementary filter for roll/pitch estimation from accelerometer + gyro.

    Parameters
    ----------
    alpha:
        Gyroscope weight in [0, 1]. Higher values trust the gyro more.
    sample_rate_hz:
        Expected sampling rate in Hz (used as default dt).
    """

    def __init__(self, alpha: float = 0.98, sample_rate_hz: float = 100.0) -> None:
        self.alpha = alpha
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
        mag: np.ndarray | None = None,  # noqa: ARG002
        dt: float | None = None,
    ) -> np.ndarray:
        dt = dt or self._dt
        dq = quat_from_gyro(gyro_rad, dt)
        q_gyro = quat_normalize(quat_multiply(self._q, dq))

        an = np.linalg.norm(acc)
        if np.isfinite(an) and an > 1e-6:
            q_acc = tilt_quat_from_acc(acc)
            self._q = quat_slerp(q_gyro, q_acc, 1.0 - self.alpha)
        else:
            self._q = q_gyro
        return self._q.copy()
