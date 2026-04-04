"""Complementary filter for IMU orientation estimation.

Blends gyroscope integration (fast, drift-prone) with accelerometer-derived
tilt correction (slow, noise-prone) using a simple alpha/1-alpha weighting
via quaternion SLERP.
"""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_identity,
    quat_normalize,
    quat_multiply,  # used for gyro integration: q ⊗ dq
    quat_from_gyro,
    quat_slerp,
    tilt_quat_from_acc,
)


class ComplementaryFilter:
    """Complementary filter for roll/pitch estimation from accelerometer + gyro.

    Parameters
    ----------
    alpha:
        Gyroscope weight in [0, 1].  Higher values trust the gyro more and
        are less susceptible to accelerometer noise, but accumulate drift.
        Typical value: 0.98.
    sample_rate_hz:
        Expected sampling rate in Hz (used to compute dt when not supplied).
    """

    def __init__(self, alpha: float = 0.98, sample_rate_hz: float = 100.0) -> None:
        self.alpha = alpha
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
        dt:
            Time step in seconds.  Defaults to ``1 / sample_rate_hz``.

        Returns
        -------
        np.ndarray
            Updated unit quaternion [w, x, y, z].
        """
        if dt is None:
            dt = self._dt

        acc = np.asarray(acc, dtype=float)
        gyro_rad = np.asarray(gyro_rad, dtype=float)

        # -- Gyroscope integration (body frame) --------------------------------
        dq_gyro = quat_from_gyro(gyro_rad, dt)
        q_gyro = quat_normalize(quat_multiply(self._q, dq_gyro))

        # -- Accelerometer-derived tilt ----------------------------------------
        acc_norm = np.linalg.norm(acc)
        if not (np.isfinite(acc_norm) and acc_norm > 1e-6):
            # No valid acc measurement (NaN or near-zero) — trust gyro entirely.
            self._q = q_gyro
            return self._q.copy()

        q_acc = tilt_quat_from_acc(acc)

        # -- Blend via SLERP ---------------------------------------------------
        # SLERP(q_gyro, q_acc, 1 - alpha) puts weight (1-alpha) on the acc.
        self._q = quat_slerp(q_gyro, q_acc, 1.0 - self.alpha)
        return self._q.copy()
