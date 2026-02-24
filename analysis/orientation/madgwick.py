"""Madgwick orientation filter.

This implementation delegates the core update step to the `ahrs` Python package
(Mayitzin/ahrs), while keeping the same interface and accelerometer gating used
elsewhere in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ahrs.filters import AngularRate as _AhrsAngularRate
from ahrs.filters import Madgwick as _AhrsMadgwick

from .quaternion import quat_identity, quat_normalize


@dataclass
class MadgwickConfig:
    """Configuration parameters for the Madgwick orientation filter."""

    # `ahrs.filters.Madgwick` uses `gain` as the gradient-descent parameter.
    # We keep the name `beta` to match common Madgwick notation.
    beta: float = 0.1

    # Approximate gravity magnitude used for gating (m/s^2).
    gravity: float = 9.81

    # Gating on accelerometer magnitude, same semantics as complementary filter.
    accel_norm_tolerance_g: float = 0.2
    use_accel_gating: bool = True


class MadgwickOrientationFilter:
    """Madgwick IMU filter (gyro + accel, no magnetometer)."""

    def __init__(self, config: Optional[MadgwickConfig] = None):
        self.config = config or MadgwickConfig()
        self.q_bw = quat_identity()
        self._madgwick = _AhrsMadgwick(gain=float(self.config.beta))
        self._angular = _AhrsAngularRate()

    def reset(self, initial_quaternion: Optional[np.ndarray] = None) -> None:
        """Reset the filter state to a given quaternion or identity."""
        if initial_quaternion is None:
            self.q_bw = quat_identity()
        else:
            self.q_bw = quat_normalize(np.asarray(initial_quaternion, dtype=float))

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
        acc_body_ms2: Optional[Iterable[float]],
    ) -> np.ndarray:
        """Advance the filter by one IMU sample."""
        q_prev = self.q_bw
        gyr = np.asarray(gyro_body_rad, dtype=float).reshape(3)

        if acc_body_ms2 is None:
            q_new = self._angular.update(q_prev, gyr=gyr, dt=float(dt))
            self.q_bw = quat_normalize(q_new)
            return self.q_bw

        acc = np.asarray(acc_body_ms2, dtype=float).reshape(3)
        if self._accel_is_trustworthy(acc):
            q_new = self._madgwick.updateIMU(q_prev, gyr=gyr, acc=acc, dt=float(dt))
        else:
            # High dynamics: skip accelerometer correction, propagate with gyro only.
            q_new = self._angular.update(q_prev, gyr=gyr, dt=float(dt))

        self.q_bw = quat_normalize(q_new)
        return self.q_bw

