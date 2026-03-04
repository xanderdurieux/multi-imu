"""Per-sensor calibration from static windows.

Computes:
- Gyroscope zero-rate bias (mean angular velocity during static).
- Gravity vector in sensor frame (mean accelerometer during static).
- Magnetometer hard-iron offset (mean magnetometer during static).

All quantities are estimated purely from the sensor's own static windows —
no inter-sensor reference is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .static_windows import StaticWindows

log = logging.getLogger(__name__)

_MAG_MIN_SAMPLES = 5


@dataclass
class SensorCalibration:
    """Calibration parameters for one IMU sensor, estimated from static windows.

    Attributes
    ----------
    gyro_bias_deg_per_s:
        Constant zero-rate gyroscope bias in **deg/s** (sensor frame).
        Subtract from raw gyro readings before use.
    gravity_vector_m_per_s2:
        Mean accelerometer reading during static (m/s²) in sensor frame.
        This vector points in the direction of gravity as seen by the sensor.
        Its magnitude should be close to 9.81 m/s².
    gravity_magnitude_m_per_s2:
        Magnitude of the measured gravity vector (m/s²). Values far from 9.81
        indicate poor calibration quality.
    mag_hard_iron_uT:
        Mean magnetometer reading during static windows in **µT** (sensor
        frame).  This equals (Earth field + hard-iron offset) in the sensor's
        orientation at calibration time, and serves as:

        1. The apparent-field reference vector for TRIAD orientation (passed
           directly as ``b2``).
        2. A DC bias reference to subtract from raw mag data when applying
           calibration to the full dataset (removes the constant component,
           leaving only field variation).

        ``None`` when fewer than *mag_min_samples* valid magnetometer samples
        were available.
    n_static_samples:
        Number of acc/gyro samples used for bias/gravity estimation.
    n_mag_samples:
        Number of magnetometer samples used for hard-iron estimation.
    yaw_calibrated:
        ``True`` when *mag_hard_iron_uT* is not None and a full TRIAD
        orientation (including yaw) can be computed downstream.
    """

    gyro_bias_deg_per_s: np.ndarray       # shape (3,)
    gravity_vector_m_per_s2: np.ndarray   # shape (3,)
    gravity_magnitude_m_per_s2: float
    mag_hard_iron_uT: np.ndarray | None   # shape (3,) or None
    n_static_samples: int
    n_mag_samples: int
    yaw_calibrated: bool


def calibrate_sensor(
    windows: StaticWindows,
    *,
    mag_min_samples: int = _MAG_MIN_SAMPLES,
) -> SensorCalibration:
    """Estimate calibration parameters from static window data.

    Parameters
    ----------
    windows:
        :class:`StaticWindows` produced by
        :func:`calibration.static_windows.extract_static_windows`.
    mag_min_samples:
        Minimum number of valid magnetometer samples required to compute a
        hard-iron offset.  Below this threshold *mag_hard_iron_uT* is ``None``
        and *yaw_calibrated* is ``False``.

    Returns
    -------
    SensorCalibration

    Raises
    ------
    ValueError
        If *windows.combined* is empty or lacks the expected columns.
    """
    df = windows.combined
    if df.empty:
        raise ValueError("No static window data available; cannot calibrate sensor.")

    required = {"ax", "ay", "az", "gx", "gy", "gz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Static window DataFrame missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Gyroscope zero-rate bias
    # ------------------------------------------------------------------
    gyro_raw = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    gyro_bias = np.nanmean(gyro_raw, axis=0)

    # ------------------------------------------------------------------
    # Gravity vector (mean accelerometer in sensor frame)
    # ------------------------------------------------------------------
    acc_raw = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gravity_vector = np.nanmean(acc_raw, axis=0)
    gravity_magnitude = float(np.linalg.norm(gravity_vector))

    deviation = abs(gravity_magnitude - 9.81)
    if deviation > 0.5:
        log.warning(
            "Gravity magnitude %.4f m/s² deviates by %.4f m/s² from 9.81. "
            "Check that the static windows are truly static.",
            gravity_magnitude, deviation,
        )

    n_static = int(np.sum(np.all(np.isfinite(acc_raw), axis=1)))

    # ------------------------------------------------------------------
    # Magnetometer hard-iron offset
    # ------------------------------------------------------------------
    mag_df = windows.mag_subset(min_samples=mag_min_samples)
    n_mag = len(mag_df)

    if n_mag >= mag_min_samples:
        mag_raw = mag_df[["mx", "my", "mz"]].to_numpy(dtype=float)
        mag_hard_iron: np.ndarray | None = np.nanmean(mag_raw, axis=0)
        yaw_calibrated = True
        log.info(
            "Magnetometer hard-iron offset estimated from %d samples: "
            "[%.2f, %.2f, %.2f] µT.",
            n_mag, *mag_hard_iron,
        )
    else:
        mag_hard_iron = None
        yaw_calibrated = False
        log.warning(
            "Only %d magnetometer samples available (min=%d). "
            "Skipping hard-iron offset; yaw will not be calibrated.",
            n_mag, mag_min_samples,
        )

    log.info(
        "Sensor calibration: gyro_bias=[%.4f, %.4f, %.4f] deg/s, "
        "gravity=[%.4f, %.4f, %.4f] m/s² (|g|=%.4f), "
        "n_static=%d, n_mag=%d.",
        *gyro_bias, *gravity_vector, gravity_magnitude, n_static, n_mag,
    )

    return SensorCalibration(
        gyro_bias_deg_per_s=gyro_bias,
        gravity_vector_m_per_s2=gravity_vector,
        gravity_magnitude_m_per_s2=gravity_magnitude,
        mag_hard_iron_uT=mag_hard_iron,
        n_static_samples=n_static,
        n_mag_samples=n_mag,
        yaw_calibrated=yaw_calibrated,
    )
