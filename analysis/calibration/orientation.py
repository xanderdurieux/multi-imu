"""World-frame orientation calibration using the TRIAD method.

World frame convention: ENU (East-North-Up)
  - X = East
  - Y = magnetic North (horizontal projection of Earth's magnetic field)
  - Z = Up

TRIAD uses two non-collinear reference vectors to solve for the rotation from
sensor frame to world frame:

  Reference 1 — specific force (accelerometer reading at rest):
    - In world frame:  r1 = [0, 0, +1]  (specific force points up, i.e. +Z in ENU)
    - In sensor frame: b1 = normalize(mean_acc_static)

  Reference 2 — magnetic north:
    - In world frame:  r2 = [0, 1, 0]  (Y = magnetic north by convention)
    - In sensor frame: b2 = normalize(mean_mag_static - hard_iron_offset)

When the magnetometer is unavailable, only pitch and roll are determined via a
minimum-rotation (gravity-only) approach; yaw is set to zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .per_sensor import SensorCalibration

log = logging.getLogger(__name__)

# World-frame reference directions (ENU)
#
# The accelerometer measures **specific force** (reaction to gravity), which
# points *upward* when the sensor is at rest.  In ENU, up = +Z.  So the
# reference for the primary TRIAD vector is [0, 0, +1], and after calibration
# the world-frame accelerometer output satisfies az ≈ +9.81 m/s² at rest.
_GRAVITY_WORLD = np.array([0.0, 0.0, 1.0])    # specific-force direction (up = +Z in ENU)
_MAG_NORTH_WORLD = np.array([0.0, 1.0, 0.0])  # magnetic north (Y in ENU)


@dataclass
class OrientationCalibration:
    """Rotation matrix from sensor frame to world (ENU) frame.

    Attributes
    ----------
    rotation_sensor_to_world:
        3×3 rotation matrix *R* such that ``v_world = R @ v_sensor``.
    yaw_calibrated:
        ``True`` when magnetometer data was used to resolve yaw.  ``False``
        when only gravity was available; yaw is then implicitly zero (sensor
        heading assumed to coincide with world Y at calibration time).
    gravity_residual_m_per_s2:
        Residual between the rotated gravity vector and the ideal world gravity
        ``[0, 0, -9.81]``.  Small values indicate good alignment.
    """

    rotation_sensor_to_world: np.ndarray   # shape (3, 3)
    yaw_calibrated: bool
    gravity_residual_m_per_s2: float


# ---------------------------------------------------------------------------
# Low-level rotation primitives
# ---------------------------------------------------------------------------

def _triad(
    r1: np.ndarray,
    r2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Compute a rotation matrix from body to reference frame using TRIAD.

    Parameters
    ----------
    r1, r2:
        Two reference vectors in the **world** frame (need not be unit or
        mutually perpendicular).  *r1* is treated as the primary reference
        (usually gravity).
    b1, b2:
        The same two vectors measured in the **sensor** frame.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix such that ``R @ b ≈ r`` for the reference pairs.

    Raises
    ------
    ValueError
        If either pair of vectors is nearly collinear (ill-conditioned).
    """
    # Normalise primary vectors
    r1_n = r1 / np.linalg.norm(r1)
    b1_n = b1 / np.linalg.norm(b1)

    # Build orthonormal triads
    # World triad
    t1_r = r1_n
    cross_r = np.cross(r1_n, r2)
    norm_r = np.linalg.norm(cross_r)
    if norm_r < 1e-10:
        raise ValueError("r1 and r2 are nearly collinear; TRIAD is ill-conditioned.")
    t2_r = cross_r / norm_r
    t3_r = np.cross(t1_r, t2_r)

    # Sensor triad
    t1_b = b1_n
    cross_b = np.cross(b1_n, b2)
    norm_b = np.linalg.norm(cross_b)
    if norm_b < 1e-10:
        raise ValueError(
            "b1 and b2 are nearly collinear in sensor frame; "
            "cannot resolve full orientation."
        )
    t2_b = cross_b / norm_b
    t3_b = np.cross(t1_b, t2_b)

    # Column matrices
    M_r = np.column_stack([t1_r, t2_r, t3_r])
    M_b = np.column_stack([t1_b, t2_b, t3_b])

    return M_r @ M_b.T


def _gravity_only_rotation(gravity_sensor: np.ndarray) -> np.ndarray:
    """Compute pitch+roll rotation that maps *gravity_sensor* to world +Z.

    *gravity_sensor* is the mean static accelerometer reading (specific force,
    pointing upward when at rest).  The rotation aligns it to world +Z (up in
    ENU).  Yaw is set to zero.  Uses Rodrigues' rotation formula for a
    minimum-angle rotation.

    Parameters
    ----------
    gravity_sensor:
        Mean accelerometer reading during static (specific force, m/s²).
        Need not be unit length.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    b = gravity_sensor / np.linalg.norm(gravity_sensor)
    r = _GRAVITY_WORLD  # [0, 0, -1]

    c = float(np.dot(b, r))

    if c > 1.0 - 1e-10:
        # Already aligned
        return np.eye(3)

    if c < -1.0 + 1e-10:
        # Anti-parallel: 180° rotation around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(b, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for θ = π: R = 2*axis*axis^T - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    v = np.cross(b, r)
    s = float(np.linalg.norm(v))
    K = np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])
    return np.eye(3) + K + K @ K * (1.0 - c) / (s * s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_orientation(cal: SensorCalibration) -> OrientationCalibration:
    """Compute the sensor-to-world rotation matrix from a :class:`SensorCalibration`.

    Uses the **TRIAD** method when magnetometer data is available, otherwise
    falls back to gravity-only alignment (pitch + roll only, yaw = 0).

    The world frame is ENU:
    - Z up  (gravity points to −Z)
    - Y magnetic North  (horizontal projection of Earth's magnetic field)
    - X East  (completing the right-handed frame)

    Parameters
    ----------
    cal:
        Sensor calibration produced by :func:`calibration.per_sensor.calibrate_sensor`.

    Returns
    -------
    OrientationCalibration
    """
    gravity_sensor = cal.gravity_vector_m_per_s2

    if cal.yaw_calibrated and cal.mag_hard_iron_uT is not None:
        # We need the mean corrected magnetometer vector from the static windows.
        # This is recomputed later in session.py from the raw static mag mean
        # minus the hard-iron offset. Here we pass through a sentinel; the
        # actual TRIAD call is performed in compute_orientation_from_vectors().
        raise RuntimeError(
            "Use compute_orientation_from_vectors() when mag data is available. "
            "compute_orientation() is for the gravity-only fallback path."
        )

    R = _gravity_only_rotation(gravity_sensor)
    gravity_residual = _gravity_residual(R, gravity_sensor)

    log.info(
        "Gravity-only orientation: gravity_residual=%.4f m/s² (yaw_calibrated=False).",
        gravity_residual,
    )
    return OrientationCalibration(
        rotation_sensor_to_world=R,
        yaw_calibrated=False,
        gravity_residual_m_per_s2=gravity_residual,
    )


def compute_orientation_from_vectors(
    gravity_sensor: np.ndarray,
    mag_sensor_corrected: np.ndarray | None,
) -> OrientationCalibration:
    """Compute sensor-to-world rotation from raw reference vectors.

    Parameters
    ----------
    gravity_sensor:
        Mean accelerometer vector during static (m/s²), in sensor frame.
    mag_sensor_corrected:
        Mean magnetometer vector during static (µT) with hard-iron offset
        already subtracted, in sensor frame.  Pass ``None`` to use the
        gravity-only fallback.

    Returns
    -------
    OrientationCalibration
    """
    if mag_sensor_corrected is not None and np.all(np.isfinite(mag_sensor_corrected)):
        # Project mag perpendicular to gravity before TRIAD.  At high magnetic
        # inclinations (e.g. Belgium ~65°) the raw mag vector is nearly parallel
        # to gravity in the sensor frame, making the cross-product ill-conditioned.
        # Projecting onto the plane perpendicular to gravity extracts the
        # horizontal (north-pointing) component, which is the correct second
        # reference for heading determination.
        b1_n = gravity_sensor / np.linalg.norm(gravity_sensor)
        mag_horiz = mag_sensor_corrected - np.dot(mag_sensor_corrected, b1_n) * b1_n
        mag_horiz_norm = np.linalg.norm(mag_horiz)

        if mag_horiz_norm < 1e-6:
            log.warning(
                "Horizontal magnetic component is near-zero (norm=%.2e). "
                "Magnetic field may be nearly vertical. Falling back to gravity-only.",
                mag_horiz_norm,
            )
            mag_sensor_corrected = None
        else:
            mag_sensor_corrected = mag_horiz

    if mag_sensor_corrected is not None and np.all(np.isfinite(mag_sensor_corrected)):
        # Attempt full TRIAD
        try:
            R = _triad(
                r1=_GRAVITY_WORLD,
                r2=_MAG_NORTH_WORLD,
                b1=gravity_sensor,
                b2=mag_sensor_corrected,
            )
            gravity_residual = _gravity_residual(R, gravity_sensor)
            log.info(
                "TRIAD orientation: gravity_residual=%.4f m/s² (yaw_calibrated=True).",
                gravity_residual,
            )
            return OrientationCalibration(
                rotation_sensor_to_world=R,
                yaw_calibrated=True,
                gravity_residual_m_per_s2=gravity_residual,
            )
        except ValueError as exc:
            log.warning(
                "TRIAD failed (%s). Falling back to gravity-only rotation.", exc
            )

    # Gravity-only fallback
    R = _gravity_only_rotation(gravity_sensor)
    gravity_residual = _gravity_residual(R, gravity_sensor)
    log.info(
        "Gravity-only orientation: gravity_residual=%.4f m/s² (yaw_calibrated=False).",
        gravity_residual,
    )
    return OrientationCalibration(
        rotation_sensor_to_world=R,
        yaw_calibrated=False,
        gravity_residual_m_per_s2=gravity_residual,
    )


def _gravity_residual(R: np.ndarray, gravity_sensor: np.ndarray) -> float:
    """Euclidean distance between *R @ gravity_sensor* and ideal world specific force."""
    g_ideal = _GRAVITY_WORLD * 9.81          # [0, 0, +9.81] in ENU (specific force = up)
    g_rotated = R @ gravity_sensor
    return float(np.linalg.norm(g_rotated - g_ideal))
