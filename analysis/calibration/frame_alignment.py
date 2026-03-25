"""Optional horizontal-frame refinement after gravity alignment.

Assumption (documented for thesis): longitudinal bicycle / rider direction is approximated
by the mean horizontal specific-force (acceleration minus gravity) over samples that are
not in the static calibration window and exceed a small motion threshold. This is a
practical baseline when GPS or magnetometer heading is unavailable; it fails for
sustained steady-speed straight-line coasting (near-zero horizontal acceleration) and
falls back to gravity-only alignment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.81


def gravity_rotation_matrix(g_hat: np.ndarray) -> np.ndarray:
    """Rotation ``R`` with ``R @ g_hat ≈ [0, 0, g]`` (+Z world up), matching ``calibrate.py``."""
    g = np.asarray(g_hat, dtype=float)
    if np.linalg.norm(g) < 1e-6:
        return np.eye(3)
    target = np.array([[0.0, 0.0, GRAVITY_M_S2]])
    src = g.reshape(1, 3)
    rot, _ = Rotation.align_vectors(target, src)
    return rot.as_matrix()


def estimate_yaw_align_forward(
    acc_world: np.ndarray,
    *,
    static_indices: slice | np.ndarray,
    min_motion_ms2: float = 0.35,
    min_samples: int = 40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate rotation about world Z so mean horizontal motion lies along +X.

    Parameters
    ----------
    acc_world:
        Accelerometer samples already rotated into gravity-aligned world frame (+Z up).
    static_indices:
        Rows belonging to the static calibration window (excluded from the mean).
    min_motion_ms2:
        Minimum horizontal norm of (acc - g) to treat a sample as motion (rough gate).
    min_samples:
        Minimum motion samples required; otherwise fallback to identity yaw.

    Returns
    -------
    R_yaw
        3×3 rotation about Z to apply *after* gravity alignment: ``acc_final = R_yaw @ acc_world``.
    meta
        JSON-serialisable quality flags and diagnostics.
    """
    acc_world = np.asarray(acc_world, dtype=float)
    n = len(acc_world)
    meta: dict[str, Any] = {
        "method": "mean_horizontal_specific_force",
        "fallback": False,
        "n_motion_samples": 0,
        "horizontal_norm_mean_ms2": None,
        "yaw_deg": 0.0,
    }

    if n == 0:
        meta["fallback"] = True
        meta["reason"] = "empty"
        return np.eye(3), meta

    mask_static = np.zeros(n, dtype=bool)
    if isinstance(static_indices, slice):
        mask_static[static_indices] = True
    else:
        idx = np.asarray(static_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        mask_static[idx] = True

    g_vec = np.array([0.0, 0.0, GRAVITY_M_S2], dtype=float)
    spec = acc_world - g_vec
    horiz = spec[:, :2]
    hn = np.linalg.norm(horiz, axis=1)

    motion = (~mask_static) & np.isfinite(hn) & (hn >= min_motion_ms2)
    n_mot = int(np.sum(motion))
    meta["n_motion_samples"] = n_mot

    if n_mot < min_samples:
        meta["fallback"] = True
        meta["reason"] = "insufficient_motion_samples"
        return np.eye(3), meta

    v = np.nanmean(horiz[motion], axis=0)
    norm = float(np.linalg.norm(v))
    meta["horizontal_norm_mean_ms2"] = norm

    if norm < min_motion_ms2:
        meta["fallback"] = True
        meta["reason"] = "weak_horizontal_mean"
        return np.eye(3), meta

    # Align 2D vector v to +X in the XY plane: rotate by -atan2(vy, vx)
    ang = float(np.arctan2(v[1], v[0]))
    yaw_deg = float(np.degrees(-ang))
    meta["yaw_deg"] = yaw_deg

    R_yaw = Rotation.from_euler("z", -ang).as_matrix()
    return R_yaw, meta


def compose_gravity_and_yaw(R_grav: np.ndarray, R_yaw: np.ndarray) -> np.ndarray:
    """Combined rotation: sensor → gravity world → yaw-refined world."""
    return R_yaw @ R_grav
