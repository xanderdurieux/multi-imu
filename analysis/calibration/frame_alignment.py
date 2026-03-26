"""Section-level horizontal frame alignment after gravity alignment.

Why this module exists
----------------------
Gravity alignment alone gives an interpretable vertical axis (+Z), but leaves yaw
underdetermined. For cycling sections this makes X/Y features hard to interpret because
their meaning can drift with mounting orientation.

Candidate approaches considered (no heavy magnetometer reliance)
---------------------------------------------------------------
1) Mean horizontal specific-force direction (legacy baseline):
   - Estimate forward from mean ``(acc_world - g)`` projected into the horizontal plane.
   - Pros: simple, fast.
   - Cons: weak during steady-speed cruising (near-zero longitudinal acceleration).
2) Horizontal PCA of dynamic specific force:
   - Use dominant horizontal variance direction as a section-local motion axis.
   - Pros: robust to zero-mean dynamics and does not need heading.
   - Cons: PCA sign ambiguity, can lock onto cornering if section is mostly turns.
3) Bicycle-sensor referenced frame transfer:
   - Estimate horizontal frame from bicycle-mounted sensor (Sporsa) and apply to
     both sensors in the section.
   - Pros: consistent cross-sensor frame for rider-vs-bike comparisons.
   - Cons: depends on reference stream quality.

Default strategy in this module combines (2) + (3): estimate a section frame from the
reference (bike) sensor using PCA and optional mean-force fallback, then apply the same
yaw to all sensors. If confidence is low we gracefully fall back to gravity-only.
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


def _yaw_rot_matrix(theta_rad: float) -> np.ndarray:
    return Rotation.from_euler("z", theta_rad).as_matrix()


def _mask_from_static(n: int, static_indices: slice | np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if isinstance(static_indices, slice):
        mask[static_indices] = True
    else:
        idx = np.asarray(static_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        mask[idx] = True
    return mask


def _mean_heading_from_horiz(h: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.nanmean(h[mask], axis=0) if np.any(mask) else np.array([np.nan, np.nan], dtype=float)
    mag = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
    return v, mag


def estimate_section_horizontal_frame(
    acc_world_ref: np.ndarray,
    *,
    static_indices: slice | np.ndarray,
    min_motion_ms2: float = 0.35,
    min_samples: int = 40,
    min_pca_ratio: float = 1.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate section yaw around +Z from reference-sensor horizontal dynamics.

    The returned rotation maps gravity-aligned world frame to a section frame where:
    +Z remains up, +X≈longitudinal (dominant horizontal motion), +Y=lateral.
    """
    acc_world_ref = np.asarray(acc_world_ref, dtype=float)
    n = len(acc_world_ref)
    meta: dict[str, Any] = {
        "method": "reference_horizontal_pca_with_mean_fallback",
        "candidate_methods": [
            "mean_horizontal_specific_force",
            "horizontal_pca_dominant_axis",
            "bicycle_reference_transfer",
        ],
        "fallback": False,
        "fallback_reason": "",
        "n_samples": int(n),
        "n_motion_samples": 0,
        "straight_motion_confidence": 0.0,
        "heading_stability": 0.0,
        "horizontal_axis_reliability": 0.0,
        "confidence_score": 0.0,
        "yaw_deg": 0.0,
    }
    if n == 0:
        meta["fallback"] = True
        meta["fallback_reason"] = "empty_reference_signal"
        return np.eye(3), meta

    g_vec = np.array([0.0, 0.0, GRAVITY_M_S2], dtype=float)
    spec = acc_world_ref - g_vec
    horiz = spec[:, :2]
    hn = np.linalg.norm(horiz, axis=1)
    mask_static = _mask_from_static(n, static_indices)
    motion = (~mask_static) & np.all(np.isfinite(horiz), axis=1) & (hn >= min_motion_ms2)
    n_motion = int(np.sum(motion))
    meta["n_motion_samples"] = n_motion
    if n_motion < min_samples:
        meta["fallback"] = True
        meta["fallback_reason"] = "insufficient_motion_samples"
        return np.eye(3), meta

    h_m = horiz[motion]
    # Candidate A: mean direction (stable in straight accelerative riding)
    mean_v, mean_mag = _mean_heading_from_horiz(horiz, motion)
    mean_unit = mean_v / max(mean_mag, 1e-12) if mean_mag > 0 else np.array([1.0, 0.0], dtype=float)

    # Candidate B: dominant PCA axis (robust to near-zero-mean oscillatory signals)
    cov = np.cov(h_m.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis = eigvecs[:, order[0]]
    if np.dot(axis, mean_unit) < 0:
        axis = -axis
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        meta["fallback"] = True
        meta["fallback_reason"] = "degenerate_horizontal_covariance"
        return np.eye(3), meta
    axis = axis / axis_norm

    pca_ratio = float((eigvals[0] + 1e-12) / (eigvals[1] + 1e-12))
    straight_conf = float(np.clip((pca_ratio - 1.0) / max(min_pca_ratio - 1.0, 1e-6), 0.0, 1.0))

    # Heading consistency from normalized vectors projected on chosen axis
    u = h_m / np.clip(np.linalg.norm(h_m, axis=1, keepdims=True), 1e-9, None)
    alignment = np.abs(u @ axis.reshape(2, 1)).ravel()
    heading_stability = float(np.nanmean(alignment)) if len(alignment) else 0.0

    mean_strength = float(np.clip(mean_mag / max(min_motion_ms2, 1e-6), 0.0, 2.0) / 2.0)
    horizontal_reliability = float(0.5 * straight_conf + 0.5 * mean_strength)
    confidence = float(0.45 * straight_conf + 0.35 * heading_stability + 0.20 * horizontal_reliability)

    meta["straight_motion_confidence"] = straight_conf
    meta["heading_stability"] = heading_stability
    meta["horizontal_axis_reliability"] = horizontal_reliability
    meta["pca_eigen_ratio"] = pca_ratio
    meta["mean_horizontal_norm_ms2"] = mean_mag
    meta["confidence_score"] = confidence

    if pca_ratio < 1.02 and mean_mag < min_motion_ms2:
        meta["fallback"] = True
        meta["fallback_reason"] = "weak_horizontal_structure"
        return np.eye(3), meta

    use_mean = mean_mag >= min_motion_ms2 and heading_stability >= 0.55 and straight_conf < 0.25
    chosen = mean_unit if use_mean else axis
    meta["chosen_axis_source"] = "mean_horizontal" if use_mean else "horizontal_pca"

    ang = float(np.arctan2(chosen[1], chosen[0]))
    yaw = -ang
    meta["yaw_deg"] = float(np.degrees(yaw))
    return _yaw_rot_matrix(yaw), meta
