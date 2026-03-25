"""Protocol-aware orientation analysis for a controlled recording.

Produces three figures for any recording that has both sensor orientation
CSVs.  All relative-orientation metrics use **quaternion algebra** — not
Euler-angle subtraction — so the result is geometrically exact and free
from representation singularities.

Sensor axis conventions (recording 2)
--------------------------------------
These conventions are fixed by the physical sensor mounting:

* **Sporsa** (handlebar):  X → Front,  Y → Left,  Z → Down
* **Arduino** (helmet):    Y → Front,  X → Right, Z → Down

When both sensors are placed with the handlebar they are rotated 90° around
the Down (Z) axis relative to each other.  This **mounting rotation** is:

    q_mount = +90° around Z  →  [w=1/√2, x=0, y=0, z=1/√2]

It is used analytically to define the "zero reference" for relative
orientation.  0° in all relative-orientation panels means the two sensors
are in the physically expected co-aligned pose.

Figures
-------
orientation_sync.png
    Both sensors' accelerometer norm overlaid on a shared time axis.
    Identical peaks → the sensors moved together (expected during aligned
    calibration shakes or co-aligned cycling).  Protocol phases are shaded.

orientation_relative_quat.png
    Quaternion relative orientation corrected for the known mounting offset:

        q_rel(t)    = q_sporsa(t)⁻¹ ⊗ q_arduino(t)
        q_change(t) = q_mount⁻¹ ⊗ q_rel(t)

    Four panels:
      1. Angular distance from the analytically co-aligned pose (°).
         0° = sensors in the expected co-aligned pose (matching handlebar).
      2. ΔPitch  — head nodding (most reliable, acc-observable).
      3. ΔRoll   — head tilting left/right (reliable, acc-observable).
      4. ΔYaw    — head turning left/right (unreliable without magnetometer).
    Protocol phases are shaded.

orientation_az_compare.png
    World-frame az (Up) component for both sensors overlaid.
    At rest both should track +9.81 m/s².

Usage
-----
    uv run -m visualization.plot_orientation_protocol 2026-02-26_2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from common import load_dataframe, recording_stage_dir
from ._utils import mask_valid_plot_x, nan_mask_invalid_plot_x

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_COLORS  = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_LABELS  = {"sporsa": "Sporsa (handlebar)", "arduino": "Arduino (helmet)"}
_GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Mounting rotation
# ---------------------------------------------------------------------------
# Sporsa : X=Front, Y=Left,  Z=Down
# Arduino: Y=Front, X=Right, Z=Down
#
# Express Arduino axes in Sporsa frame:
#   Arduino-Front (Y_a) = Sporsa-Front (X_s) → Y_a maps to X_s
#   Arduino-Right (X_a) = Sporsa-Right (-Y_s) → X_a maps to -Y_s
#   Arduino-Down  (Z_a) = Sporsa-Down  (Z_s)  → Z_a maps to Z_s
#
# The rotation R (Sporsa body → Arduino body) is a +90° rotation around Z:
#   R * [1,0,0] = [0,1,0]   (X_s → Y_a)
#   R * [0,1,0] = [-1,0,0]  (Y_s → -X_a = left → -right ✓)
#   R * [0,0,1] = [0,0,1]   (Z_s → Z_a)
#
# As quaternion: +90° around Z → [w=1/√2, x=0, y=0, z=1/√2]
_SQRT2_INV = 1.0 / np.sqrt(2.0)
_MOUNTING_QUAT = np.array([[_SQRT2_INV, 0.0, 0.0, _SQRT2_INV]])  # shape (1,4)


# ---------------------------------------------------------------------------
# Quaternion helpers (vectorised)
# ---------------------------------------------------------------------------

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batch quaternion multiplication q1 ⊗ q2.  Both (N,4) or (4,)."""
    q1 = np.atleast_2d(q1); q2 = np.atleast_2d(q2)
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)


def _qconj(q: np.ndarray) -> np.ndarray:
    """Batch conjugate.  (N,4) → (N,4)."""
    q = np.atleast_2d(q.copy())
    q[:, 1:] *= -1
    return q


def _qnorm(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(q)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def _euler_from_quat(q: np.ndarray):
    """Z-Y-X Euler angles (yaw, pitch, roll) in degrees, batch version."""
    q = _qnorm(np.atleast_2d(q))
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    sinp  = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.degrees(np.arcsin(sinp))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return yaw, pitch, roll


def _rotate_vecs(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors to world frame (vectorised)."""
    if vecs.ndim == 1:
        vecs = np.tile(vecs, (len(quats), 1))
    w, x, y, z = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    vx, vy, vz = vecs[:,0], vecs[:,1], vecs[:,2]
    cx = y*vz - z*vy;  cy = z*vx - x*vz;  cz = x*vy - y*vx
    cx2 = y*cz - z*cy; cy2 = z*cx - x*cz; cz2 = x*cy - y*cx
    return np.stack([vx+2*(w*cx+cx2), vy+2*(w*cy+cy2), vz+2*(w*cz+cz2)], axis=1)


def _unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    fin = np.isfinite(out)
    if fin.sum() > 1:
        out[fin] = np.degrees(np.unwrap(np.radians(out[fin])))
    return out


# ---------------------------------------------------------------------------
# Protocol phase detection
# ---------------------------------------------------------------------------

def _detect_shake_clusters(
    t: np.ndarray,
    norm: np.ndarray,
    shake_threshold: float = 18.0,
    cluster_gap_s: float = 3.0,
) -> list[tuple[float, float]]:
    """Return the time intervals of each shake cluster.

    A "shake cluster" is a contiguous group of samples where acc_norm exceeds
    *shake_threshold*, with gaps < *cluster_gap_s* merged together.

    Returns a list of ``(t_start, t_end)`` tuples, one per cluster.
    """
    above = np.isfinite(norm) & (norm > shake_threshold)
    if not above.any():
        return []

    # Build contiguous on-segments
    changes = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]

    segments: list[tuple[float, float]] = [
        (float(t[s]), float(t[min(e, len(t)-1)]))
        for s, e in zip(starts, ends)
    ]

    # Merge segments within cluster_gap_s of each other
    if not segments:
        return []
    merged: list[tuple[float, float]] = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] < cluster_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return merged


def _protocol_phases(
    t: np.ndarray,
    norm: np.ndarray,
) -> dict:
    """Infer recording phases from acc_norm.

    Returns a dict with keys:
    - ``pre_align``:  (t_start, t_end) of first shake cluster ± buffer
    - ``post_align``: (t_start, t_end) of last shake cluster ± buffer
    - ``ride``:       (t_start, t_end) between the two clusters
    - ``clusters``:   list of all detected shake cluster intervals
    """
    clusters = _detect_shake_clusters(t, norm)
    buf = 6.0   # seconds of padding around each shake cluster

    if len(clusters) < 2:
        return {"pre_align": None, "post_align": None, "ride": None,
                "clusters": clusters}

    pre_t0  = max(t[0], clusters[0][0] - buf)
    pre_t1  = min(t[-1], clusters[0][1] + buf)
    post_t0 = max(t[0], clusters[-1][0] - buf)
    post_t1 = min(t[-1], clusters[-1][1] + buf)
    ride_t0 = clusters[0][1]
    ride_t1 = clusters[-1][0]

    return {
        "pre_align":  (pre_t0,  pre_t1),
        "post_align": (post_t0, post_t1),
        "ride":       (ride_t0, ride_t1) if ride_t1 > ride_t0 else None,
        "clusters":   clusters,
    }


def _shade_phases(ax, phases: dict, t_max: float) -> None:
    """Add protocol phase shading and legend patches to *ax*."""
    if phases["pre_align"] is not None:
        t0, t1 = phases["pre_align"]
        ax.axvspan(t0, t1, color="#2ecc71", alpha=0.12, zorder=0)
    if phases["post_align"] is not None:
        t0, t1 = phases["post_align"]
        ax.axvspan(t0, t1, color="#2ecc71", alpha=0.12, zorder=0)
    if phases["ride"] is not None:
        t0, t1 = phases["ride"]
        ax.axvspan(t0, t1, color="#3498db", alpha=0.07, zorder=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(csv_path: Path):
    df = load_dataframe(csv_path)
    if df.empty:
        return None, None
    t = df["timestamp"].to_numpy(dtype=float)
    return df, (t - t[0]) / 1000.0


def _find_csv(stage_dir: Path, sensor: str) -> Path | None:
    """Return the best orientation CSV for *sensor* in *stage_dir*.

    Prefers ``complementary/<sensor>_orientation.csv`` (new per-method layout),
    then falls back to legacy flat naming patterns.
    """
    # New per-method layout: prefer complementary, fall back to any method.
    for method in ("complementary", "madgwick"):
        candidate = stage_dir / method / f"{sensor}_orientation.csv"
        if candidate.exists():
            return candidate
    # Legacy flat layout
    for pat in (f"{sensor}*__complementary_orientation.csv",
                f"{sensor}*_orientation.csv"):
        hits = sorted(stage_dir.glob(pat))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Phase legend helper
# ---------------------------------------------------------------------------

def _phase_legend_patches() -> list:
    return [
        mpatches.Patch(facecolor="#2ecc71", alpha=0.35, label="Alignment (helmet off, aligned with handlebar)"),
        mpatches.Patch(facecolor="#3498db", alpha=0.25, label="Bike ride (helmet on)"),
    ]


# ---------------------------------------------------------------------------
# Figure 1 — acc norm overlay (synchronisation check)
# ---------------------------------------------------------------------------

def plot_sync(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Overlay both sensors' acc_body norm with protocol phase shading."""
    stage_dir = recording_stage_dir(recording_name, stage)
    data = {}
    for sensor in _SENSORS:
        p = _find_csv(stage_dir, sensor)
        if p is None:
            continue
        df, t = _load(p)
        if df is None:
            continue
        acc_cols = ["ax","ay","az"]
        if not all(c in df.columns for c in acc_cols):
            continue
        acc  = df[acc_cols].to_numpy(dtype=float)
        norm = np.linalg.norm(acc, axis=1)
        data[sensor] = (t, norm)

    if len(data) < 2:
        log.warning("[%s] need both sensors — skipping sync plot.", recording_name)
        return None

    # Infer phases from Sporsa (handlebar) acc_norm
    phases = _protocol_phases(*data["sporsa"]) if "sporsa" in data else {}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             constrained_layout=True)

    ax_main = axes[0]
    for sensor, (t, norm) in data.items():
        fin = mask_valid_plot_x(t) & np.isfinite(norm)
        ax_main.plot(t[fin], norm[fin], color=_COLORS[sensor], linewidth=0.7,
                     alpha=0.8, label=_LABELS[sensor])
    ax_main.axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.8,
                    alpha=0.5, label=f"|g| = {_GRAVITY}")
    _shade_phases(ax_main, phases, max(v[0][-1] for v in data.values()))
    ax_main.set_ylabel("‖acc_body‖ [m/s²]", fontsize=9)
    ax_main.set_ylim(0, min(60, max(max(np.nanmax(v[1]) for v in data.values()), 15)))
    ax_main.grid(True, alpha=0.25)
    legend_handles = [
        plt.Line2D([], [], color=_COLORS[s], label=_LABELS[s]) for s in data
    ] + [
        plt.Line2D([], [], color="k", linestyle="--", label=f"|g| = {_GRAVITY}")
    ] + _phase_legend_patches()
    ax_main.legend(handles=legend_handles, fontsize=8, loc="upper right")

    for ax, sensor in zip(axes[1:], _SENSORS):
        if sensor not in data:
            ax.set_visible(False)
            continue
        t, norm = data[sensor]
        fin = mask_valid_plot_x(t) & np.isfinite(norm)
        ax.plot(t[fin], norm[fin], color=_COLORS[sensor], linewidth=0.6, alpha=0.8)
        ax.axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
        _shade_phases(ax, phases, t[-1])
        ax.set_ylabel(_LABELS[sensor].split()[0], fontsize=8)
        ax.set_ylim(0, 50)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time [s]", fontsize=9)
    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "Acc-norm synchronisation — matching peaks = sensors moved together\n"
        "Green = alignment periods (helmet off, co-aligned with handlebar)  "
        "| Blue = bike ride",
        fontsize=10,
    )

    out = stage_dir / "orientation_sync.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 — quaternion relative orientation
# ---------------------------------------------------------------------------

def plot_relative_quat(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Quaternion-based relative orientation using the analytic mounting offset.

    The relative orientation is:
        q_rel(t) = q_sporsa(t)⁻¹ ⊗ q_arduino(t)

    To express head rotation relative to handlebar in physically meaningful
    angles the known mounting rotation is removed analytically:
        q_change(t) = q_mount⁻¹ ⊗ q_rel(t)

    where q_mount = +90° around Z encodes the known axis difference between
    the two sensors (Sporsa X=Front vs Arduino Y=Front).

    0° in all panels means the sensors are in the expected co-aligned pose
    (both aligned with the handlebar).  Any deviation during the bike ride
    represents actual head rotation relative to the handlebar.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    dfs: dict[str, tuple] = {}
    for sensor in _SENSORS:
        p = _find_csv(stage_dir, sensor)
        if p is None:
            continue
        df, t = _load(p)
        if df is None:
            continue
        dfs[sensor] = (df, t)

    if len(dfs) < 2:
        log.warning("[%s] need both sensors — skipping relative-quat plot.", recording_name)
        return None

    df_s, t_s = dfs["sporsa"]
    df_a, t_a = dfs["arduino"]

    q_s     = _qnorm(df_s[["qw","qx","qy","qz"]].to_numpy(dtype=float))
    q_a_raw = df_a[["qw","qx","qy","qz"]].to_numpy(dtype=float)

    # Find overlapping time window
    overlap_min = max(t_s[0], t_a[0])
    overlap_max = min(t_s[-1], t_a[-1])
    mask_s = (t_s >= overlap_min) & (t_s <= overlap_max)
    mask_a = (t_a >= overlap_min) & (t_a <= overlap_max)

    if mask_s.sum() < 10 or mask_a.sum() < 10:
        log.warning("[%s] insufficient time overlap — skipping.", recording_name)
        return None

    # Resample arduino onto sporsa time grid via linear interpolation per component
    q_a_interp = np.zeros((mask_s.sum(), 4))
    for i in range(4):
        q_a_interp[:, i] = np.interp(
            t_s[mask_s], t_a[mask_a], q_a_raw[mask_a, i]
        )
    q_a_interp = _qnorm(q_a_interp)
    q_s_sub    = q_s[mask_s]
    t_ref      = t_s[mask_s]

    # Relative quaternion: arduino relative to sporsa in sporsa's world frame
    q_rel = _qnorm(_qmul(_qconj(q_s_sub), q_a_interp))

    # Remove analytic mounting rotation (Sporsa X=Front, Y=Left, Z=Down vs
    # Arduino Y=Front, X=Right, Z=Down → +90° around Z)
    q_change = _qnorm(_qmul(_qconj(_MOUNTING_QUAT), q_rel))

    # Angular distance from perfectly co-aligned pose (scalar, degrees)
    ang_dist = 2.0 * np.degrees(np.arccos(np.clip(np.abs(q_change[:, 0]), 0, 1)))

    # Directional decomposition
    yaw_ch, pitch_ch, roll_ch = _euler_from_quat(q_change)
    yaw_ch   = _unwrap_deg(yaw_ch)
    pitch_ch = _unwrap_deg(pitch_ch)
    roll_ch  = _unwrap_deg(roll_ch)

    # Protocol phases from sporsa acc_norm
    acc_s    = df_s[["ax","ay","az"]].to_numpy(dtype=float)
    norm_s   = np.linalg.norm(acc_s, axis=1)
    phases   = _protocol_phases(t_s, norm_s)

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                             constrained_layout=True)

    def _shade_and_zero(ax):
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
        _shade_phases(ax, phases, t_ref[-1])
        ax.grid(True, alpha=0.25)

    # Panel 1: angular distance
    ax = axes[0]
    ta, ad = nan_mask_invalid_plot_x(t_ref, ang_dist)
    ax.fill_between(ta, ad, alpha=0.25, color="#555")
    ax.plot(ta, ad, color="#333", linewidth=0.7)
    _shade_and_zero(ax)
    ax.set_ylabel("Angular distance [°]", fontsize=9)
    ax.text(0.01, 0.95,
            "0° = sensors perfectly co-aligned with handlebar  "
            "| rises when head and handlebar diverge",
            transform=ax.transAxes, fontsize=7.5, color="#444",
            va="top")

    # Panel 2: ΔPitch (nodding)
    ax = axes[1]
    tp, pp = nan_mask_invalid_plot_x(t_ref, pitch_ch)
    ax.plot(tp, pp, color="#e67e22", linewidth=0.7)
    _shade_and_zero(ax)
    ax.set_ylabel("ΔPitch [°]\n(head nod up/down)", fontsize=9)

    # Panel 3: ΔRoll (tilting)
    ax = axes[2]
    tr, rr = nan_mask_invalid_plot_x(t_ref, roll_ch)
    ax.plot(tr, rr, color="#27ae60", linewidth=0.7)
    _shade_and_zero(ax)
    ax.set_ylabel("ΔRoll [°]\n(head tilt left/right)", fontsize=9)

    # Panel 4: ΔYaw (turning)
    ax = axes[3]
    ty, yy = nan_mask_invalid_plot_x(t_ref, yaw_ch)
    ax.plot(ty, yy, color="#8e44ad", linewidth=0.7)
    _shade_and_zero(ax)
    ax.set_ylabel("ΔYaw [°]\n(head turn left/right)", fontsize=9)
    ax.text(0.98, 0.05, "Yaw drifts without\nmagnetometer",
            transform=ax.transAxes, fontsize=8, color="#8e44ad",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#8e44ad", alpha=0.7))
    ax.set_xlabel("Time [s]", fontsize=9)

    phase_patches = _phase_legend_patches()
    fig.legend(handles=phase_patches, loc="upper right", fontsize=8,
               framealpha=0.85, bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "Head orientation relative to handlebar  (analytic mounting offset removed)\n"
        "Sporsa: X=Front, Y=Left, Z=Down  |  Arduino: Y=Front, X=Right, Z=Down",
        fontsize=10,
    )

    out = stage_dir / "orientation_relative_quat.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 3 — world-frame az comparison
# ---------------------------------------------------------------------------

def plot_az_world_compare(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Both sensors' world-frame az (Up) component overlaid with phase shading."""
    stage_dir = recording_stage_dir(recording_name, stage)
    data = {}
    acc_norms = {}
    for sensor in _SENSORS:
        p = _find_csv(stage_dir, sensor)
        if p is None:
            continue
        df, t = _load(p)
        if df is None:
            continue
        qcols = ["qw","qx","qy","qz"]; acols = ["ax","ay","az"]
        if not all(c in df.columns for c in qcols + acols):
            continue
        q    = _qnorm(df[qcols].to_numpy(dtype=float))
        acc  = df[acols].to_numpy(dtype=float)
        aw   = _rotate_vecs(q, acc)
        data[sensor]      = (t, aw[:, 2])
        acc_norms[sensor] = (t, np.linalg.norm(acc, axis=1))

    if len(data) < 2:
        log.warning("[%s] need both sensors — skipping az compare.", recording_name)
        return None

    phases = _protocol_phases(*acc_norms["sporsa"]) if "sporsa" in acc_norms else {}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             constrained_layout=True)

    ax = axes[0]
    for sensor, (t, az) in data.items():
        fin = mask_valid_plot_x(t) & np.isfinite(az)
        ax.plot(t[fin], az[fin], color=_COLORS[sensor], linewidth=0.6,
                alpha=0.75, label=_LABELS[sensor])
    ax.axhline(_GRAVITY, color="k", linestyle="--", linewidth=1.0,
               label=f"+g = {_GRAVITY}")
    ax.fill_between(
        [0, max(v[0][-1] for v in data.values())],
        _GRAVITY * 0.9, _GRAVITY * 1.1,
        color="k", alpha=0.06, label="±10 %",
    )
    _shade_phases(ax, phases, max(v[0][-1] for v in data.values()))
    ax.set_ylabel("az_world [m/s²]", fontsize=9)
    ax.set_ylim(-30, 50)
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([], [], color=_COLORS[s], label=_LABELS[s]) for s in data
    ] + [
        plt.Line2D([], [], color="k", linestyle="--", label=f"+g = {_GRAVITY}"),
        mpatches.Patch(facecolor="k", alpha=0.15, label="±10 %"),
    ] + _phase_legend_patches()
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    # Difference panel
    s_t, s_az = data.get("sporsa", (None, None))
    a_t, a_az = data.get("arduino", (None, None))
    if s_t is not None and a_t is not None:
        a_interp = np.interp(s_t, a_t[np.isfinite(a_az)], a_az[np.isfinite(a_az)])
        diff = a_interp - s_az
        stp, dfp = nan_mask_invalid_plot_x(s_t, diff)
        axes[1].plot(stp, dfp, color="#555", linewidth=0.6)
        axes[1].axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
        _shade_phases(axes[1], phases, s_t[-1])
        axes[1].set_ylabel("Arduino − Sporsa\n[m/s²]", fontsize=8)
        axes[1].set_ylim(-20, 20)
        axes[1].grid(True, alpha=0.2)

    # Acc-norm panel
    for sensor, (t, norm) in acc_norms.items():
        fin = mask_valid_plot_x(t) & np.isfinite(norm)
        axes[2].plot(t[fin], norm[fin], color=_COLORS[sensor],
                     linewidth=0.5, alpha=0.7, label=_LABELS[sensor])
    axes[2].axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
    _shade_phases(axes[2], phases, max(v[0][-1] for v in acc_norms.values()))
    axes[2].set_ylabel("‖acc_body‖ [m/s²]", fontsize=8)
    axes[2].set_ylim(0, 50)
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("Time [s]", fontsize=9)

    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "World-frame az comparison — both should track +9.81 m/s² at rest\n"
        "Middle panel: Arduino − Sporsa az_world (should be ≈ 0 at rest)",
        fontsize=10,
    )

    out = stage_dir / "orientation_az_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_protocol_analysis(
    recording_name: str,
    stage: str = "orientation",
) -> None:
    """Generate all three protocol-verification figures for a recording."""
    plot_sync(recording_name, stage)
    plot_relative_quat(recording_name, stage)
    plot_az_world_compare(recording_name, stage)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_orientation_protocol",
        description=(
            "Protocol-verification orientation plots "
            "(sync check, quaternion relative orientation, az comparison)."
        ),
    )
    parser.add_argument("recording_names", nargs="+",
                        help="Recording names (e.g. 2026-02-26_2).")
    parser.add_argument("--stage", default="orientation")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for rec in args.recording_names:
        plot_protocol_analysis(rec, stage=args.stage)


if __name__ == "__main__":
    main()
