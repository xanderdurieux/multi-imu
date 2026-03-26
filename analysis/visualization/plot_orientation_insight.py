"""Orientation insight plots for recording verification.

Four targeted plots that let you check whether the calibration,
synchronisation, and orientation filter produced correct results.

1.  **gravity_accuracy.png**
    At every auto-detected static calibration stop the world-frame
    az component (az_world) is summarised as mean ± std and compared to
    the expected +9.81 m/s².  A well-calibrated, correctly filtered sensor
    should hit within ± 0.2 m/s² at rest.

2.  **alignment_quality.png**
    During the helmet-on-handlebar alignment phases the relative orientation
    (helmet vs handlebar) should be ≈ 0°.  Panels show angular distance,
    ΔPitch, and ΔRoll for each detected pre/post alignment window.

3.  **head_movement_peaks.png**
    ΔYaw and ΔPitch (quaternion relative orientation) with auto-detected
    peaks annotated.  Useful for recordings with known repeated head
    movements (e.g. rec 4: 5× left/right + 5× down, rec 5: 5× shoulder).

4.  **fall_analysis_<sensor>.png**
    60-s window around the auto-detected fall event (largest acc spike).
    Panels: acc_norm, gyro_norm, pitch/roll angles, world-frame az.
    For recordings 9 (fall right) and 10 (fall left).

CLI::

    # Gravity + alignment for every recording in a session
    uv run python -m visualization.plot_orientation_insight 2026-02-26 --all

    # Head-movement peaks for recording 4
    uv run python -m visualization.plot_orientation_insight 2026-02-26_4 --head-movements

    # Fall analysis for recording 9
    uv run python -m visualization.plot_orientation_insight 2026-02-26_9 --fall
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import find_peaks

from common import load_dataframe, recordings_root
from ._utils import mask_valid_plot_x, nan_mask_invalid_plot_x
from ._utils import resolve_stage_dir

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_COLORS  = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_LABELS  = {"sporsa": "Sporsa (handlebar)", "arduino": "Arduino (helmet)"}
_GRAVITY = 9.81

# Mounting rotation: +90° around Z (Sporsa X=Front → Arduino Y=Front)
_sq2 = float(np.sqrt(2.0)) / 2.0
_MOUNTING_QUAT = np.array([_sq2, 0.0, 0.0, _sq2])  # [w, x, y, z]


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    c = q.copy()
    c[..., 1:] *= -1
    return c


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 0, q / n, q)


def _euler_from_quat(q: np.ndarray):
    """Return (yaw, pitch, roll) in degrees — ZYX convention."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return yaw, pitch, roll


def _rotate_vecs(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    v_q = np.concatenate([np.zeros((len(vecs), 1)), vecs], axis=1)
    return _qmul(_qmul(quats, v_q), _qconj(quats))[:, 1:]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(csv_path: Path):
    df = load_dataframe(csv_path)
    if df.empty:
        return None, None
    t = df["timestamp"].to_numpy(dtype=float)
    return df, (t - t[0]) / 1000.0


def _find_csv(stage_dir: Path, sensor: str) -> Path | None:
    for method in ("complementary", "madgwick"):
        c = stage_dir / method / f"{sensor}_orientation.csv"
        if c.exists():
            return c
    for pat in (f"{sensor}*__complementary_orientation.csv", f"{sensor}*_orientation.csv"):
        hits = sorted(stage_dir.glob(pat))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Static-period detection
# ---------------------------------------------------------------------------

def _detect_static_windows(
    t: np.ndarray,
    norm: np.ndarray,
    threshold: float = 10.5,
    min_duration_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Return time windows where acc_norm stays below threshold for ≥ min_duration_s."""
    below = np.isfinite(norm) & (norm < threshold)
    changes = np.diff(below.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    windows = []
    for s, e in zip(starts, ends):
        t0, t1 = float(t[s]), float(t[min(e, len(t) - 1)])
        if t1 - t0 >= min_duration_s:
            windows.append((t0, t1))
    return windows


# ---------------------------------------------------------------------------
# Protocol phase helpers (inline to avoid cross-module import)
# ---------------------------------------------------------------------------

def _detect_shake_clusters(t, norm, shake_threshold=18.0, cluster_gap_s=3.0):
    above = np.isfinite(norm) & (norm > shake_threshold)
    if not above.any():
        return []
    changes = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    segments = [(float(t[s]), float(t[min(e, len(t)-1)])) for s, e in zip(starts, ends)]
    if not segments:
        return []
    merged = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] < cluster_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _protocol_phases(t, norm):
    clusters = _detect_shake_clusters(t, norm)
    buf = 6.0
    if len(clusters) < 2:
        return {"pre_align": None, "post_align": None, "ride": None, "clusters": clusters}
    return {
        "pre_align":  (max(t[0], clusters[0][0] - buf),  min(t[-1], clusters[0][1]  + buf)),
        "post_align": (max(t[0], clusters[-1][0] - buf), min(t[-1], clusters[-1][1] + buf)),
        "ride":       (clusters[0][1], clusters[-1][0]),
        "clusters":   clusters,
    }


# ---------------------------------------------------------------------------
# Quaternion column compatibility
# ---------------------------------------------------------------------------


def _qcols_for_df(df) -> list[str] | None:
    """Quaternion columns in w,x,y,z order (supports qw..qz and q0..q3)."""
    cols = getattr(df, "columns", None)
    if cols is None:
        return None
    if all(c in cols for c in ("qw", "qx", "qy", "qz")):
        return ["qw", "qx", "qy", "qz"]
    if all(c in cols for c in ("q0", "q1", "q2", "q3")):
        return ["q0", "q1", "q2", "q3"]
    return None


# ---------------------------------------------------------------------------
# Shared relative-orientation computation
# ---------------------------------------------------------------------------

def _compute_relative_orientation(recording_name: str, stage_dir: Path):
    """Compute q_change, t_ref, and protocol phases. Returns None on failure."""
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
        return None

    df_s, t_s = dfs["sporsa"]
    df_a, t_a = dfs["arduino"]

    qcols_s = _qcols_for_df(df_s)
    qcols_a = _qcols_for_df(df_a)
    if qcols_s is None or qcols_a is None:
        return None

    q_s     = _qnorm(df_s[qcols_s].to_numpy(dtype=float))
    q_a_raw = df_a[qcols_a].to_numpy(dtype=float)

    overlap_min = max(t_s[0], t_a[0])
    overlap_max = min(t_s[-1], t_a[-1])
    mask_s = (t_s >= overlap_min) & (t_s <= overlap_max)
    mask_a = (t_a >= overlap_min) & (t_a <= overlap_max)
    if mask_s.sum() < 10 or mask_a.sum() < 10:
        return None

    q_a_interp = np.zeros((mask_s.sum(), 4))
    for i in range(4):
        q_a_interp[:, i] = np.interp(t_s[mask_s], t_a[mask_a], q_a_raw[mask_a, i])
    q_a_interp = _qnorm(q_a_interp)
    q_s_sub    = q_s[mask_s]
    t_ref      = t_s[mask_s]

    q_rel    = _qnorm(_qmul(_qconj(q_s_sub), q_a_interp))
    q_change = _qnorm(_qmul(_qconj(_MOUNTING_QUAT), q_rel))
    ang_dist = 2.0 * np.degrees(np.arccos(np.clip(np.abs(q_change[:, 0]), 0, 1)))
    yaw_ch, pitch_ch, roll_ch = _euler_from_quat(q_change)

    phases = {"pre_align": None, "post_align": None, "ride": None, "clusters": []}
    acc_cols = ["ax", "ay", "az"]
    if all(c in df_s.columns for c in acc_cols):
        acc_s = df_s[acc_cols].to_numpy(dtype=float)
        norm_s = np.linalg.norm(acc_s, axis=1)
        phases = _protocol_phases(t_s[mask_s], norm_s[mask_s])

    return {
        "t_ref": t_ref,
        "ang_dist": ang_dist,
        "yaw": yaw_ch,
        "pitch": pitch_ch,
        "roll": roll_ch,
        "phases": phases,
        "dfs": dfs,
    }


# ---------------------------------------------------------------------------
# 1. Gravity accuracy at static segments
# ---------------------------------------------------------------------------

def plot_gravity_accuracy(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """At each static calibration stop, verify az_world ≈ +9.81 m/s².

    A well-calibrated and correctly filtered sensor must track gravity to
    within ± 0.2 m/s² whenever the bike is stationary.  Errors reveal
    calibration bias, world-frame rotation issues, or filter initialisation
    problems.
    """
    stage_dir = resolve_stage_dir(recording_name, stage)
    results: dict[str, list] = {}

    for sensor in _SENSORS:
        p = _find_csv(stage_dir, sensor)
        if p is None:
            continue
        df, t = _load(p)
        if df is None:
            continue
        acc_cols = ["ax", "ay", "az"]
        if not all(c in df.columns for c in acc_cols):
            continue
        qcols = _qcols_for_df(df)
        if qcols is None:
            continue

        acc = df[acc_cols].to_numpy(dtype=float)
        q   = _qnorm(df[qcols].to_numpy(dtype=float))
        acc_norm = np.linalg.norm(acc, axis=1)
        aw       = _rotate_vecs(q, acc)

        windows  = _detect_static_windows(t, acc_norm)
        if not windows:
            log.info("[%s/%s] %s: no static windows.", recording_name, stage, sensor)
            continue

        segs = []
        for w_t0, w_t1 in windows:
            m = (t >= w_t0) & (t <= w_t1)
            if m.sum() < 5:
                continue
            az_vals = aw[m, 2]
            segs.append({
                "t_mid": (w_t0 + w_t1) / 2,
                "mean":  float(np.nanmean(az_vals)),
                "std":   float(np.nanstd(az_vals)),
                "error": float(np.nanmean(az_vals)) - _GRAVITY,
            })
        if segs:
            results[sensor] = segs

    if not results:
        log.warning("[%s/%s] no static segments found — skipping gravity accuracy.", recording_name, stage)
        return None

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), constrained_layout=True, squeeze=False)

    for ax_row, (sensor, segs) in zip(axes, results.items()):
        ax = ax_row[0]
        t_mids = np.array([s["t_mid"] for s in segs], dtype=float)
        means  = np.array([s["mean"] for s in segs], dtype=float)
        stds   = np.array([s["std"] for s in segs], dtype=float)
        errors = [s["error"] for s in segs]
        xm = mask_valid_plot_x(t_mids)
        if not xm.any():
            continue
        t_mids, means, stds = t_mids[xm], means[xm], stds[xm]
        errors = [errors[i] for i in np.flatnonzero(xm)]
        segs_good = [segs[i] for i in np.flatnonzero(xm)]

        ax.axhline(_GRAVITY, color="k", linestyle="--", linewidth=1.0,
                   label=f"+g = {_GRAVITY:.2f} m/s²")
        ax.fill_between(
            [t_mids[0] - 10, t_mids[-1] + 10],
            _GRAVITY * 0.98, _GRAVITY * 1.02,
            color="k", alpha=0.05, label="±2 %",
        )
        ax.errorbar(t_mids, means, yerr=stds, fmt="o",
                    color=_COLORS.get(sensor, "#555"), capsize=4,
                    markersize=6, label=f"{_LABELS.get(sensor, sensor)} — mean ± std")
        ax.set_xlim(t_mids[0] - 15, t_mids[-1] + 15)
        ax.set_xlabel("Time at segment mid-point [s]", fontsize=9)
        ax.set_ylabel("az_world [m/s²]", fontsize=9)
        ax.set_ylim(_GRAVITY - 3, _GRAVITY + 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        rms = float(np.sqrt(np.mean([e**2 for e in errors])))
        ax.set_title(
            f"{_LABELS.get(sensor, sensor)}  |  {len(segs_good)} static segments  |  "
            f"RMS error vs +g: {rms:.3f} m/s²",
            fontsize=9,
        )

        for s in segs_good:
            c = "#27ae60" if abs(s["error"]) < 0.2 else (
                "#e67e22" if abs(s["error"]) < 0.5 else "#e74c3c")
            ax.annotate(
                f"{s['error']:+.2f}",
                xy=(s["t_mid"], s["mean"]),
                xytext=(s["t_mid"], s["mean"] + s["std"] + 0.2),
                fontsize=7, color=c, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.6),
            )

    fig.suptitle(
        f"{recording_name} / {stage}  —  Gravity tracking accuracy at static segments\n"
        "az_world should be ≈ +9.81 m/s² when stationary  |  "
        "Green ≤ 0.2 m/s²  ·  Orange ≤ 0.5  ·  Red > 0.5",
        fontsize=10,
    )
    out = stage_dir / "gravity_accuracy.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# 2. Alignment quality
# ---------------------------------------------------------------------------

def plot_alignment_quality(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Relative orientation during helmet-on-handlebar alignment phases.

    When the helmet is co-aligned with the handlebar (between calibration
    shakes) the relative angular distance should be ≈ 0°, ΔPitch ≈ 0°,
    and ΔRoll ≈ 0°.  Persistent offsets indicate a mounting-offset error
    or filter warm-up drift.  The pre- and post-ride phases are shown
    side by side for comparison.
    """
    stage_dir = resolve_stage_dir(recording_name, stage)
    result    = _compute_relative_orientation(recording_name, stage_dir)
    if result is None:
        log.warning("[%s/%s] cannot compute relative orientation — skipping alignment quality.", recording_name, stage)
        return None

    phases   = result["phases"]
    t_ref    = result["t_ref"]
    ang_dist = result["ang_dist"]
    pitch_ch = result["pitch"]
    roll_ch  = result["roll"]

    align_windows: list[tuple[float, float, str]] = []
    if phases.get("pre_align"):
        align_windows.append((*phases["pre_align"], "Pre-ride alignment"))
    if phases.get("post_align"):
        align_windows.append((*phases["post_align"], "Post-ride alignment"))

    if not align_windows:
        log.warning("[%s/%s] no alignment phases detected — skipping alignment quality.", recording_name, stage)
        return None

    n = len(align_windows)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.5 * n),
                              constrained_layout=True, squeeze=False)

    summary_parts = []
    for row, (w_t0, w_t1, label) in enumerate(align_windows):
        m = (t_ref >= w_t0) & (t_ref <= w_t1)
        if m.sum() < 5:
            continue
        tc = t_ref[m]

        col_specs = [
            (ang_dist, "Angular distance [°]", "#555555"),
            (pitch_ch, "ΔPitch [°]",           "#e67e22"),
            (roll_ch,  "ΔRoll [°]",             "#27ae60"),
        ]
        for col, (signal, ylabel, color) in enumerate(col_specs):
            ax = axes[row][col]
            sig = signal[m]
            tp, sp = nan_mask_invalid_plot_x(tc, sig)
            ax.fill_between(tp, sp, alpha=0.22, color=color)
            ax.plot(tp, sp, color=color, linewidth=0.8)
            ax.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
            mean_v = float(np.nanmean(signal[m]))
            std_v  = float(np.nanstd(signal[m]))
            ax.set_title(f"{label}  —  {ylabel}\nmean = {mean_v:.1f}°,  std = {std_v:.1f}°",
                         fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Time [s]", fontsize=9)
            ax.grid(True, alpha=0.3)
            lim = max(20.0, float(np.nanmax(np.abs(signal[m]))) * 1.15)
            ax.set_ylim(0 if col == 0 else -lim, lim)

        summary_parts.append(
            f"{label}: ∠{np.nanmean(ang_dist[m]):.1f}° ± {np.nanstd(ang_dist[m]):.1f}°"
        )

    fig.suptitle(
        f"{recording_name} / {stage}  —  Alignment quality\n"
        "Helmet co-aligned with handlebar: angular distance and ΔPitch / ΔRoll should be ≈ 0°\n"
        + "  |  ".join(summary_parts),
        fontsize=10,
    )
    out = stage_dir / "alignment_quality.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# 3. Head movement peaks
# ---------------------------------------------------------------------------

def plot_head_movement_peaks(
    recording_name: str,
    stage: str = "orientation",
    min_yaw_peak_deg: float = 15.0,
    min_pitch_peak_deg: float = 10.0,
) -> Path | None:
    """ΔYaw and ΔPitch relative orientation with auto-detected peaks annotated.

    Useful for recordings with known deliberate head movements (e.g. rec 4:
    5× left/right turns and 5× look-down, rec 5: 5× shoulder looks).
    Detected peak count should match the number noted in the protocol.

    Grey bands mark auto-detected static calibration segments.
    """
    stage_dir = resolve_stage_dir(recording_name, stage)
    result    = _compute_relative_orientation(recording_name, stage_dir)
    if result is None:
        log.warning("[%s/%s] cannot compute relative orientation — skipping head movement peaks.", recording_name, stage)
        return None

    t_ref    = result["t_ref"]
    yaw_ch   = result["yaw"]
    pitch_ch = result["pitch"]
    roll_ch  = result["roll"]
    ang_dist = result["ang_dist"]
    phases   = result["phases"]

    # Static segments from sporsa acc_norm for shading
    acc_s    = result["dfs"]["sporsa"][0][["ax", "ay", "az"]].to_numpy(dtype=float)
    norm_s   = np.linalg.norm(acc_s, axis=1)
    static_windows = _detect_static_windows(t_ref, norm_s[: len(t_ref)])

    def _signed_peaks(signal, min_prom, min_dist_s=5.0):
        dt = float(np.median(np.diff(t_ref)))
        dist = max(1, int(min_dist_s / dt))
        pos, _ = find_peaks( signal, prominence=min_prom, distance=dist)
        neg, _ = find_peaks(-signal, prominence=min_prom, distance=dist)
        return sorted(list(pos) + list(neg))

    yaw_peaks   = _signed_peaks(yaw_ch,   min_yaw_peak_deg)
    pitch_peaks = _signed_peaks(pitch_ch, min_pitch_peak_deg)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, constrained_layout=True)

    def _draw_panel(ax, signal, peaks, color, ylabel):
        tp, sp = nan_mask_invalid_plot_x(t_ref, signal)
        # Static shading
        for w0, w1 in static_windows:
            ax.axvspan(w0, w1, color="#aaaaaa", alpha=0.12, zorder=0)
        # Alignment phase shading
        if phases.get("pre_align"):
            ax.axvspan(*phases["pre_align"], color="#2ecc71", alpha=0.10, zorder=0)
        if phases.get("post_align"):
            ax.axvspan(*phases["post_align"], color="#2ecc71", alpha=0.10, zorder=0)
        if phases.get("ride"):
            ax.axvspan(*phases["ride"], color="#3498db", alpha=0.05, zorder=0)

        ax.plot(tp, sp, color=color, linewidth=0.75, alpha=0.9)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.4)

        for idx in peaks:
            val = signal[idx]
            if not (mask_valid_plot_x(np.array([t_ref[idx]]))[0] and np.isfinite(val)):
                continue
            ax.axvline(t_ref[idx], color=color, linewidth=0.8, alpha=0.35, linestyle=":")
            ax.annotate(
                f"{val:+.0f}°",
                xy=(t_ref[idx], val),
                xytext=(t_ref[idx], val + np.sign(val) * 7),
                fontsize=6.5, color=color, ha="center", va="bottom" if val > 0 else "top",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
            )
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(f"{ylabel}\n({len(peaks)} peaks detected)", fontsize=9)

    _draw_panel(axes[0], yaw_ch,   yaw_peaks,   "#8e44ad", f"ΔYaw [°]  head turn L/R  (≥ {min_yaw_peak_deg:.0f}°)")
    _draw_panel(axes[1], pitch_ch, pitch_peaks, "#e67e22", f"ΔPitch [°]  head nod down/up  (≥ {min_pitch_peak_deg:.0f}°)")

    # Bottom panel: angular distance overview
    for w0, w1 in static_windows:
        axes[2].axvspan(w0, w1, color="#aaaaaa", alpha=0.12, zorder=0)
    if phases.get("pre_align"):
        axes[2].axvspan(*phases["pre_align"], color="#2ecc71", alpha=0.10, zorder=0)
    if phases.get("post_align"):
        axes[2].axvspan(*phases["post_align"], color="#2ecc71", alpha=0.10, zorder=0)
    if phases.get("ride"):
        axes[2].axvspan(*phases["ride"], color="#3498db", alpha=0.05, zorder=0)
    ta, ad = nan_mask_invalid_plot_x(t_ref, ang_dist)
    axes[2].fill_between(ta, ad, alpha=0.20, color="#555")
    axes[2].plot(ta, ad, color="#555", linewidth=0.7)
    axes[2].set_ylabel("Angular distance [°]", fontsize=9)
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlabel("Time [s]", fontsize=9)

    legend_patches = [
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.3, label="Static segment"),
        mpatches.Patch(facecolor="#2ecc71", alpha=0.25, label="Alignment (helmet on handlebar)"),
        mpatches.Patch(facecolor="#3498db", alpha=0.15, label="Riding"),
    ]
    axes[0].legend(handles=legend_patches, fontsize=8, loc="upper right")

    fig.suptitle(
        f"{recording_name} / {stage}  —  Head movement peaks\n"
        f"ΔYaw: {len(yaw_peaks)} peaks ≥ {min_yaw_peak_deg:.0f}°  |  "
        f"ΔPitch: {len(pitch_peaks)} peaks ≥ {min_pitch_peak_deg:.0f}°\n"
        "0° = head and handlebar co-aligned  |  peaks = deliberate head movements",
        fontsize=10,
    )
    out = stage_dir / "head_movement_peaks.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# 4. Fall analysis
# ---------------------------------------------------------------------------

def plot_fall_analysis(
    recording_name: str,
    stage: str = "orientation",
    sensor: str = "arduino",
    context_s: float = 30.0,
) -> Path | None:
    """60-s window around the auto-detected fall event.

    The fall is identified as the largest acc_norm spike after the first 5 s
    (to skip calibration shakes).  Four panels show the physical change at
    impact: acc spike, gyro rate spike, pitch/roll step, and world-frame az.

    Falls back to the other sensor if the requested one has no data.
    """
    stage_dir = resolve_stage_dir(recording_name, stage)
    p = _find_csv(stage_dir, sensor)
    if p is None:
        fallback = "sporsa" if sensor == "arduino" else "arduino"
        p = _find_csv(stage_dir, fallback)
        if p is not None:
            log.info("[%s] %s not found, using %s for fall analysis.", recording_name, sensor, fallback)
            sensor = fallback
    if p is None:
        log.warning("[%s/%s] no orientation CSV found for fall analysis.", recording_name, stage)
        return None

    df, t = _load(p)
    if df is None:
        return None

    acc_cols = ["ax", "ay", "az"]
    if not all(c in df.columns for c in acc_cols):
        log.warning("[%s/%s] missing acc columns for fall analysis.", recording_name, stage)
        return None
    qcols = _qcols_for_df(df)
    if qcols is None:
        log.warning("[%s/%s] missing quaternion columns for fall analysis.", recording_name, stage)
        return None

    acc      = df[acc_cols].to_numpy(dtype=float)
    q        = _qnorm(df[qcols].to_numpy(dtype=float))
    acc_norm = np.linalg.norm(acc, axis=1)
    aw       = _rotate_vecs(q, acc)
    has_gyro = all(c in df.columns for c in ["gx", "gy", "gz"])
    gyro_norm = np.linalg.norm(df[["gx", "gy", "gz"]].to_numpy(dtype=float), axis=1) if has_gyro else None
    _, pitch, roll = _euler_from_quat(q)

    # Detect fall: largest acc spike after t > 5 s
    post = t > 5.0
    if not post.any():
        log.warning("[%s] recording too short for fall detection.", recording_name)
        return None
    idx_in_post = int(np.argmax(acc_norm[post]))
    idx_fall    = int(np.where(post)[0][idx_in_post])
    t_fall      = float(t[idx_fall])

    t0   = max(t[0], t_fall - context_s)
    t1   = min(t[-1], t_fall + context_s)
    mask = (t >= t0) & (t <= t1) & mask_valid_plot_x(t)
    tc   = t[mask]

    n_panels = 4 if has_gyro else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels),
                              sharex=True, constrained_layout=True)

    def _vline(ax, label=None):
        ax.axvline(t_fall, color="#e74c3c", linewidth=1.5, linestyle="--",
                   alpha=0.85, label=label or f"Fall ≈ {t_fall:.1f} s")
        ax.grid(True, alpha=0.25)

    panel = 0
    # Acc norm
    ax = axes[panel]; panel += 1
    ax.plot(tc, acc_norm[mask], color=_COLORS.get(sensor, "#555"), linewidth=0.8)
    ax.axhline(_GRAVITY, color="k", linestyle=":", linewidth=0.8, alpha=0.5,
               label=f"|g| = {_GRAVITY:.2f}")
    _vline(ax, f"Fall detected at t = {t_fall:.1f} s  (peak = {acc_norm[idx_fall]:.1f} m/s²)")
    ax.set_ylabel("‖acc_body‖ [m/s²]", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    # Gyro norm
    if has_gyro:
        ax = axes[panel]; panel += 1
        ax.plot(tc, gyro_norm[mask], color="#888888", linewidth=0.7)
        _vline(ax)
        ax.set_ylabel("‖gyro‖ [°/s]", fontsize=9)

    # Pitch and Roll
    ax = axes[panel]; panel += 1
    ax.plot(tc, pitch[mask], color="#e67e22", linewidth=0.9, label="Pitch")
    ax.plot(tc, roll[mask],  color="#27ae60", linewidth=0.9, label="Roll")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
    _vline(ax)
    ax.set_ylabel("Angle [°]", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    # Pre/post pitch+roll values annotated
    pre_idx  = max(0, idx_fall - int(2.0 / float(np.median(np.diff(t)))))
    post_idx = min(len(t) - 1, idx_fall + int(5.0 / float(np.median(np.diff(t)))))
    for val, label, col in [
        (pitch[pre_idx],  f"pitch before\n{pitch[pre_idx]:.0f}°",  "#e67e22"),
        (pitch[post_idx], f"pitch after\n{pitch[post_idx]:.0f}°",  "#e67e22"),
        (roll[pre_idx],   f"roll before\n{roll[pre_idx]:.0f}°",    "#27ae60"),
        (roll[post_idx],  f"roll after\n{roll[post_idx]:.0f}°",    "#27ae60"),
    ]:
        t_ann = t[pre_idx] if "before" in label else t[post_idx]
        ax.annotate(label, xy=(t_ann, val),
                    xytext=(t_ann, val + 15 * np.sign(val or 1)),
                    fontsize=7, color=col, ha="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5))

    # World-frame az
    ax = axes[panel]; panel += 1
    ax.plot(tc, aw[mask, 2], color=_COLORS.get(sensor, "#555"), linewidth=0.7)
    ax.axhline(_GRAVITY, color="k", linestyle=":", linewidth=0.8, alpha=0.5,
               label=f"+g = {_GRAVITY:.2f}")
    _vline(ax)
    ax.set_ylabel("az_world [m/s²]", fontsize=9)
    ax.set_ylim(-30, 30)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Time [s]", fontsize=9)

    # Summary stats
    pre_window  = (t >= t_fall - 3) & (t < t_fall)
    post_window = (t > t_fall + 2) & (t <= t_fall + 10)
    pre_az  = float(np.nanmean(aw[pre_window,  2])) if pre_window.any()  else float("nan")
    post_az = float(np.nanmean(aw[post_window, 2])) if post_window.any() else float("nan")
    dpitch  = float(pitch[post_idx] - pitch[pre_idx])
    droll   = float(roll[post_idx]  - roll[pre_idx])

    fig.suptitle(
        f"{recording_name} / {stage}  —  Fall analysis ({sensor})\n"
        f"Fall at t = {t_fall:.1f} s  |  peak acc = {acc_norm[idx_fall]:.1f} m/s²\n"
        f"az_world: {pre_az:.2f} → {post_az:.2f} m/s²  |  "
        f"ΔPitch = {dpitch:+.1f}°  |  ΔRoll = {droll:+.1f}°",
        fontsize=10,
    )
    out = stage_dir / f"fall_analysis_{sensor}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def plot_insight_stage(
    recording_name: str,
    stage: str = "orientation",
    *,
    fall: bool = False,
    head_movements: bool = False,
) -> None:
    """Generate all applicable insight plots for a single recording."""
    plot_gravity_accuracy(recording_name, stage)
    plot_alignment_quality(recording_name, stage)
    if fall:
        plot_fall_analysis(recording_name, stage=stage)
    if head_movements:
        plot_head_movement_peaks(recording_name, stage)


# Recording-specific defaults based on 2026-02-26 session notes
_FALL_SUFFIXES      = frozenset({"9", "10"})
_HEADMOVE_SUFFIXES  = frozenset({"4", "5"})


def plot_insight_session(session_name: str, stage: str = "orientation") -> None:
    """Generate insight plots for all recordings in a session."""
    root = recordings_root()
    recordings = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_name}_")
    )
    if not recordings:
        log.warning("No recordings found for session '%s'.", session_name)
        return
    for rec in recordings:
        suffix = rec.split("_")[-1]
        plot_insight_stage(
            rec, stage,
            fall=suffix in _FALL_SUFFIXES,
            head_movements=suffix in _HEADMOVE_SUFFIXES,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_orientation_insight",
        description="Orientation insight and verification plots.",
    )
    parser.add_argument("names", nargs="+",
                        help="Recording name(s) or session prefix with --all.")
    parser.add_argument("--all", action="store_true", dest="all_recordings",
                        help="Treat each NAME as a session prefix.")
    parser.add_argument("--stage", default="orientation",
                        help="Orientation stage directory (default: orientation).")
    parser.add_argument("--fall", action="store_true",
                        help="Include fall analysis plot.")
    parser.add_argument("--head-movements", action="store_true", dest="head_movements",
                        help="Include head movement peak detection plot.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for name in args.names:
        if args.all_recordings:
            plot_insight_session(name, args.stage)
        else:
            plot_insight_stage(
                name, args.stage,
                fall=args.fall,
                head_movements=args.head_movements,
            )


if __name__ == "__main__":
    main()
