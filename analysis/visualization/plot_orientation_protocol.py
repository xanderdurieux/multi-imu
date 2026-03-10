"""Protocol-aware orientation analysis for a controlled recording.

Produces three figures for any recording that has both sensor orientation
CSVs.  All relative-orientation metrics use **quaternion algebra** — not
Euler-angle subtraction — so the result is geometrically exact and free
from representation singularities.

Figures
-------
orientation_sync.png
    Both sensors' accelerometer norm overlaid on a shared time axis.
    Identical peaks → the sensors moved together (expected during
    aligned calibration shakes or co-aligned cycling).

orientation_relative_quat.png
    Proper quaternion relative orientation: q_rel = q_sporsa⁻¹ ⊗ q_arduino.
    Three panels:
      1. Angular distance from the initial relative pose (scalar, deg).
         0 → both sensors in the same relative orientation as at t=0.
         Rising value → the two sensors have diverged from their initial
         relative pose (e.g. head turns, helmet removed).
      2. Tilt components (ΔPitch, ΔRoll) of q_change — the rotation that
         describes how the *change* in relative orientation is distributed.
         These are observable from the accelerometer and meaningful even
         without a magnetometer.
      3. ΔYaw from q_change — interpretable as independent heading drift
         plus any deliberate head rotation.  Annotated as "unreliable
         without magnetometer" to manage expectations.

orientation_acc_world_compare.png
    World-frame az component for both sensors overlaid.
    At rest both should track +9.81 m/s²; any systematic offset reveals
    a calibration or orientation error.

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

log = logging.getLogger(__name__)

_SENSORS   = ("sporsa", "arduino")
_COLORS    = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_LABELS    = {"sporsa": "Sporsa (handlebar)", "arduino": "Arduino (helmet)"}
_GRAVITY   = 9.81


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
# Data loading
# ---------------------------------------------------------------------------

def _load(csv_path: Path):
    df = load_dataframe(csv_path)
    if df.empty:
        return None, None
    t = df["timestamp"].to_numpy(dtype=float)
    return df, (t - t[0]) / 1000.0


def _find_csv(stage_dir: Path, sensor: str) -> Path | None:
    for pat in (f"{sensor}*__complementary_orientation.csv",
                f"{sensor}*_orientation.csv"):
        hits = sorted(stage_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _get_quats_and_time(df, time_s) -> tuple[np.ndarray, np.ndarray]:
    q = df[["qw","qx","qy","qz"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(q), axis=1)
    return _qnorm(q), time_s, finite


# ---------------------------------------------------------------------------
# Figure 1 — acc norm overlay (synchronisation check)
# ---------------------------------------------------------------------------

def plot_sync(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Overlay both sensors' acc_body norm — shared peaks confirm synchronisation."""
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
        acc = df[acc_cols].to_numpy(dtype=float)
        norm = np.linalg.norm(acc, axis=1)
        data[sensor] = (t, norm)

    if len(data) < 2:
        log.warning("[%s] need both sensors — skipping sync plot.", recording_name)
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             constrained_layout=True)

    # Main overlay panel
    ax_main = axes[0]
    for sensor, (t, norm) in data.items():
        fin = np.isfinite(norm)
        ax_main.plot(t[fin], norm[fin], color=_COLORS[sensor], linewidth=0.7,
                     alpha=0.8, label=_LABELS[sensor])
    ax_main.axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.8,
                    alpha=0.5, label=f"|g| = {_GRAVITY}")
    ax_main.set_ylabel("‖acc_body‖ [m/s²]", fontsize=9)
    ax_main.set_ylim(0, min(60, max(
        max(np.nanmax(v[1]) for v in data.values()), 15)))
    ax_main.grid(True, alpha=0.25)
    ax_main.legend(fontsize=9, loc="upper right")

    # Separate per-sensor panels (clipped to show detail)
    for ax, sensor in zip(axes[1:], _SENSORS):
        if sensor not in data:
            ax.set_visible(False)
            continue
        t, norm = data[sensor]
        fin = np.isfinite(norm)
        ax.plot(t[fin], norm[fin], color=_COLORS[sensor], linewidth=0.6, alpha=0.8)
        ax.axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
        ax.set_ylabel(_LABELS[sensor].split()[0], fontsize=8)
        ax.set_ylim(0, 50)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time [s]", fontsize=9)
    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "Acc-norm synchronisation — matching peaks = sensors moved together",
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
    """Quaternion-based relative orientation (arduino relative to sporsa).

    q_rel(t) = q_sporsa(t)⁻¹ ⊗ q_arduino(t)

    All angles represent the rotation that takes the Sporsa body frame into
    the Arduino body frame.  When both sensors move together (head aligned
    with handlebar), q_rel should stay constant.

    To remove the static mounting offset, q_change(t) = q_rel_0⁻¹ ⊗ q_rel(t)
    is shown in the lower panels, where q_rel_0 is the mean relative
    quaternion computed over the first BASELINE_S seconds.
    """
    BASELINE_S = 5.0   # seconds used to estimate the initial relative pose

    stage_dir = recording_stage_dir(recording_name, stage)
    dfs = {}
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

    q_s = _qnorm(df_s[["qw","qx","qy","qz"]].to_numpy(dtype=float))
    q_a_raw = df_a[["qw","qx","qy","qz"]].to_numpy(dtype=float)

    # Resample arduino quaternions to sporsa time grid
    t_s_arr = t_s
    t_a_arr = t_a

    # For each component, interpolate arduino onto sporsa time axis
    overlap_min = max(t_s_arr[0], t_a_arr[0])
    overlap_max = min(t_s_arr[-1], t_a_arr[-1])
    mask_s = (t_s_arr >= overlap_min) & (t_s_arr <= overlap_max)
    mask_a = (t_a_arr >= overlap_min) & (t_a_arr <= overlap_max)

    if mask_s.sum() < 10 or mask_a.sum() < 10:
        log.warning("[%s] insufficient overlap — skipping.", recording_name)
        return None

    q_a_interp = np.zeros((mask_s.sum(), 4))
    for i in range(4):
        q_a_interp[:, i] = np.interp(
            t_s_arr[mask_s], t_a_arr[mask_a], q_a_raw[mask_a, i]
        )
    q_a_interp = _qnorm(q_a_interp)
    q_s_sub    = q_s[mask_s]
    t_ref      = t_s_arr[mask_s]

    # Relative quaternion: how does arduino differ from sporsa?
    q_rel = _qnorm(_qmul(_qconj(q_s_sub), q_a_interp))

    # Baseline: mean q_rel over first BASELINE_S seconds
    baseline_mask = t_ref <= (t_ref[0] + BASELINE_S)
    if baseline_mask.sum() < 2:
        baseline_mask[:5] = True
    q_rel_0 = _qnorm(np.nanmean(q_rel[baseline_mask], axis=0, keepdims=True))

    # Change relative to baseline
    q_change = _qnorm(_qmul(_qconj(q_rel_0), q_rel))

    # Angular distance from baseline (scalar, deg) — 0 = same relative pose as start
    ang_dist = 2.0 * np.degrees(np.arccos(np.clip(np.abs(q_change[:, 0]), 0, 1)))

    # Directional decomposition of q_change
    yaw_ch, pitch_ch, roll_ch = _euler_from_quat(q_change)
    yaw_ch   = _unwrap_deg(yaw_ch)
    pitch_ch = _unwrap_deg(pitch_ch)
    roll_ch  = _unwrap_deg(roll_ch)

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                             constrained_layout=True)

    # Panel 1: angular distance
    ax = axes[0]
    ax.fill_between(t_ref, ang_dist, alpha=0.25, color="#555")
    ax.plot(t_ref, ang_dist, color="#333", linewidth=0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Angular distance\nfrom baseline [°]", fontsize=9)
    ax.grid(True, alpha=0.25)
    _annotate_expected(ax, t_ref[-1])

    # Panel 2: ΔPitch
    ax = axes[1]
    ax.plot(t_ref, pitch_ch, color="#e67e22", linewidth=0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("ΔPitch change [°]", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 3: ΔRoll
    ax = axes[2]
    ax.plot(t_ref, roll_ch, color="#27ae60", linewidth=0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("ΔRoll change [°]", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 4: ΔYaw (with drift warning)
    ax = axes[3]
    ax.plot(t_ref, yaw_ch, color="#8e44ad", linewidth=0.7)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("ΔYaw change [°]", fontsize=9)
    ax.text(0.98, 0.05, "Yaw drifts without\nmagnetometer",
            transform=ax.transAxes, fontsize=8, color="#8e44ad",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#8e44ad", alpha=0.7))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time [s]", fontsize=9)

    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "Quaternion relative orientation change from baseline  "
        "(arduino − sporsa frame, offset removed)\n"
        "0° = sensors in same relative pose as at recording start",
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
    """Both sensors' world-frame az (Up) component overlaid.

    At rest: should both track +9.81 m/s².  Any systematic vertical offset
    between the two traces reveals a calibration or orientation error.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    data = {}
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
        q   = _qnorm(df[qcols].to_numpy(dtype=float))
        acc = df[acols].to_numpy(dtype=float)
        aw  = _rotate_vecs(q, acc)
        data[sensor] = (t, aw[:, 2])   # az_world

    if len(data) < 2:
        log.warning("[%s] need both sensors — skipping az compare.", recording_name)
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             constrained_layout=True)

    # Overlay
    ax = axes[0]
    for sensor, (t, az) in data.items():
        fin = np.isfinite(az)
        ax.plot(t[fin], az[fin], color=_COLORS[sensor], linewidth=0.6,
                alpha=0.75, label=_LABELS[sensor])
    ax.axhline(_GRAVITY, color="k", linestyle="--", linewidth=1.0,
               label=f"+g = {_GRAVITY}")
    ax.fill_between(
        [0, max(v[0][-1] for v in data.values())],
        _GRAVITY * 0.9, _GRAVITY * 1.1,
        color="k", alpha=0.06, label="±10 %",
    )
    ax.set_ylabel("az_world [m/s²]", fontsize=9)
    ax.set_ylim(-30, 50)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    # Difference panel
    s_t, s_az = data.get("sporsa", (None, None))
    a_t, a_az = data.get("arduino", (None, None))
    if s_t is not None and a_t is not None:
        t_common = s_t
        a_interp = np.interp(t_common, a_t[np.isfinite(a_az)],
                             a_az[np.isfinite(a_az)])
        diff = a_interp - s_az
        axes[1].plot(t_common, diff, color="#555", linewidth=0.6)
        axes[1].axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
        axes[1].set_ylabel("Arduino − Sporsa\n[m/s²]", fontsize=8)
        axes[1].set_ylim(-20, 20)
        axes[1].grid(True, alpha=0.2)

    # Acc-norm overlay
    for sensor, (t, _) in data.items():
        p = _find_csv(stage_dir, sensor)
        df, _ = _load(p)
        acc = df[["ax","ay","az"]].to_numpy(dtype=float)
        norm = np.linalg.norm(acc, axis=1)
        fin = np.isfinite(norm)
        axes[2].plot(t[fin], norm[fin], color=_COLORS[sensor],
                     linewidth=0.5, alpha=0.7, label=_LABELS[sensor])
    axes[2].axhline(_GRAVITY, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
    axes[2].set_ylabel("‖acc_body‖ [m/s²]", fontsize=8)
    axes[2].set_ylim(0, 50)
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("Time [s]", fontsize=9)

    fig.suptitle(
        f"{recording_name} / {stage}\n"
        "World-frame az comparison — both should track +9.81 m/s² at rest\n"
        "Middle panel: Arduino − Sporsa az_world (should be ≈0 at rest)",
        fontsize=10,
    )

    out = stage_dir / "orientation_az_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def _annotate_expected(ax, t_max: float) -> None:
    """Add a subtle 'expected: near 0° when aligned' annotation."""
    ax.text(0.01, 0.96,
            "≈ 0° when sensors co-aligned  |  rises when head/handlebar diverge",
            transform=ax.transAxes, fontsize=7.5, color="#666",
            va="top", ha="left")


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
