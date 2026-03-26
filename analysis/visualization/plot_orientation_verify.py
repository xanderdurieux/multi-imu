"""Orientation verification plots — gravity tracking and sensor-axes sphere.

Produces two figures per orientation CSV:

1. **<stem>_gravity_world.png** — "Does the filter correctly track gravity?"
   ─────────────────────────────────────────────────────────────────────────
   World-frame accelerometer components (ax_w, ay_w, az_w) obtained by
   rotating the body-frame acc through the filter quaternion.
   At rest, a correct orientation estimate gives ax_w ≈ ay_w ≈ 0 and
   az_w ≈ +9.81 m/s².  Deviations reveal filter drift or calibration error.
   The bottom panel shows the total acc norm vs. the expected 9.81 m/s²
   so impulsive events (bumps, braking) are also visible.

2. **<stem>_axes_sphere.png** — "Where does each sensor axis point over time?"
   ─────────────────────────────────────────────────────────────────────────
   Three panels on a unit sphere in the ENU world frame showing the
   time-coloured trajectory of the sensor's body X (forward/right),
   Y (left), and Z (up) axes.  For a handlebar sensor at rest on an upright
   bike, body Z should cluster near world +Z.  For a helmet sensor, looking
   left/right sweeps body X along the horizontal equator.

In addition, an orchestrator `plot_orientation_verify_stage` generates a
**combined comparison figure** (`orientation_verify_compare.png`) overlaying
sporsa and arduino on shared sphere panels (for the complementary filter).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from common import load_dataframe
from ._utils import mask_valid_plot_x, nan_mask_invalid_plot_x, resolve_stage_dir

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_SENSOR_LABELS = {"sporsa": "Sporsa (handlebar)", "arduino": "Arduino (helmet)"}
_AXIS_COLORS = {"X": "#d62728", "Y": "#2ca02c", "Z": "#1f77b4"}
_GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path):
    """Return (df, time_s) or (empty_df, empty_series) on failure."""
    df = load_dataframe(csv_path)
    if df.empty or "timestamp" not in df.columns:
        return df, np.array([])
    t = df["timestamp"].to_numpy(dtype=float)
    return df, (t - t[0]) / 1000.0


def _rotate_vecs(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors to world frame using quaternions.

    Parameters
    ----------
    quats : (N, 4) float  [w, x, y, z]
    vecs  : (N, 3) or (3,) float

    Returns
    -------
    (N, 3) float  world-frame vectors
    """
    if vecs.ndim == 1:
        vecs = np.tile(vecs, (len(quats), 1))
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    vx, vy, vz = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    # Rodrigues / quaternion sandwich product — vectorised
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    cx2 = y * cz - z * cy
    cy2 = z * cx - x * cz
    cz2 = x * cy - y * cx
    return np.stack(
        [vx + 2 * (w * cx + cx2),
         vy + 2 * (w * cy + cy2),
         vz + 2 * (w * cz + cz2)],
        axis=1,
    )


def _quat_cols(df) -> list[str] | None:
    """Quaternion column names in [w, x, y, z] order.

    Supports both:
    - ``qw, qx, qy, qz``
    - ``q0, q1, q2, q3`` (assumed w,x,y,z)
    """
    if all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        return ["qw", "qx", "qy", "qz"]
    if all(c in df.columns for c in ("q0", "q1", "q2", "q3")):
        return ["q0", "q1", "q2", "q3"]
    return None


def _static_mask(acc_body: np.ndarray, gyro_body_deg: np.ndarray) -> np.ndarray:
    """Boolean mask for samples that are likely static (at rest)."""
    acc_norm = np.linalg.norm(acc_body, axis=1)
    gyro_norm = np.linalg.norm(gyro_body_deg, axis=1)
    return (
        np.isfinite(acc_norm)
        & np.isfinite(gyro_norm)
        & (np.abs(acc_norm - _GRAVITY) < 0.15 * _GRAVITY)
        & (gyro_norm < 5.73)                              # ≈ 0.1 rad/s in deg/s
    )


def _add_sphere_wireframe(ax, alpha: float = 0.07, color: str = "gray") -> None:
    """Draw a unit-sphere wireframe on a 3-D axis."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, zorder=0)


def _add_world_axes(ax, scale: float = 1.15) -> None:
    """Draw ENU world-frame reference arrows and labels on a 3-D axis."""
    for direction, color, label in [
        ([1, 0, 0], "#888", "E"),
        ([0, 1, 0], "#888", "N"),
        ([0, 0, 1], "#333", "Up"),
    ]:
        d = np.array(direction, dtype=float) * scale
        ax.quiver(0, 0, 0, d[0], d[1], d[2],
                  color=color, linewidth=1.2, arrow_length_ratio=0.15)
        ax.text(d[0], d[1], d[2], f" {label}", fontsize=7, color=color)


# ---------------------------------------------------------------------------
# Plot 1 — gravity tracking in world frame
# ---------------------------------------------------------------------------

def plot_gravity_world(csv_path: Path, gravity: float = _GRAVITY) -> Path | None:
    """World-frame acc components over time — verifies orientation correctness.

    A well-calibrated, well-estimated orientation yields:
    - ax_world ≈ 0  (no East force at rest)
    - ay_world ≈ 0  (no North force at rest)
    - az_world ≈ +9.81 m/s² (Up = specific force from ground reaction)
    """
    df, time_s = _load_csv(csv_path)
    if df.empty or len(time_s) == 0:
        return None

    quat_cols = _quat_cols(df)
    if quat_cols is None:
        return None
    acc_cols  = ["ax", "ay", "az"]
    gyro_cols = ["gx", "gy", "gz"]
    if not all(c in df.columns for c in quat_cols + acc_cols):
        return None

    quats     = df[quat_cols].to_numpy(dtype=float)
    acc_body  = df[acc_cols].to_numpy(dtype=float)
    acc_world = _rotate_vecs(quats, acc_body)
    acc_norm  = np.linalg.norm(acc_world, axis=1)

    gyro_body = df[gyro_cols].to_numpy(dtype=float) if all(
        c in df.columns for c in gyro_cols
    ) else None

    static = _static_mask(acc_body, gyro_body) if gyro_body is not None else None

    labels_w = ["East (ax) [m/s²]", "North (ay) [m/s²]", "Up (az) [m/s²]"]
    refs     = [0.0, 0.0, gravity]
    data     = [acc_world[:, 0], acc_world[:, 1], acc_world[:, 2]]

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
    )

    for i, (ax, series, label, ref) in enumerate(
        zip(axes[:3], data, labels_w, refs)
    ):
        finite = mask_valid_plot_x(time_s) & np.isfinite(series)
        ax.plot(time_s[finite], series[finite], linewidth=0.6, color="#444", alpha=0.8)
        ax.axhline(ref, color="#c0392b", linestyle="--", linewidth=1.1,
                   label=f"ref = {ref:.1f}")

        if static is not None:
            # Shade static windows
            in_static = False
            t0 = 0.0
            for k in range(len(time_s)):
                if static[k] and not in_static:
                    t0 = time_s[k]; in_static = True
                elif not static[k] and in_static:
                    ax.axvspan(t0, time_s[k], color="#27ae60", alpha=0.08)
                    in_static = False
            if in_static:
                ax.axvspan(t0, time_s[-1], color="#27ae60", alpha=0.08)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    # Bottom panel: acc norm
    ax_norm = axes[3]
    tn, an = nan_mask_invalid_plot_x(time_s, acc_norm)
    ax_norm.plot(tn, an,
                 linewidth=0.6, color="#555", alpha=0.8, label="‖acc_world‖")
    ax_norm.axhline(gravity, color="#c0392b", linestyle="--", linewidth=1.1,
                    label=f"|g| = {gravity:.2f}")
    ax_norm.fill_between(
        tn,
        gravity * 0.9, gravity * 1.1,
        color="#c0392b", alpha=0.07, label="±10 %",
    )
    ax_norm.set_ylabel("‖acc‖ [m/s²]", fontsize=9)
    ax_norm.set_xlabel("Time [s]", fontsize=9)
    ax_norm.grid(True, alpha=0.25)
    ax_norm.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{csv_path.stem}\nWorld-frame acc — gravity tracking "
        f"(green shading = detected-static windows)",
        fontsize=10,
    )

    out = csv_path.with_name(f"{csv_path.stem}_gravity_world.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{csv_path.parent.name}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Plot 2 — sensor-axes sphere
# ---------------------------------------------------------------------------

def plot_axes_sphere(
    csv_path: Path,
    n_samples: int = 2000,
) -> Path | None:
    """Show where each sensor body axis points in the ENU world frame.

    The sphere plot reveals the full 3-D orientation history:
    - Body Z (sensor Up): should cluster near world +Z for an upright bike.
    - Body X (sensor Forward/Right): traces the heading direction.
    - Body Y (sensor Left): completes the right-hand frame.

    Points are coloured by time (dark blue = start → yellow = end).
    Static samples are additionally marked with larger markers.
    """
    df, time_s = _load_csv(csv_path)
    if df.empty or len(time_s) == 0:
        return None

    quat_cols = _quat_cols(df)
    if quat_cols is None:
        return None
    acc_cols  = ["ax", "ay", "az"]
    gyro_cols = ["gx", "gy", "gz"]
    quats = df[quat_cols].to_numpy(dtype=float)
    acc_body  = df[acc_cols].to_numpy(dtype=float) if all(
        c in df.columns for c in acc_cols
    ) else None
    gyro_body = df[gyro_cols].to_numpy(dtype=float) if all(
        c in df.columns for c in gyro_cols
    ) else None

    static = (
        _static_mask(acc_body, gyro_body)
        if (acc_body is not None and gyro_body is not None)
        else np.zeros(len(quats), dtype=bool)
    )

    # Subsample for performance
    N = len(quats)
    step = max(1, N // n_samples)
    idx  = np.arange(0, N, step)
    q_s  = quats[idx]
    t_s  = time_s[idx]
    st_s = static[idx]

    # Body axes in world frame
    bx_w = _rotate_vecs(q_s, np.array([1.0, 0.0, 0.0]))  # Forward / East-aligned
    by_w = _rotate_vecs(q_s, np.array([0.0, 1.0, 0.0]))  # Left
    bz_w = _rotate_vecs(q_s, np.array([0.0, 0.0, 1.0]))  # Up

    # Time normalised for colourmap
    t_norm = (t_s - t_s[0]) / max(t_s[-1] - t_s[0], 1e-6)
    cmap   = cm.plasma

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    titles = ["Body X (forward/right)", "Body Y (left)", "Body Z (up)"]
    axes_data = [bx_w, by_w, bz_w]
    ax_colors = [_AXIS_COLORS["X"], _AXIS_COLORS["Y"], _AXIS_COLORS["Z"]]

    for col, (body_w, title, _color) in enumerate(
        zip(axes_data, titles, ax_colors)
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        _add_sphere_wireframe(ax)
        _add_world_axes(ax)

        sc = ax.scatter(
            body_w[:, 0], body_w[:, 1], body_w[:, 2],
            c=t_norm, cmap=cmap, s=3, alpha=0.45, linewidths=0,
            vmin=0, vmax=1,
        )
        # Highlight static samples
        if st_s.sum() > 0:
            ax.scatter(
                body_w[st_s, 0], body_w[st_s, 1], body_w[st_s, 2],
                c="lime", s=8, alpha=0.7, linewidths=0,
                label="static",
            )

        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel("East", fontsize=7, labelpad=1)
        ax.set_ylabel("North", fontsize=7, labelpad=1)
        ax.set_zlabel("Up", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6, pad=0)
        ax.set_box_aspect([1, 1, 1])

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.04)
    cbar.set_label("Time →", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["start", "mid", "end"])

    fig.suptitle(
        f"{csv_path.stem}\nSensor axes in ENU world frame "
        f"(plasma = time, lime = detected static)",
        fontsize=10,
    )

    out = csv_path.with_name(f"{csv_path.stem}_axes_sphere.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{csv_path.parent.name}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Plot 3 — 3-D sensor-frame arrows at key moments
# ---------------------------------------------------------------------------

def plot_3d_frames(
    csv_path: Path,
    n_frames: int = 8,
) -> Path | None:
    """Show the sensor coordinate frame at N evenly-spaced moments.

    Each mini-subplot is a 3-D view of the world frame showing three arrows:
    - Red  = body X axis (sensor forward / right)
    - Green = body Y axis (sensor left)
    - Blue  = body Z axis (sensor up)

    The world frame background axes (East, North, Up) are shown in gray.
    This directly visualises how the physical sensor is oriented at each
    selected instant and whether the orientation evolves sensibly over time.
    """
    df, time_s = _load_csv(csv_path)
    if df.empty or len(time_s) == 0:
        return None

    quat_cols = _quat_cols(df)
    if quat_cols is None:
        return None

    quats = df[quat_cols].to_numpy(dtype=float)

    # Pick n_frames evenly spaced indices
    indices = np.linspace(0, len(quats) - 1, n_frames, dtype=int)
    n_cols  = 4
    n_rows  = (n_frames + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.5), constrained_layout=True)

    body_vecs = {
        "X": (np.array([1.0, 0.0, 0.0]), _AXIS_COLORS["X"]),
        "Y": (np.array([0.0, 1.0, 0.0]), _AXIS_COLORS["Y"]),
        "Z": (np.array([0.0, 0.0, 1.0]), _AXIS_COLORS["Z"]),
    }
    world_ref = {
        "E": ([1.15, 0, 0], "#aaa"),
        "N": ([0, 1.15, 0], "#aaa"),
        "↑": ([0, 0, 1.15], "#555"),
    }

    for panel, idx in enumerate(indices):
        ax = fig.add_subplot(n_rows, n_cols, panel + 1, projection="3d")
        q  = quats[idx]
        t  = time_s[idx]

        # World-frame reference (thin gray)
        for label, (pos, col) in world_ref.items():
            ax.quiver(0, 0, 0, *pos, color=col, linewidth=0.8,
                      arrow_length_ratio=0.12, alpha=0.6)
            ax.text(*[p * 1.05 for p in pos], label, fontsize=6, color=col)

        # Sensor body axes
        for name, (vec, color) in body_vecs.items():
            bv = _rotate_vecs(q[np.newaxis, :], vec[np.newaxis, :])[0]
            ax.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                      color=color, linewidth=2, arrow_length_ratio=0.15)
            ax.text(bv[0] * 1.12, bv[1] * 1.12, bv[2] * 1.12,
                    name, fontsize=8, color=color, fontweight="bold")

        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel("E", fontsize=6, labelpad=0)
        ax.set_ylabel("N", fontsize=6, labelpad=0)
        ax.set_zlabel("↑", fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5, pad=0)
        ax.set_box_aspect([1, 1, 1])
        pct = 100.0 * idx / max(len(quats) - 1, 1)
        ax.set_title(f"t = {t:.0f} s  ({pct:.0f}%)", fontsize=8, pad=2)

    # Legend in the last empty subplot if any
    ax_leg = fig.add_subplot(n_rows, n_cols, n_frames + 1) if (
        n_frames < n_rows * n_cols
    ) else None
    if ax_leg:
        ax_leg.axis("off")
        for name, (_, color) in body_vecs.items():
            ax_leg.plot([], [], color=color, linewidth=3, label=f"Body {name}")
        ax_leg.plot([], [], color="#aaa", linewidth=1.5, label="World ref")
        ax_leg.legend(loc="center", fontsize=10)

    fig.suptitle(
        f"{csv_path.stem}\nSensor body frame at {n_frames} moments  "
        f"(R=X, G=Y, B=Z  in ENU world)",
        fontsize=10,
    )

    out = csv_path.with_name(f"{csv_path.stem}_3d_frames.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{csv_path.parent.name}] {out.name}")
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _find_orientation_csv(search_dir: Path, sensor: str) -> Path | None:
    """Return the orientation CSV for *sensor* in *search_dir*.

    Looks for the new per-method naming (``<sensor>_orientation.csv``) first,
    then falls back to legacy flat patterns.
    """
    direct = search_dir / f"{sensor}_orientation.csv"
    if direct.exists():
        return direct
    for pat in (
        f"{sensor}*__complementary_orientation.csv",
        f"{sensor}*_orientation.csv",
    ):
        hits = sorted(search_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def plot_orientation_verify_stage(
    recording_name: str,
    stage: str = "orientation",
) -> None:
    """Generate all verification plots for every orientation CSV in *stage*.

    Iterates over method subdirectories (``complementary/``, ``madgwick/``, …).
    Falls back to the flat layout if no subdirectories contain orientation CSVs.
    """
    stage_dir = resolve_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    # Collect CSV files from method subdirectories first.
    csv_files: list[Path] = []
    for method_dir in sorted(d for d in stage_dir.iterdir() if d.is_dir()):
        csv_files.extend(sorted(method_dir.glob("*_orientation.csv")))

    # Fallback: flat layout.
    if not csv_files:
        csv_files = sorted(stage_dir.glob("*_orientation.csv"))

    if not csv_files:
        log.warning("[%s/%s] no orientation CSVs found.", recording_name, stage)
        return

    for csv_path in csv_files:
        plot_gravity_world(csv_path)
        plot_axes_sphere(csv_path)
        plot_3d_frames(csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_orientation_verify",
        description=(
            "Generate orientation verification plots "
            "(gravity tracking, axes sphere, 3-D frames) for one or more recordings."
        ),
    )
    parser.add_argument(
        "recording_names",
        nargs="+",
        help="One or more recording names (e.g. 2026-02-26_r5).",
    )
    parser.add_argument(
        "--stage",
        default="orientation",
        help="Orientation stage directory name (default: orientation).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for recording_name in args.recording_names:
        plot_orientation_verify_stage(recording_name, stage=args.stage)


if __name__ == "__main__":
    main()
