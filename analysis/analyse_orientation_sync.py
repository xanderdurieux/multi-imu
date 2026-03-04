"""Orientation-space crosscheck of sync method quality.

For each recording that has both synced_lida and synced_cal stages, runs
a complementary orientation filter on both sensors, then computes the
relative quaternion trajectory between the two sensors and reports:

  - Mean / RMSD of the inter-sensor angular distance
  - Short-window (1 s) relative-orientation stability (std of angular distance)

Better temporal synchronization is expected to reduce artificial jitter in
the relative orientation at short timescales.

Usage::

    uv run analyse_orientation_sync.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe, recordings_root
from orientation.pipeline import run_complementary_on_dataframe
from sync.common import resample_stream

SESSIONS = ["2026-02-26", "2025-12-04"]
OUT_DIR = Path(__file__).parent / "data"
SAMPLE_RATE_HZ = 10.0
WINDOW_S = 2.0          # short window for stability metric
MIN_OVERLAP_S = 30.0    # skip recordings with less overlap


def _quat_angular_distance_deg(q1_arr: np.ndarray, q2_arr: np.ndarray) -> np.ndarray:
    """Angular distance in degrees between paired unit quaternions.

    q1_arr, q2_arr: shape (N, 4) in [w, x, y, z] order.
    Returns array of shape (N,) with angles in degrees.
    """
    dots = np.einsum("ij,ij->i", q1_arr, q2_arr)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _run_orientation_on_csv(csv_path: Path) -> pd.DataFrame | None:
    """Load a sensor CSV, run complementary filter, return oriented DataFrame.

    NaN rows (dropped packets) are forward-filled before filtering so that
    the orientation integration step does not fail on sensors like the Arduino
    that can have intermittent gaps.  Rows are not dropped because removing
    them would create spurious large time steps in dt estimation.
    """
    try:
        df = load_dataframe(csv_path)
        if df.empty or len(df) < 10:
            return None
        imu_cols = [c for c in ("ax", "ay", "az", "gx", "gy", "gz") if c in df.columns]
        if imu_cols:
            df[imu_cols] = df[imu_cols].ffill().bfill()
        return run_complementary_on_dataframe(df, calibration=None)
    except Exception as e:
        print(f"  [WARN] orientation failed for {csv_path.name}: {e}")
        return None


def _resample_to_common_grid(
    df_ref: pd.DataFrame,
    df_tgt: pd.DataFrame,
    rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample two oriented DataFrames to a common uniform time grid.

    Returns (t_common_ms, quats_ref, quats_tgt) where quats are (N,4).
    """
    ref_r = resample_stream(df_ref, rate_hz)
    tgt_r = resample_stream(df_tgt, rate_hz)

    ref_ts = ref_r["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_r["timestamp"].to_numpy(dtype=float)
    lo = max(ref_ts[0], tgt_ts[0])
    hi = min(ref_ts[-1], tgt_ts[-1])

    if (hi - lo) / 1000.0 < MIN_OVERLAP_S:
        return np.array([]), np.zeros((0, 4)), np.zeros((0, 4))

    mask_r = (ref_ts >= lo) & (ref_ts <= hi)
    mask_t = (tgt_ts >= lo) & (tgt_ts <= hi)
    ref_slice = ref_r.loc[mask_r]
    tgt_slice = tgt_r.loc[mask_t]

    n = min(len(ref_slice), len(tgt_slice))
    if n < 10:
        return np.array([]), np.zeros((0, 4)), np.zeros((0, 4))

    t_common = ref_slice["timestamp"].to_numpy(dtype=float)[:n]
    q_ref = ref_slice[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)[:n]
    q_tgt = tgt_slice[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)[:n]
    return t_common, q_ref, q_tgt


def _short_window_stability(
    angles_deg: np.ndarray,
    rate_hz: float,
    window_s: float = WINDOW_S,
) -> float:
    """Mean standard deviation of angular distance within sliding windows."""
    n_win = max(2, int(round(window_s * rate_hz)))
    stds = []
    for i in range(0, len(angles_deg) - n_win + 1, max(1, n_win // 2)):
        stds.append(float(np.std(angles_deg[i:i + n_win])))
    return float(np.mean(stds)) if stds else float("nan")


def analyse_recording(
    recording_name: str,
) -> dict | None:
    """Compute orientation crosscheck metrics for one recording."""
    root = recordings_root()
    result = {"recording": recording_name}

    for stage in ("synced_lida", "synced_cal"):
        stage_dir = root / recording_name / stage
        ref_csv = stage_dir / "sporsa.csv"
        tgt_csv = stage_dir / "arduino.csv"

        if not ref_csv.exists() or not tgt_csv.exists():
            result[stage] = None
            continue

        df_ref = _run_orientation_on_csv(ref_csv)
        df_tgt = _run_orientation_on_csv(tgt_csv)

        if df_ref is None or df_tgt is None:
            result[stage] = None
            continue

        t_common, q_ref, q_tgt = _resample_to_common_grid(df_ref, df_tgt)
        if t_common.size == 0:
            result[stage] = None
            continue

        angles = _quat_angular_distance_deg(q_ref, q_tgt)
        stability = _short_window_stability(angles, SAMPLE_RATE_HZ)
        result[stage] = {
            "n_samples": int(len(angles)),
            "duration_s": float((t_common[-1] - t_common[0]) / 1000.0),
            "angular_dist_mean_deg": float(np.mean(angles)),
            "angular_dist_std_deg": float(np.std(angles)),
            "angular_dist_rmsd_deg": float(np.sqrt(np.mean(angles ** 2))),
            "short_window_stability_deg": stability,
            "angles_deg": angles.tolist(),
            "t_s": ((t_common - t_common[0]) / 1000.0).tolist(),
        }

    if result.get("synced_lida") is None and result.get("synced_cal") is None:
        return None
    return result


def print_crosscheck_table(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("  ORIENTATION-SPACE SYNC CROSSCHECK")
    print("  Metric: inter-sensor angular distance (deg) after complementary filter")
    print("=" * 90)
    hdr = (f"  {'Recording':<20} {'Method':<14} {'Mean':>8} {'Std':>8} "
           f"{'RMSD':>8} {'Stability':>12} {'Duration':>10}")
    print(hdr)
    print("  " + "-" * 84)
    for r in results:
        rec = r["recording"]
        for stage_key, label in [("synced_lida", "SDA+LIDA"), ("synced_cal", "Cal-sync")]:
            m = r.get(stage_key)
            if m is None:
                print(f"  {rec:<20} {label:<14}   N/A")
                continue
            print(f"  {rec:<20} {label:<14} "
                  f"{m['angular_dist_mean_deg']:>8.2f} "
                  f"{m['angular_dist_std_deg']:>8.2f} "
                  f"{m['angular_dist_rmsd_deg']:>8.2f} "
                  f"{m['short_window_stability_deg']:>12.3f} "
                  f"{m['duration_s']:>10.1f} s")
    print("=" * 90 + "\n")


def plot_crosscheck(results: list[dict], out_dir: Path) -> Path:
    """Save orientation crosscheck comparison figure."""
    valid = [r for r in results if r.get("synced_lida") or r.get("synced_cal")]
    n = len(valid)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle("Orientation crosscheck: inter-sensor angular distance", fontsize=13)

    for row, r in enumerate(valid):
        rec = r["recording"]
        for col, (stage_key, label, color) in enumerate([
            ("synced_lida", "SDA+LIDA", "steelblue"),
            ("synced_cal", "Cal-sync", "tomato"),
        ]):
            ax = axes[row, col]
            m = r.get(stage_key)
            if m is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{rec} — {label} (N/A)")
                continue
            t = np.array(m["t_s"], dtype=float)
            ang = np.array(m["angles_deg"], dtype=float)
            ax.plot(t, ang, lw=0.7, alpha=0.8, color=color)
            ax.axhline(m["angular_dist_mean_deg"], color="black", lw=1, linestyle="--",
                       label=f"mean={m['angular_dist_mean_deg']:.1f}°")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angular distance (°)")
            ax.set_title(
                f"{rec} — {label}  "
                f"RMSD={m['angular_dist_rmsd_deg']:.1f}°  "
                f"stab={m['short_window_stability_deg']:.2f}°"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "orientation_crosscheck.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Orientation crosscheck plot saved -> {out_path}")
    return out_path


def save_results_json(results: list[dict], out_dir: Path) -> Path:
    serialisable = []
    for r in results:
        entry = {"recording": r["recording"]}
        for stage in ("synced_lida", "synced_cal"):
            m = r.get(stage)
            if m is None:
                entry[stage] = None
            else:
                entry[stage] = {k: v for k, v in m.items() if k not in ("angles_deg", "t_s")}
        serialisable.append(entry)
    out_path = out_dir / "orientation_crosscheck_results.json"
    out_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"Orientation crosscheck JSON saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and any(d.name.startswith(f"{s}_") for s in SESSIONS)
    )

    # Only process recordings that have at least one synced stage with both sensors
    to_process = []
    for rec in recordings:
        has_lida = (root / rec / "synced_lida" / "sporsa.csv").exists()
        has_cal = (root / rec / "synced_cal" / "sporsa.csv").exists()
        if has_lida or has_cal:
            to_process.append(rec)

    print(f"Processing {len(to_process)} recordings...")
    results = []
    for rec in to_process:
        print(f"  [{rec}]")
        r = analyse_recording(rec)
        if r is not None:
            results.append(r)

    print_crosscheck_table(results)
    plot_crosscheck(results, OUT_DIR)
    save_results_json(results, OUT_DIR)
