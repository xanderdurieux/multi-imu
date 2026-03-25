"""Orchestrate orientation estimation for section data.

Calibrated CSVs store acc and gyro in world frame (rotated by a per-section R).
Classical filters work in body frame, so R.T is applied internally before each filter,
and the resulting quaternion represents body→world.  This is transparent for thesis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_dataframe
from common.quaternion import euler_from_quat, quat_rotate

from .complementary import complementary_orientation
from .ekf import ekf_orientation
from .madgwick import madgwick_acc_only, madgwick_9dof

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.81
FILTER_VARIANTS = [
    "madgwick_acc_only",
    "madgwick_9dof",
    "complementary_orientation",
    "ekf_orientation",
]


def _world_to_body(df: pd.DataFrame, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-frame acc and gyro to body frame using R.T."""
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    R = np.asarray(R, dtype=float)
    acc_body = (R.T @ acc.T).T
    gyro_body = (R.T @ gyro.T).T
    return acc_body, gyro_body


def _upsample_mag_to_imu(mag_df: pd.DataFrame, ts_imu: np.ndarray) -> np.ndarray:
    """Zero-order hold: upsample mag to IMU timestamps."""
    ts_mag = mag_df["timestamp"].to_numpy(dtype=float)
    mag_vals = mag_df[["mx", "my", "mz"]].to_numpy(dtype=float)
    n = len(ts_imu)
    out = np.full((n, 3), np.nan)
    last_idx = 0
    for i in range(n):
        t = ts_imu[i]
        while last_idx + 1 < len(ts_mag) and ts_mag[last_idx + 1] <= t:
            last_idx += 1
        if ts_mag[last_idx] <= t or last_idx == 0:
            out[i] = mag_vals[last_idx]
    for i in range(n):
        if np.any(np.isnan(out[i])):
            for j in range(i + 1, n):
                if not np.any(np.isnan(out[j])):
                    out[i] = out[j]
                    break
    return out


def _compute_dt(timestamps_ms: np.ndarray) -> float:
    """Mean dt in seconds."""
    ts = np.asarray(timestamps_ms, dtype=float)
    if len(ts) < 2:
        return 0.01
    return float(np.nanmean(np.diff(ts)) / 1000.0)

def _compute_effective_dt_from_valid_gyro(
    timestamps_ms: np.ndarray,
    gyro_body: np.ndarray,
) -> np.ndarray:
    """Per-sample dt that accumulates across dropout gaps.

    dt[i] is non-zero only when gyro[i] is finite; then it equals elapsed time since
    the previous *valid gyro* sample. For invalid gyro rows dt[i]=0 so filters hold state.
    """
    ts = np.asarray(timestamps_ms, dtype=float)
    n = len(ts)
    if n == 0:
        return np.array([], dtype=float)

    dt = np.zeros(n, dtype=float)
    valid = np.all(np.isfinite(gyro_body), axis=1)

    # Fallback nominal dt from timestamps
    diffs = np.diff(ts)
    nominal = float(np.nanmedian(diffs) / 1000.0) if len(diffs) else 0.01
    nominal = max(nominal, 1e-6)

    last_valid_ts = None
    for i in range(n):
        if not valid[i]:
            continue
        if last_valid_ts is None:
            dt[i] = nominal
        else:
            dt[i] = max((ts[i] - last_valid_ts) / 1000.0, 0.0)
        last_valid_ts = ts[i]

    return dt


def _g_err_abs_mean(
    acc_body: np.ndarray,
    quats: np.ndarray,
    static_n: int | None = None,
) -> float:
    """Mean |world-frame az - 9.81| over the static window.

    Rotates body-frame acc to world frame using the estimated quaternion and
    checks that the z-component equals 9.81 m/s^2 (gravity up, +Z convention).
    Evaluated only over the first static_n samples to avoid dynamic-motion bias.
    """
    n = min(static_n, len(quats)) if static_n is not None else len(quats)
    errs = []
    for i in range(n):
        if not np.all(np.isfinite(acc_body[i])):
            continue
        v_world = quat_rotate(quats[i], acc_body[i])
        errs.append(abs(v_world[2] - GRAVITY_M_S2))
    return float(np.nanmean(errs)) if errs else float("nan")


def _static_pitch_roll_std(
    quats: np.ndarray,
    static_n: int,
) -> tuple[float, float]:
    """Std of pitch and roll over first static_n samples (degrees).

    np.unwrap is applied before computing std to handle the +/-180 deg discontinuity
    that occurs when sensors are mounted inverted (roll~180 deg at rest).
    """
    pitches_rad = []
    rolls_rad = []
    for i in range(min(static_n, len(quats))):
        yaw, pitch, roll = euler_from_quat(quats[i])
        pitches_rad.append(pitch)
        rolls_rad.append(roll)
    if not pitches_rad:
        return 0.0, 0.0
    pitches_unwrapped = np.unwrap(pitches_rad)
    rolls_unwrapped = np.unwrap(rolls_rad)
    return float(np.nanstd(np.degrees(pitches_unwrapped))), float(np.nanstd(np.degrees(rolls_unwrapped)))


def _quality_tag(g_err: float, pitch_std: float, roll_std: float) -> str:
    if g_err <= 0.3 and pitch_std <= 2.0 and roll_std <= 2.0:
        return "good"
    if pitch_std <= 5.0 and roll_std <= 5.0:
        return "marginal"
    return "poor"


def estimate_section(
    section_path: Path,
    *,
    write_plots: bool = True,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    """Run orientation estimation for a section; write CSVs and stats.

    Parameters
    ----------
    variants:
        Subset of :data:`FILTER_VARIANTS` to run. Default ``None`` runs all filters.
    """
    section_path = Path(section_path)
    calibrated_dir = section_path / "calibrated"
    cal_json = calibrated_dir / "calibration.json"

    if not cal_json.exists():
        log.warning("No calibration.json at %s — skipping orientation", cal_json)
        return {}

    with cal_json.open("r", encoding="utf-8") as f:
        calibration = json.load(f)

    orient_dir = section_path / "orientation"
    orient_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    for sensor in ("sporsa", "arduino"):
        cal_csv = calibrated_dir / f"{sensor}.csv"
        if not cal_csv.exists():
            continue

        df = load_dataframe(cal_csv)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            continue

        meta = calibration.get(sensor, {})
        R = np.array(meta.get("rotation_matrix", np.eye(3)))
        n_static = meta.get("n_static_samples", min(100, len(df)))

        acc_body, gyro_body = _world_to_body(df, R)
        ts = df["timestamp"].to_numpy(dtype=float)
        dt = _compute_effective_dt_from_valid_gyro(ts, gyro_body)

        mag = df[["mx", "my", "mz"]].to_numpy(dtype=float)
        if np.any(np.isfinite(mag)):
            mag_valid = mag.copy()
            mag_valid[~np.isfinite(mag_valid)] = 0.0
        else:
            mag_valid = np.zeros_like(acc_body)

        results[sensor] = {}

        run_variants = variants if variants is not None else list(FILTER_VARIANTS)
        for variant in run_variants:
            if variant not in FILTER_VARIANTS:
                log.warning("Unknown orientation variant %r — skipping", variant)
                continue
            if variant == "madgwick_acc_only":
                quats = madgwick_acc_only(acc_body, gyro_body, dt)
            elif variant == "madgwick_9dof":
                quats = madgwick_9dof(acc_body, gyro_body, mag_valid, dt)
            elif variant == "complementary_orientation":
                quats = complementary_orientation(acc_body, gyro_body, dt)
            elif variant == "ekf_orientation":
                quats = ekf_orientation(acc_body, gyro_body, dt)
            else:
                continue

            g_err = _g_err_abs_mean(acc_body, quats, static_n=n_static)
            pitch_std, roll_std = _static_pitch_roll_std(quats, n_static)
            quality = _quality_tag(g_err, pitch_std, roll_std)

            yaw_rad, pitch_rad, roll_rad = zip(
                *[euler_from_quat(quats[i]) for i in range(len(quats))]
            )
            out_df = pd.DataFrame({
                "timestamp": ts,
                "q0": quats[:, 0],
                "q1": quats[:, 1],
                "q2": quats[:, 2],
                "q3": quats[:, 3],
                "roll_deg": np.degrees(roll_rad),
                "pitch_deg": np.degrees(pitch_rad),
                "yaw_deg": np.degrees(yaw_rad),
            })
            out_path = orient_dir / f"{sensor}__{variant}.csv"
            out_df.to_csv(out_path, index=False)

            results[sensor][f"__{variant}"] = {
                "g_err_abs_mean": g_err,
                "static_pitch_std_deg": pitch_std,
                "static_roll_std_deg": roll_std,
                "quality": quality,
            }

    stats_path = orient_dir / "orientation_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if write_plots and results:
        from .plots import plot_orientation_diagnostics
        plot_orientation_diagnostics(section_path, orient_dir, results)

    return results


def _parse_section_arg(arg: str) -> tuple[str, str | None]:
    parts = arg.strip().rstrip("/").replace("\\", "/").split("/")
    rec = parts[0]
    sec = None
    for i, p in enumerate(parts):
        if p == "sections" and i + 1 < len(parts):
            sec = parts[i + 1]
            break
    return rec, sec


def estimate_sections_from_args(
    name: str,
    *,
    all_sections: bool = False,
    variants: list[str] | None = None,
) -> list[Path]:
    """Run orientation for section(s) from CLI-style args."""
    rec, sec = _parse_section_arg(name)

    if all_sections:
        from common.paths import iter_sections_for_recording

        dirs = iter_sections_for_recording(rec)
    else:
        if sec is None:
            raise ValueError("Specify section path or use --all-sections")
        sec_s = str(sec).strip()
        if not sec_s.startswith("section_"):
            raise ValueError(
                f"Expected legacy section id like 'section_<idx>', got {sec_s!r}"
            )
        sec_idx = int(sec_s.split("_", 1)[1])
        from common.paths import section_dir

        d0 = section_dir(rec, sec_idx)
        if not d0.exists():
            raise FileNotFoundError(f"Section not found: {d0}")
        dirs = [d0]

    done = []
    for d in dirs:
        cal_dir = d / "calibrated"
        if not (cal_dir / "calibration.json").exists():
            log.warning("Skipping %s (no calibration)", d.name)
            continue
        estimate_section(d, variants=variants)
        done.append(d)
    return done


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser(prog="python -m orientation.estimate")
    parser.add_argument("name", help="Section path or recording with --all-sections")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        metavar="NAME",
        help=(
            "Orientation filter to run (repeatable). "
            "One of: madgwick_acc_only, madgwick_9dof, complementary_orientation, ekf_orientation. "
            "Default: all."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    estimate_sections_from_args(args.name, all_sections=args.all_sections, variants=args.variants)
