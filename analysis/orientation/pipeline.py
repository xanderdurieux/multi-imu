"""Orientation estimation pipeline for calibrated IMU sections.

Reads calibrated sensor CSVs, runs orientation filters (Madgwick and/or
complementary), writes per-variant orientation CSVs and a JSON stats file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.csv_schema import load_dataframe, write_dataframe
from common.paths import iter_sections_for_recording, sections_root
from common.quaternion import euler_from_quat
from .madgwick import MadgwickFilter
from .complementary import ComplementaryFilter

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_DEG_PER_RAD = 180.0 / np.pi
_RAD_PER_DEG = np.pi / 180.0  # 1/57.2958
_DEFAULT_VARIANTS = ["madgwick", "complementary"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OrientationStats:
    """Quality statistics for a single sensor + filter variant."""

    sensor: str
    variant: str
    gravity_alignment: float  # Pearson r of az_world vs expected constant g
    pitch_std_deg: float
    roll_std_deg: float
    quality: str  # "good", "marginal", or "poor"


# ---------------------------------------------------------------------------
# Core filter runner
# ---------------------------------------------------------------------------

def run_orientation_filters(
    df: pd.DataFrame,
    *,
    sensor_name: str,
    sample_rate_hz: float = 100.0,
    variants: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run orientation filters on calibrated IMU data.

    Parameters
    ----------
    df:
        Calibrated IMU DataFrame with columns ``timestamp``,
        ``ax``, ``ay``, ``az``, ``gx``, ``gy``, ``gz`` (gyro in deg/s).
        World-frame columns ``ax_world``, ``ay_world``, ``az_world`` are used
        when available, falling back to body-frame columns otherwise.
    sensor_name:
        Identifier used only for logging.
    sample_rate_hz:
        Sampling rate in Hz.
    variants:
        List of filter names to run.  Supported: ``"madgwick"``,
        ``"complementary"``.  Defaults to both.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping variant name -> DataFrame with columns:
        ``timestamp``, ``qw``, ``qx``, ``qy``, ``qz``,
        ``yaw_deg``, ``pitch_deg``, ``roll_deg``.
    """
    if variants is None:
        variants = list(_DEFAULT_VARIANTS)

    # Determine which acc/gyro columns to use.
    has_world = all(c in df.columns for c in ("ax_world", "ay_world", "az_world"))
    if has_world:
        log.debug("%s: using world-frame acc columns", sensor_name)
        acc_cols = ("ax_world", "ay_world", "az_world")
    else:
        log.debug("%s: world-frame acc not found, falling back to body frame", sensor_name)
        acc_cols = ("ax", "ay", "az")

    has_world_gyro = all(c in df.columns for c in ("gx_world", "gy_world", "gz_world"))
    if has_world_gyro:
        gyro_cols = ("gx_world", "gy_world", "gz_world")
    else:
        gyro_cols = ("gx", "gy", "gz")

    # Extract arrays.
    timestamps = df["timestamp"].to_numpy(dtype=float)
    acc_arr = df[list(acc_cols)].to_numpy(dtype=float)   # shape (N, 3)
    gyro_arr = df[list(gyro_cols)].to_numpy(dtype=float)  # shape (N, 3), deg/s

    # Convert gyro from deg/s -> rad/s.
    gyro_rad_arr = gyro_arr * _RAD_PER_DEG

    n_samples = len(timestamps)
    if n_samples == 0:
        log.warning("%s: empty DataFrame, returning empty results", sensor_name)
        return {v: pd.DataFrame(columns=["timestamp", "qw", "qx", "qy", "qz",
                                         "yaw_deg", "pitch_deg", "roll_deg"])
                for v in variants}

    # Compute per-sample dt from timestamps (ms -> s).
    # For first sample use nominal dt.
    dt_nominal = 1.0 / sample_rate_hz
    dt_arr = np.diff(timestamps, prepend=timestamps[0] - dt_nominal * 1000.0) / 1000.0
    # Clamp wildly off values (e.g. gaps) to avoid filter explosion.
    dt_arr = np.clip(dt_arr, dt_nominal * 0.1, dt_nominal * 10.0)

    # Initialize filters from first accelerometer sample.
    first_acc = acc_arr[0]

    results: dict[str, pd.DataFrame] = {}

    for variant in variants:
        if variant == "madgwick":
            filt = MadgwickFilter(beta=0.1, sample_rate_hz=sample_rate_hz)
        elif variant == "complementary":
            filt = ComplementaryFilter(alpha=0.98, sample_rate_hz=sample_rate_hz)
        else:
            log.warning("Unknown variant '%s' — skipping", variant)
            continue

        filt.initialize_from_acc(first_acc)

        rows = []
        for i in range(n_samples):
            acc = acc_arr[i]
            gyro_rad = gyro_rad_arr[i]
            dt = float(dt_arr[i])

            q = filt.update(acc, gyro_rad, dt=dt)

            yaw_r, pitch_r, roll_r = euler_from_quat(q)
            rows.append({
                "timestamp": timestamps[i],
                "qw": q[0],
                "qx": q[1],
                "qy": q[2],
                "qz": q[3],
                "yaw_deg": float(yaw_r * _DEG_PER_RAD),
                "pitch_deg": float(pitch_r * _DEG_PER_RAD),
                "roll_deg": float(roll_r * _DEG_PER_RAD),
            })

        results[variant] = pd.DataFrame(rows)
        log.debug("%s/%s: processed %d samples", sensor_name, variant, n_samples)

    return results


# ---------------------------------------------------------------------------
# Quality statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    df_raw: pd.DataFrame,
    df_orient: pd.DataFrame,
    sensor: str,
    variant: str,
) -> OrientationStats:
    """Compute OrientationStats for a sensor + variant combination."""
    # Gravity alignment: Pearson r of az_world with its mean (should be ~-g).
    # We compare the az_world column against a constant equal to its median,
    # but a more meaningful metric is the correlation of ||acc_world|| with g.
    # We use: r(az_world, mean(az_world)) = consistency of z-axis acc magnitude.
    # Simpler approach: use the z-acc channel correlation with a constant 9.81.
    acc_col = "az_world" if "az_world" in df_raw.columns else "az"
    az = df_raw[acc_col].dropna().to_numpy(dtype=float)

    if len(az) < 2:
        gravity_alignment = 0.0
    else:
        # Pearson r between az and its mean (all-same = perfect).
        # Instead, compute correlation between az and a constant equal to its mean.
        # Pearson r of [x] with constant is 0; meaningful signal = low variance.
        # Better: compare az magnitude std relative to mean (coefficient of variation).
        mean_az = np.mean(az)
        std_az = np.std(az)
        if abs(mean_az) < 1e-6:
            gravity_alignment = 0.0
        else:
            # 1 - (std/mean)^2 clamped to [0,1]: 1 means perfectly constant.
            cv_sq = (std_az / mean_az) ** 2
            gravity_alignment = float(np.clip(1.0 - cv_sq, 0.0, 1.0))

    pitch_std = float(df_orient["pitch_deg"].std()) if "pitch_deg" in df_orient.columns else 0.0
    roll_std = float(df_orient["roll_deg"].std()) if "roll_deg" in df_orient.columns else 0.0

    # Quality thresholds.
    if gravity_alignment < 0.8:
        quality = "poor"
    elif gravity_alignment < 0.95:
        quality = "marginal"
    else:
        quality = "good"

    return OrientationStats(
        sensor=sensor,
        variant=variant,
        gravity_alignment=round(gravity_alignment, 4),
        pitch_std_deg=round(pitch_std, 4),
        roll_std_deg=round(roll_std, 4),
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Section-level pipeline
# ---------------------------------------------------------------------------

def process_section_orientation(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    canonical_variant: str = "madgwick",
    variants: list[str] | None = None,
) -> dict:
    """Process orientation for all sensors in one section.

    Reads:
        ``<section_dir>/calibrated/<sensor>.csv``

    Writes:
        ``<section_dir>/orientation/<sensor>__<variant>.csv``
        ``<section_dir>/orientation/orientation_stats.json``

    Parameters
    ----------
    section_dir:
        Path to section directory (e.g. ``data/sections/2026-02-26_r1s1``).
    sample_rate_hz:
        Sampling rate in Hz.
    force:
        Overwrite existing outputs when True.
    canonical_variant:
        The filter variant considered the primary one.
    variants:
        Filter variants to run. Defaults to ``["madgwick", "complementary"]``.

    Returns
    -------
    dict
        The orientation_stats dict that was written to JSON.
    """
    if variants is None:
        variants = list(_DEFAULT_VARIANTS)

    orient_dir = section_dir / "orientation"
    stats_json = orient_dir / "orientation_stats.json"

    if stats_json.exists() and not force:
        log.info(
            "Orientation already exists for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        return json.loads(stats_json.read_text(encoding="utf-8"))

    orient_dir.mkdir(parents=True, exist_ok=True)

    stats_dict: dict = {}

    for sensor in _SENSORS:
        cal_csv = section_dir / "calibrated" / f"{sensor}.csv"
        if not cal_csv.exists():
            log.warning(
                "Calibrated CSV not found for sensor '%s' in %s — skipping",
                sensor,
                section_dir.name,
            )
            continue

        df = load_dataframe(cal_csv)
        if df.empty:
            log.warning("Empty calibrated CSV for %s/%s — skipping", section_dir.name, sensor)
            continue

        orient_results = run_orientation_filters(
            df,
            sensor_name=sensor,
            sample_rate_hz=sample_rate_hz,
            variants=variants,
        )

        sensor_stats: dict[str, dict] = {}
        for variant, df_orient in orient_results.items():
            # Write orientation CSV.
            out_csv = orient_dir / f"{sensor}__{variant}.csv"
            write_dataframe(df_orient, out_csv)
            log.info(
                "Wrote orientation %s/%s → %s (%d rows)",
                sensor, variant, out_csv, len(df_orient),
            )

            # Compute stats.
            s = _compute_stats(df, df_orient, sensor, variant)
            sensor_stats[variant] = {
                "gravity_alignment": s.gravity_alignment,
                "pitch_std_deg": s.pitch_std_deg,
                "roll_std_deg": s.roll_std_deg,
                "quality": s.quality,
            }

        stats_dict[sensor] = sensor_stats

    stats_dict["canonical_variant"] = canonical_variant
    stats_dict["created_at_utc"] = datetime.now(UTC).isoformat()

    stats_json.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
    log.info("Wrote orientation_stats.json → %s", orient_dir)

    return stats_dict


# ---------------------------------------------------------------------------
# Recording-level pipeline
# ---------------------------------------------------------------------------

def process_recording_orientation(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    canonical_variant: str = "madgwick",
    variants: list[str] | None = None,
) -> list[dict]:
    """Process orientation for all sections of a recording.

    Parameters
    ----------
    recording_name:
        Recording name (e.g. ``"2026-02-26_r1"``).
    sample_rate_hz:
        Sampling rate in Hz.
    force:
        Overwrite existing outputs when True.
    canonical_variant:
        The filter variant considered the primary one.
    variants:
        Filter variants to run. Defaults to ``["madgwick", "complementary"]``.

    Returns
    -------
    list[dict]
        List of orientation_stats dicts (one per section).
    """
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'", recording_name)
        return []

    results: list[dict] = []
    for sec_dir in section_dirs:
        log.info("Processing orientation for section %s ...", sec_dir.name)
        try:
            stats = process_section_orientation(
                sec_dir,
                sample_rate_hz=sample_rate_hz,
                force=force,
                canonical_variant=canonical_variant,
                variants=variants,
            )
            results.append(stats)
        except Exception as exc:
            log.error("Failed to process orientation for %s: %s", sec_dir.name, exc)

    return results
