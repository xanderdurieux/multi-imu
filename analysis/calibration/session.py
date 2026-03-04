"""Recording-level calibration pipeline.

Estimates world-frame orientation calibration parameters independently for each
sensor in a recording, then optionally applies those parameters to produce
bias-corrected, world-rotated CSVs.

CLI::

    python -m calibration.session 2026-02-26_5
    python -m calibration.session 2026-02-26_5 --stage parsed --no-apply

Algorithm
---------
For each sensor CSV found in ``data/recordings/<recording>/<stage>/``:

1. Detect calibration sequences with
   :func:`parser.split_sections.find_calibration_segments`.
2. Extract pre/post-peak static windows via
   :func:`calibration.static_windows.extract_static_windows`.
3. Estimate gyro bias, gravity vector, and magnetometer hard-iron offset via
   :func:`calibration.per_sensor.calibrate_sensor`.
4. Compute the sensor-to-world rotation matrix via
   :func:`calibration.orientation.compute_orientation_from_vectors` (TRIAD
   when mag data is available, gravity-only otherwise).
5. Write ``calibrated/calibration.json`` with all parameters and quality
   metrics.
6. Unless ``--no-apply`` is given, write corrected sensor CSVs to
   ``calibrated/`` with:
   - Gyro bias subtracted from ``gx, gy, gz``.
   - Magnetometer hard-iron offset subtracted from ``mx, my, mz``.
   - All vectors (acc, gyro, mag) rotated to the world frame.

Output layout::

    data/recordings/<recording>/calibrated/
        calibration.json
        sporsa.csv
        arduino.csv
"""

from __future__ import annotations

import json
import logging
import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_dataframe, write_dataframe
from common.paths import find_sensor_csv, recording_stage_dir

from .static_windows import StaticWindows, extract_static_windows
from .per_sensor import SensorCalibration, calibrate_sensor
from .orientation import OrientationCalibration, compute_orientation_from_vectors

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def _apply_calibration_to_df(
    df: pd.DataFrame,
    sensor_cal: SensorCalibration,
    orientation: OrientationCalibration,
) -> pd.DataFrame:
    """Return a new DataFrame with bias corrections and world-frame rotation applied.

    Steps
    -----
    1. Subtract gyroscope bias from ``gx, gy, gz``.
    2. Subtract magnetometer hard-iron offset from ``mx, my, mz`` (only for
       rows where mag data is present).
    3. Rotate every sensor vector to the world frame via
       ``orientation.rotation_sensor_to_world``.

    The ``timestamp`` and any extra columns are preserved unchanged.

    Parameters
    ----------
    df:
        Parsed IMU DataFrame with standard columns.
    sensor_cal:
        Per-sensor calibration parameters.
    orientation:
        World-frame rotation calibration.

    Returns
    -------
    pd.DataFrame
        Calibrated DataFrame with the same column layout as the input.
    """
    out = df.copy()
    R = orientation.rotation_sensor_to_world

    # 1. Subtract gyro bias
    gyro_bias = sensor_cal.gyro_bias_deg_per_s
    for col, bias_val in zip(["gx", "gy", "gz"], gyro_bias):
        if col in out.columns:
            out[col] = out[col] - bias_val

    # 2. Subtract mag hard-iron offset (only rows with valid mag)
    if sensor_cal.mag_hard_iron_uT is not None:
        hard_iron = sensor_cal.mag_hard_iron_uT
        mag_mask = out["mx"].notna() if "mx" in out.columns else pd.Series(False, index=out.index)
        for col, offset in zip(["mx", "my", "mz"], hard_iron):
            if col in out.columns:
                out.loc[mag_mask, col] = out.loc[mag_mask, col] - offset

    # 3. Rotate all sensor vectors to world frame
    for cols in (["ax", "ay", "az"], ["gx", "gy", "gz"]):
        if all(c in out.columns for c in cols):
            raw = out[cols].to_numpy(dtype=float).copy()
            valid = np.all(np.isfinite(raw), axis=1)
            if valid.any():
                raw[valid] = (R @ raw[valid].T).T
            out[cols] = raw

    # Mag rotation (only non-NaN rows)
    mag_cols = ["mx", "my", "mz"]
    if all(c in out.columns for c in mag_cols):
        raw = out[mag_cols].to_numpy(dtype=float).copy()
        valid = np.all(np.isfinite(raw), axis=1)
        if valid.any():
            raw[valid] = (R @ raw[valid].T).T
        out[mag_cols] = raw

    return out


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _ndarray_to_list(arr: np.ndarray) -> list:
    return arr.tolist()


def _sensor_cal_to_dict(
    sensor_cal: SensorCalibration,
    orientation: OrientationCalibration,
) -> dict[str, Any]:
    return {
        "gyro_bias_deg_per_s": _ndarray_to_list(sensor_cal.gyro_bias_deg_per_s),
        "gravity_vector_m_per_s2": _ndarray_to_list(sensor_cal.gravity_vector_m_per_s2),
        "gravity_magnitude_m_per_s2": round(sensor_cal.gravity_magnitude_m_per_s2, 6),
        "mag_hard_iron_offset_uT": (
            _ndarray_to_list(sensor_cal.mag_hard_iron_uT)
            if sensor_cal.mag_hard_iron_uT is not None else None
        ),
        "rotation_sensor_to_world": _ndarray_to_list(orientation.rotation_sensor_to_world),
        "quality": {
            "gravity_residual_m_per_s2": round(orientation.gravity_residual_m_per_s2, 6),
            "n_static_samples": sensor_cal.n_static_samples,
            "n_mag_samples": sensor_cal.n_mag_samples,
            "yaw_calibrated": orientation.yaw_calibrated,
        },
    }


# ---------------------------------------------------------------------------
# Per-sensor pipeline
# ---------------------------------------------------------------------------

def _calibrate_one_sensor(
    csv_path: Path,
    *,
    sample_rate_hz: float,
    buffer_samples: int,
    mag_min_samples: int,
) -> tuple[SensorCalibration, OrientationCalibration, StaticWindows]:
    """Run the full calibration pipeline for one sensor CSV.

    Returns
    -------
    (sensor_cal, orientation_cal, windows)
    """
    df = load_dataframe(csv_path)
    if df.empty:
        raise ValueError(f"Sensor CSV is empty: {csv_path}")

    # Step 1 & 2: detect calibration sequences and extract static windows
    windows = extract_static_windows(
        df,
        sample_rate_hz=sample_rate_hz,
        buffer_samples=buffer_samples,
    )

    if windows.n_calibration_segments == 0:
        raise ValueError(
            f"No calibration segments found in {csv_path.name}. "
            "This recording may lack a calibration sequence (e.g. rec1 or rec6)."
        )

    # Step 3: per-sensor bias and gravity estimation
    sensor_cal = calibrate_sensor(windows, mag_min_samples=mag_min_samples)

    # Step 4: orientation to world frame.
    #
    # For TRIAD we need the apparent magnetic field direction in the sensor
    # frame.  From a single static position, the best estimate is the raw mean
    # magnetometer reading: mean(mag_static) = Earth_field + hard_iron_offset.
    # We use this vector directly as b2 — NOT "mean − mean = 0".
    # The stored mag_hard_iron_uT is the same value and is used later as a DC
    # reference when applying the calibration to the full dataset.
    mag_for_triad: np.ndarray | None = sensor_cal.mag_hard_iron_uT  # raw mean in sensor frame

    orientation = compute_orientation_from_vectors(
        gravity_sensor=sensor_cal.gravity_vector_m_per_s2,
        mag_sensor_corrected=mag_for_triad,
    )

    return sensor_cal, orientation, windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_recording(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    apply: bool = True,
    plot: bool = True,
    sample_rate_hz: float = 100.0,
    buffer_samples: int = 10,
    mag_min_samples: int = 5,
) -> Path:
    """Run the calibration pipeline for one recording.

    Reads sensor CSVs from ``data/recordings/<recording_name>/<stage_in>/``,
    computes calibration parameters for each sensor, and writes results to
    ``data/recordings/<recording_name>/calibrated/``.

    Parameters
    ----------
    recording_name:
        Recording identifier, e.g. ``"2026-02-26_5"``.
    stage_in:
        Input stage containing the parsed sensor CSVs (default: ``"parsed"``).
    apply:
        When ``True`` (default), write calibrated sensor CSVs in addition to
        the ``calibration.json``.  Set to ``False`` to only compute and save
        the calibration parameters.
    plot:
        When ``True`` (default), generate world-frame plots in ``calibrated/``
        after writing the calibrated CSVs.
    sample_rate_hz:
        Approximate sensor sampling rate in Hz, used for calibration segment
        detection.
    buffer_samples:
        Guard band (samples) excluded around peak edges when cutting static
        windows.
    mag_min_samples:
        Minimum number of valid magnetometer samples required to compute a
        hard-iron offset.

    Returns
    -------
    Path
        Path to the written ``calibration.json`` file.

    Raises
    ------
    FileNotFoundError
        If the input stage directory or expected sensor CSVs are not found.
    ValueError
        If no calibration segments are found in any sensor CSV.
    """
    out_dir = recording_stage_dir(recording_name, "calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_json: dict[str, Any] = {
        "metadata": {
            "recording": recording_name,
            "stage": stage_in,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    }

    for sensor_name in _SENSORS:
        try:
            csv_path = find_sensor_csv(recording_name, stage_in, sensor_name)
        except FileNotFoundError:
            log.warning("No CSV found for sensor '%s' in stage '%s'. Skipping.", sensor_name, stage_in)
            continue

        print(f"[{recording_name}/calibrated] calibrating {sensor_name} ← {csv_path.name}")

        try:
            sensor_cal, orientation, windows = _calibrate_one_sensor(
                csv_path,
                sample_rate_hz=sample_rate_hz,
                buffer_samples=buffer_samples,
                mag_min_samples=mag_min_samples,
            )
        except ValueError as exc:
            log.warning(
                "Calibration skipped for %s/%s: %s",
                recording_name, sensor_name, exc,
            )
            print(f"[{recording_name}/calibrated] WARNING: skipping {sensor_name} — {exc}")
            continue

        cal_json[sensor_name] = _sensor_cal_to_dict(sensor_cal, orientation)

        if apply:
            df = load_dataframe(csv_path)
            calibrated_df = _apply_calibration_to_df(df, sensor_cal, orientation)
            out_csv = out_dir / f"{sensor_name}.csv"
            write_dataframe(calibrated_df, out_csv)
            print(f"[{recording_name}/calibrated] wrote {out_csv.name}")

        _log_summary(recording_name, sensor_name, sensor_cal, orientation)

    json_path = out_dir / "calibration.json"
    json_path.write_text(json.dumps(cal_json, indent=2), encoding="utf-8")
    print(f"[{recording_name}/calibrated] calibration.json")

    if apply and plot:
        plot_calibration(recording_name)

    return json_path


def plot_calibration(recording_name: str) -> None:
    """Generate calibration plots for *recording_name* (delegates to visualization)."""
    from visualization.plot_calibration import plot_calibration_stage
    plot_calibration_stage(recording_name)


def _log_summary(
    recording: str,
    sensor: str,
    sensor_cal: SensorCalibration,
    orientation: OrientationCalibration,
) -> None:
    g_vec = sensor_cal.gravity_vector_m_per_s2
    log.info(
        "[%s/%s] gyro_bias=[%.3f, %.3f, %.3f] deg/s | "
        "gravity=[%.3f, %.3f, %.3f] m/s² (|g|=%.4f) | "
        "residual=%.4f m/s² | yaw_calibrated=%s | "
        "n_static=%d, n_mag=%d",
        recording, sensor,
        *sensor_cal.gyro_bias_deg_per_s,
        *g_vec, sensor_cal.gravity_magnitude_m_per_s2,
        orientation.gravity_residual_m_per_s2,
        orientation.yaw_calibrated,
        sensor_cal.n_static_samples,
        sensor_cal.n_mag_samples,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calibration.session",
        description=(
            "Estimate world-frame orientation calibration parameters "
            "independently for each sensor in a recording, then optionally "
            "apply them to produce bias-corrected, world-rotated CSVs.\n\n"
            "Reads from data/recordings/<recording_name>/<stage>/ and writes "
            "to data/recordings/<recording_name>/calibrated/."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording identifier, e.g. '2026-02-26_5'.",
    )
    parser.add_argument(
        "--stage",
        default="parsed",
        help="Input stage directory containing sensor CSVs (default: parsed).",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help=(
            "Only write calibration.json; skip producing calibrated sensor CSVs."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating calibration plots after writing CSVs.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Approximate sensor sampling rate in Hz (default: 100.0).",
    )
    parser.add_argument(
        "--buffer-samples",
        type=int,
        default=10,
        help=(
            "Guard band (samples) excluded around calibration peaks when "
            "cutting static windows (default: 10)."
        ),
    )
    parser.add_argument(
        "--mag-min-samples",
        type=int,
        default=5,
        help=(
            "Minimum valid magnetometer samples needed to compute a "
            "hard-iron offset (default: 5)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    json_path = calibrate_recording(
        recording_name=args.recording_name,
        stage_in=args.stage,
        apply=not args.no_apply,
        plot=not args.no_plot,
        sample_rate_hz=args.sample_rate_hz,
        buffer_samples=args.buffer_samples,
        mag_min_samples=args.mag_min_samples,
    )
    print(f"\ncalibration parameters: {json_path}")


if __name__ == "__main__":
    main()
