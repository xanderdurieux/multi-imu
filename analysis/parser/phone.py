"""Parser for phone IMU logs exported by phyphox."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.paths import CSV_COLUMNS, write_csv
from common.signals import add_imu_norms
from .gps import write_gps_csv

_NS_TO_MS = 1e-6          # nanoseconds → milliseconds
_RAD_TO_DEG = 180.0 / np.pi  # rad/s → deg/s

# Maximum time gap (ms) allowed when merging sensor streams by nearest timestamp.
# At 100 Hz the nominal interval is 10 ms; 50 ms covers ≥5 samples of jitter.
_MERGE_TOLERANCE_MS = 50.0

# Target sample interval after resampling, matching the sporsa device rate.
# The phone hardware delivers ~16 ms (62.5 Hz); resampling to 10 ms ensures the
# calibration-segment detector's MIN_PEAK_DISTANCE_MS=150 ms converts to the
# same 15-sample minimum it was tuned for, preventing double-detection of taps.
_TARGET_INTERVAL_MS = 10.0

_SENSOR_COLS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")


def _load_sensor_csv(path: Path) -> pd.DataFrame:
    """Load sensor csv."""
    if not path.exists():
        return pd.DataFrame(columns=["ts_ms", "x", "y", "z"])
    try:
        df = pd.read_csv(path, usecols=["time", "x", "y", "z"], dtype=float)
    except Exception:
        return pd.DataFrame(columns=["ts_ms", "x", "y", "z"])
    df = df.rename(columns={"time": "ts_ms"})
    df["ts_ms"] = df["ts_ms"] * _NS_TO_MS
    return df.dropna(subset=["ts_ms"]).sort_values("ts_ms").reset_index(drop=True)


def _merge_sensors(
    accel: pd.DataFrame,
    gyro: pd.DataFrame,
    mag: pd.DataFrame,
) -> pd.DataFrame:
    """Return merge sensors."""
    if not accel.empty:
        base = accel.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    elif not gyro.empty:
        base = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})
        gyro = pd.DataFrame(columns=["ts_ms", "x", "y", "z"])
    else:
        base = mag.rename(columns={"x": "mx", "y": "my", "z": "mz"})
        mag = pd.DataFrame(columns=["ts_ms", "x", "y", "z"])

    result = base

    if not gyro.empty:
        g = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})
        result = pd.merge_asof(
            result, g, on="ts_ms", direction="nearest", tolerance=_MERGE_TOLERANCE_MS
        )

    if not mag.empty:
        m = mag.rename(columns={"x": "mx", "y": "my", "z": "mz"})
        result = pd.merge_asof(
            result, m, on="ts_ms", direction="nearest", tolerance=_MERGE_TOLERANCE_MS
        )

    return result


def _resample_to_target(merged: pd.DataFrame) -> pd.DataFrame:
    """Return resample to target."""
    ts = merged["ts_ms"].to_numpy(dtype=float)
    t_new = np.arange(ts[0], ts[-1], _TARGET_INTERVAL_MS)
    out: dict[str, np.ndarray] = {"ts_ms": t_new}
    for col in _SENSOR_COLS:
        if col in merged.columns:
            vals = merged[col].to_numpy(dtype=float)
            out[col] = np.interp(t_new, ts, vals)
    return pd.DataFrame(out)


def parse_phone_recording(folder: Path) -> pd.DataFrame:
    """Parse phone recording."""
    accel = _load_sensor_csv(folder / "TotalAcceleration.csv")
    gyro = _load_sensor_csv(folder / "Gyroscope.csv")
    mag = _load_sensor_csv(folder / "Magnetometer.csv")

    if accel.empty and gyro.empty and mag.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    for col in ("x", "y", "z"):
        if col in gyro.columns:
            gyro[col] = gyro[col] * _RAD_TO_DEG

    merged = _resample_to_target(_merge_sensors(accel, gyro, mag))

    out = pd.DataFrame({col: pd.NA for col in CSV_COLUMNS}, index=range(len(merged)))
    out["timestamp"] = merged["ts_ms"].values
    for col in _SENSOR_COLS:
        if col in merged.columns:
            out[col] = merged[col].values

    for col in CSV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return add_imu_norms(out)


def parse_phone_gps(folder: Path) -> pd.DataFrame:
    """Parse phone gps."""
    path = folder / "Location.csv"
    _empty = pd.DataFrame(
        columns=["latitude", "longitude", "elevation_m", "time_utc", "speed_m_s"]
    )
    if not path.exists():
        return _empty
    try:
        df = pd.read_csv(
            path,
            usecols=["time", "latitude", "longitude", "altitude", "speed"],
            dtype={"time": "int64", "latitude": float, "longitude": float,
                   "altitude": float, "speed": float},
        )
    except Exception:
        return _empty

    df = df.dropna(subset=["latitude", "longitude", "time"])
    if df.empty:
        return _empty

    time_utc = pd.to_datetime(df["time"], unit="ns", utc=True)
    return pd.DataFrame({
        "latitude": df["latitude"].values,
        "longitude": df["longitude"].values,
        "elevation_m": df["altitude"].values,
        "time_utc": time_utc,
        "speed_m_s": df["speed"].values,
    }).reset_index(drop=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build arg parser."""
    parser = argparse.ArgumentParser(
        prog="python -m parser.phone",
        description="Convert a phyphox phone recording folder to processed IMU CSV.",
    )
    parser.add_argument("source_folder", type=Path, help="Path to phyphox recording folder.")
    parser.add_argument("destination_csv", type=Path, help="Output CSV path.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Run the command-line interface."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    df = parse_phone_recording(args.source_folder)
    write_csv(df, args.destination_csv)


if __name__ == "__main__":
    main()
