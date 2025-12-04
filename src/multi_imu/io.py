"""I/O helpers for IMU datasets."""
from __future__ import annotations

import csv
import os
import re
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .data_models import IMUSensorData


DEFAULT_COLUMN_MAP = {
    "time": "timestamp",
    "t": "timestamp",
    "acc_x": "ax",
    "acc_y": "ay",
    "acc_z": "az",
    "gyro_x": "gx",
    "gyro_y": "gy",
    "gyro_z": "gz",
    "mag_x": "mx",
    "mag_y": "my",
    "mag_z": "mz",
}

ACCEL_SENS = {
    "4G": 0.122,
    "8G": 0.244,
    "16G": 0.488,
}

GYRO_SENS = {
    "125DPS": 4.375,
    "250DPS": 8.750,
    "500DPS": 17.500,
    "1000DPS": 35.000,
    "2000DPS": 70.000,
}


@dataclass
class _SensorData:
    ts: datetime
    x: float
    y: float
    z: float


def load_imu_csv(
    path: str,
    name: str,
    sample_rate_hz: float,
    time_column: str = "timestamp",
    column_map: Optional[Mapping[str, str]] = None,
    required_columns: Optional[Iterable[str]] = None,
) -> IMUSensorData:
    """Load an IMU CSV file into a standardized :class:`IMUSensorData`.

    Args:
        path: Path to the CSV file.
        name: Label for the sensor (e.g., "arduino", "custom").
        sample_rate_hz: Nominal sampling rate of the sensor.
        time_column: Name of the column containing timestamps in seconds.
        column_map: Optional mapping to rename columns to standard keys.
        required_columns: Optional iterable of required columns to keep.

    Returns:
        IMUSensorData with renamed columns and timestamps sorted.
    """

    df = pd.read_csv(path)
    rename_map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    df = df.rename(columns=rename_map)

    if time_column != "timestamp":
        df = df.rename(columns={time_column: "timestamp"})

    if required_columns:
        df = df[[col for col in df.columns if col in set(required_columns) or col == "timestamp"]]

    df = df.sort_values("timestamp").reset_index(drop=True)

    return IMUSensorData(name=name, data=df, sample_rate_hz=sample_rate_hz)


def _ts_to_seconds(ts: datetime) -> float:
    """Convert ``datetime`` to seconds since epoch, assuming UTC when naive."""

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.timestamp()


def _standardize_sensor_frame(
    df: pd.DataFrame, sensor_type: str, add_magnitude: bool = True
) -> pd.DataFrame:
    """Convert parsed dataframes into the library's standard CSV schema."""

    if df.empty:
        return pd.DataFrame()

    axis_map = {
        "acc": ("ax", "ay", "az"),
        "gyro": ("gx", "gy", "gz"),
        "mag": ("mx", "my", "mz"),
    }

    if sensor_type not in axis_map:
        raise ValueError(f"Unsupported sensor type: {sensor_type}")

    axis_labels = axis_map[sensor_type]
    standardized = df.copy()
    standardized["timestamp"] = standardized["ts"].apply(_ts_to_seconds)
    standardized = standardized.rename(
        columns={"x": axis_labels[0], "y": axis_labels[1], "z": axis_labels[2]}
    ).drop(columns=["ts"])

    if add_magnitude:
        standardized["total"] = np.sqrt(sum(standardized[label] ** 2 for label in axis_labels))

    ordered_cols = ["timestamp", *axis_labels]
    if add_magnitude:
        ordered_cols.append("total")

    return standardized[ordered_cols]


def _export_to_csv(sensor_frames: Dict[str, pd.DataFrame], output_dir: str, add_magnitude: bool = True):
    """Write standardized sensor CSVs for downstream analysis."""

    os.makedirs(output_dir, exist_ok=True)

    for sensor_type, df in sensor_frames.items():
        standardized = _standardize_sensor_frame(df, sensor_type, add_magnitude)
        standardized.to_csv(os.path.join(output_dir, f"{sensor_type}.csv"), index=False)


def _get_arduino_date(log_file: str) -> datetime.date:
    with open(log_file) as f:
        first_line = f.readline()
        date_str = first_line.split(",")[1]
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()


def parse_arduino_log(log_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse Arduino log and return accelerometer, gyro, and magnetometer frames."""

    acc_data = []
    gyro_data = []
    mag_data = []

    with open(log_file) as f:
        for line in f:
            if not line.startswith("A"):
                continue

            linelist = line.split("\t")
            ts_str = linelist[1]
            d = _get_arduino_date(log_file)
            t = datetime.strptime(ts_str, "%H:%M:%S.%f").time()
            ts = datetime.combine(d, t)

            data_str = linelist[2].replace("received", "")
            if "Notifications" in data_str:
                continue

            hex_bytes = re.findall(r"[0-9A-Fa-f]{2}", data_str)
            raw = bytes(int(h, 16) for h in hex_bytes)
            sensor_type, x, y, z, _ = struct.unpack("<B3xfffI", raw)

            if sensor_type == 1:
                acc_data.append(_SensorData(ts=ts, x=x * 9.81, y=y * 9.81, z=z * 9.81))
            elif sensor_type == 2:
                gyro_data.append(_SensorData(ts=ts, x=x, y=y, z=z))
            elif sensor_type == 4:
                mag_data.append(_SensorData(ts=ts, x=x, y=y, z=z))

    return pd.DataFrame(acc_data), pd.DataFrame(gyro_data), pd.DataFrame(mag_data)


def parse_sporsa_log(log_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Sporsa log file and return accelerometer and gyroscope frames."""

    acc_data = []
    gyro_data = []

    with open(log_file) as f:
        for line in f:
            linelist = line.split(",")
            if len(linelist) != 7:
                continue

            ts_str = linelist[0].replace("uart:~$ ", "")
            ts = datetime.fromtimestamp(int(ts_str) / 1000.0, UTC) + timedelta(hours=1)

            acc_x = int(linelist[1]) * ACCEL_SENS["16G"] * 9.81 / 1000
            acc_y = int(linelist[2]) * ACCEL_SENS["16G"] * 9.81 / 1000
            acc_z = int(linelist[3]) * ACCEL_SENS["16G"] * 9.81 / 1000

            gyro_x = int(linelist[4]) * GYRO_SENS["2000DPS"] / 1000
            gyro_y = int(linelist[5]) * GYRO_SENS["2000DPS"] / 1000
            gyro_z = int(linelist[6]) * GYRO_SENS["2000DPS"] / 1000

            acc_data.append(_SensorData(ts=ts, x=acc_x, y=acc_y, z=acc_z))
            gyro_data.append(_SensorData(ts=ts, x=gyro_x, y=gyro_y, z=gyro_z))

    return pd.DataFrame(acc_data), pd.DataFrame(gyro_data)


def parse_phone_log(log_file: str) -> pd.DataFrame:
    """Parse a smartphone CSV log with timestamp and axis columns."""

    data = []

    with open(log_file) as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if reader.line_num == 1:
                continue

            ts = datetime.fromtimestamp(int(line[0]) / 1_000_000_000.0, UTC)
            data.append(_SensorData(ts=ts, x=float(line[2]), y=float(line[3]), z=float(line[4])))

    return pd.DataFrame(data)


def export_raw_session(
    sensor_frames: Dict[str, pd.DataFrame], output_dir: str, add_magnitude: bool = True
):
    """Export parsed raw frames to standardized CSVs ready for ``load_imu_csv``."""

    _export_to_csv(sensor_frames, output_dir, add_magnitude=add_magnitude)
