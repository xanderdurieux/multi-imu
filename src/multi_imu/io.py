"""Raw log parsing and export helpers."""
from __future__ import annotations

import csv
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from multi_imu.data_models import IMUSensorData

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

# =============================================== Helper functions ===============================================
@dataclass
class _SensorData:
    device_time_ms: int
    x: float
    y: float
    z: float


def _compute_common_base_ms(frames: Iterable[pd.DataFrame]) -> Optional[int]:
    """Return the earliest device timestamp across provided frames, if available."""

    mins = [int(df["device_time_ms"].min()) for df in frames if "device_time_ms" in df.columns and not df.empty]
    return min(mins) if mins else None


def _relative_timestamp_from_device(df: pd.DataFrame, base_ms: Optional[int] = None) -> pd.Series:
    """Return timestamps in seconds using the device clock."""

    if "device_time_ms" not in df.columns:
        raise ValueError("Expected 'device_time_ms' in dataframe")

    base = base_ms if base_ms is not None else df["device_time_ms"].min()
    return (df["device_time_ms"] - base) / 1000


def _standardize_sensor_frame(
    df: pd.DataFrame, sensor_type: str, add_magnitude: bool = True, base_ms: Optional[int] = None
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
    standardized["timestamp"] = _relative_timestamp_from_device(standardized, base_ms=base_ms)
    standardized = standardized.rename(
        columns={"x": axis_labels[0], "y": axis_labels[1], "z": axis_labels[2]}
    ).drop(columns=["device_time_ms"])

    if add_magnitude:
        standardized["total"] = np.sqrt(sum(standardized[label] ** 2 for label in axis_labels))

    ordered_cols = ["timestamp", *axis_labels]
    if add_magnitude:
        ordered_cols.append("total")

    return standardized[ordered_cols]


# =============================================== Log parsers ===============================================
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

            data_str = linelist[2].replace("received", "")
            if "Notifications" in data_str:
                continue

            hex_bytes = re.findall(r"[0-9A-Fa-f]{2}", data_str)
            raw = bytes(int(h, 16) for h in hex_bytes)
            sensor_type, x, y, z, sensor_time_ms = struct.unpack("<B3xfffI", raw)

            if sensor_type == 1:
                acc_data.append(
                    _SensorData(device_time_ms=sensor_time_ms, x=x * 9.81, y=y * 9.81, z=z * 9.81)
                )
            elif sensor_type == 2:
                gyro_data.append(
                    _SensorData(device_time_ms=sensor_time_ms, x=x, y=y, z=z)
                )
            elif sensor_type == 4:
                mag_data.append(
                    _SensorData(device_time_ms=sensor_time_ms, x=x, y=y, z=z)
                )

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
            device_time_ms = int(ts_str)

            acc_x = int(linelist[1]) * ACCEL_SENS["16G"] * 9.81 / 1000
            acc_y = int(linelist[2]) * ACCEL_SENS["16G"] * 9.81 / 1000
            acc_z = int(linelist[3]) * ACCEL_SENS["16G"] * 9.81 / 1000

            gyro_x = int(linelist[4]) * GYRO_SENS["2000DPS"] / 1000
            gyro_y = int(linelist[5]) * GYRO_SENS["2000DPS"] / 1000
            gyro_z = int(linelist[6]) * GYRO_SENS["2000DPS"] / 1000

            acc_data.append(
                _SensorData(device_time_ms=device_time_ms, x=acc_x, y=acc_y, z=acc_z)
            )
            gyro_data.append(
                _SensorData(device_time_ms=device_time_ms, x=gyro_x, y=gyro_y, z=gyro_z)
            )

    return pd.DataFrame(acc_data), pd.DataFrame(gyro_data)


def parse_phone_log(log_file: str) -> pd.DataFrame:
    """Parse a smartphone CSV log with timestamp and axis columns."""

    data = []

    with open(log_file) as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if reader.line_num == 1:
                continue

            data.append(
                _SensorData(device_time_ms=int(float(line[1]) * 1000), x=float(line[2]), y=float(line[3]), z=float(line[4]))
            )

    return pd.DataFrame(data)


# =============================================== Export dataframes ===============================================
def export_sensor_frames(sensor_frames: Dict[str, pd.DataFrame], output_dir: str, add_magnitude: bool = True):
    """Write standardized sensor CSVs for downstream analysis.

    Each key in ``sensor_frames`` should be ``"acc"``, ``"gyro"``, or ``"mag"`` with
    corresponding dataframes. Files are written as ``acc.csv``, ``gyro.csv``, etc.
    """

    os.makedirs(output_dir, exist_ok=True)

    base_ms = _compute_common_base_ms(sensor_frames.values())

    for sensor_type, df in sensor_frames.items():
        standardized = _standardize_sensor_frame(df, sensor_type, add_magnitude, base_ms=base_ms)
        standardized.to_csv(os.path.join(output_dir, f"{sensor_type}.csv"), index=False)


def export_joined_imu_frame(sensor_frames: Dict[str, pd.DataFrame], output_dir: str, add_magnitude: bool = True) -> pd.DataFrame:
    """Merge accelerometer and gyroscope frames and write a single CSV to ``output_dir``.

    The output contains ``timestamp``, ``ax/ay/az``, ``gx/gy/gz``, and optional ``mx/my/mz`` columns
    sorted by time. The merged dataframe is returned for convenience.
    """

    base_ms = _compute_common_base_ms(sensor_frames.values())

    standardized_frames = []
    for sensor_type, df in sensor_frames.items():
        standardized = _standardize_sensor_frame(df, sensor_type, add_magnitude, base_ms=base_ms)
        standardized_frames.append(standardized)

    combined = pd.merge(standardized_frames[0], standardized_frames[1], on="timestamp", how="outer")
    if len(standardized_frames) > 2:
        combined = pd.merge(combined, standardized_frames[2], on="timestamp", how="outer")
    combined = combined[[col for col in ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"] if col in combined]]
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    combined.to_csv(os.path.join(output_dir, "imu.csv"), index=False)
    return combined


def export_imu_csv(sensor: IMUSensorData, output_path: str) -> None:
    """Export an :class:`IMUSensorData` to a CSV file.
    
    Args:
        sensor: IMUSensorData to export
        output_path: Path where the CSV file will be written
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sensor.data.to_csv(output_path, index=False)


# =============================================== Load dataframes ===============================================
def load_sensor_csv(path: str) -> pd.DataFrame:
    """Load a sensor CSV into a dataframe."""
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_imu_csv(path: str, name: str) -> IMUSensorData:
    """Load a standardized IMU CSV into :class:`IMUSensorData`."""

    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    sample_rate_hz = 1.0 / df["timestamp"].diff().median()

    return IMUSensorData(name=name, data=df, sample_rate_hz=sample_rate_hz)
