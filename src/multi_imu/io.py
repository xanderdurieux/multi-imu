"""I/O helpers for IMU datasets."""
from __future__ import annotations
import pandas as pd
from typing import Iterable, Mapping, Optional

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
