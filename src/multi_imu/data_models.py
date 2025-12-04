"""Data models for IMU streams."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class IMUSensorData:
    """Container for a single IMU stream.

    Attributes:
        name: Human-readable sensor name.
        data: DataFrame with a ``timestamp`` column in seconds and IMU columns
            such as ``ax``, ``ay``, ``az``, ``gx``, ``gy``, ``gz``, ``mx``,
            ``my``, and ``mz``.
        sample_rate_hz: Nominal sampling rate of the stream.
    """

    name: str
    data: pd.DataFrame
    sample_rate_hz: float

    def copy(self) -> "IMUSensorData":
        return IMUSensorData(name=self.name, data=self.data.copy(), sample_rate_hz=self.sample_rate_hz)


@dataclass
class SyncedIMUData:
    """Paired IMU streams after synchronization and alignment."""

    reference: IMUSensorData
    target: IMUSensorData
    offset_seconds: float
    alignment_matrix: Optional[pd.DataFrame] = None
