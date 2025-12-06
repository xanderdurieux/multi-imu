"""Data model for IMU streams."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class IMUSensorData:
    """Container for a single IMU stream with axis data and timestamps."""

    name: str
    data: pd.DataFrame
    sample_rate_hz: float

    def copy(self) -> "IMUSensorData":
        return IMUSensorData(name=self.name, data=self.data.copy(), sample_rate_hz=self.sample_rate_hz)
