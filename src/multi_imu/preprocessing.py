"""Pre-processing helpers for IMU signals."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import signal

from .data_models import IMUSensorData


def resample_signal(stream: IMUSensorData, target_rate_hz: float) -> IMUSensorData:
    """Resample an IMU stream to a target sampling rate using interpolation."""
    df = stream.data
    time = df["timestamp"].to_numpy()
    duration = time[-1] - time[0]
    num_samples = int(duration * target_rate_hz) + 1
    new_time = np.linspace(time[0], time[-1], num_samples)

    resampled = {"timestamp": new_time}
    for col in df.columns:
        if col == "timestamp":
            continue
        resampled[col] = np.interp(new_time, time, df[col].to_numpy())

    return IMUSensorData(name=f"{stream.name}_resampled", data=pd.DataFrame(resampled), sample_rate_hz=target_rate_hz)


def _magnitude(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    vectors = df[list(columns)].to_numpy()
    return np.linalg.norm(vectors, axis=1)


def remove_gravity(stream: IMUSensorData, cutoff_hz: float = 0.3, order: int = 2) -> IMUSensorData:
    """High-pass filter acceleration to remove gravity components."""
    nyquist = 0.5 * stream.sample_rate_hz
    normalized_cutoff = cutoff_hz / nyquist
    sos = signal.butter(order, normalized_cutoff, btype="highpass", output="sos")

    df = stream.data.copy()
    for axis in ["ax", "ay", "az"]:
        if axis in df:
            df[axis] = signal.sosfiltfilt(sos, df[axis])

    return IMUSensorData(name=f"{stream.name}_gravity_removed", data=df, sample_rate_hz=stream.sample_rate_hz)


def normalize_axes(stream: IMUSensorData, axes: Sequence[str]) -> IMUSensorData:
    """Scale axes to unit variance to aid alignment and correlation."""
    df = stream.data.copy()
    for axis in axes:
        if axis in df:
            std = df[axis].std() or 1.0
            df[axis] = df[axis] / std
    return IMUSensorData(name=f"{stream.name}_normalized", data=df, sample_rate_hz=stream.sample_rate_hz)


__all__ = ["resample_signal", "remove_gravity", "normalize_axes"]
