"""Analytics for motion events such as falls or harsh braking."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from .data_models import IMUSensorData


EventList = List[Dict[str, float]]


def _event_from_indices(indices, timestamps, label: str) -> EventList:
    events = []
    for idx in indices:
        events.append({"label": label, "timestamp": float(timestamps[idx])})
    return events


def detect_falls(stream: IMUSensorData, threshold_g: float = 2.5) -> EventList:
    """Detect peaks in acceleration magnitude that exceed the given threshold."""
    df = stream.data
    accel_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not accel_cols:
        return []
    magnitude = np.linalg.norm(df[accel_cols].to_numpy(), axis=1)
    indices = np.where(magnitude > threshold_g * 9.81)[0]
    return _event_from_indices(indices, df["timestamp"], "fall")


def detect_braking_events(stream: IMUSensorData, axis: str = "ax", threshold_m_s2: float = -5.0) -> EventList:
    """Detect deceleration events on a chosen axis."""
    if axis not in stream.data:
        return []
    indices = np.where(stream.data[axis] < threshold_m_s2)[0]
    return _event_from_indices(indices, stream.data["timestamp"], "brake")


def detect_turns(stream: IMUSensorData, axis: str = "gz", threshold_rad_s: float = 2.0) -> EventList:
    """Detect turn events using gyroscope rate around a principal axis."""
    if axis not in stream.data:
        return []
    indices = np.where(np.abs(stream.data[axis]) > threshold_rad_s)[0]
    return _event_from_indices(indices, stream.data["timestamp"], "turn")


__all__ = ["detect_falls", "detect_braking_events", "detect_turns"]
