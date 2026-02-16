"""Core IMU data models for processed sensor streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IMUSample:
    """Single normalized IMU sample with timestamp and sensor readings."""

    timestamp: int
    ax: Optional[float] = None
    ay: Optional[float] = None
    az: Optional[float] = None
    gx: Optional[float] = None
    gy: Optional[float] = None
    gz: Optional[float] = None
    mx: Optional[float] = None
    my: Optional[float] = None
    mz: Optional[float] = None


