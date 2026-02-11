"""
Core IMU data models used across the analysis tools.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IMUSample:
    """
    A single, normalized IMU sample.

    Use None for values that are not present in a given source.
    """

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


