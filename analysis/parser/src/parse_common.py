"""
Common IMU data structures and CSV writer.

Target CSV columns (in order):
timestamp, ax, ay, az, gx, gy, gz, mx, my, mz
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CSV_COLUMNS: List[str] = [
    "timestamp",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
]


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

    def as_row(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "gx": self.gx,
            "gy": self.gy,
            "gz": self.gz,
            "mx": self.mx,
            "my": self.my,
            "mz": self.mz,
        }


def write_csv(samples: Iterable[IMUSample], csv_path: Path) -> None:
    """
    Write IMU samples to CSV with the fixed schema.

    Notes:
    - Missing values (None) are written as empty fields.
    - Output uses newline='' so CSV is correct on all platforms.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for s in samples:
            writer.writerow(s.as_row())


