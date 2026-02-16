"""CSV schema and I/O utilities for processed IMU streams."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, TYPE_CHECKING

from .model import IMUSample

if TYPE_CHECKING:
    import pandas as pd


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


def write_csv(samples: Iterable[IMUSample], csv_path: Path) -> None:
    """Write IMU samples to CSV with standardized schema."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for s in samples:
            row: Dict[str, object] = {
                "timestamp": s.timestamp,
                "ax": s.ax,
                "ay": s.ay,
                "az": s.az,
                "gx": s.gx,
                "gy": s.gy,
                "gz": s.gz,
                "mx": s.mx,
                "my": s.my,
                "mz": s.mz,
            }
            writer.writerow(row)


def load_dataframe(csv_path: Path) -> "pd.DataFrame":
    """Load processed IMU CSV into DataFrame with standardized columns."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Ensure all expected columns exist; missing ones are filled with NaN.
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert to numeric where possible (non-numeric values become NaN).
    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


