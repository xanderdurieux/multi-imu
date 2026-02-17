"""CSV schema and I/O utilities for processed IMU streams."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


CSV_COLUMNS: list[str] = [
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


def write_dataframe(df: pd.DataFrame, csv_path: Path) -> None:
    """Write processed IMU DataFrame to CSV with standardized columns."""
    out = df.copy()

    # Ensure all expected columns exist; missing ones are filled with NA.
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # Reorder with schema columns first, then keep any extra columns at the end.
    extra = [c for c in out.columns if c not in CSV_COLUMNS]
    out = out[CSV_COLUMNS + extra]

    out.to_csv(csv_path, index=False)

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load processed IMU CSV into DataFrame with standardized columns."""

    df = pd.read_csv(csv_path)

    # Ensure all expected columns exist; missing ones are filled with NaN.
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert to numeric where possible (non-numeric values become NaN).
    for col in CSV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


