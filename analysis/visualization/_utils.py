"""Internal visualization utilities — shared constants, IO, and plot helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SENSORS: tuple[str, ...] = ("sporsa", "arduino")

SENSOR_COLORS: dict[str, str] = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}

ACC_COLS: tuple[str, ...] = ("ax", "ay", "az")
GYRO_COLS: tuple[str, ...] = ("gx", "gy", "gz")
MAG_COLS: tuple[str, ...] = ("mx", "my", "mz")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _as_float_vector(values: np.ndarray) -> np.ndarray:
    """Return a flattened float vector for plotting helpers."""
    return np.ravel(np.asarray(values, dtype=float))


def _mask_valid_plot_x(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite, monotonically non-decreasing values."""
    x = _as_float_vector(x)
    if x.size == 0:
        return np.array([], dtype=bool)
    finite = np.isfinite(x)
    diffs = np.diff(x, prepend=x[0])
    monotone = diffs >= 0
    return finite & monotone


def filter_valid_plot_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays already filtered to values safe to plot."""
    x = _as_float_vector(x)
    y = _as_float_vector(y)
    if x.shape != y.shape:
        raise ValueError(f"Plot arrays must have the same shape, got {x.shape} and {y.shape}")
    valid = _mask_valid_plot_x(x) & np.isfinite(y)
    return x[valid], y[valid]


def strict_vector_norm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return row-wise vector norm while preserving NaN for incomplete rows."""
    if not cols:
        return np.array([], dtype=float)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for vector norm: {missing}")

    arr = np.column_stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in cols])
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    out[valid] = np.linalg.norm(arr[valid], axis=1)
    return out


def acc_norm(df: pd.DataFrame) -> np.ndarray | None:
    """Accelerometer magnitude — uses pre-computed ``acc_norm`` column when present."""
    if "acc_norm" in df.columns:
        return pd.to_numeric(df["acc_norm"], errors="coerce").to_numpy(dtype=float)
    cols = [c for c in ACC_COLS if c in df.columns]
    return strict_vector_norm(df, cols) if cols else None


def gyro_norm(df: pd.DataFrame) -> np.ndarray | None:
    """Gyroscope magnitude — uses pre-computed ``gyro_norm`` column when present."""
    if "gyro_norm" in df.columns:
        return pd.to_numeric(df["gyro_norm"], errors="coerce").to_numpy(dtype=float)
    cols = [c for c in GYRO_COLS if c in df.columns]
    return strict_vector_norm(df, cols) if cols else None


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def timestamps_to_relative_seconds(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert timestamp-like values in ms to relative seconds from first valid value."""
    ts = _as_float_vector(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    if ts.size == 0:
        return ts
    finite = np.isfinite(ts)
    if not finite.any():
        return np.full_like(ts, np.nan, dtype=float)
    t0 = ts[finite][0]
    out = (ts - t0) / 1000.0
    out[~finite] = np.nan
    return out


def shared_t0_ms(*dfs: pd.DataFrame) -> float:
    """Return earliest finite ``timestamp`` (ms) across multiple DataFrames."""
    starts: list[float] = []
    for df in dfs:
        if "timestamp" not in df.columns:
            continue
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        finite = ts[np.isfinite(ts)]
        if finite.size > 0:
            starts.append(float(finite[0]))
    return min(starts) if starts else 0.0


def relative_seconds(ts_ms: np.ndarray, t0_ms: float) -> np.ndarray:
    """Convert timestamp array (ms) to seconds relative to *t0_ms*."""
    return (np.asarray(ts_ms, dtype=float) - t0_ms) / 1000.0


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Load a JSON file. Returns None if the file is missing or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def prepare_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN timestamps, sort, reset index — standard pre-plot cleanup."""
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_sensor_df(stage_dir: Path, sensor: str) -> pd.DataFrame | None:
    """Load a sensor CSV, coerce to numeric, and prepare for plotting.

    Returns None if the CSV does not exist.
    """
    csv = stage_dir / f"{sensor}.csv"
    if not csv.exists():
        return None
    df = read_csv(csv)
    df = prepare_sensor_df(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 120) -> Path:
    """Save *fig* to *path*, close it, log the result, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(path))
    return path
