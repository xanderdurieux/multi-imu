"""Internal visualization utilities — shared constants, IO, and plot helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv, read_json_file
from common.signals import vector_norm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SENSORS: tuple[str, ...] = ("sporsa", "arduino")

SENSOR_COLORS: dict[str, str] = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}

# Qualitative palette reused by reporting/visualization plots needing many
# distinguishable colors (e.g. label strips, scenario legends).
QUALITATIVE_PALETTE: tuple[str, ...] = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000",
)
UNKNOWN_LABEL_COLOR = "#90A4AE"

# Per-axis colors used by sensor plots.
AXIS_COLORS: dict[str, str] = {"x": "#e41a1c", "y": "#4daf4a", "z": "#377eb8"}
NORM_COLOR: str = "black"

ACC_COLS: tuple[str, ...] = ("ax", "ay", "az")
GYRO_COLS: tuple[str, ...] = ("gx", "gy", "gz")
MAG_COLS: tuple[str, ...] = ("mx", "my", "mz")


def label_color(label: str, label_list: list[str]) -> str:
    """Return a stable color for ``label`` given its position in ``label_list``.

    Unknown labels fall back to ``UNKNOWN_LABEL_COLOR``.
    """
    try:
        return QUALITATIVE_PALETTE[label_list.index(label) % len(QUALITATIVE_PALETTE)]
    except ValueError:
        return UNKNOWN_LABEL_COLOR


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


def acc_norm(df: pd.DataFrame) -> np.ndarray | None:
    """Accelerometer magnitude — uses pre-computed ``acc_norm`` column when present."""
    if "acc_norm" in df.columns:
        return pd.to_numeric(df["acc_norm"], errors="coerce").to_numpy(dtype=float)
    cols = [c for c in ACC_COLS if c in df.columns]
    return vector_norm(df, cols) if cols else None


def gyro_norm(df: pd.DataFrame) -> np.ndarray | None:
    """Gyroscope magnitude — uses pre-computed ``gyro_norm`` column when present."""
    if "gyro_norm" in df.columns:
        return pd.to_numeric(df["gyro_norm"], errors="coerce").to_numpy(dtype=float)
    cols = [c for c in GYRO_COLS if c in df.columns]
    return vector_norm(df, cols) if cols else None


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
        return read_json_file(path)
    except (OSError, ValueError) as exc:
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
