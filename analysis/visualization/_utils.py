"""Shared low-level utilities for visualization modules."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common.paths import recording_stage_dir as _recording_stage_dir
from common.paths import section_dir as _section_dir


def mask_dropout_packets(df: pd.DataFrame, epsilon_fraction: float = 0.1) -> pd.DataFrame:
    """Set sensor columns to NaN for rows where acc_norm is near-zero.

    The timestamp column is preserved so dropout periods appear as visible
    gaps rather than zero-value spikes on the time axis.
    """
    acc_cols = ["ax", "ay", "az"]
    if not all(c in df.columns for c in acc_cols):
        return df
    acc_norm = np.sqrt((df[acc_cols].to_numpy(dtype=float) ** 2).sum(axis=1))
    g_approx = float(np.nanmedian(acc_norm))
    if g_approx <= 0:
        return df
    dropout = acc_norm < epsilon_fraction * g_approx
    if not dropout.any():
        return df
    out = df.copy()
    sensor_cols = [c for c in df.columns if c != "timestamp"]
    out.loc[dropout, sensor_cols] = np.nan
    return out


def time_axis_seconds(timestamps_ms: pd.Series) -> pd.Series:
    """Convert a millisecond timestamp series to seconds starting at zero."""
    ts = timestamps_ms.astype(float)
    return (ts - float(ts.iloc[0])) / 1000.0


def acc_norm_series(df: pd.DataFrame) -> np.ndarray:
    """Compute accelerometer vector norm for each row."""
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt((acc ** 2).sum(axis=1))


def mask_valid_plot_x(x: np.ndarray | pd.Series) -> np.ndarray:
    """True where *x* is safe to use as a time-like plot abscissa.

    Drops non-finite values. Drops **accidental** ``x == 0``: zeros are kept
    only in the leading run before any positive *x* (typical ``t=0`` at
    recording start); a zero after a positive time is treated as invalid.
    """
    xv = np.asarray(x, dtype=float)
    n = xv.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(xv)
    if not finite.any():
        return finite
    xf = np.where(finite, xv, -np.inf)
    cmax = np.maximum.accumulate(xf)
    prev_max = np.empty(n, dtype=float)
    prev_max[0] = -np.inf
    if n > 1:
        prev_max[1:] = cmax[:-1]
    bad_zero = finite & (xv == 0.0) & (prev_max > 0.0)
    return finite & ~bad_zero


def nan_mask_invalid_plot_x(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` as float arrays with NaNs where the abscissa is invalid.

    Use for :meth:`~matplotlib.axes.Axes.plot` / :meth:`~matplotlib.axes.Axes.fill_between`
    so segments break at bad timestamps instead of drawing spikes to ``x=0``.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    xm = mask_valid_plot_x(xa)
    return np.where(xm, xa, np.nan), np.where(xm, ya, np.nan)


_SECTION_RE = re.compile(r"^(?P<rec>.+_r\d+)s(?P<idx>\d+)$")


def parse_recording_or_section_id(name: str) -> tuple[str, int | None]:
    """Parse either a recording id or a section id.

    - recording: ``<session>_r<idx>`` (e.g. ``2026-02-26_r2``) → ``(name, None)``
    - section:   ``<recording>s<section_idx>`` (e.g. ``2026-02-26_r2s1``) → ``(<recording>, section_idx)``
    """
    n = name.strip()
    m = _SECTION_RE.match(n)
    if not m:
        return n, None
    return str(m.group("rec")), int(m.group("idx"))


def resolve_stage_dir(recording_or_section_id: str, stage: str):
    """Resolve a stage directory for both recording-level and section-level layouts.

    This lets visualization scripts accept either:
    - a recording id (e.g. ``2026-02-26_r2``) with stages like ``orientation`` / ``calibrated`` / ``synced``
    - a section id (e.g. ``2026-02-26_r2s1``) with the same stage names, located under ``data/sections/``.
    """
    from pathlib import Path

    rec, section_idx = parse_recording_or_section_id(recording_or_section_id)
    if section_idx is None:
        return _recording_stage_dir(rec, stage)
    return _section_dir(rec, section_idx) / stage
