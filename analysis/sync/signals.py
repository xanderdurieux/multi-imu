"""Stream I/O, activity-signal construction, and post-fit correlation scoring.

This module owns every signal-level operation used by the sync stage:

- Stream loading and uniform-grid resampling (`load_stream`, `resample_stream`)
- 1-D activity-signal construction from an IMU frame (`build_activity_signal`)
- Single-call pipeline used by strategies (`build_resampled_activity_signal`)
- Post-fit Pearson correlation of acc_norm (`compute_sync_correlations`)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import read_csv
from common.signals import (
    add_imu_norms,
    add_vector_norms,
    first_difference,
    zscore_finite,
)

from .config import SIGNAL_MODES
from .model import SyncModel, apply_sync_model


# ---------------------------------------------------------------------------
# Stream I/O and resampling
# ---------------------------------------------------------------------------


def load_stream(csv_path: Path | str) -> pd.DataFrame:
    """Load an IMU CSV, coerce types, sort by timestamp."""
    df = read_csv(Path(csv_path)).copy()
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_stream(
    df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    columns: list[str] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> pd.DataFrame:
    """Resample stream to a uniform time grid via linear interpolation."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    base = df.copy()
    base[timestamp_col] = pd.to_numeric(base[timestamp_col], errors="coerce")
    base = base.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if base.empty:
        return pd.DataFrame(columns=[timestamp_col])

    ts = base[timestamp_col].to_numpy(dtype=float)
    step_ms = 1000.0 / float(sample_rate_hz)
    lo = float(ts[0] if start_ms is None else start_ms)
    hi = float(ts[-1] if end_ms is None else end_ms)

    if hi <= lo:
        out = pd.DataFrame({timestamp_col: np.asarray([lo], dtype=float)})
    else:
        grid = np.arange(lo, hi + 0.5 * step_ms, step_ms, dtype=float)
        out = pd.DataFrame({timestamp_col: grid})

    if columns is None:
        columns = [
            c for c in base.columns
            if c != timestamp_col and pd.api.types.is_numeric_dtype(base[c])
        ]

    for col in columns:
        values = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ts) & np.isfinite(values)
        if valid.sum() >= 2:
            out[col] = np.interp(
                out[timestamp_col].to_numpy(dtype=float), ts[valid], values[valid]
            )
        elif valid.sum() == 1:
            out[col] = values[valid][0]
        else:
            out[col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Activity-signal construction
# ---------------------------------------------------------------------------


def _norm_diff(signal: np.ndarray) -> np.ndarray:
    return zscore_finite(first_difference(signal))


def build_activity_signal(
    df: pd.DataFrame,
    *,
    signal_mode: str,
) -> np.ndarray:
    """Build a 1-D activity signal from an IMU frame."""
    if signal_mode not in SIGNAL_MODES:
        raise ValueError(
            f"Unknown signal_mode {signal_mode!r}; expected one of {SIGNAL_MODES}."
        )

    base = add_vector_norms(df)
    if signal_mode == "acc_norm_diff":
        return _norm_diff(base["acc_norm"].to_numpy(dtype=float))
    if signal_mode == "gyro_norm_diff":
        return _norm_diff(base["gyro_norm"].to_numpy(dtype=float))
    # acc_gyro_fused_diff
    acc = _norm_diff(base["acc_norm"].to_numpy(dtype=float))
    gyro = _norm_diff(base["gyro_norm"].to_numpy(dtype=float))
    return zscore_finite(np.nanmean(np.vstack([acc, gyro]), axis=0))


def build_resampled_activity_signal(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
    signal_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a stream and return ``(timestamps_seconds, activity_signal)``."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    resampled = resample_stream(df, sample_rate_hz=sample_rate_hz)
    if resampled.empty:
        empty = np.asarray([], dtype=float)
        return empty, empty

    signal = build_activity_signal(resampled, signal_mode=signal_mode)
    timestamps_seconds = (
        pd.to_numeric(resampled["timestamp"], errors="coerce").to_numpy(dtype=float)
        / 1000.0
    )
    return timestamps_seconds, signal


# ---------------------------------------------------------------------------
# Post-fit correlation scoring
# ---------------------------------------------------------------------------


def _acc_norm_correlation(
    ref_df: pd.DataFrame, tgt_df: pd.DataFrame, *, sample_rate_hz: float
) -> float | None:
    """Pearson r of acc_norm over the overlapping time region."""
    ref_r = add_imu_norms(resample_stream(ref_df, sample_rate_hz))
    tgt_r = add_imu_norms(resample_stream(tgt_df, sample_rate_hz))

    ref_ts = ref_r["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_r["timestamp"].to_numpy(dtype=float)
    if ref_ts.size == 0 or tgt_ts.size == 0:
        return None

    lo = max(float(ref_ts[0]), float(tgt_ts[0]))
    hi = min(float(ref_ts[-1]), float(tgt_ts[-1]))
    if lo >= hi:
        return None

    ref_acc = ref_r.loc[(ref_ts >= lo) & (ref_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    tgt_acc = tgt_r.loc[(tgt_ts >= lo) & (tgt_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    n = min(len(ref_acc), len(tgt_acc))
    if n < 10:
        return None

    x, y = ref_acc[:n], tgt_acc[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def compute_sync_correlations(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    model: SyncModel,
    *,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Pearson r of acc_norm before (offset only) and after (offset+drift) sync."""
    offset_only = replace(model, drift_seconds_per_second=0.0)
    offset_df = apply_sync_model(tgt_df, offset_only, replace_timestamp=True)
    full_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    corr_offset = _acc_norm_correlation(ref_df, offset_df, sample_rate_hz=sample_rate_hz)
    corr_full = _acc_norm_correlation(ref_df, full_df, sample_rate_hz=sample_rate_hz)
    return {
        "offset_only": round(corr_offset, 4) if corr_offset is not None else None,
        "offset_and_drift": round(corr_full, 4) if corr_full is not None else None,
    }
