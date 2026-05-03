"""Model helpers for align arduino timestamps to the sporsa reference clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyncModel:
    """Linear offset + drift clock model."""

    target_time_origin_seconds: float
    offset_seconds: float
    drift_seconds_per_second: float


def target_to_reference_seconds(
    target_seconds: pd.Series | np.ndarray | float,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Return target to reference seconds."""
    t_tgt = np.asarray(target_seconds, dtype=float)
    return (
        t_tgt
        + float(offset_seconds)
        + float(drift_seconds_per_second) * (t_tgt - float(target_origin_seconds))
    )


def reference_to_target_seconds(
    reference_seconds: pd.Series | np.ndarray | float,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Return reference to target seconds."""
    t_ref = np.asarray(reference_seconds, dtype=float)
    a = float(drift_seconds_per_second)
    denom = 1.0 + a
    if abs(denom) < 1e-9:
        raise ValueError("Invalid drift: 1 + drift_seconds_per_second is near zero.")
    return (t_ref - float(offset_seconds) + a * float(target_origin_seconds)) / denom


def apply_linear_time_transform(
    timestamp_ms: pd.Series | np.ndarray,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Map target timestamps (ms) to reference time via offset + linear drift."""
    ts_sec = np.asarray(timestamp_ms, dtype=float) / 1000.0
    aligned_sec = target_to_reference_seconds(
        ts_sec,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift_seconds_per_second,
        target_origin_seconds=target_origin_seconds,
    )
    return aligned_sec * 1000.0


def apply_sync_model(
    target_df: pd.DataFrame,
    model: SyncModel,
    *,
    replace_timestamp: bool = True,
) -> pd.DataFrame:
    """Apply ``model`` to a target DataFrame's ``timestamp`` column (ms)."""
    out = target_df.copy()
    out["timestamp_orig"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["timestamp_aligned"] = apply_linear_time_transform(
        out["timestamp_orig"],
        offset_seconds=model.offset_seconds,
        drift_seconds_per_second=model.drift_seconds_per_second,
        target_origin_seconds=model.target_time_origin_seconds,
    )
    if replace_timestamp:
        out["timestamp"] = out["timestamp_aligned"]
    return out
