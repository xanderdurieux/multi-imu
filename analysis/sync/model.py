"""SyncModel dataclass and linear clock-transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyncModel:
    """Linear offset + drift clock model: t_ref = t_tgt + b + a*(t_tgt - t0)."""

    reference_csv: str
    target_csv: str
    target_time_origin_seconds: float
    offset_seconds: float
    drift_seconds_per_second: float
    sample_rate_hz: float
    max_lag_seconds: float
    created_at_utc: str


def make_sync_model(
    *,
    reference_name: str,
    target_name: str,
    target_origin_seconds: float,
    offset_seconds: float,
    drift_seconds_per_second: float,
    sample_rate_hz: float,
    max_lag_seconds: float,
) -> SyncModel:
    return SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift_seconds_per_second,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        created_at_utc=datetime.now(UTC).isoformat(),
    )




def target_to_reference_seconds(
    target_seconds: pd.Series | np.ndarray | float,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Apply model equation in seconds: ``t_ref = t_tgt + b + a*(t_tgt - t0)``."""
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
    """Invert the linear model: solve ``t_ref = t_tgt + b + a*(t_tgt - t0)`` for ``t_tgt``."""
    t_ref = np.asarray(reference_seconds, dtype=float)
    a = float(drift_seconds_per_second)
    denom = 1.0 + a
    if abs(denom) < 1e-9:
        raise ValueError("Invalid drift model: 1 + drift_seconds_per_second is near zero.")

    num = t_ref - float(offset_seconds) + a * float(target_origin_seconds)
    return num / denom

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
    """Apply the offset+drift model to a target DataFrame's timestamps."""
    out = target_df.copy()
    out["timestamp_orig"] = pd.to_numeric(out["timestamp"], errors="coerce")
    aligned = apply_linear_time_transform(
        out["timestamp_orig"],
        offset_seconds=model.offset_seconds,
        drift_seconds_per_second=model.drift_seconds_per_second,
        target_origin_seconds=model.target_time_origin_seconds,
    )
    out["timestamp_aligned"] = aligned
    if replace_timestamp:
        out["timestamp"] = out["timestamp_aligned"]
    return out
