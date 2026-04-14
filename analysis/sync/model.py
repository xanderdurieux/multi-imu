"""SyncModel dataclass, linear time transform, and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

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


def apply_linear_time_transform(
    timestamp_ms: pd.Series | np.ndarray,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Map target timestamps (ms) to reference time via offset + linear drift."""
    ts_sec = np.asarray(timestamp_ms, dtype=float) / 1000.0
    aligned_sec = (
        ts_sec
        + float(offset_seconds)
        + float(drift_seconds_per_second) * (ts_sec - float(target_origin_seconds))
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


def save_sync_model(model: SyncModel, path: Path | str) -> Path:
    """Serialize a SyncModel to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(model), indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_sync_model(path: Path | str) -> SyncModel:
    """Deserialize a SyncModel from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_keys = {f.name for f in fields(SyncModel)}
    filtered = {k: v for k, v in data.items() if k in allowed_keys}
    return SyncModel(**filtered)
