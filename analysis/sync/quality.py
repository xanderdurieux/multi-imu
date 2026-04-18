"""Sync quality metrics: acc_norm correlation before and after alignment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from common.signals import add_imu_norms

from .model import SyncModel, apply_sync_model
from .stream_io import resample_stream


def acc_norm_correlation(
    ref_df, tgt_df, *, sample_rate_hz: float
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

    ref_acc = ref_r.loc[
        (ref_ts >= lo) & (ref_ts <= hi), "acc_norm"
    ].to_numpy(dtype=float)
    tgt_acc = tgt_r.loc[
        (tgt_ts >= lo) & (tgt_ts <= hi), "acc_norm"
    ].to_numpy(dtype=float)
    n = min(len(ref_acc), len(tgt_acc))
    if n < 10:
        return None

    x, y = ref_acc[:n], tgt_acc[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def compute_sync_correlations(
    ref_df, tgt_df, model: SyncModel, *, sample_rate_hz: float
) -> dict[str, Any]:
    """Correlation of acc_norm before (offset only) and after (offset+drift) sync."""
    offset_only = replace(model, drift_seconds_per_second=0.0)
    offset_df = apply_sync_model(tgt_df, offset_only, replace_timestamp=True)
    full_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    corr_offset = acc_norm_correlation(ref_df, offset_df, sample_rate_hz=sample_rate_hz)
    corr_full = acc_norm_correlation(ref_df, full_df, sample_rate_hz=sample_rate_hz)

    return {
        "offset_only": round(corr_offset, 4) if corr_offset is not None else None,
        "offset_and_drift": round(corr_full, 4) if corr_full is not None else None,
    }
