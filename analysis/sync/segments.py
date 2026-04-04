"""Shared calibration-window detection and masking helpers for synchronization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from parser.calibration_segments import CalibrationSegment, find_calibration_segments

CALIBRATION_USAGE_FULL_STREAM = "full_stream"
CALIBRATION_USAGE_MASK_SEGMENTS = "mask_calibration_segments"
CALIBRATION_USAGE_ONLY = "calibration_only"
CALIBRATION_USAGE_STRATEGIES: tuple[str, ...] = (
    CALIBRATION_USAGE_FULL_STREAM,
    CALIBRATION_USAGE_MASK_SEGMENTS,
    CALIBRATION_USAGE_ONLY,
)


@dataclass(frozen=True)
class SegmentSelection:
    """Masking outcome for one uniformly sampled alignment stream."""

    valid_mask: np.ndarray
    segment_mask: np.ndarray
    segment_windows_seconds: list[tuple[float, float]]
    segment_count: int
    segment_aware_used: bool
    usable_duration_seconds: float
    percentage_removed: float
    percentage_used: float


def detect_calibration_segments_for_stream(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
) -> list[CalibrationSegment]:
    """Detect calibration segments in one stream, returning an empty list on failure."""
    try:
        return find_calibration_segments(df, sample_rate_hz=sample_rate_hz)
    except Exception:
        return []


def calibration_segments_to_windows_seconds(
    df: pd.DataFrame,
    segments: list[CalibrationSegment],
) -> list[tuple[float, float]]:
    """Convert segment indices into absolute timestamp windows in seconds."""
    if "timestamp" not in df.columns or not segments:
        return []
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float) / 1000.0
    windows: list[tuple[float, float]] = []
    for seg in segments:
        if seg.start_idx < 0 or seg.end_idx >= len(ts):
            continue
        start_s = float(ts[seg.start_idx])
        end_s = float(ts[seg.end_idx])
        if np.isfinite(start_s) and np.isfinite(end_s) and end_s >= start_s:
            windows.append((start_s, end_s))
    return windows


def build_segment_selection(
    timestamps_seconds: np.ndarray,
    *,
    sample_rate_hz: float,
    segment_windows_seconds: list[tuple[float, float]],
    calibration_usage_strategy: str,
    segment_aware: bool,
) -> SegmentSelection:
    """Build a valid-sample mask for a synchronization strategy."""
    if calibration_usage_strategy not in CALIBRATION_USAGE_STRATEGIES:
        raise ValueError(
            "Unknown calibration_usage_strategy "
            f"{calibration_usage_strategy!r}; expected one of {CALIBRATION_USAGE_STRATEGIES}."
        )

    ts = np.asarray(timestamps_seconds, dtype=float)
    if ts.size == 0:
        empty = np.asarray([], dtype=bool)
        return SegmentSelection(
            valid_mask=empty,
            segment_mask=empty,
            segment_windows_seconds=[],
            segment_count=0,
            segment_aware_used=False,
            usable_duration_seconds=0.0,
            percentage_removed=0.0,
            percentage_used=0.0,
        )

    all_valid = np.isfinite(ts)
    segment_mask = np.zeros(ts.shape, dtype=bool)
    for start_s, end_s in segment_windows_seconds:
        segment_mask |= (ts >= float(start_s)) & (ts <= float(end_s))

    strategy = calibration_usage_strategy
    aware_used = bool(segment_aware and segment_windows_seconds)
    if not segment_aware:
        strategy = CALIBRATION_USAGE_FULL_STREAM

    if calibration_usage_strategy == CALIBRATION_USAGE_ONLY and not segment_windows_seconds:
        raise ValueError("calibration_only requested but no calibration segments were found.")

    if strategy == CALIBRATION_USAGE_FULL_STREAM or not aware_used:
        valid_mask = all_valid.copy()
    elif strategy == CALIBRATION_USAGE_MASK_SEGMENTS:
        valid_mask = all_valid & ~segment_mask
    else:
        valid_mask = all_valid & segment_mask
        if not valid_mask.any():
            raise ValueError("calibration_only requested but no usable calibration windows were found.")

    sample_period_s = 1.0 / float(sample_rate_hz) if sample_rate_hz > 0 else 0.0
    usable_duration_s = float(valid_mask.sum() * sample_period_s)
    percentage_used = float(100.0 * valid_mask.mean()) if valid_mask.size else 0.0
    return SegmentSelection(
        valid_mask=valid_mask,
        segment_mask=segment_mask,
        segment_windows_seconds=list(segment_windows_seconds),
        segment_count=len(segment_windows_seconds),
        segment_aware_used=aware_used,
        usable_duration_seconds=usable_duration_s,
        percentage_removed=float(max(0.0, 100.0 - percentage_used)),
        percentage_used=percentage_used,
    )
