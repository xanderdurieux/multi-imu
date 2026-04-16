"""Calibration-anchor matching and per-segment refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from parser.calibration_segments import (
    CalibrationSegment,
    find_calibration_segments,
    load_calibration_segments_from_json,
)

from .activity import build_alignment_series
from .stream_io import (
    lowpass_filter,
    resample_stream,
)
from .xcorr import estimate_lag

log = logging.getLogger(__name__)

DEFAULT_ANCHOR_SAMPLE_RATE_HZ = 100.0
DEFAULT_ANCHOR_SEARCH_SECONDS = 5.0
DEFAULT_ANCHOR_PEAK_BUFFER_SECONDS = 1.0


def _segments_fit_dataframe(
    df: pd.DataFrame,
    segments: list[CalibrationSegment],
) -> bool:
    """Return True when all stored segment indices are valid for *df*."""
    n_rows = len(df)
    if n_rows <= 0:
        return False
    for seg in segments:
        if seg.start_idx < 0 or seg.end_idx >= n_rows or seg.start_idx > seg.end_idx:
            return False
        if any(idx < 0 or idx >= n_rows for idx in seg.peak_indices):
            return False
    return True


def _load_segments_for_current_frame(
    df: pd.DataFrame,
    *,
    recording_name: str,
    sensor: str,
) -> tuple[list[CalibrationSegment], str]:
    """Load recording-level JSON segments when compatible, else detect locally."""
    if recording_name:
        try:
            json_segments = load_calibration_segments_from_json(recording_name, sensor)
        except (FileNotFoundError, KeyError) as exc:
            log.info(
                "Calibration JSON unavailable for %s/%s; detecting local segments (%s).",
                recording_name,
                sensor,
                exc,
            )
        else:
            if _segments_fit_dataframe(df, json_segments):
                return json_segments, "json_calibration_segments"
            log.info(
                "Calibration JSON indices do not fit current %s frame (%d rows); "
                "detecting local segments instead.",
                sensor,
                len(df),
            )

    return find_calibration_segments(df, sensor=sensor), "local_calibration_detection"


@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured at one calibration window."""

    offset_seconds: float
    t_tgt_seconds: float
    correlation_score: float
    window_duration_s: float
    t_ref_peak_s: float = 0.0


@dataclass(frozen=True)
class CalibrationAnchorExtraction:
    """Shared result of extracting matched calibration anchors."""

    anchors: list[CalibrationWindowResult]
    coarse_offset_seconds: float
    coarse_method: str
    reference_segments_detected: int
    reference_segments_matched: int


# ---------------------------------------------------------------------------
# Per-segment offset refinement via windowed cross-correlation
# ---------------------------------------------------------------------------


def refine_offset_at_calibration(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    seg: CalibrationSegment,
    *,
    coarse_offset_s: float,
    sample_rate_hz: float = 100.0,
    peak_buffer_s: float = 1.0,
    search_s: float = 3.0,
    lowpass_cutoff_hz: float | None = None,
) -> CalibrationWindowResult:
    """Refine the offset at one calibration segment using cross-correlation."""
    buf = int(sample_rate_hz * peak_buffer_s)
    p_start = max(0, seg.peak_indices[0] - buf)
    p_end = min(len(ref_df) - 1, seg.peak_indices[-1] + buf)
    ref_window = ref_df.iloc[p_start : p_end + 1].reset_index(drop=True)
    window_duration_s = float(
        (ref_window["timestamp"].iloc[-1] - ref_window["timestamp"].iloc[0]) / 1000.0
    )

    ref_start_ms = float(ref_window["timestamp"].iloc[0])
    ref_end_ms = float(ref_window["timestamp"].iloc[-1])
    search_ms = search_s * 1000.0

    tgt_mask = (
        (tgt_df["timestamp"] >= ref_start_ms - coarse_offset_s * 1000.0 - search_ms)
        & (tgt_df["timestamp"] <= ref_end_ms - coarse_offset_s * 1000.0 + search_ms)
    )
    tgt_window_raw = tgt_df.loc[tgt_mask].reset_index(drop=True)
    if len(tgt_window_raw) < 10:
        raise ValueError(
            f"Target window too small ({len(tgt_window_raw)} samples) for calibration "
            f"at t_ref=[{ref_start_ms / 1000:.1f}, {ref_end_ms / 1000:.1f}] s."
        )

    sr = min(sample_rate_hz, 100.0)
    ref_filt = ref_window
    tgt_filt = tgt_window_raw
    if lowpass_cutoff_hz is not None:
        ref_filt = lowpass_filter(resample_stream(ref_window, sr), lowpass_cutoff_hz, sr)
        tgt_filt = lowpass_filter(resample_stream(tgt_window_raw, sr), lowpass_cutoff_hz, sr)

    ref_series = build_alignment_series(ref_filt, sample_rate_hz=sr)
    tgt_series = build_alignment_series(tgt_filt, sample_rate_hz=sr)

    max_lag_samples = int(round((search_s + 1.0) * sr))
    lag_samples, score = estimate_lag(
        ref_series.signal, tgt_series.signal, max_lag_samples=max_lag_samples
    )
    lag_seconds = float(lag_samples) / sr
    ref_start_s = float(ref_series.timestamps_seconds[0])
    tgt_start_s = float(tgt_series.timestamps_seconds[0])
    offset_seconds = (ref_start_s - tgt_start_s) + lag_seconds

    ref_peak_ts = ref_df.iloc[seg.peak_indices]["timestamp"].to_numpy(dtype=float)
    t_ref_center_s = float(np.median(ref_peak_ts)) / 1000.0
    t_tgt_center_s = t_ref_center_s - float(offset_seconds)

    return CalibrationWindowResult(
        offset_seconds=float(offset_seconds),
        t_tgt_seconds=t_tgt_center_s,
        correlation_score=float(score),
        window_duration_s=float(window_duration_s),
        t_ref_peak_s=t_ref_center_s,
    )


# ---------------------------------------------------------------------------
# Anchor extraction: load from JSON, match in order, refine
# ---------------------------------------------------------------------------


def extract_calibration_anchors(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_ANCHOR_SAMPLE_RATE_HZ,
    search_seconds: float = DEFAULT_ANCHOR_SEARCH_SECONDS,
    peak_buffer_seconds: float = DEFAULT_ANCHOR_PEAK_BUFFER_SECONDS,
    lowpass_cutoff_hz: float | None = None,
) -> CalibrationAnchorExtraction:
    """Load compatible calibration segments, match in order, and refine via xcorr.

    Recording-level calibration JSON is preferred when its stored indices fit
    the current DataFrame. When syncing a shorter section window, those JSON
    indices are recording-global and may no longer be valid, so we fall back to
    local calibration detection on the provided frame.
    """
    if not recording_name:
        raise ValueError(
            "recording_name is required to load calibration segments from JSON."
        )

    ref_cals, ref_source = _load_segments_for_current_frame(
        ref_df,
        recording_name=recording_name,
        sensor=reference_sensor,
    )
    tgt_cals, tgt_source = _load_segments_for_current_frame(
        tgt_df,
        recording_name=recording_name,
        sensor=target_sensor,
    )

    if not ref_cals:
        raise ValueError(
            f"No calibration segments available for reference sensor {reference_sensor!r}."
        )
    if not tgt_cals:
        raise ValueError(
            f"No calibration segments available for target sensor {target_sensor!r}."
        )

    n_pairs = min(len(ref_cals), len(tgt_cals))
    pairs = list(zip(ref_cals[:n_pairs], tgt_cals[:n_pairs]))

    log.debug(
        "Loaded %d ref segment(s) (%s) and %d tgt segment(s) (%s); "
        "matching %d pair(s).",
        len(ref_cals),
        ref_source,
        len(tgt_cals),
        tgt_source,
        n_pairs,
    )

    # Coarse offset from the first matched pair.
    ref0_peak_ts = ref_df.iloc[pairs[0][0].peak_indices]["timestamp"].to_numpy(dtype=float)
    tgt0_peak_ts = tgt_df.iloc[pairs[0][1].peak_indices]["timestamp"].to_numpy(dtype=float)
    coarse_offset_s = (float(np.median(ref0_peak_ts)) - float(np.median(tgt0_peak_ts))) / 1000.0

    log.info(
        "Coarse offset from %s/%s segments: %.3f s "
        "(ref_cal[0] at t_ref=%.1f s, tgt_cal[0] at t_tgt=%.1f s)",
        ref_source,
        tgt_source,
        coarse_offset_s,
        float(np.median(ref0_peak_ts)) / 1000.0,
        float(np.median(tgt0_peak_ts)) / 1000.0,
    )

    anchors: list[CalibrationWindowResult] = []
    for ref_seg, tgt_seg in pairs:
        # Per-pair coarse offset derived from stored target peak timestamps.
        ref_peak_ts = ref_df.iloc[ref_seg.peak_indices]["timestamp"].to_numpy(dtype=float)
        tgt_peak_ts = tgt_df.iloc[tgt_seg.peak_indices]["timestamp"].to_numpy(dtype=float)
        pair_coarse_s = (float(np.median(ref_peak_ts)) - float(np.median(tgt_peak_ts))) / 1000.0

        try:
            anchor = refine_offset_at_calibration(
                ref_df,
                tgt_df,
                ref_seg,
                coarse_offset_s=pair_coarse_s,
                sample_rate_hz=sample_rate_hz,
                peak_buffer_s=peak_buffer_seconds,
                search_s=search_seconds,
                lowpass_cutoff_hz=lowpass_cutoff_hz,
            )
        except Exception as exc:
            log.warning("Skipping calibration pair during refinement: %s", exc)
            continue
        anchors.append(anchor)

    if not anchors:
        raise ValueError("No calibration pairs were refined successfully.")

    anchors = sorted(anchors, key=lambda a: (a.t_tgt_seconds, a.t_ref_peak_s))
    return CalibrationAnchorExtraction(
        anchors=anchors,
        coarse_offset_seconds=coarse_offset_s,
        coarse_method=(
            "json_calibration_segments"
            if ref_source == "json_calibration_segments" and tgt_source == "json_calibration_segments"
            else "local_calibration_detection"
        ),
        reference_segments_detected=len(ref_cals),
        reference_segments_matched=n_pairs,
    )


def calibration_anchor_to_dict(
    anchor: CalibrationWindowResult,
    *,
    index: int | None = None,
) -> dict[str, Any]:
    """Serialize one matched anchor for sync metadata."""
    data: dict[str, Any] = {
        "offset_s": round(float(anchor.offset_seconds), 6),
        "t_ref_s": round(float(anchor.t_ref_peak_s), 3),
        "t_tgt_s": round(float(anchor.t_tgt_seconds), 3),
        "score": round(float(anchor.correlation_score), 4),
        "window_duration_s": round(float(anchor.window_duration_s), 2),
    }
    if index is not None:
        data["index"] = int(index)
    return data
