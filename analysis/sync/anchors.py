"""Calibration-anchor matching and per-segment refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.signals import find_peaks_simple, smooth_moving_average
from parser.calibration_segments import (
    CalibrationSegment,
    find_calibration_segments,
    load_calibration_segments_from_json,
)

from .activity import SIGNAL_MODE_ACC_NORM_DIFF, build_alignment_series
from .stream_io import (
    lowpass_filter,
    remove_dropouts,
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
    reference_segments_in_range: int


# ---------------------------------------------------------------------------
# Coarse offset from the opening calibration cluster in the target
# ---------------------------------------------------------------------------


def _cluster_peaks(
    peaks: np.ndarray,
    ts: np.ndarray,
    *,
    min_inter_peak_s: float = 0.3,
    max_inter_peak_s: float = 2.5,
    peak_min_count: int = 3,
    peak_max_count: int = 8,
    max_cluster_duration_s: float = 20.0,
) -> list[list[int]]:
    """Group peaks into timing-consistent clusters."""
    clusters: list[list[int]] = []
    current: list[int] = [int(peaks[0])]

    for p in peaks[1:]:
        gap_s = (ts[int(p)] - ts[current[-1]]) / 1000.0
        if min_inter_peak_s <= gap_s <= max_inter_peak_s:
            current.append(int(p))
        else:
            if peak_min_count <= len(current) <= peak_max_count:
                dur = (ts[current[-1]] - ts[current[0]]) / 1000.0
                if dur <= max_cluster_duration_s:
                    clusters.append(list(current))
            current = [int(p)]

    if peak_min_count <= len(current) <= peak_max_count:
        dur = (ts[current[-1]] - ts[current[0]]) / 1000.0
        if dur <= max_cluster_duration_s:
            clusters.append(list(current))

    return clusters


def coarse_offset_from_opening_calibration(
    ref_df: pd.DataFrame,
    tgt_df_clean: pd.DataFrame,
    ref_cal: CalibrationSegment,
    *,
    search_first_s: float = 180.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_count: int = 8,
    min_inter_peak_s: float = 0.3,
    max_inter_peak_s: float = 2.5,
    max_cluster_duration_s: float = 20.0,
) -> float:
    """Estimate coarse offset by matching the opening calibration cluster."""
    ts = tgt_df_clean["timestamp"].to_numpy(dtype=float)
    if len(ts) < 2:
        raise ValueError("Target DataFrame is too short.")

    dt_ms = float(np.median(np.diff(ts)))
    actual_sr = max(1.0, 1000.0 / dt_ms)
    search_end_ms = ts[0] + search_first_s * 1000.0
    n_search = int(np.sum(ts <= search_end_ms))

    norm = tgt_df_clean["acc_norm"].to_numpy(dtype=float)
    g = float(np.nanmedian(norm))
    smooth_win = max(3, int(actual_sr * 0.1))
    dyn_smooth = np.abs(smooth_moving_average(norm[:n_search], smooth_win) - g)

    dist = max(1, int(actual_sr * min_inter_peak_s))
    peaks = find_peaks_simple(dyn_smooth, height=peak_min_height, distance=dist)
    if len(peaks) == 0:
        raise ValueError(
            f"No peaks above {peak_min_height} m/s² in first {search_first_s}s of target."
        )

    clusters = _cluster_peaks(
        peaks,
        ts,
        min_inter_peak_s=min_inter_peak_s,
        max_inter_peak_s=max_inter_peak_s,
        peak_min_count=peak_min_count,
        peak_max_count=peak_max_count,
        max_cluster_duration_s=max_cluster_duration_s,
    )
    if not clusters:
        raise ValueError(
            f"No calibration-like cluster ({peak_min_count}-{peak_max_count} peaks, "
            f"{min_inter_peak_s}-{max_inter_peak_s}s spacing) in first {search_first_s}s."
        )

    tgt_median_ms = float(np.median(ts[clusters[0]]))
    ref_peak_ts = ref_df.iloc[ref_cal.peak_indices]["timestamp"].to_numpy(dtype=float)
    ref_median_ms = float(np.median(ref_peak_ts))
    coarse_offset_s = (ref_median_ms - tgt_median_ms) / 1000.0

    log.info(
        "Coarse offset from opening calibration: %.3f s (target cluster at t_tgt=%.1f s)",
        coarse_offset_s,
        tgt_median_ms / 1000.0,
    )
    return coarse_offset_s


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
# Helpers
# ---------------------------------------------------------------------------


def filter_segments_in_target_range(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    segments: list[CalibrationSegment],
    *,
    coarse_offset_s: float,
    margin_s: float = 15.0,
) -> list[CalibrationSegment]:
    """Keep only reference segments whose predicted target centre is in range."""
    tgt_ts = tgt_df["timestamp"].to_numpy(dtype=float)
    tgt_lo_ms, tgt_hi_ms = float(tgt_ts[0]), float(tgt_ts[-1])
    margin_ms = margin_s * 1000.0

    kept: list[CalibrationSegment] = []
    for seg in segments:
        ref_center_ms = float(
            np.median(ref_df.iloc[seg.peak_indices]["timestamp"].to_numpy(dtype=float))
        )
        tgt_center_ms = ref_center_ms - coarse_offset_s * 1000.0
        if tgt_lo_ms - margin_ms <= tgt_center_ms <= tgt_hi_ms + margin_ms:
            kept.append(seg)
    return kept


def bootstrap_coarse_offset(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    ref_cals: list[CalibrationSegment],
    *,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    sda_sample_rate_hz: float = 5.0,
    sda_max_lag_s: float = 120.0,
    signal_mode: str = SIGNAL_MODE_ACC_NORM_DIFF,
) -> tuple[float, str]:
    """Determine coarse offset from the opening calibration only."""
    if not ref_cals:
        raise ValueError("No reference calibration sequences found.")

    tgt_clean = remove_dropouts(tgt_df)
    offset = coarse_offset_from_opening_calibration(
        ref_df,
        tgt_clean,
        ref_cals[0],
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
    )
    return offset, "opening_calibration"


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
    sda_fallback_max_lag_s: float = 120.0,
) -> CalibrationAnchorExtraction:
    """Load reference calibrations, bootstrap coarse offset, then refine on target.

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

    if not ref_cals:
        raise ValueError(
            f"No calibration segments available for reference sensor {reference_sensor!r}."
        )

    coarse_offset_s, coarse_method = bootstrap_coarse_offset(
        ref_df,
        tgt_df,
        ref_cals,
        sda_max_lag_s=sda_fallback_max_lag_s,
    )
    in_range = filter_segments_in_target_range(
        ref_df,
        tgt_df,
        ref_cals,
        coarse_offset_s=coarse_offset_s,
    )
    if not in_range:
        raise ValueError("No calibration sequences map into the target stream.")

    anchors: list[CalibrationWindowResult] = []
    current_coarse = coarse_offset_s
    for ref_seg in in_range:
        try:
            anchor = refine_offset_at_calibration(
                ref_df,
                tgt_df,
                ref_seg,
                coarse_offset_s=current_coarse,
                sample_rate_hz=sample_rate_hz,
                peak_buffer_s=peak_buffer_seconds,
                search_s=search_seconds,
                lowpass_cutoff_hz=lowpass_cutoff_hz,
            )
        except Exception as exc:
            log.warning("Skipping calibration window during refinement: %s", exc)
            continue
        anchors.append(anchor)
        current_coarse = anchor.offset_seconds

    if not anchors:
        raise ValueError("No calibration sequences were refined successfully.")

    anchors = sorted(anchors, key=lambda a: (a.t_tgt_seconds, a.t_ref_peak_s))
    return CalibrationAnchorExtraction(
        anchors=anchors,
        coarse_offset_seconds=coarse_offset_s,
        coarse_method=f"{ref_source}+{coarse_method}",
        reference_segments_detected=len(ref_cals),
        reference_segments_in_range=len(in_range),
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
