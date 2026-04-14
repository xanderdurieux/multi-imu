"""Calibration-anchor detection, coarse offset, and per-segment refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.signals import acc_norm_from_imu_df, find_peaks_simple, smooth_moving_average
from parser.calibration_segments import CalibrationSegment, find_calibration_segments

from .activity import build_alignment_series, SIGNAL_MODE_ACC_NORM_DIFF
from .stream_io import (
    load_stream,
    lowpass_filter,
    remove_dropouts,
    resample_stream,
)
from .xcorr import estimate_lag

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured at one calibration window."""

    offset_seconds: float
    t_tgt_seconds: float
    correlation_score: float
    window_duration_s: float


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
    """Estimate coarse offset by matching the opening calibration cluster.

    Detects peak clusters in the first ``search_first_s`` seconds of the
    target, then aligns the median peak time to the reference calibration.
    """
    ts = tgt_df_clean["timestamp"].to_numpy(dtype=float)
    if len(ts) < 2:
        raise ValueError("Target DataFrame is too short.")

    dt_ms = float(np.median(np.diff(ts)))
    actual_sr = max(1.0, 1000.0 / dt_ms)
    search_end_ms = ts[0] + search_first_s * 1000.0
    n_search = int(np.sum(ts <= search_end_ms))

    norm = acc_norm_from_imu_df(tgt_df_clean)
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
        peaks, ts,
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

    p_mid = (seg.peak_indices[0] + seg.peak_indices[-1]) // 2
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

    t_tgt_center_ms = float(ref_df["timestamp"].iloc[p_mid]) - coarse_offset_s * 1000.0
    return CalibrationWindowResult(
        offset_seconds=float(offset_seconds),
        t_tgt_seconds=float(t_tgt_center_ms / 1000.0),
        correlation_score=float(score),
        window_duration_s=float(window_duration_s),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_reference_calibrations(
    ref_df: pd.DataFrame, *, sample_rate_hz: float = 100.0, **kwargs
) -> list[CalibrationSegment]:
    """Detect calibration segments in the reference stream."""
    return find_calibration_segments(ref_df, sample_rate_hz=sample_rate_hz, **kwargs)


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
    """Determine coarse offset from opening calibration or SDA fallback.

    Returns (coarse_offset_seconds, method_used).
    """
    if ref_cals:
        tgt_clean = remove_dropouts(tgt_df)
        try:
            offset = coarse_offset_from_opening_calibration(
                ref_df, tgt_clean, ref_cals[0],
                peak_min_height=peak_min_height,
                peak_min_count=peak_min_count,
            )
            return offset, "opening_calibration"
        except ValueError as exc:
            log.warning("Opening-calibration coarse offset failed (%s); trying SDA.", exc)

    ref_series = build_alignment_series(
        ref_df, sample_rate_hz=sda_sample_rate_hz, signal_mode=signal_mode
    )
    tgt_clean = remove_dropouts(tgt_df)
    tgt_series = build_alignment_series(
        tgt_clean, sample_rate_hz=sda_sample_rate_hz, signal_mode=signal_mode
    )
    max_lag_samples = int(round(sda_max_lag_s * sda_sample_rate_hz))
    lag_samples, _ = estimate_lag(
        ref_series.signal, tgt_series.signal, max_lag_samples=max_lag_samples
    )
    lag_seconds = float(lag_samples) / sda_sample_rate_hz
    ref_start = float(ref_series.timestamps_seconds[0])
    tgt_start = float(tgt_series.timestamps_seconds[0])
    offset = (ref_start - tgt_start) + lag_seconds
    return offset, "sda_fallback"
