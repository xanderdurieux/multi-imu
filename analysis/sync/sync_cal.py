"""Calibration-anchor synchronisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.calibration_segments import _acc_norm, _find_peaks, _smooth, find_calibration_segments

from ._shared import (
    DEFAULT_REFERENCE_SENSOR,
    DEFAULT_TARGET_SENSOR,
    SyncArtifacts,
    SyncOutputs,
    load_recording_inputs,
    stage_output_dir,
    write_sync_outputs,
)
from .core import (
    SyncModel,
    estimate_offset,
    load_stream,
    lowpass_filter,
    remove_dropouts,
    resample_stream,
)

log = logging.getLogger(__name__)
METHOD_NAME = "cal"
DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_COARSE_SAMPLE_RATE_HZ = 5.0
DEFAULT_COARSE_MAX_LAG_SECONDS = 120.0
DEFAULT_CAL_SEARCH_SECONDS = 5.0
DEFAULT_PEAK_BUFFER_SECONDS = 1.0
_MAX_PLAUSIBLE_DRIFT = 0.01


@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured around one calibration anchor."""

    offset_seconds: float
    target_time_seconds: float
    correlation_score: float
    window_duration_s: float


def _coarse_offset_from_opening_calibration(
    reference_df: pd.DataFrame,
    target_df_clean: pd.DataFrame,
    reference_segment,
    *,
    search_first_s: float = 180.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_count: int = 8,
    min_inter_peak_s: float = 0.3,
    max_inter_peak_s: float = 2.5,
    max_cluster_duration_s: float = 20.0,
) -> float:
    """Estimate a coarse opening offset by matching the first tap cluster."""
    timestamps = target_df_clean["timestamp"].to_numpy(dtype=float)
    if len(timestamps) < 2:
        raise ValueError("Target DataFrame is too short.")

    sample_interval_ms = float(np.median(np.diff(timestamps)))
    actual_rate_hz = max(1.0, 1000.0 / sample_interval_ms)
    search_mask = timestamps <= timestamps[0] + search_first_s * 1000.0
    n_search = int(np.sum(search_mask))

    acc_norm = _acc_norm(target_df_clean)
    gravity = float(np.nanmedian(acc_norm))
    smooth_window = max(3, int(actual_rate_hz * 0.1))
    dynamic_signal = np.abs(_smooth(acc_norm[:n_search], smooth_window) - gravity)
    peaks = _find_peaks(
        dynamic_signal,
        height=peak_min_height,
        distance=max(1, int(actual_rate_hz * min_inter_peak_s)),
    )
    if len(peaks) == 0:
        raise ValueError(
            f"No peaks above {peak_min_height} m/s² found in first {search_first_s}s of target."
        )

    clusters: list[list[int]] = []
    current: list[int] = [int(peaks[0])]
    for peak in peaks[1:]:
        gap_s = (timestamps[int(peak)] - timestamps[current[-1]]) / 1000.0
        if min_inter_peak_s <= gap_s <= max_inter_peak_s:
            current.append(int(peak))
            continue
        _maybe_add_cluster(
            clusters,
            current,
            timestamps,
            peak_min_count=peak_min_count,
            peak_max_count=peak_max_count,
            max_duration_s=max_cluster_duration_s,
        )
        current = [int(peak)]
    _maybe_add_cluster(
        clusters,
        current,
        timestamps,
        peak_min_count=peak_min_count,
        peak_max_count=peak_max_count,
        max_duration_s=max_cluster_duration_s,
    )
    if not clusters:
        raise ValueError("No calibration-like cluster found in the target opening window.")

    target_cluster = clusters[0]
    target_median_ms = float(np.median(timestamps[target_cluster]))
    reference_median_ms = float(
        np.median(reference_df.iloc[reference_segment.peak_indices]["timestamp"].to_numpy(dtype=float))
    )
    return (reference_median_ms - target_median_ms) / 1000.0


def _maybe_add_cluster(
    clusters: list[list[int]],
    cluster: list[int],
    timestamps: np.ndarray,
    *,
    peak_min_count: int,
    peak_max_count: int,
    max_duration_s: float,
) -> None:
    if not (peak_min_count <= len(cluster) <= peak_max_count):
        return
    duration_s = (timestamps[cluster[-1]] - timestamps[cluster[0]]) / 1000.0
    if duration_s <= max_duration_s:
        clusters.append(list(cluster))


def _estimate_drift_from_duration(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> float:
    reference_duration_s = (
        float(reference_df["timestamp"].iloc[-1]) - float(reference_df["timestamp"].iloc[0])
    ) / 1000.0
    target_duration_s = (
        float(target_df["timestamp"].iloc[-1]) - float(target_df["timestamp"].iloc[0])
    ) / 1000.0
    if target_duration_s <= 0:
        return 0.0
    return (reference_duration_s / target_duration_s) - 1.0


def _refine_offset_at_calibration(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    segment,
    *,
    coarse_offset_s: float,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    peak_buffer_s: float = DEFAULT_PEAK_BUFFER_SECONDS,
    search_s: float = DEFAULT_CAL_SEARCH_SECONDS,
    lowpass_cutoff_hz: float | None = None,
) -> CalibrationWindowResult:
    """Refine a calibration anchor with a narrow-window SDA pass."""
    buffer_n = int(sample_rate_hz * peak_buffer_s)
    start_idx = max(0, segment.peak_indices[0] - buffer_n)
    end_idx = min(len(reference_df) - 1, segment.peak_indices[-1] + buffer_n)
    reference_window = reference_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    window_duration_s = float(
        (reference_window["timestamp"].iloc[-1] - reference_window["timestamp"].iloc[0]) / 1000.0
    )

    mid_idx = (segment.peak_indices[0] + segment.peak_indices[-1]) // 2
    reference_center_ms = float(reference_df["timestamp"].iloc[mid_idx])
    target_center_ms = reference_center_ms - coarse_offset_s * 1000.0

    search_ms = search_s * 1000.0
    target_window = target_df.loc[
        (target_df["timestamp"] >= float(reference_window["timestamp"].iloc[0]) - coarse_offset_s * 1000.0 - search_ms)
        & (target_df["timestamp"] <= float(reference_window["timestamp"].iloc[-1]) - coarse_offset_s * 1000.0 + search_ms)
    ].reset_index(drop=True)
    if len(target_window) < 10:
        raise ValueError("Target window too small for calibration refinement.")

    if lowpass_cutoff_hz is not None:
        rate_hz = min(sample_rate_hz, 100.0)
        reference_window = lowpass_filter(resample_stream(reference_window, rate_hz), lowpass_cutoff_hz, rate_hz)
        target_window = lowpass_filter(resample_stream(target_window, rate_hz), lowpass_cutoff_hz, rate_hz)

    refined = estimate_offset(
        reference_window,
        target_window,
        sample_rate_hz=min(sample_rate_hz, 100.0),
        max_lag_seconds=search_s + 1.0,
        use_acc=True,
        use_gyro=False,
        differentiate=False,
    )
    return CalibrationWindowResult(
        offset_seconds=float(refined.offset_seconds),
        target_time_seconds=float(target_center_ms / 1000.0),
        correlation_score=float(refined.score),
        window_duration_s=window_duration_s,
    )


def estimate_sync_from_calibration(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    coarse_sample_rate_hz: float = DEFAULT_COARSE_SAMPLE_RATE_HZ,
    cal_search_s: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_s: float = DEFAULT_PEAK_BUFFER_SECONDS,
    lowpass_cutoff_hz: float | None = None,
) -> tuple[SyncOutputs, CalibrationWindowResult, CalibrationWindowResult]:
    """Estimate offset and drift from opening/closing calibration anchors."""
    detect_kwargs = dict(
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )
    reference_segments = find_calibration_segments(reference_df, **detect_kwargs)
    if len(reference_segments) < 2:
        raise ValueError(f"Need at least 2 calibration sequences in reference, found {len(reference_segments)}.")

    target_clean = remove_dropouts(target_df)
    try:
        coarse_offset_s = _coarse_offset_from_opening_calibration(
            reference_df,
            target_clean,
            reference_segments[0],
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError:
        coarse = estimate_offset(
            reference_df,
            target_clean,
            sample_rate_hz=coarse_sample_rate_hz,
            max_lag_seconds=coarse_max_lag_s,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    target_timestamps = target_df["timestamp"].to_numpy(dtype=float)
    margin_ms = (cal_search_s + 10.0) * 1000.0
    in_range_segments = []
    for segment in reference_segments:
        reference_center_ms = float(
            np.median(reference_df.iloc[segment.peak_indices]["timestamp"].to_numpy(dtype=float))
        )
        target_center_ms = reference_center_ms - coarse_offset_s * 1000.0
        if float(target_timestamps[0]) - margin_ms <= target_center_ms <= float(target_timestamps[-1]) + margin_ms:
            in_range_segments.append(segment)
    if len(in_range_segments) < 2:
        raise ValueError("Need at least 2 reference calibration anchors within the target time span.")

    opening = _refine_offset_at_calibration(
        reference_df,
        target_df,
        in_range_segments[0],
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    closing = _refine_offset_at_calibration(
        reference_df,
        target_df,
        in_range_segments[-1],
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    anchor_span_s = closing.target_time_seconds - opening.target_time_seconds
    if abs(anchor_span_s) < 1.0:
        raise ValueError("Opening and closing calibrations are too close for a stable drift estimate.")

    drift_from_anchors = (closing.offset_seconds - opening.offset_seconds) / anchor_span_s
    if abs(drift_from_anchors) > _MAX_PLAUSIBLE_DRIFT:
        drift_seconds_per_second = _estimate_drift_from_duration(reference_df, target_df)
        drift_source = "duration_ratio"
    else:
        drift_seconds_per_second = drift_from_anchors
        drift_source = "calibration_windows"

    target_origin_seconds = float(target_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = opening.offset_seconds - drift_seconds_per_second * (
        opening.target_time_seconds - target_origin_seconds
    )
    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset_at_origin_s),
        drift_seconds_per_second=float(drift_seconds_per_second),
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(coarse_max_lag_s),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    outputs = SyncOutputs(
        model=model,
        metadata={
            "sync_method": "calibration_windows",
            "drift_source": drift_source,
            "calibration": {
                "opening": {
                    "offset_s": round(opening.offset_seconds, 6),
                    "t_tgt_s": round(opening.target_time_seconds, 3),
                    "score": round(opening.correlation_score, 4),
                    "window_duration_s": round(opening.window_duration_s, 2),
                },
                "closing": {
                    "offset_s": round(closing.offset_seconds, 6),
                    "t_tgt_s": round(closing.target_time_seconds, 3),
                    "score": round(closing.correlation_score, 4),
                    "window_duration_s": round(closing.window_duration_s, 2),
                },
                "calibration_span_s": round(anchor_span_s, 1),
            },
        },
    )
    return outputs, opening, closing


def synchronize_pair(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    coarse_sample_rate_hz: float = DEFAULT_COARSE_SAMPLE_RATE_HZ,
    cal_search_s: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_s: float = DEFAULT_PEAK_BUFFER_SECONDS,
    lowpass_cutoff_hz: float | None = None,
) -> SyncArtifacts:
    """Synchronize one reference/target CSV pair from calibration anchors."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    outputs, _, _ = estimate_sync_from_calibration(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
        coarse_max_lag_s=coarse_max_lag_s,
        coarse_sample_rate_hz=coarse_sample_rate_hz,
        cal_search_s=cal_search_s,
        peak_buffer_s=peak_buffer_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    return write_sync_outputs(
        reference_csv=reference_path,
        target_csv=target_path,
        reference_df=reference_df,
        target_df=target_df,
        outputs=outputs,
        out_dir=Path(output_dir),
        correlation_rate_hz=sample_rate_hz,
    )


def synchronize_recording(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = DEFAULT_REFERENCE_SENSOR,
    target_sensor: str = DEFAULT_TARGET_SENSOR,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    coarse_sample_rate_hz: float = DEFAULT_COARSE_SAMPLE_RATE_HZ,
    cal_search_s: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_s: float = DEFAULT_PEAK_BUFFER_SECONDS,
    lowpass_cutoff_hz: float | None = None,
) -> SyncArtifacts:
    """Synchronize one recording's parsed streams from calibration anchors."""
    inputs = load_recording_inputs(
        recording_name,
        stage_in,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
    outputs, _, _ = estimate_sync_from_calibration(
        inputs.reference_df,
        inputs.target_df,
        reference_name=str(inputs.reference_csv),
        target_name=str(inputs.target_csv),
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
        coarse_max_lag_s=coarse_max_lag_s,
        coarse_sample_rate_hz=coarse_sample_rate_hz,
        cal_search_s=cal_search_s,
        peak_buffer_s=peak_buffer_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    return write_sync_outputs(
        reference_csv=inputs.reference_csv,
        target_csv=inputs.target_csv,
        reference_df=inputs.reference_df,
        target_df=inputs.target_df,
        outputs=outputs,
        out_dir=stage_output_dir(recording_name, METHOD_NAME),
        correlation_rate_hz=sample_rate_hz,
    )
