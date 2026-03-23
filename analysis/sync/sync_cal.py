"""Calibration-anchor synchronisation using opening and closing tap bursts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import recording_stage_dir
from calibration.segments import find_calibration_segments

from .align_df import estimate_offset
from .common import load_stream, remove_dropouts
from .drift_estimator import SyncModel
from .helpers import (
    REFERENCE_SENSOR,
    TARGET_SENSOR,
    SyncArtifacts,
    load_recording_streams,
    write_sync_outputs,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_COARSE_MAX_LAG_SECONDS = 60.0
DEFAULT_CAL_SEARCH_SECONDS = 3.0
DEFAULT_PEAK_BUFFER_SECONDS = 1.0
MAX_REASONABLE_DRIFT = 0.01


@dataclass(frozen=True)
class CalibrationAnchor:
    offset_seconds: float
    target_time_seconds: float
    score: float
    window_duration_seconds: float


def _opening_cluster_offset(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    opening_segment,
    *,
    search_first_seconds: float = 180.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_seconds: float = 2.5,
) -> float:
    timestamps = target_df['timestamp'].to_numpy(dtype=float)
    if len(timestamps) < 2:
        raise ValueError('Target stream is too short for calibration matching.')

    cutoff = timestamps[0] + search_first_seconds * 1000.0
    search_mask = timestamps <= cutoff
    search_df = target_df.loc[search_mask].reset_index(drop=True)
    search_segments = find_calibration_segments(
        search_df,
        sample_rate_hz=max(1.0, 1000.0 / float(np.median(np.diff(search_df['timestamp'].to_numpy(dtype=float))))),
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_seconds,
    )
    if not search_segments:
        raise ValueError('Could not find an opening calibration cluster in the target stream.')

    target_segment = search_segments[0]
    reference_mid_ms = float(np.median(reference_df.iloc[opening_segment.peak_indices]['timestamp'].to_numpy(dtype=float)))
    target_mid_ms = float(np.median(search_df.iloc[target_segment.peak_indices]['timestamp'].to_numpy(dtype=float)))
    return (reference_mid_ms - target_mid_ms) / 1000.0


def _refine_anchor(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    segment,
    *,
    coarse_offset_seconds: float,
    sample_rate_hz: float,
    peak_buffer_seconds: float,
    search_seconds: float,
) -> CalibrationAnchor:
    buffer_samples = max(1, int(round(sample_rate_hz * peak_buffer_seconds)))
    start = max(0, segment.peak_indices[0] - buffer_samples)
    end = min(len(reference_df) - 1, segment.peak_indices[-1] + buffer_samples)
    reference_window = reference_df.iloc[start : end + 1].reset_index(drop=True)

    reference_mid_index = (segment.peak_indices[0] + segment.peak_indices[-1]) // 2
    reference_mid_ms = float(reference_df.iloc[reference_mid_index]['timestamp'])
    expected_target_mid_ms = reference_mid_ms - coarse_offset_seconds * 1000.0

    target_timestamps = target_df['timestamp'].to_numpy(dtype=float)
    target_mask = (
        target_timestamps >= expected_target_mid_ms - search_seconds * 1000.0
    ) & (
        target_timestamps <= expected_target_mid_ms + search_seconds * 1000.0
    )
    target_window = target_df.loc[target_mask].reset_index(drop=True)
    if target_window.empty:
        raise ValueError('Target calibration search window is empty.')

    refined = estimate_offset(
        reference_window,
        target_window,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=search_seconds,
        use_acc=True,
        use_gyro=False,
        use_mag=False,
        differentiate=False,
    )

    window_target_mid_ms = expected_target_mid_ms
    return CalibrationAnchor(
        offset_seconds=float(refined.offset_seconds),
        target_time_seconds=float(window_target_mid_ms) / 1000.0,
        score=float(refined.score),
        window_duration_seconds=float(
            reference_window['timestamp'].iloc[-1] - reference_window['timestamp'].iloc[0]
        ) / 1000.0,
    )


def _duration_ratio_drift(reference_df: pd.DataFrame, target_df: pd.DataFrame) -> float:
    ref_duration = (float(reference_df['timestamp'].iloc[-1]) - float(reference_df['timestamp'].iloc[0])) / 1000.0
    tgt_duration = (float(target_df['timestamp'].iloc[-1]) - float(target_df['timestamp'].iloc[0])) / 1000.0
    if tgt_duration <= 0:
        return 0.0
    return ref_duration / tgt_duration - 1.0


def estimate_sync_from_calibration(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    reference_name: str,
    target_name: str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    coarse_max_lag_seconds: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    cal_search_seconds: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_seconds: float = DEFAULT_PEAK_BUFFER_SECONDS,
) -> tuple[SyncModel, dict]:
    reference_segments = find_calibration_segments(reference_df, sample_rate_hz=sample_rate_hz)
    if len(reference_segments) < 2:
        raise ValueError('Calibration sync requires at least two reference calibration segments.')

    target_clean = remove_dropouts(target_df)
    opening_segment = reference_segments[0]
    closing_segment = reference_segments[-1]

    try:
        coarse_offset = _opening_cluster_offset(reference_df, target_clean, opening_segment)
    except ValueError:
        coarse_offset = float(
            estimate_offset(
                reference_df,
                target_clean,
                sample_rate_hz=5.0,
                max_lag_seconds=coarse_max_lag_seconds,
                use_acc=True,
                use_gyro=False,
                differentiate=False,
            ).offset_seconds
        )

    opening = _refine_anchor(
        reference_df,
        target_clean,
        opening_segment,
        coarse_offset_seconds=coarse_offset,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_seconds=peak_buffer_seconds,
        search_seconds=cal_search_seconds,
    )
    closing = _refine_anchor(
        reference_df,
        target_clean,
        closing_segment,
        coarse_offset_seconds=coarse_offset,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_seconds=peak_buffer_seconds,
        search_seconds=cal_search_seconds,
    )

    delta_t = closing.target_time_seconds - opening.target_time_seconds
    if abs(delta_t) < 1e-6:
        drift = 0.0
    else:
        drift = (closing.offset_seconds - opening.offset_seconds) / delta_t
    if abs(drift) > MAX_REASONABLE_DRIFT:
        drift = _duration_ratio_drift(reference_df, target_df)

    target_origin_seconds = float(target_df['timestamp'].iloc[0]) / 1000.0
    offset_at_origin = opening.offset_seconds - drift * (opening.target_time_seconds - target_origin_seconds)
    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset_at_origin),
        drift_seconds_per_second=float(drift),
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(coarse_max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    extra = {
        'sync_method': 'cal',
        'calibration': {
            'opening': opening.__dict__,
            'closing': closing.__dict__,
            'calibration_span_s': round(float(delta_t), 6),
            'coarse_offset_seconds': round(float(coarse_offset), 6),
            'n_reference_calibrations': len(reference_segments),
        },
    }
    return model, extra


def synchronize_streams(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    coarse_max_lag_seconds: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    cal_search_seconds: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_seconds: float = DEFAULT_PEAK_BUFFER_SECONDS,
) -> SyncArtifacts:
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    output_path = Path(output_dir)

    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError('Reference and target streams must both be non-empty.')

    model, extra = estimate_sync_from_calibration(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        sample_rate_hz=sample_rate_hz,
        coarse_max_lag_seconds=coarse_max_lag_seconds,
        cal_search_seconds=cal_search_seconds,
        peak_buffer_seconds=peak_buffer_seconds,
    )
    return write_sync_outputs(
        method='cal',
        reference_csv=reference_path,
        reference_df=reference_df,
        target_csv=target_path,
        target_df=target_df,
        model=model,
        output_dir=output_path,
        sample_rate_hz=sample_rate_hz,
        extra_info=extra,
    )


def synchronize_recording(
    recording_name: str,
    stage_in: str = 'parsed',
    *,
    reference_sensor: str = REFERENCE_SENSOR,
    target_sensor: str = TARGET_SENSOR,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    coarse_max_lag_seconds: float = DEFAULT_COARSE_MAX_LAG_SECONDS,
    cal_search_seconds: float = DEFAULT_CAL_SEARCH_SECONDS,
    peak_buffer_seconds: float = DEFAULT_PEAK_BUFFER_SECONDS,
) -> SyncArtifacts:
    reference_csv, target_csv = load_recording_streams(
        recording_name,
        stage_in,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
    return synchronize_streams(
        reference_csv,
        target_csv,
        output_dir=recording_stage_dir(recording_name, 'synced/cal'),
        sample_rate_hz=sample_rate_hz,
        coarse_max_lag_seconds=coarse_max_lag_seconds,
        cal_search_seconds=cal_search_seconds,
        peak_buffer_seconds=peak_buffer_seconds,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m sync.sync_cal')
    parser.add_argument('recording_name_stage', help="<recording_name>/<stage>")
    parser.add_argument('--sample-rate-hz', type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument('--coarse-max-lag-seconds', type=float, default=DEFAULT_COARSE_MAX_LAG_SECONDS)
    parser.add_argument('--cal-search-seconds', type=float, default=DEFAULT_CAL_SEARCH_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    recording_name, stage_in = args.recording_name_stage.split('/', 1)
    synchronize_recording(
        recording_name,
        stage_in,
        sample_rate_hz=args.sample_rate_hz,
        coarse_max_lag_seconds=args.coarse_max_lag_seconds,
        cal_search_seconds=args.cal_search_seconds,
    )


if __name__ == '__main__':
    main()
