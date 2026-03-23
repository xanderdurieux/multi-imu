"""Single-anchor synchronisation for online use."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

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
from .sync_cal import DEFAULT_PEAK_BUFFER_SECONDS, _opening_cluster_offset, _refine_anchor

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_DRIFT_PPM = 400.0
DRIFT_CHARACTERISATION_JSON = Path(__file__).resolve().parents[1] / 'data' / 'drift_characterisation.json'


def load_characterised_drift_ppm(json_path: Path = DRIFT_CHARACTERISATION_JSON) -> float:
    if not json_path.exists():
        return DEFAULT_DRIFT_PPM
    data = json.loads(json_path.read_text(encoding='utf-8'))
    for key in ('cal_span_ge_60s', 'lida'):
        block = data.get(key)
        if isinstance(block, dict) and block.get('median_ppm') is not None:
            return float(block['median_ppm'])
    return DEFAULT_DRIFT_PPM


def estimate_sync_from_opening_anchor(
    reference_df,
    target_df,
    *,
    reference_name: str,
    target_name: str,
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    cal_search_seconds: float = 3.0,
    peak_buffer_seconds: float = DEFAULT_PEAK_BUFFER_SECONDS,
) -> tuple[SyncModel, dict]:
    drift_ppm = load_characterised_drift_ppm() if drift_ppm is None else drift_ppm
    reference_segments = find_calibration_segments(reference_df, sample_rate_hz=sample_rate_hz)
    if not reference_segments:
        raise ValueError('Online sync requires an opening calibration segment in the reference stream.')

    opening_segment = reference_segments[0]
    target_clean = remove_dropouts(target_df)
    try:
        coarse_offset = _opening_cluster_offset(reference_df, target_clean, opening_segment)
    except ValueError:
        coarse_offset = float(
            estimate_offset(
                reference_df,
                target_clean,
                sample_rate_hz=5.0,
                max_lag_seconds=120.0,
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

    drift_seconds_per_second = float(drift_ppm) * 1e-6
    target_origin_seconds = float(target_df['timestamp'].iloc[0]) / 1000.0
    offset_at_origin = opening.offset_seconds - drift_seconds_per_second * (
        opening.target_time_seconds - target_origin_seconds
    )
    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset_at_origin),
        drift_seconds_per_second=drift_seconds_per_second,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(cal_search_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    extra = {
        'sync_method': 'online',
        'online': {
            'opening': opening.__dict__,
            'drift_ppm': float(drift_ppm),
            'coarse_offset_seconds': round(float(coarse_offset), 6),
        },
    }
    return model, extra


def synchronize_streams(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    cal_search_seconds: float = 3.0,
    peak_buffer_seconds: float = DEFAULT_PEAK_BUFFER_SECONDS,
) -> SyncArtifacts:
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    output_path = Path(output_dir)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError('Reference and target streams must both be non-empty.')

    model, extra = estimate_sync_from_opening_anchor(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        cal_search_seconds=cal_search_seconds,
        peak_buffer_seconds=peak_buffer_seconds,
    )
    return write_sync_outputs(
        method='online',
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
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    cal_search_seconds: float = 3.0,
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
        output_dir=recording_stage_dir(recording_name, 'synced/online'),
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        cal_search_seconds=cal_search_seconds,
        peak_buffer_seconds=peak_buffer_seconds,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m sync.sync_online')
    parser.add_argument('recording_name_stage', help="<recording_name>/<stage>")
    parser.add_argument('--drift-ppm', type=float, default=None)
    parser.add_argument('--sample-rate-hz', type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    recording_name, stage_in = args.recording_name_stage.split('/', 1)
    synchronize_recording(
        recording_name,
        stage_in,
        drift_ppm=args.drift_ppm,
        sample_rate_hz=args.sample_rate_hz,
    )


if __name__ == '__main__':
    main()
