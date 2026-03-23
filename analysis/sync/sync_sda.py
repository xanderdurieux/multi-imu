"""SDA-based offset-only synchronisation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from common import recording_stage_dir

from .align_df import estimate_offset
from .common import load_stream
from .drift_estimator import SyncModel
from .helpers import (
    REFERENCE_SENSOR,
    TARGET_SENSOR,
    SyncArtifacts,
    load_recording_streams,
    write_sync_outputs,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


def estimate_sync_model_sda(
    reference_df,
    target_df,
    *,
    reference_name: str,
    target_name: str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> tuple[SyncModel, dict]:
    offset = estimate_offset(
        reference_df,
        target_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    target_origin_seconds = float(target_df['timestamp'].iloc[0]) / 1000.0
    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset.offset_seconds),
        drift_seconds_per_second=0.0,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    details = {
        'sync_method': 'sda',
        'sda': {
            'lag_samples': offset.lag_samples,
            'lag_seconds': round(float(offset.lag_seconds), 6),
            'score': round(float(offset.score), 6),
        },
    }
    return model, details


def synchronize_streams(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> SyncArtifacts:
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    output_path = Path(output_dir)

    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError('Reference and target streams must both be non-empty.')

    model, extra = estimate_sync_model_sda(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    return write_sync_outputs(
        method='sda',
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
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
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
        output_dir=recording_stage_dir(recording_name, 'synced/sda'),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m sync.sync_sda')
    parser.add_argument('recording_name_stage', help="<recording_name>/<stage>")
    parser.add_argument('--sample-rate-hz', type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument('--max-lag-seconds', type=float, default=DEFAULT_MAX_LAG_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    recording_name, stage_in = args.recording_name_stage.split('/', 1)
    synchronize_recording(
        recording_name,
        stage_in,
        sample_rate_hz=args.sample_rate_hz,
        max_lag_seconds=args.max_lag_seconds,
    )


if __name__ == '__main__':
    main()
