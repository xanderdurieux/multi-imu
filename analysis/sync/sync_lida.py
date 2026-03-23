"""SDA + LIDA synchronisation with linear drift fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import recording_stage_dir

from .common import load_stream
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    estimate_sync_model,
)
from .helpers import (
    REFERENCE_SENSOR,
    TARGET_SENSOR,
    SyncArtifacts,
    load_recording_streams,
    write_sync_outputs,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


def synchronize_streams(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
) -> SyncArtifacts:
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    output_path = Path(output_dir)

    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError('Reference and target streams must both be non-empty.')

    model = estimate_sync_model(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_fit_r2=min_fit_r2,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    extra = {
        'sync_method': 'lida',
        'lida': {
            'window_seconds': window_seconds,
            'window_step_seconds': window_step_seconds,
            'local_search_seconds': local_search_seconds,
            'min_window_score': min_window_score,
            'min_fit_r2': min_fit_r2,
        },
    }
    return write_sync_outputs(
        method='lida',
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
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
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
        output_dir=recording_stage_dir(recording_name, 'synced/lida'),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_fit_r2=min_fit_r2,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m sync.sync_lida')
    parser.add_argument('recording_name_stage', help="<recording_name>/<stage>")
    parser.add_argument('--sample-rate-hz', type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument('--max-lag-seconds', type=float, default=DEFAULT_MAX_LAG_SECONDS)
    parser.add_argument('--window-seconds', type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument('--window-step-seconds', type=float, default=DEFAULT_WINDOW_STEP_SECONDS)
    parser.add_argument('--local-search-seconds', type=float, default=DEFAULT_LOCAL_SEARCH_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    recording_name, stage_in = args.recording_name_stage.split('/', 1)
    synchronize_recording(
        recording_name,
        stage_in,
        sample_rate_hz=args.sample_rate_hz,
        max_lag_seconds=args.max_lag_seconds,
        window_seconds=args.window_seconds,
        window_step_seconds=args.window_step_seconds,
        local_search_seconds=args.local_search_seconds,
    )


if __name__ == '__main__':
    main()
