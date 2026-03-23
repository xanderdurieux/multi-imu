"""Offline SDA + LIDA synchronisation."""

from __future__ import annotations

from pathlib import Path

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
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    estimate_sync_model,
    load_stream,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0
METHOD_NAME = "lida"


def estimate_sync_from_lida(
    reference_df,
    target_df,
    *,
    reference_name: str = "",
    target_name: str = "",
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
) -> SyncOutputs:
    """Estimate a full offset+drift model using SDA followed by LIDA refinement."""
    model = estimate_sync_model(
        reference_df,
        target_df,
        reference_name=reference_name,
        target_name=target_name,
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
    return SyncOutputs(model=model, metadata={"sync_method": "sda_lida"})


def synchronize_pair(
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
    """Synchronize one reference/target CSV pair with SDA + LIDA."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    outputs = estimate_sync_from_lida(
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
    """Synchronize one recording's parsed streams with SDA + LIDA."""
    inputs = load_recording_inputs(
        recording_name,
        stage_in,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
    outputs = estimate_sync_from_lida(
        inputs.reference_df,
        inputs.target_df,
        reference_name=str(inputs.reference_csv),
        target_name=str(inputs.target_csv),
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
    return write_sync_outputs(
        reference_csv=inputs.reference_csv,
        target_csv=inputs.target_csv,
        reference_df=inputs.reference_df,
        target_df=inputs.target_df,
        outputs=outputs,
        out_dir=stage_output_dir(recording_name, METHOD_NAME),
        correlation_rate_hz=sample_rate_hz,
    )
