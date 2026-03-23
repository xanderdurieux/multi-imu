"""SDA-only synchronisation."""

from __future__ import annotations

from datetime import UTC, datetime
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
from .core import SyncModel, estimate_offset, load_stream

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0
METHOD_NAME = "sda"


def estimate_sync_from_sda(
    reference_df,
    target_df,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> SyncOutputs:
    """Estimate an offset-only sync model using SDA cross-correlation."""
    offset = estimate_offset(
        reference_df,
        target_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    target_origin_seconds = float(target_df["timestamp"].iloc[0]) / 1000.0
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
    return SyncOutputs(
        model=model,
        metadata={
            "sync_method": "sda_offset_only",
            "sda_score": round(float(offset.score), 4),
        },
    )


def synchronize_pair(
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
    """Synchronize one reference/target CSV pair with SDA."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    outputs = estimate_sync_from_sda(
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
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> SyncArtifacts:
    """Synchronize one recording's parsed streams with SDA."""
    inputs = load_recording_inputs(
        recording_name,
        stage_in,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
    outputs = estimate_sync_from_sda(
        inputs.reference_df,
        inputs.target_df,
        reference_name=str(inputs.reference_csv),
        target_name=str(inputs.target_csv),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
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
