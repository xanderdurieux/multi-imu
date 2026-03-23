"""Online synchronisation using the opening calibration anchor and characterised drift."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.calibration_segments import find_calibration_segments

from ._shared import (
    DEFAULT_REFERENCE_SENSOR,
    DEFAULT_TARGET_SENSOR,
    SyncArtifacts,
    SyncOutputs,
    load_recording_inputs,
    stage_output_dir,
    write_sync_outputs,
)
from .core import SyncModel, estimate_offset, load_stream, remove_dropouts
from .sync_cal import _coarse_offset_from_opening_calibration, _refine_offset_at_calibration

log = logging.getLogger(__name__)
METHOD_NAME = "online"
DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_DRIFT_PPM = 400.0
DRIFT_CHAR_JSON = Path(__file__).resolve().parents[1] / "data" / "drift_characterisation.json"


def load_characterised_drift(json_path: Path = DRIFT_CHAR_JSON) -> float:
    """Load the median drift rate (ppm) from offline characterisation output."""
    if not json_path.exists():
        return DEFAULT_DRIFT_PPM
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load %s: %s", json_path, exc)
        return DEFAULT_DRIFT_PPM

    for key in ("cal_span_ge_60s", "lida"):
        block = payload.get(key)
        if isinstance(block, dict) and block.get("median_ppm") is not None:
            return float(block["median_ppm"])
    return DEFAULT_DRIFT_PPM


def estimate_sync_from_opening_anchor(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    reference_name: str = "",
    target_name: str = "",
) -> SyncOutputs:
    """Estimate sync from the opening calibration plus pre-characterised drift."""
    if drift_ppm is None:
        drift_ppm = load_characterised_drift()
    drift_seconds_per_second = float(drift_ppm) * 1e-6

    reference_segments = find_calibration_segments(
        reference_df,
        sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
    )
    if not reference_segments:
        raise ValueError("No calibration sequence found in the reference sensor.")

    opening_segment = reference_segments[0]
    target_clean = remove_dropouts(target_df)
    try:
        coarse_offset_s = _coarse_offset_from_opening_calibration(
            reference_df,
            target_clean,
            opening_segment,
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError:
        coarse = estimate_offset(
            reference_df,
            target_clean,
            sample_rate_hz=5.0,
            max_lag_seconds=120.0,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    opening = _refine_offset_at_calibration(
        reference_df,
        target_df,
        opening_segment,
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
    )

    target_origin_seconds = float(target_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = opening.offset_seconds - drift_seconds_per_second * (
        opening.target_time_seconds - target_origin_seconds
    )
    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=offset_at_origin_s,
        drift_seconds_per_second=drift_seconds_per_second,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(cal_search_s + 1.0),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return SyncOutputs(
        model=model,
        metadata={
            "sync_method": "online_opening_anchor",
            "drift_ppm_source": "pre_characterised",
            "drift_ppm_applied": float(drift_ppm),
            "opening_anchor": {
                "offset_s": round(opening.offset_seconds, 6),
                "t_tgt_s": round(opening.target_time_seconds, 3),
                "score": round(opening.correlation_score, 4),
                "window_duration_s": round(opening.window_duration_s, 2),
            },
        },
    )


def synchronize_pair(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> SyncArtifacts:
    """Synchronize one reference/target CSV pair with the online method."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    if reference_df.empty or target_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    outputs = estimate_sync_from_opening_anchor(
        reference_df,
        target_df,
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        reference_name=str(reference_path),
        target_name=str(target_path),
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
    drift_ppm: float | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> SyncArtifacts:
    """Synchronize one recording's parsed streams with the online method."""
    inputs = load_recording_inputs(
        recording_name,
        stage_in,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
    outputs = estimate_sync_from_opening_anchor(
        inputs.reference_df,
        inputs.target_df,
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        reference_name=str(inputs.reference_csv),
        target_name=str(inputs.target_csv),
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
