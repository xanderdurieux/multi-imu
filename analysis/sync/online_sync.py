"""Online (single-anchor) synchronisation using opening calibration + characterised drift."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common import write_dataframe
from common.calibration_segments import find_calibration_segments
from common.paths import find_sensor_csv, recording_stage_dir

from .calibration_sync import (
    _coarse_offset_from_opening_calibration,
    _refine_offset_at_calibration,
)
from .core import (
    SyncModel,
    apply_sync_model,
    compute_sync_correlations,
    estimate_offset,
    load_stream,
    remove_dropouts,
    save_sync_model,
)

log = logging.getLogger(__name__)

DEFAULT_DRIFT_PPM = 400.0
DRIFT_CHAR_JSON = Path(__file__).parent.parent / "data" / "drift_characterisation.json"


def load_characterised_drift(json_path: Path = DRIFT_CHAR_JSON) -> float:
    """Load the median drift rate (ppm) from the offline drift characterisation."""
    if not json_path.exists():
        log.info(
            "Drift characterisation file not found (%s). "
            "Using default %.0f ppm.",
            json_path, DEFAULT_DRIFT_PPM,
        )
        return DEFAULT_DRIFT_PPM
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("cal_span_ge_60s", "lida"):
            block = data.get(key)
            if block and block.get("median_ppm") is not None:
                ppm = float(block["median_ppm"])
                log.info("Loaded characterised drift: %.1f ppm (from '%s')", ppm, key)
                return ppm
    except Exception as exc:
        log.warning("Could not load drift characterisation: %s. Using default.", exc)
    return DEFAULT_DRIFT_PPM


def estimate_sync_from_opening_anchor(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    drift_ppm: float | None = None,
    sample_rate_hz: float = 100.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    reference_name: str = "",
    target_name: str = "",
) -> SyncModel:
    if drift_ppm is None:
        drift_ppm = load_characterised_drift()
    drift_s_per_s = drift_ppm * 1e-6

    ref_cals = find_calibration_segments(
        ref_df,
        sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
    )
    if len(ref_cals) == 0:
        raise ValueError(
            "No calibration sequence found in reference sensor. "
            "Ensure the opening calibration tap has been recorded."
        )

    opening_cal = ref_cals[0]
    log.info(
        "Opening calibration detected in reference at ~%.1f s "
        "(peaks: %d)",
        float(ref_df.iloc[opening_cal.peak_indices[0]]["timestamp"]) / 1000.0,
        len(opening_cal.peak_indices),
    )

    tgt_clean = remove_dropouts(tgt_df)
    try:
        coarse_offset_s = _coarse_offset_from_opening_calibration(
            ref_df,
            tgt_clean,
            opening_cal,
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError:
        log.warning(
            "Opening-calibration coarse offset failed; falling back to SDA at 5 Hz."
        )
        coarse = estimate_offset(
            ref_df, tgt_clean,
            sample_rate_hz=5.0,
            max_lag_seconds=120.0,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    cal_result = _refine_offset_at_calibration(
        ref_df, tgt_df, opening_cal,
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
    )

    log.info(
        "Opening anchor: offset=%.6f s (score=%.3f), drift=%.0f ppm (pre-characterised)",
        cal_result.offset_seconds,
        cal_result.correlation_score,
        drift_ppm,
    )

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = (
        cal_result.offset_seconds
        - drift_s_per_s * (cal_result.t_tgt_seconds - tgt_origin_s)
    )

    return SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=tgt_origin_s,
        offset_seconds=offset_at_origin_s,
        drift_seconds_per_second=drift_s_per_s,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(cal_search_s + 1.0),
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def synchronize_recording_online(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    drift_ppm: float | None = None,
    sample_rate_hz: float = 100.0,
) -> tuple[Path, Path, Path]:
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)
    out_dir = recording_stage_dir(recording_name, "synced/online")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced/online] {reference_sensor} (ref) <- {ref_csv.name}")
    print(f"[{recording_name}/synced/online] {target_sensor} (target) <- {tgt_csv.name}")

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)

    if drift_ppm is None:
        drift_ppm = load_characterised_drift()

    model = estimate_sync_from_opening_anchor(
        ref_df, tgt_df,
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        reference_name=str(ref_csv),
        target_name=str(tgt_csv),
    )

    sync_json_path = out_dir / "sync_info.json"
    save_sync_model(model, sync_json_path)

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    correlations = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)

    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "online_opening_anchor"
    sync_data["drift_ppm_source"] = "pre_characterised"
    sync_data["drift_ppm_applied"] = drift_ppm
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    ref_out = out_dir / f"{reference_sensor}.csv"
    tgt_out = out_dir / f"{target_sensor}.csv"
    shutil.copy2(ref_csv, ref_out)
    write_dataframe(aligned_df, tgt_out)

    print(f"[{recording_name}/synced/online] {ref_out.name}")
    print(f"[{recording_name}/synced/online] {tgt_out.name}")
    print(f"[{recording_name}/synced/online] {sync_json_path.name}")

    return ref_out, tgt_out, sync_json_path
