"""Concrete synchronization methods and recording/file I/O helpers.

This module contains the four synchronization strategies used in the
Multi-IMU analysis:

- ``sda``: coarse offset only
- ``lida``: SDA + linear drift fit
- ``calibration``: calibration-sequence anchors (all available windows)
- ``online``: opening anchor + pre-characterised drift
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import project_relative_path, write_csv
from parser.calibration_segments import _acc_norm, _find_peaks, _smooth, find_calibration_segments
from common.paths import find_sensor_csv, recording_stage_dir

from .core import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    SyncModel,
    _adaptive_windowed_refinement,
    _fit_offset_drift,
    apply_sync_model,
    build_alignment_series,
    compute_sync_correlations,
    estimate_offset,
    estimate_sync_model,
    load_stream,
    lowpass_filter,
    remove_dropouts,
    resample_aligned_stream,
    resample_stream,
    save_sync_model,
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0
DEFAULT_DRIFT_PPM = 300.0

SYNC_METHODS: tuple[str, ...] = ("sda", "lida", "calibration", "online", "adaptive")
METHOD_STAGES: dict[str, str] = {
    "sda": "synced/sda",
    "lida": "synced/lida",
    "calibration": "synced/cal",
    "online": "synced/online",
    "adaptive": "synced/adaptive",
}
METHOD_LABELS: dict[str, str] = {
    "sda": "SDA only",
    "lida": "SDA + LIDA",
    "calibration": "Calibration",
    "online": "Online",
    "adaptive": "Adaptive",
}
DRIFT_CHAR_JSON = Path(__file__).parent.parent / "data" / "drift_characterisation.json"


def method_stage(method: str) -> str:
    """Return the output stage for a sync method."""
    try:
        return METHOD_STAGES[method]
    except KeyError as exc:
        raise ValueError(f"Unknown sync method {method!r}; expected one of {SYNC_METHODS}") from exc


def method_label(method: str) -> str:
    """Return a human-readable label for a sync method."""
    return METHOD_LABELS.get(method, method)


def _drop_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in df.columns
    ]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _prepare_recording_io(
    recording_name: str,
    stage_in: str,
    method: str,
    reference_sensor: str,
    target_sensor: str,
) -> tuple[Path, Path, Path]:
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)
    out_dir = recording_stage_dir(recording_name, method_stage(method))
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "sync %s/%s: reference=%s target=%s",
        recording_name,
        method_stage(method),
        ref_csv.name,
        tgt_csv.name,
    )
    return ref_csv, tgt_csv, out_dir


def _write_recording_outputs(
    *,
    recording_name: str,
    method: str,
    ref_csv: Path,
    tgt_csv: Path,
    out_dir: Path,
    aligned_df: pd.DataFrame,
    sync_payload: dict,
    reference_sensor: str,
    target_sensor: str,
    resample_rate_hz: float | None = None,
) -> tuple[Path, Path, Path]:
    ref_out = out_dir / f"{reference_sensor}.csv"
    tgt_out = out_dir / f"{target_sensor}.csv"
    sync_json = out_dir / "sync_info.json"

    shutil.copy2(ref_csv, ref_out)
    write_csv(_drop_alignment_columns(aligned_df), tgt_out)
    sync_json.write_text(json.dumps(sync_payload, indent=2), encoding="utf-8")

    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(aligned_df, resample_rate_hz=float(resample_rate_hz))
        uniform_out = out_dir / f"{target_sensor}_uniform.csv"
        write_csv(_drop_alignment_columns(uniform_df), uniform_out)

    return ref_out, tgt_out, sync_json


def _sync_payload(
    model: SyncModel,
    *,
    sync_method: str,
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    sample_rate_hz: float,
    extra: dict | None = None,
) -> dict:
    payload = asdict(model)
    payload["sync_method"] = sync_method
    payload["correlation"] = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    if extra:
        payload.update(extra)
    return payload


def synchronize_recording_sda(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronize two sensor streams using SDA (offset only)."""
    ref_csv, tgt_csv, out_dir = _prepare_recording_io(
        recording_name, stage_in, "sda", reference_sensor, target_sensor
    )

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    offset = estimate_offset(
        ref_df,
        tgt_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    model = SyncModel(
        reference_csv=str(ref_csv),
        target_csv=str(tgt_csv),
        target_time_origin_seconds=float(tgt_df["timestamp"].iloc[0]) / 1000.0,
        offset_seconds=float(offset.offset_seconds),
        drift_seconds_per_second=0.0,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    payload = _sync_payload(
        model,
        sync_method="sda_offset_only",
        ref_df=ref_df,
        tgt_df=tgt_df,
        sample_rate_hz=sample_rate_hz,
        extra={"sda_score": round(float(offset.score), 4)},
    )
    return _write_recording_outputs(
        recording_name=recording_name,
        method="sda",
        ref_csv=ref_csv,
        tgt_csv=tgt_csv,
        out_dir=out_dir,
        aligned_df=aligned_df,
        sync_payload=payload,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )


def synchronize(
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
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """Synchronize target stream to reference stream using SDA + LIDA."""
    ref_path = Path(reference_csv)
    tgt_path = Path(target_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = load_stream(ref_path)
    tgt_df = load_stream(tgt_path)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(ref_path),
        target_name=str(tgt_path),
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

    sync_json_path = out_dir / f"{tgt_path.stem}_to_{ref_path.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "sda_lida"
    sync_data["correlation"] = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    aligned_df = _drop_alignment_columns(apply_sync_model(tgt_df, model, replace_timestamp=True))
    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_csv(aligned_df, synced_csv_path)

    uniform_csv_path: Path | None = None
    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(aligned_df, resample_rate_hz=float(resample_rate_hz))
        uniform_csv_path = out_dir / f"{tgt_path.stem}_synced_uniform.csv"
        write_csv(_drop_alignment_columns(uniform_df), uniform_csv_path)

    return sync_json_path, synced_csv_path, uniform_csv_path


def synchronize_recording(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronize a recording using SDA + LIDA."""
    ref_csv, tgt_csv, out_dir = _prepare_recording_io(
        recording_name, stage_in, "lida", reference_sensor, target_sensor
    )
    tmp_dir = out_dir / "_tmp"
    try:
        sync_json_raw, synced_csv_raw, uniform_csv_raw = synchronize(
            reference_csv=ref_csv,
            target_csv=tgt_csv,
            output_dir=tmp_dir,
            sample_rate_hz=sample_rate_hz,
            max_lag_seconds=max_lag_seconds,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
            local_search_seconds=local_search_seconds,
            min_window_score=min_window_score,
            min_fit_r2=min_fit_r2,
            resample_rate_hz=resample_rate_hz,
            use_acc=use_acc,
            use_gyro=use_gyro,
            use_mag=use_mag,
        )

        ref_out = out_dir / f"{reference_sensor}.csv"
        tgt_out = out_dir / f"{target_sensor}.csv"
        sync_json_out = out_dir / "sync_info.json"
        shutil.copy2(ref_csv, ref_out)
        shutil.move(str(synced_csv_raw), tgt_out)
        shutil.move(str(sync_json_raw), sync_json_out)

        if uniform_csv_raw is not None:
            uniform_out = out_dir / f"{target_sensor}_uniform.csv"
            shutil.move(str(uniform_csv_raw), uniform_out)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return ref_out, tgt_out, sync_json_out


@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured by cross-correlating one calibration window."""

    offset_seconds: float
    t_tgt_seconds: float
    correlation_score: float
    window_duration_s: float


def _maybe_add(
    clusters: list[list[int]],
    current: list[int],
    ts: np.ndarray,
    peak_min_count: int,
    peak_max_count: int,
    max_duration_s: float,
) -> None:
    if peak_min_count <= len(current) <= peak_max_count:
        duration_s = (ts[current[-1]] - ts[current[0]]) / 1000.0
        if duration_s <= max_duration_s:
            clusters.append(list(current))


def _coarse_offset_from_opening_calibration(
    ref_df: pd.DataFrame,
    tgt_df_clean: pd.DataFrame,
    ref_cal,
    *,
    search_first_s: float = 180.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_count: int = 8,
    min_inter_peak_s: float = 0.3,
    max_inter_peak_s: float = 2.5,
    max_cluster_duration_s: float = 20.0,
) -> float:
    """Estimate coarse offset from the opening calibration cluster in the target."""
    ts = tgt_df_clean["timestamp"].to_numpy(dtype=float)
    if len(ts) < 2:
        raise ValueError("Target DataFrame is too short.")

    dt_ms = float(np.median(np.diff(ts)))
    actual_sr = max(1.0, 1000.0 / dt_ms)
    search_end_ms = ts[0] + search_first_s * 1000.0
    n_search = int(np.sum(ts <= search_end_ms))

    norm = _acc_norm(tgt_df_clean)
    g = float(np.nanmedian(norm))
    smooth_win = max(3, int(actual_sr * 0.1))
    dyn_smooth = np.abs(_smooth(norm[:n_search], smooth_win) - g)

    dist = max(1, int(actual_sr * min_inter_peak_s))
    peaks = _find_peaks(dyn_smooth, height=peak_min_height, distance=dist)
    if len(peaks) == 0:
        raise ValueError(
            f"No peaks above {peak_min_height} m/s² found in first {search_first_s}s of target."
        )

    clusters: list[list[int]] = []
    current: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        gap_s = (ts[int(p)] - ts[current[-1]]) / 1000.0
        if min_inter_peak_s <= gap_s <= max_inter_peak_s:
            current.append(int(p))
        else:
            _maybe_add(clusters, current, ts, peak_min_count, peak_max_count, max_cluster_duration_s)
            current = [int(p)]
    _maybe_add(clusters, current, ts, peak_min_count, peak_max_count, max_cluster_duration_s)

    if not clusters:
        raise ValueError(
            f"No calibration-like cluster (3-{peak_max_count} peaks, "
            f"{min_inter_peak_s}-{max_inter_peak_s}s spacing) found in first {search_first_s}s of target."
        )

    tgt_median_ms = float(np.median(ts[clusters[0]]))
    ref_median_ms = float(np.median(ref_df.iloc[ref_cal.peak_indices]["timestamp"].to_numpy(dtype=float)))
    coarse_offset_s = (ref_median_ms - tgt_median_ms) / 1000.0
    log.info(
        "Coarse offset from opening calibration cluster: %.3f s (target cluster at t_tgt=%.1f s)",
        coarse_offset_s,
        tgt_median_ms / 1000.0,
    )
    return coarse_offset_s


def _estimate_drift_from_duration(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    ref_dur_s = (float(ref_df["timestamp"].iloc[-1]) - float(ref_df["timestamp"].iloc[0])) / 1000.0
    tgt_dur_s = (float(tgt_df["timestamp"].iloc[-1]) - float(tgt_df["timestamp"].iloc[0])) / 1000.0
    if tgt_dur_s <= 0.0:
        log.warning("Target recording has zero or negative duration; returning drift = 0.")
        return 0.0
    return (ref_dur_s / tgt_dur_s) - 1.0


def _refine_offset_at_calibration(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    seg,
    *,
    coarse_offset_s: float,
    sample_rate_hz: float = 100.0,
    peak_buffer_s: float = 1.0,
    search_s: float = 3.0,
    lowpass_cutoff_hz: float | None = None,
) -> CalibrationWindowResult:
    buf = int(sample_rate_hz * peak_buffer_s)
    p_start = max(0, seg.peak_indices[0] - buf)
    p_end = min(len(ref_df) - 1, seg.peak_indices[-1] + buf)

    ref_window = ref_df.iloc[p_start : p_end + 1].reset_index(drop=True)
    window_duration_s = float((ref_window["timestamp"].iloc[-1] - ref_window["timestamp"].iloc[0]) / 1000.0)

    p_mid = (seg.peak_indices[0] + seg.peak_indices[-1]) // 2
    t_ref_center_ms = float(ref_df["timestamp"].iloc[p_mid])
    t_tgt_center_ms = t_ref_center_ms - coarse_offset_s * 1000.0

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
            f"at t_ref=[{ref_start_ms / 1000:.1f}, {ref_end_ms / 1000:.1f}] s "
            f"(coarse offset={coarse_offset_s:.3f} s)."
        )

    sr = min(sample_rate_hz, 100.0)
    ref_window_filtered = ref_window
    tgt_window_filtered = tgt_window_raw
    if lowpass_cutoff_hz is not None:
        ref_uniform = resample_stream(ref_window, sr)
        tgt_uniform = resample_stream(tgt_window_raw, sr)
        ref_window_filtered = lowpass_filter(ref_uniform, lowpass_cutoff_hz, sr)
        tgt_window_filtered = lowpass_filter(tgt_uniform, lowpass_cutoff_hz, sr)

    refined = estimate_offset(
        ref_window_filtered,
        tgt_window_filtered,
        sample_rate_hz=sr,
        max_lag_seconds=search_s + 1.0,
        use_acc=True,
        use_gyro=False,
        differentiate=False,
    )
    return CalibrationWindowResult(
        offset_seconds=float(refined.offset_seconds),
        t_tgt_seconds=float(t_tgt_center_ms / 1000.0),
        correlation_score=float(refined.score),
        window_duration_s=float(window_duration_s),
    )


def estimate_sync_from_calibration(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = 120.0,
    coarse_sample_rate_hz: float = 5.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    lowpass_cutoff_hz: float | None = None,
) -> tuple[SyncModel, CalibrationWindowResult, CalibrationWindowResult, str, list[CalibrationWindowResult], float | None]:
    detect_kwargs = dict(
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )
    ref_cals = find_calibration_segments(ref_df, **detect_kwargs)
    if len(ref_cals) < 2:
        raise ValueError(f"Need >= 2 calibration sequences in reference, found {len(ref_cals)}.")

    tgt_clean = remove_dropouts(tgt_df)
    try:
        coarse_offset_s = _coarse_offset_from_opening_calibration(
            ref_df,
            tgt_clean,
            ref_cals[0],
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError as exc:
        log.warning(
            "Opening-calibration coarse offset failed (%s); falling back to SDA at %.0f Hz.",
            exc,
            coarse_sample_rate_hz,
        )
        coarse = estimate_offset(
            ref_df,
            tgt_clean,
            sample_rate_hz=coarse_sample_rate_hz,
            max_lag_seconds=coarse_max_lag_s,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    tgt_ts = tgt_df["timestamp"].to_numpy(dtype=float)
    tgt_lo_ms = float(tgt_ts[0])
    tgt_hi_ms = float(tgt_ts[-1])
    margin_ms = (cal_search_s + 10.0) * 1000.0

    in_range_cals = []
    for seg in ref_cals:
        ref_center_ms = float(np.median(ref_df.iloc[seg.peak_indices]["timestamp"].to_numpy(dtype=float)))
        tgt_center_ms = ref_center_ms - coarse_offset_s * 1000.0
        if tgt_lo_ms - margin_ms <= tgt_center_ms <= tgt_hi_ms + margin_ms:
            in_range_cals.append(seg)

    if len(in_range_cals) < 2:
        raise ValueError(
            f"Only {len(in_range_cals)} calibration segment(s) from the reference fall within the target range."
        )

    cal_windows: list[CalibrationWindowResult] = []
    for seg in in_range_cals:
        try:
            cal_windows.append(
                _refine_offset_at_calibration(
                    ref_df,
                    tgt_df,
                    seg,
                    coarse_offset_s=coarse_offset_s,
                    sample_rate_hz=sample_rate_hz,
                    peak_buffer_s=peak_buffer_s,
                    search_s=cal_search_s,
                    lowpass_cutoff_hz=lowpass_cutoff_hz,
                )
            )
        except Exception as exc:
            log.warning("Skipping one calibration window during refinement: %s", exc)

    if len(cal_windows) < 2:
        raise ValueError(
            f"Only {len(cal_windows)} calibration segment(s) could be refined successfully."
        )

    cal_windows = sorted(cal_windows, key=lambda w: w.t_tgt_seconds)
    cal_open = cal_windows[0]
    cal_close = cal_windows[-1]

    dt_tgt_s = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
    if abs(dt_tgt_s) < 1.0:
        raise ValueError(
            f"Opening and closing calibrations are only {dt_tgt_s:.2f} s apart; drift estimate is unreliable."
        )

    drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt_tgt_s
    max_plausible_drift = 0.01
    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    fit_r2: float | None = None

    # Prefer a weighted all-anchor linear fit when 3+ windows are available.
    # Fall back to the previous first/last estimate if fit quality is poor.
    if len(cal_windows) >= 3:
        x = np.asarray([w.t_tgt_seconds - tgt_origin_s for w in cal_windows], dtype=float)
        y = np.asarray([w.offset_seconds for w in cal_windows], dtype=float)
        w = np.clip(np.asarray([w.correlation_score for w in cal_windows], dtype=float), 0.0, None)
        try:
            if np.any(w > 0):
                slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
            else:
                slope, intercept = np.polyfit(x, y, 1)
            y_hat = intercept + slope * x
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
            fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            if abs(float(slope)) <= max_plausible_drift and fit_r2 >= 0.2:
                drift = float(slope)
                offset_at_origin_s = float(intercept)
                drift_source = "calibration_windows"
            elif abs(drift_raw) <= max_plausible_drift:
                drift = drift_raw
                offset_at_origin_s = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                drift_source = "calibration_windows"
            else:
                drift = _estimate_drift_from_duration(ref_df, tgt_df)
                offset_at_origin_s = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                drift_source = "duration_ratio"
        except Exception as exc:
            log.warning("All-anchor calibration fit failed (%s); falling back to first/last.", exc)
            if abs(drift_raw) <= max_plausible_drift:
                drift = drift_raw
                offset_at_origin_s = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                drift_source = "calibration_windows"
            else:
                drift = _estimate_drift_from_duration(ref_df, tgt_df)
                offset_at_origin_s = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                drift_source = "duration_ratio"
    else:
        if abs(drift_raw) > max_plausible_drift:
            drift = _estimate_drift_from_duration(ref_df, tgt_df)
            drift_source = "duration_ratio"
        else:
            drift = drift_raw
            drift_source = "calibration_windows"
        offset_at_origin_s = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)

    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=tgt_origin_s,
        offset_seconds=offset_at_origin_s,
        drift_seconds_per_second=drift,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(coarse_max_lag_s),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return model, cal_open, cal_close, drift_source, cal_windows, fit_r2


def synchronize_from_calibration(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = 120.0,
    coarse_sample_rate_hz: float = 5.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    lowpass_cutoff_hz: float | None = None,
    resample_rate_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    ref_path = Path(reference_csv)
    tgt_path = Path(target_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = load_stream(ref_path)
    tgt_df = load_stream(tgt_path)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model, cal_open, cal_close, drift_source, cal_windows, fit_r2 = estimate_sync_from_calibration(
        ref_df,
        tgt_df,
        reference_name=str(ref_path),
        target_name=str(tgt_path),
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

    sync_json_path = out_dir / "sync_info.json"
    save_sync_model(model, sync_json_path)
    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "calibration_windows"
    sync_data["drift_source"] = drift_source
    sync_data["calibration"] = {
        "opening": {
            "offset_s": round(cal_open.offset_seconds, 6),
            "t_tgt_s": round(cal_open.t_tgt_seconds, 3),
            "score": round(cal_open.correlation_score, 4),
            "window_duration_s": round(cal_open.window_duration_s, 2),
        },
        "closing": {
            "offset_s": round(cal_close.offset_seconds, 6),
            "t_tgt_s": round(cal_close.t_tgt_seconds, 3),
            "score": round(cal_close.correlation_score, 4),
            "window_duration_s": round(cal_close.window_duration_s, 2),
        },
        "calibration_span_s": round(cal_close.t_tgt_seconds - cal_open.t_tgt_seconds, 1),
        "n_windows_used": len(cal_windows),
        "fit_r2": round(float(fit_r2), 4) if fit_r2 is not None else None,
        "anchors": [
            {
                "offset_s": round(w.offset_seconds, 6),
                "t_tgt_s": round(w.t_tgt_seconds, 3),
                "score": round(w.correlation_score, 4),
                "window_duration_s": round(w.window_duration_s, 2),
            }
            for w in cal_windows
        ],
    }
    sync_data["correlation"] = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    aligned_df = _drop_alignment_columns(apply_sync_model(tgt_df, model, replace_timestamp=True))
    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_csv(aligned_df, synced_csv_path)

    uniform_csv_path: Path | None = None
    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(aligned_df, resample_rate_hz=float(resample_rate_hz))
        uniform_csv_path = out_dir / f"{tgt_path.stem}_synced_uniform.csv"
        write_csv(_drop_alignment_columns(uniform_df), uniform_csv_path)

    return sync_json_path, synced_csv_path, uniform_csv_path


def synchronize_recording_from_calibration(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    coarse_max_lag_s: float = 120.0,
    coarse_sample_rate_hz: float = 5.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    lowpass_cutoff_hz: float | None = None,
    resample_rate_hz: float | None = None,
) -> tuple[Path, Path, Path]:
    """Synchronize a recording using calibration anchors with robust fallbacks."""
    ref_csv, tgt_csv, out_dir = _prepare_recording_io(
        recording_name, stage_in, "calibration", reference_sensor, target_sensor
    )
    tmp_dir = out_dir / "_tmp"
    try:
        sync_json_raw, synced_csv_raw, uniform_csv_raw = synchronize_from_calibration(
            reference_csv=ref_csv,
            target_csv=tgt_csv,
            output_dir=tmp_dir,
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
            resample_rate_hz=resample_rate_hz,
        )

        ref_out = out_dir / f"{reference_sensor}.csv"
        tgt_out = out_dir / f"{target_sensor}.csv"
        sync_json_out = out_dir / "sync_info.json"
        shutil.copy2(ref_csv, ref_out)
        shutil.move(str(synced_csv_raw), tgt_out)
        shutil.move(str(sync_json_raw), sync_json_out)

        if uniform_csv_raw is not None:
            uniform_out = out_dir / f"{target_sensor}_uniform.csv"
            shutil.move(str(uniform_csv_raw), uniform_out)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return ref_out, tgt_out, sync_json_out


def load_characterised_drift(json_path: Path = DRIFT_CHAR_JSON) -> float:
    """Load the median drift rate (ppm) from offline characterisation."""
    if not json_path.exists():
        log.info("Drift characterisation file not found (%s). Using default %.0f ppm.", json_path, DEFAULT_DRIFT_PPM)
        return DEFAULT_DRIFT_PPM
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("cal_span_ge_60s", "lida"):
            block = data.get(key)
            if block and block.get("median_ppm") is not None:
                return float(block["median_ppm"])
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
    """Estimate sync from the opening calibration anchor and known drift."""
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
    if not ref_cals:
        raise ValueError("No calibration sequence found in reference sensor.")

    opening_cal = ref_cals[0]
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
        coarse = estimate_offset(
            ref_df,
            tgt_clean,
            sample_rate_hz=5.0,
            max_lag_seconds=120.0,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    cal_result = _refine_offset_at_calibration(
        ref_df,
        tgt_df,
        opening_cal,
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
    )

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = cal_result.offset_seconds - drift_s_per_s * (cal_result.t_tgt_seconds - tgt_origin_s)
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
    """Synchronize a recording using one opening calibration anchor plus characterised drift."""
    ref_csv, tgt_csv, out_dir = _prepare_recording_io(
        recording_name, stage_in, "online", reference_sensor, target_sensor
    )

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)
    if drift_ppm is None:
        drift_ppm = load_characterised_drift()

    model = estimate_sync_from_opening_anchor(
        ref_df,
        tgt_df,
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        reference_name=str(ref_csv),
        target_name=str(tgt_csv),
    )
    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    payload = _sync_payload(
        model,
        sync_method="online_opening_anchor",
        ref_df=ref_df,
        tgt_df=tgt_df,
        sample_rate_hz=sample_rate_hz,
        extra={
            "drift_ppm_source": "pre_characterised",
            "drift_ppm_applied": drift_ppm,
        },
    )
    return _write_recording_outputs(
        recording_name=recording_name,
        method="online",
        ref_csv=ref_csv,
        tgt_csv=tgt_csv,
        out_dir=out_dir,
        aligned_df=aligned_df,
        sync_payload=payload,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )


def estimate_sync_adaptive(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = 100.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
) -> tuple[SyncModel, dict]:
    """Estimate sync using the adaptive online algorithm.

    Bootstraps an initial offset from the opening calibration anchor (falling back
    to SDA if no calibration is found), then processes activity-signal windows in
    strict chronological order.  After each accepted window the running linear model
    (offset + drift) is updated, so no future data is ever used.

    Returns
    -------
    model : SyncModel
        Final linear sync model fitted to all accumulated window observations.
    meta : dict
        Metadata for the ``adaptive`` block in ``sync_info.json``:
        ``initial_offset_s``, ``initial_method``, ``opening_anchor`` (if found),
        ``n_windows_used``, ``fit_r2``, ``drift_source``.
    """
    # 1. Bootstrap: opening calibration anchor or SDA fallback.
    ref_cals = find_calibration_segments(
        ref_df,
        sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
    )

    opening_anchor: CalibrationWindowResult | None = None
    initial_method = "sda_fallback"
    initial_offset_s = 0.0

    if ref_cals:
        tgt_clean = remove_dropouts(tgt_df)
        try:
            coarse_offset_s = _coarse_offset_from_opening_calibration(
                ref_df,
                tgt_clean,
                ref_cals[0],
                peak_min_height=peak_min_height,
                peak_min_count=peak_min_count,
            )
            opening_anchor = _refine_offset_at_calibration(
                ref_df,
                tgt_df,
                ref_cals[0],
                coarse_offset_s=coarse_offset_s,
                sample_rate_hz=sample_rate_hz,
                peak_buffer_s=peak_buffer_s,
                search_s=cal_search_s,
            )
            initial_offset_s = opening_anchor.offset_seconds
            initial_method = "calibration_anchor"
            log.info("Adaptive sync: initial offset from calibration anchor: %.4f s", initial_offset_s)
        except Exception as exc:
            log.warning("Adaptive sync: opening calibration failed (%s); falling back to SDA.", exc)
            opening_anchor = None

    if opening_anchor is None:
        tgt_clean = remove_dropouts(tgt_df)
        coarse = estimate_offset(
            ref_df,
            tgt_clean,
            sample_rate_hz=5.0,
            max_lag_seconds=120.0,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        initial_offset_s = float(coarse.offset_seconds)
        initial_method = "sda_fallback"
        log.info("Adaptive sync: initial offset from SDA fallback: %.4f s", initial_offset_s)

    # 2. Build activity signals for the windowed pass.
    ref_series = build_alignment_series(
        ref_df, sample_rate_hz=sample_rate_hz, use_acc=True, use_gyro=False
    )
    tgt_series = build_alignment_series(
        tgt_df, sample_rate_hz=sample_rate_hz, use_acc=True, use_gyro=False
    )
    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0

    # 3. Causal windowed refinement — updates its own running model after each window.
    target_times, offsets, scores = _adaptive_windowed_refinement(
        ref_series,
        tgt_series,
        initial_offset_seconds=initial_offset_s,
        initial_drift_seconds_per_second=0.0,
        target_origin_seconds=tgt_origin_s,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
    )

    # 4. Fit final linear model to all accumulated observations.
    fit_r2 = 0.0
    if offsets.size == 0:
        final_offset_s = initial_offset_s
        final_drift_s_per_s = 0.0
        drift_source = "initial_only"
    else:
        weights = np.clip(scores, 0.0, None)
        final_offset_s, final_drift_s_per_s, fit_r2 = _fit_offset_drift(
            target_times,
            offsets,
            target_origin_seconds=tgt_origin_s,
            weights=weights,
        )
        if fit_r2 < min_fit_r2:
            final_drift_s_per_s = 0.0
            drift_source = "initial_only"
        else:
            drift_source = "adaptive_windowed_fit"

    log.info(
        "Adaptive sync: %d windows accepted, drift=%.1f ppm (R²=%.3f, source=%s)",
        int(offsets.size),
        final_drift_s_per_s * 1e6,
        fit_r2,
        drift_source,
    )

    model = SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=tgt_origin_s,
        offset_seconds=final_offset_s,
        drift_seconds_per_second=final_drift_s_per_s,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(cal_search_s + 1.0),
        created_at_utc=datetime.now(UTC).isoformat(),
    )

    meta: dict = {
        "initial_offset_s": round(float(initial_offset_s), 6),
        "initial_method": initial_method,
        "n_windows_used": int(offsets.size),
        "fit_r2": round(float(fit_r2), 4),
        "drift_source": drift_source,
    }
    if opening_anchor is not None:
        meta["opening_anchor"] = {
            "offset_s": round(opening_anchor.offset_seconds, 6),
            "t_tgt_s": round(opening_anchor.t_tgt_seconds, 3),
            "score": round(opening_anchor.correlation_score, 4),
            "window_duration_s": round(opening_anchor.window_duration_s, 2),
        }
    return model, meta


def synchronize_recording_adaptive(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 100.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
) -> tuple[Path, Path, Path]:
    """Synchronize a recording using adaptive online estimation.

    Bootstraps from the opening calibration anchor (or SDA fallback if absent),
    then fits offset + drift by processing activity-signal windows in chronological
    order — no future data is used at any step.
    """
    ref_csv, tgt_csv, out_dir = _prepare_recording_io(
        recording_name, stage_in, "adaptive", reference_sensor, target_sensor
    )

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model, meta = estimate_sync_adaptive(
        ref_df,
        tgt_df,
        reference_name=str(ref_csv),
        target_name=str(tgt_csv),
        sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        cal_search_s=cal_search_s,
        peak_buffer_s=peak_buffer_s,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_fit_r2=min_fit_r2,
    )

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    drift_source = meta.pop("drift_source")
    payload = _sync_payload(
        model,
        sync_method="adaptive_online",
        ref_df=ref_df,
        tgt_df=tgt_df,
        sample_rate_hz=sample_rate_hz,
        extra={
            "drift_source": drift_source,
            "adaptive": meta,
        },
    )
    return _write_recording_outputs(
        recording_name=recording_name,
        method="adaptive",
        ref_csv=ref_csv,
        tgt_csv=tgt_csv,
        out_dir=out_dir,
        aligned_df=aligned_df,
        sync_payload=payload,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )
