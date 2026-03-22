"""Calibration-sequence based synchronisation."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_dataframe
from common.calibration_segments import find_calibration_segments, _acc_norm, _smooth, _find_peaks
from common.paths import find_sensor_csv, recording_stage_dir

from .core import (
    SyncModel,
    apply_sync_model,
    compute_sync_correlations,
    estimate_offset,
    load_stream,
    lowpass_filter,
    remove_dropouts,
    resample_aligned_stream,
    resample_stream,
    save_sync_model,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured by cross-correlating one calibration window."""

    offset_seconds: float
    t_tgt_seconds: float
    correlation_score: float
    window_duration_s: float


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
    search_mask = ts <= search_end_ms
    n_search = int(np.sum(search_mask))

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
            f"No calibration-like cluster (3–{peak_max_count} peaks, "
            f"{min_inter_peak_s}–{max_inter_peak_s}s spacing) found in first "
            f"{search_first_s}s of target."
        )

    tgt_cluster = clusters[0]
    tgt_median_ms = float(np.median(ts[tgt_cluster]))
    ref_median_ms = float(
        np.median(ref_df.iloc[ref_cal.peak_indices]["timestamp"].to_numpy(dtype=float))
    )

    coarse_offset_s = (ref_median_ms - tgt_median_ms) / 1000.0
    log.info(
        "Coarse offset from opening calibration cluster: %.3f s "
        "(target cluster at t_tgt=%.1f s)",
        coarse_offset_s,
        tgt_median_ms / 1000.0,
    )
    return coarse_offset_s


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


def _estimate_drift_from_duration(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
) -> float:
    ref_dur_s = (
        float(ref_df["timestamp"].iloc[-1]) - float(ref_df["timestamp"].iloc[0])
    ) / 1000.0
    tgt_dur_s = (
        float(tgt_df["timestamp"].iloc[-1]) - float(tgt_df["timestamp"].iloc[0])
    ) / 1000.0
    if tgt_dur_s <= 0.0:
        log.warning("Target recording has zero or negative duration; returning drift = 0.")
        return 0.0
    alpha = ref_dur_s / tgt_dur_s
    return alpha - 1.0


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
    window_duration_s = float(
        (ref_window["timestamp"].iloc[-1] - ref_window["timestamp"].iloc[0]) / 1000.0
    )

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
            f"(coarse offset={coarse_offset_s:.3f} s). "
            "Check that the streams overlap and the coarse offset is reasonable."
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

    log.debug(
        "Calibration window at t_ref=%.1f s: refined offset=%.6f s (score=%.4f, lag=%.3f s)",
        t_ref_center_ms / 1000.0,
        refined.offset_seconds,
        refined.score,
        refined.lag_seconds,
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
) -> tuple[SyncModel, CalibrationWindowResult, CalibrationWindowResult, str]:
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
        raise ValueError(
            f"Need ≥ 2 calibration sequences in reference, found {len(ref_cals)}."
        )

    log.info(
        "Found %d calibration segments in reference. Using first and last.",
        len(ref_cals),
    )

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
            "Opening-calibration coarse offset failed (%s); "
            "falling back to SDA at %.0f Hz.",
            exc, coarse_sample_rate_hz,
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
        log.info("SDA fallback offset: %.3f s (score: %.4f)", coarse_offset_s, coarse.score)

    tgt_ts = tgt_df["timestamp"].to_numpy(dtype=float)
    tgt_lo_ms = float(tgt_ts[0])
    tgt_hi_ms = float(tgt_ts[-1])
    margin_ms = (cal_search_s + 10.0) * 1000.0

    in_range_cals = []
    for seg in ref_cals:
        ref_center_ms = float(
            np.median(ref_df.iloc[seg.peak_indices]["timestamp"].to_numpy(dtype=float))
        )
        tgt_center_ms = ref_center_ms - coarse_offset_s * 1000.0
        if tgt_lo_ms - margin_ms <= tgt_center_ms <= tgt_hi_ms + margin_ms:
            in_range_cals.append(seg)

    if len(in_range_cals) < 2:
        raise ValueError(
            f"Only {len(in_range_cals)} calibration segment(s) from the reference "
            f"fall within the target's data range "
            f"[{tgt_lo_ms / 1000:.1f}, {tgt_hi_ms / 1000:.1f}] s. "
            "Need at least 2 for drift estimation."
        )

    log.info(
        "%d/%d calibration segments are within the target range; "
        "using first and last.",
        len(in_range_cals), len(ref_cals),
    )

    cal_open = _refine_offset_at_calibration(
        ref_df, tgt_df, in_range_cals[0],
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    cal_close = _refine_offset_at_calibration(
        ref_df, tgt_df, in_range_cals[-1],
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    log.info(
        "Opening calibration: offset=%.6f s (score=%.4f)",
        cal_open.offset_seconds, cal_open.correlation_score,
    )
    log.info(
        "Closing calibration: offset=%.6f s (score=%.4f)",
        cal_close.offset_seconds, cal_close.correlation_score,
    )

    dt_tgt_s = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
    if abs(dt_tgt_s) < 1.0:
        raise ValueError(
            f"Opening and closing calibrations are only {dt_tgt_s:.2f} s apart — "
            "drift estimate would be unreliable."
        )

    drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt_tgt_s

    MAX_PLAUSIBLE_DRIFT = 0.01
    if abs(drift_raw) > MAX_PLAUSIBLE_DRIFT:
        drift = _estimate_drift_from_duration(ref_df, tgt_df)
        drift_source = "duration_ratio"
        log.warning(
            "Computed drift (%.4f s/s = %.0f ppm) exceeds plausibility threshold "
            "(%.0f ppm).  The closing-calibration refinement likely failed "
            "(score=%.3f).  Falling back to duration-ratio drift (%.3e s/s = %.0f ppm).",
            drift_raw,
            drift_raw * 1e6,
            MAX_PLAUSIBLE_DRIFT * 1e6,
            cal_close.correlation_score,
            drift,
            drift * 1e6,
        )
    else:
        drift = drift_raw
        drift_source = "calibration_windows"

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = (
        cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
    )

    log.info(
        "Sync model: offset=%.6f s, drift=%.3e s/s (span=%.1f s, source=%s)",
        offset_at_origin_s, drift, dt_tgt_s, drift_source,
    )

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
    return model, cal_open, cal_close, drift_source


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

    model, cal_open, cal_close, drift_source = estimate_sync_from_calibration(
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
    }

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    sync_data["correlation"] = compute_sync_correlations(
        ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz
    )

    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_dataframe(aligned_df, synced_csv_path)

    uniform_csv_path: Path | None = None
    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(
            aligned_df, resample_rate_hz=float(resample_rate_hz),
        )
        uniform_csv_path = out_dir / f"{tgt_path.stem}_synced_uniform.csv"
        write_dataframe(uniform_df, uniform_csv_path)

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
    plot: bool = True,
) -> tuple[Path, Path, Path]:
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)

    out_dir = recording_stage_dir(recording_name, "synced/cal")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced/cal] {reference_sensor} (ref) ← {ref_csv.name}")
    print(f"[{recording_name}/synced/cal] {target_sensor} (target) ← {tgt_csv.name}")

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
            print(f"[{recording_name}/synced/cal] {uniform_out.name}")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    print(f"[{recording_name}/synced/cal] {ref_out.name}")
    print(f"[{recording_name}/synced/cal] {tgt_out.name}")
    print(f"[{recording_name}/synced/cal] {sync_json_out.name}")

    if plot:
        from visualization import plot_comparison

        stage_ref = f"{recording_name}/synced/cal"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return ref_out, tgt_out, sync_json_out
