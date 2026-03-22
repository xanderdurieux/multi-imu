"""Calibration-sequence based synchronisation.

Instead of correlating the full activity signal (SDA + LIDA), this approach
uses the sharp acceleration peaks of the opening and closing calibration
sequences as known reference events to estimate clock offset and drift.

Algorithm
---------
1. Detect calibration segments in the **reference** sensor only.
2. Run SDA cross-correlation on the full streams to obtain a coarse offset.
3. For each calibration (opening and closing):

   a. Extract the peak window from the reference stream.
   b. Map it to the target clock using the coarse offset.
   c. Cross-correlate the two narrow windows → refined offset at that time.

4. Fit the linear model::

       t_ref = t_tgt + offset(t_tgt)
       offset(t) = offset_origin + drift × (t − t_origin)

   where ``drift = (offset_close − offset_open) / (t_close_tgt − t_open_tgt)``.

This avoids needing to find calibration segments in the target sensor (which
can be unreliable when the target has frequent dropout packets) while still
exploiting the known, high-amplitude reference events for precise alignment.

CLI::

    python -m sync.calibration_sync 2026-02-26_5/parsed
    python -m sync.calibration_sync 2026-02-26_5/sections/section_1 --no-plot
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_dataframe
from common.paths import find_sensor_csv, recording_stage_dir

from .align_df import estimate_offset
from .common import add_vector_norms, load_stream, lowpass_filter, remove_dropouts, resample_stream
from .drift_estimator import (
    SyncModel,
    apply_sync_model,
    resample_aligned_stream,
    save_sync_model,
)
from .metrics import compute_sync_correlations
from common.calibration_segments import find_calibration_segments, _acc_norm, _smooth, _find_peaks

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationWindowResult:
    """Refined offset measured by cross-correlating one calibration window."""

    offset_seconds: float         # absolute offset (t_ref = t_tgt + offset)
    t_tgt_seconds: float          # approximate target-clock time of this calibration
    correlation_score: float      # SDA score (higher = better quality)
    window_duration_s: float      # duration of the reference window used


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Estimate coarse offset from the opening calibration cluster in the target.

    Finds the first short peak cluster (typical of a calibration event) in the
    opening portion of the target recording and matches it to the reference
    opening calibration by median timestamp.

    Returns ``offset_seconds = (ref_ts_ms − tgt_ts_ms) / 1000``.

    Raises ``ValueError`` if no calibration-like cluster is found.
    """

    ts = tgt_df_clean["timestamp"].to_numpy(dtype=float)
    if len(ts) < 2:
        raise ValueError("Target DataFrame is too short.")

    # Estimate effective sample rate from median sample interval
    dt_ms = float(np.median(np.diff(ts)))
    actual_sr = max(1.0, 1000.0 / dt_ms)

    # Restrict search to the first search_first_s of target data
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

    # Group peaks into clusters by real-time inter-peak spacing
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

    # Use the first qualifying cluster as the opening calibration
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
    """Conditionally append a cluster if it meets the calibration criteria."""
    if peak_min_count <= len(current) <= peak_max_count:
        duration_s = (ts[current[-1]] - ts[current[0]]) / 1000.0
        if duration_s <= max_duration_s:
            clusters.append(list(current))



def _estimate_drift_from_duration(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
) -> float:
    """Estimate clock drift from the ratio of total recording durations.

    Implements the spec's Step 4::

        alpha = (t_ref_end - t_ref_start) / (t_tgt_end - t_tgt_start)
        drift  = alpha - 1.0

    This is a robust fallback when the closing-calibration cross-correlation
    is unreliable.  It requires no feature detection in the target stream —
    only the raw timestamp extents of both recordings.

    Returns drift in seconds per second (positive means the target clock runs
    slower than the reference clock).
    """
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
    """Cross-correlate a calibration peak window to get a precise offset.

    Parameters
    ----------
    seg:
        ``CalibrationSegment`` from the reference sensor.
    coarse_offset_s:
        Approximate offset (s) from a prior SDA run: ``t_ref ≈ t_tgt + offset``.
    peak_buffer_s:
        Seconds of data to include before/after the first/last peak in the
        reference window.
    search_s:
        The target window is extracted as ±``search_s`` around the expected
        target position, and the internal SDA search range is also ``search_s``.
    lowpass_cutoff_hz:
        If set, apply a zero-phase Butterworth low-pass filter at this cutoff
        (in Hz) to both windows after resampling, before cross-correlation.
        Recommended range: 20–30 Hz.  Reduces high-frequency noise that can
        shift the correlation peak.
    """
    buf = int(sample_rate_hz * peak_buffer_s)
    p_start = max(0, seg.peak_indices[0] - buf)
    p_end = min(len(ref_df) - 1, seg.peak_indices[-1] + buf)

    ref_window = ref_df.iloc[p_start : p_end + 1].reset_index(drop=True)
    window_duration_s = float(
        (ref_window["timestamp"].iloc[-1] - ref_window["timestamp"].iloc[0]) / 1000.0
    )

    # Centre of the calibration window in reference clock (seconds)
    p_mid = (seg.peak_indices[0] + seg.peak_indices[-1]) // 2
    t_ref_center_ms = float(ref_df["timestamp"].iloc[p_mid])

    # Expected centre in target clock (using coarse offset: t_tgt = t_ref - offset)
    t_tgt_center_ms = t_ref_center_ms - coarse_offset_s * 1000.0

    # Align the target window's time span to match the reference window, plus a
    # search margin on each side.  This ensures that after resampling both windows
    # to a uniform grid, their first timestamps differ by only ~coarse_offset ±
    # search_s, keeping the true lag well within max_lag_seconds.
    #
    # Using a symmetric margin around t_tgt_center_ms instead would shift
    # tgt_start to the very beginning of the section for the closing calibration,
    # making ref_start − tgt_start ≈ coarse_offset + margin >> max_lag_seconds
    # and causing the SDA to latch onto a spurious peak.
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

    # Use the raw target window (with dropouts intact).  The resample step in
    # estimate_offset interpolates across the irregular timestamps, which
    # blends zero-norm dropout values with their neighbours – much better than
    # removing rows (which breaks timestamp continuity) or differentiating
    # (which amplifies the zero → real → zero spike).
    sr = min(sample_rate_hz, 100.0)
    ref_window_filtered = ref_window
    tgt_window_filtered = tgt_window_raw
    if lowpass_cutoff_hz is not None:
        # Resample to uniform grid first so the filter sees a constant dt,
        # then hand the filtered windows to estimate_offset (which will
        # re-interpolate internally, but the signal is already smooth).
        from .common import resample_stream
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Estimate a linear sync model using calibration-sequence windows.

    Pipeline (matches spec Steps 1–5)
    ----------------------------------
    1. Detect calibration segments in the **reference** sensor only.
    2. Estimate a coarse offset from the opening calibration cluster in the
       target (peak-matching, Method B from spec Step 3); falls back to a
       low-rate SDA cross-correlation (Method A) if no cluster is found.
    3. For each calibration (opening and closing): cross-correlate a narrow
       peak window between reference and target → precise independent offset
       measurement at that calibration time (refined Method A).
    4. Estimate drift (spec Step 4):
       - **Primary**: linear fit from the two calibration-window offsets
         (``drift = Δoffset / Δt``), equivalent to comparing calibration
         event timestamps across both clocks.
       - **Fallback** (when primary drift exceeds 1 % plausibility threshold,
         indicating a false-peak match in the closing calibration):
         recording-duration ratio ``alpha = ref_duration / tgt_duration``,
         so ``drift = alpha − 1``.
    5. Apply the linear model: ``t_ref = t_tgt + offset + drift × (t_tgt − t₀)``.

    Returns
    -------
    model:
        Fitted ``SyncModel`` (offset + drift at target time origin).
    cal_open:
        Calibration window result for the opening calibration.
    cal_close:
        Calibration window result for the closing calibration.
    drift_source:
        ``"calibration_windows"`` when drift was estimated from the two
        calibration-window cross-correlations; ``"duration_ratio"`` when the
        fallback recording-duration ratio was used.

    Raises
    ------
    ValueError
        If fewer than 2 calibration sequences are found in the reference, or
        if a calibration window cannot be located in the target stream.
    """
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

    # Coarse offset: use the opening calibration cluster in the target to
    # directly match the reference's opening calibration.  This is more
    # robust than full-recording SDA when the target has irregular sampling
    # or frequent dropout packets.
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

    # Determine which reference calibrations are within the target's data range
    tgt_ts = tgt_df["timestamp"].to_numpy(dtype=float)
    tgt_lo_ms = float(tgt_ts[0])
    tgt_hi_ms = float(tgt_ts[-1])
    margin_ms = (cal_search_s + 10.0) * 1000.0  # generous safety margin

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

    # Refine offset at each calibration window
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

    # Fit linear drift from the two offset measurements
    dt_tgt_s = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
    if abs(dt_tgt_s) < 1.0:
        raise ValueError(
            f"Opening and closing calibrations are only {dt_tgt_s:.2f} s apart — "
            "drift estimate would be unreliable."
        )

    drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt_tgt_s

    # Sanity-check: a drift exceeding ~1% is physically implausible for typical
    # embedded clocks (Arduino crystals are usually within 1000 ppm).  If the
    # estimate is beyond this, the closing-calibration cross-correlation likely
    # landed on a false peak.  Fall back to the recording-duration ratio
    # (spec Step 4: alpha = ref_duration / tgt_duration) rather than zeroing
    # out drift, which would leave systematic time error uncorrected.
    MAX_PLAUSIBLE_DRIFT = 0.01  # 1 % = 10 000 ppm
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
    """Synchronize *target_csv* to *reference_csv* using calibration windows.

    Mirrors :func:`sync.lida_sync.synchronize` but derives the sync model
    from calibration-window cross-correlation rather than full-signal LIDA.

    Returns ``(sync_info_json, target_synced_csv, uniform_csv_or_None)``.
    """
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

    # Save model JSON annotated with calibration details and correlation stats
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

    # Compute correlation quality metric (offset-only vs full offset+drift)
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
    """Synchronize a recording using calibration windows, writing to ``synced/cal/``.

    Mirrors :func:`sync.lida_sync.synchronize_recording` but uses
    :func:`synchronize_from_calibration` instead of SDA + LIDA.

    Returns ``(reference_csv, synced_target_csv, sync_info_json)``.
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.calibration_sync",
        description=(
            "Synchronise two IMU streams using calibration-sequence windows as "
            "reference events.  Detects calibrations in the reference sensor, "
            "then cross-correlates each calibration's peak window against the "
            "target to fit a precise offset + drift model.  Writes outputs to "
            "data/recordings/<recording_name>/synced/cal/."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help=(
            "Recording name and input stage as '<recording_name>/<stage>' "
            "(e.g. '2026-02-26_5/parsed')."
        ),
    )
    parser.add_argument(
        "--reference-sensor", default="sporsa",
        help="Sensor used as time reference (default: sporsa).",
    )
    parser.add_argument(
        "--target-sensor", default="arduino",
        help="Sensor whose timestamps are corrected (default: arduino).",
    )
    parser.add_argument(
        "--sample-rate-hz", type=float, default=100.0,
        help="Approximate sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--static-min-s", type=float, default=3.0,
        help="Minimum flanking static region in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--static-threshold", type=float, default=1.5,
        help="Max |acc_norm - g| (m/s²) to classify a sample as static (default: 1.5).",
    )
    parser.add_argument(
        "--peak-min-height", type=float, default=3.0,
        help="Min |acc_norm - g| (m/s²) to count as a calibration peak (default: 3.0).",
    )
    parser.add_argument(
        "--peak-min-count", type=int, default=3,
        help="Minimum peaks in a cluster to qualify as calibration (default: 3).",
    )
    parser.add_argument(
        "--peak-max-gap-s", type=float, default=3.0,
        help="Max gap between consecutive peaks in the same cluster (default: 3.0).",
    )
    parser.add_argument(
        "--static-gap-max-s", type=float, default=5.0,
        help="Max gap between flanking static run and nearest peak (default: 5.0).",
    )
    parser.add_argument(
        "--coarse-max-lag-s", type=float, default=120.0,
        help="SDA coarse search window in seconds (default: 120).",
    )
    parser.add_argument(
        "--coarse-sample-rate-hz", type=float, default=5.0,
        help="Sample rate for the coarse SDA alignment (default: 5.0 Hz).",
    )
    parser.add_argument(
        "--cal-search-s", type=float, default=5.0,
        help="Search window (±s) for each calibration window alignment (default: 5.0).",
    )
    parser.add_argument(
        "--peak-buffer-s", type=float, default=1.0,
        help="Seconds of buffer around first/last peak in reference window (default: 1.0).",
    )
    parser.add_argument(
        "--lowpass-cutoff-hz", type=float, default=None,
        help=(
            "If set, apply a zero-phase Butterworth low-pass filter at this "
            "cutoff (Hz) to calibration windows before cross-correlation "
            "(spec Step 1 optional; recommended: 20–30 Hz)."
        ),
    )
    parser.add_argument(
        "--resample-rate-hz", type=float, default=None,
        help="If set, also write a uniformly resampled synced target CSV.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating plots for the synced/cal stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")
    recording_name, stage_in = parts

    ref_out, tgt_out, sync_json = synchronize_recording_from_calibration(
        recording_name=recording_name,
        stage_in=stage_in,
        reference_sensor=args.reference_sensor,
        target_sensor=args.target_sensor,
        sample_rate_hz=args.sample_rate_hz,
        static_min_s=args.static_min_s,
        static_threshold=args.static_threshold,
        peak_min_height=args.peak_min_height,
        peak_min_count=args.peak_min_count,
        peak_max_gap_s=args.peak_max_gap_s,
        static_gap_max_s=args.static_gap_max_s,
        coarse_max_lag_s=args.coarse_max_lag_s,
        coarse_sample_rate_hz=args.coarse_sample_rate_hz,
        cal_search_s=args.cal_search_s,
        peak_buffer_s=args.peak_buffer_s,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        resample_rate_hz=args.resample_rate_hz,
        plot=not args.no_plot,
    )

    print(f"\nreference: {ref_out}")
    print(f"synced:    {tgt_out}")
    print(f"model:     {sync_json}")


if __name__ == "__main__":
    main()
