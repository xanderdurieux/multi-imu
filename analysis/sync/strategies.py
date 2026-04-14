"""Four synchronization strategies, ordered by tier strength.

Each function takes raw DataFrames and returns ``(SyncModel, metadata_dict)``.
The metadata dict is merged into ``sync_info.json`` by the pipeline layer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .activity import SIGNAL_MODE_ACC_NORM_DIFF, build_alignment_series
from .anchors import (
    CalibrationWindowResult,
    bootstrap_coarse_offset,
    detect_reference_calibrations,
    filter_segments_in_target_range,
    refine_offset_at_calibration,
)
from .model import SyncModel, make_sync_model
from .stream_io import remove_dropouts
from .xcorr import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    adaptive_windowed_refinement,
    estimate_lag,
    fit_offset_drift,
    windowed_lag_refinement,
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0
DEFAULT_DRIFT_PPM = 300.0
MAX_PLAUSIBLE_DRIFT = 0.01
MIN_ANCHOR_SCORE = 0.5


def _anchor_to_dict(w: CalibrationWindowResult) -> dict[str, Any]:
    return {
        "offset_s": round(w.offset_seconds, 6),
        "t_tgt_s": round(w.t_tgt_seconds, 3),
        "score": round(w.correlation_score, 4),
        "window_duration_s": round(w.window_duration_s, 2),
    }


def _duration_ratio_drift(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    ref_dur = (float(ref_df["timestamp"].iloc[-1]) - float(ref_df["timestamp"].iloc[0])) / 1000.0
    tgt_dur = (float(tgt_df["timestamp"].iloc[-1]) - float(tgt_df["timestamp"].iloc[0])) / 1000.0
    if tgt_dur <= 0.0:
        return 0.0
    return (ref_dur / tgt_dur) - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Multi-anchor protocol
# ═══════════════════════════════════════════════════════════════════════════


def estimate_multi_anchor(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    lowpass_cutoff_hz: float | None = None,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
) -> tuple[SyncModel, dict[str, Any]]:
    """Multi-anchor calibration: offset + drift from >=2 protocol anchors.

    Raises ValueError when fewer than 2 usable anchors are available.
    """
    ref_cals = detect_reference_calibrations(
        ref_df, sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height, peak_min_count=peak_min_count,
    )
    if len(ref_cals) < 2:
        raise ValueError(
            f"Need >= 2 calibration sequences in reference, found {len(ref_cals)}."
        )

    coarse_offset_s, coarse_method = bootstrap_coarse_offset(
        ref_df, tgt_df, ref_cals,
        peak_min_height=peak_min_height, peak_min_count=peak_min_count,
    )
    in_range = filter_segments_in_target_range(
        ref_df, tgt_df, ref_cals, coarse_offset_s=coarse_offset_s,
    )
    if len(in_range) < 2:
        raise ValueError(
            f"Only {len(in_range)} calibration segment(s) fall within target range."
        )

    cal_windows: list[CalibrationWindowResult] = []
    for seg in in_range:
        try:
            cal_windows.append(
                refine_offset_at_calibration(
                    ref_df, tgt_df, seg,
                    coarse_offset_s=coarse_offset_s,
                    sample_rate_hz=sample_rate_hz,
                    peak_buffer_s=peak_buffer_s,
                    search_s=cal_search_s,
                    lowpass_cutoff_hz=lowpass_cutoff_hz,
                )
            )
        except Exception as exc:
            log.warning("Skipping calibration window during refinement: %s", exc)

    if len(cal_windows) < 2:
        raise ValueError(
            f"Only {len(cal_windows)} calibration window(s) refined successfully."
        )

    # Reject low-quality anchors and offset outliers.
    good_windows = [w for w in cal_windows if w.correlation_score >= MIN_ANCHOR_SCORE]
    if len(good_windows) >= 2:
        cal_windows = good_windows
    else:
        log.warning(
            "Only %d/%d anchors above score threshold %.2f; using all.",
            len(good_windows), len(cal_windows), MIN_ANCHOR_SCORE,
        )

    # Reject offsets that deviate too far from the coarse estimate:
    # plausible drift accumulates at most ~5 s per 1000 s of recording.
    if len(cal_windows) >= 3:
        offsets = np.asarray([w.offset_seconds for w in cal_windows])
        median_offset = float(np.median(offsets))
        max_deviation_s = 3.0
        consistent = [
            w for w in cal_windows
            if abs(w.offset_seconds - median_offset) <= max_deviation_s
        ]
        if len(consistent) >= 2:
            n_removed = len(cal_windows) - len(consistent)
            if n_removed:
                log.info(
                    "Removed %d offset-outlier anchor(s) (median offset=%.3fs, max dev=%.1fs).",
                    n_removed, median_offset, max_deviation_s,
                )
            cal_windows = consistent

    cal_windows = sorted(cal_windows, key=lambda w: w.t_tgt_seconds)
    cal_open, cal_close = cal_windows[0], cal_windows[-1]
    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    fit_r2: float | None = None

    if len(cal_windows) >= 3:
        x = np.asarray([w.t_tgt_seconds - tgt_origin_s for w in cal_windows])
        y = np.asarray([w.offset_seconds for w in cal_windows])
        w = np.clip(np.asarray([w.correlation_score for w in cal_windows]), 0.0, None)
        try:
            intercept, slope, fit_r2 = fit_offset_drift(
                np.asarray([w.t_tgt_seconds for w in cal_windows]),
                y, target_origin_seconds=tgt_origin_s, weights=w,
            )
            if abs(slope) <= MAX_PLAUSIBLE_DRIFT and fit_r2 >= 0.2:
                drift, offset_at_origin = slope, intercept
                drift_source = "calibration_fit"
            else:
                dt = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
                drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt if abs(dt) > 1 else 0.0
                if abs(drift_raw) <= MAX_PLAUSIBLE_DRIFT:
                    drift = drift_raw
                    offset_at_origin = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                    drift_source = "calibration_endpoints"
                else:
                    drift = _duration_ratio_drift(ref_df, tgt_df)
                    offset_at_origin = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
                    drift_source = "duration_ratio"
        except Exception:
            dt = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
            drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt if abs(dt) > 1 else 0.0
            drift = drift_raw if abs(drift_raw) <= MAX_PLAUSIBLE_DRIFT else _duration_ratio_drift(ref_df, tgt_df)
            offset_at_origin = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)
            drift_source = "calibration_endpoints"
    else:
        dt = cal_close.t_tgt_seconds - cal_open.t_tgt_seconds
        if abs(dt) < 1.0:
            raise ValueError(
                f"Anchors only {dt:.2f}s apart; drift estimate unreliable."
            )
        drift_raw = (cal_close.offset_seconds - cal_open.offset_seconds) / dt
        if abs(drift_raw) > MAX_PLAUSIBLE_DRIFT:
            drift = _duration_ratio_drift(ref_df, tgt_df)
            drift_source = "duration_ratio"
        else:
            drift = drift_raw
            drift_source = "calibration_endpoints"
        offset_at_origin = cal_open.offset_seconds - drift * (cal_open.t_tgt_seconds - tgt_origin_s)

    model = make_sync_model(
        reference_name=reference_name,
        target_name=target_name,
        target_origin_seconds=tgt_origin_s,
        offset_seconds=offset_at_origin,
        drift_seconds_per_second=drift,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=float(cal_search_s + 1.0),
    )
    meta: dict[str, Any] = {
        "sync_method": "multi_anchor",
        "drift_source": drift_source,
        "signal_mode": SIGNAL_MODE_ACC_NORM_DIFF,
        "calibration": {
            "n_anchors": len(cal_windows),
            "anchor_span_s": round(cal_close.t_tgt_seconds - cal_open.t_tgt_seconds, 1),
            "fit_r2": round(float(fit_r2), 4) if fit_r2 is not None else None,
            "opening": _anchor_to_dict(cal_open),
            "closing": _anchor_to_dict(cal_close),
            "anchors": [_anchor_to_dict(w) for w in cal_windows],
        },
    }
    return model, meta


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — One-anchor adaptive
# ═══════════════════════════════════════════════════════════════════════════


def estimate_one_anchor_adaptive(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    min_valid_fraction: float = 0.5,
    signal_mode: str = SIGNAL_MODE_ACC_NORM_DIFF,
    bootstrap_prefix_seconds: float = 60.0,
) -> tuple[SyncModel, dict[str, Any]]:
    """One-anchor adaptive: opening anchor + causal windowed drift estimation."""
    ref_cals = detect_reference_calibrations(
        ref_df, sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height, peak_min_count=peak_min_count,
    )

    opening_anchor: CalibrationWindowResult | None = None
    initial_method = "sda_fallback"
    initial_offset_s = 0.0

    if ref_cals:
        tgt_clean = remove_dropouts(tgt_df)
        try:
            from .anchors import coarse_offset_from_opening_calibration
            coarse_s = coarse_offset_from_opening_calibration(
                ref_df, tgt_clean, ref_cals[0],
                peak_min_height=peak_min_height,
                peak_min_count=peak_min_count,
            )
            opening_anchor = refine_offset_at_calibration(
                ref_df, tgt_df, ref_cals[0],
                coarse_offset_s=coarse_s,
                sample_rate_hz=sample_rate_hz,
                peak_buffer_s=peak_buffer_s,
                search_s=cal_search_s,
            )
            initial_offset_s = opening_anchor.offset_seconds
            initial_method = "calibration_anchor"
        except Exception as exc:
            log.warning("Opening calibration failed (%s); using SDA fallback.", exc)
            opening_anchor = None

    if opening_anchor is None:
        ref_start = float(ref_df["timestamp"].iloc[0])
        tgt_start = float(tgt_df["timestamp"].iloc[0])
        ref_prefix = ref_df.loc[
            ref_df["timestamp"] <= ref_start + bootstrap_prefix_seconds * 1000.0
        ].reset_index(drop=True)
        tgt_prefix = tgt_df.loc[
            tgt_df["timestamp"] <= tgt_start + bootstrap_prefix_seconds * 1000.0
        ].reset_index(drop=True)
        tgt_clean = remove_dropouts(tgt_prefix)
        ref_series = build_alignment_series(
            ref_prefix, sample_rate_hz=5.0, signal_mode=signal_mode
        )
        tgt_series = build_alignment_series(
            tgt_clean, sample_rate_hz=5.0, signal_mode=signal_mode
        )
        max_lag_samples = int(round(min(120.0, bootstrap_prefix_seconds) * 5.0))
        lag_samples, _ = estimate_lag(
            ref_series.signal, tgt_series.signal, max_lag_samples=max_lag_samples
        )
        lag_seconds = float(lag_samples) / 5.0
        initial_offset_s = (
            float(ref_series.timestamps_seconds[0])
            - float(tgt_series.timestamps_seconds[0])
            + lag_seconds
        )
        initial_method = "sda_prefix_bootstrap"

    ref_series = build_alignment_series(
        ref_df, sample_rate_hz=sample_rate_hz, signal_mode=signal_mode
    )
    tgt_series = build_alignment_series(
        tgt_df, sample_rate_hz=sample_rate_hz, signal_mode=signal_mode
    )
    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0

    target_times, offsets, scores, win_stats = adaptive_windowed_refinement(
        ref_series, tgt_series,
        initial_offset_seconds=initial_offset_s,
        initial_drift_seconds_per_second=0.0,
        target_origin_seconds=tgt_origin_s,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_valid_fraction=min_valid_fraction,
    )

    fit_r2 = 0.0
    if offsets.size == 0:
        final_offset = initial_offset_s
        final_drift = 0.0
        drift_source = "initial_only"
    else:
        weights = np.clip(scores, 0.0, None)
        final_offset, final_drift, fit_r2 = fit_offset_drift(
            target_times, offsets,
            target_origin_seconds=tgt_origin_s, weights=weights,
        )
        if fit_r2 < min_fit_r2:
            final_drift = 0.0
            drift_source = "initial_only"
        else:
            drift_source = "adaptive_windowed_fit"

    model = make_sync_model(
        reference_name=reference_name,
        target_name=target_name,
        target_origin_seconds=tgt_origin_s,
        offset_seconds=final_offset,
        drift_seconds_per_second=final_drift,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=float(cal_search_s + 1.0),
    )
    meta: dict[str, Any] = {
        "sync_method": "one_anchor_adaptive",
        "drift_source": drift_source,
        "signal_mode": signal_mode,
        "adaptive": {
            "initial_offset_s": round(float(initial_offset_s), 6),
            "initial_method": initial_method,
            "n_windows_used": int(offsets.size),
            "fit_r2": round(float(fit_r2), 4),
            "accepted_windows": int(win_stats["accepted_windows"]),
            "rejected_windows": int(win_stats["rejected_windows"]),
            "local_corr_mean": float(np.mean(scores)) if scores.size else None,
            "local_corr_median": float(np.median(scores)) if scores.size else None,
        },
    }
    if opening_anchor is not None:
        meta["adaptive"]["opening_anchor"] = _anchor_to_dict(opening_anchor)
    return model, meta


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — One-anchor prior
# ═══════════════════════════════════════════════════════════════════════════


def estimate_one_anchor_prior(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    reference_name: str = "",
    target_name: str = "",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    drift_ppm: float = DEFAULT_DRIFT_PPM,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
) -> tuple[SyncModel, dict[str, Any]]:
    """One-anchor prior: opening anchor + pre-characterised drift."""
    drift_s_per_s = drift_ppm * 1e-6

    ref_cals = detect_reference_calibrations(
        ref_df, sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height, peak_min_count=peak_min_count,
    )
    if not ref_cals:
        raise ValueError("No calibration sequence found in reference sensor.")

    tgt_clean = remove_dropouts(tgt_df)
    try:
        from .anchors import coarse_offset_from_opening_calibration
        coarse_s = coarse_offset_from_opening_calibration(
            ref_df, tgt_clean, ref_cals[0],
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError:
        ref_series = build_alignment_series(ref_df, sample_rate_hz=5.0)
        tgt_series = build_alignment_series(tgt_clean, sample_rate_hz=5.0)
        lag, _ = estimate_lag(
            ref_series.signal, tgt_series.signal,
            max_lag_samples=int(round(120.0 * 5.0)),
        )
        coarse_s = (
            float(ref_series.timestamps_seconds[0])
            - float(tgt_series.timestamps_seconds[0])
            + float(lag) / 5.0
        )

    cal_result = refine_offset_at_calibration(
        ref_df, tgt_df, ref_cals[0],
        coarse_offset_s=coarse_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
    )

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin = (
        cal_result.offset_seconds
        - drift_s_per_s * (cal_result.t_tgt_seconds - tgt_origin_s)
    )

    model = make_sync_model(
        reference_name=reference_name,
        target_name=target_name,
        target_origin_seconds=tgt_origin_s,
        offset_seconds=offset_at_origin,
        drift_seconds_per_second=drift_s_per_s,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=float(cal_search_s + 1.0),
    )
    meta: dict[str, Any] = {
        "sync_method": "one_anchor_prior",
        "drift_source": "prior_ppm",
        "signal_mode": SIGNAL_MODE_ACC_NORM_DIFF,
        "drift_ppm_prior": float(drift_ppm),
        "calibration": {
            "n_anchors": 1,
            "anchor_span_s": 0.0,
            "fit_r2": None,
            "opening": _anchor_to_dict(cal_result),
            "closing": None,
            "anchors": [_anchor_to_dict(cal_result)],
        },
    }
    return model, meta


# ═══════════════════════════════════════════════════════════════════════════
# Tier 4 — Signal-only
# ═══════════════════════════════════════════════════════════════════════════


def estimate_signal_only(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
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
    min_valid_fraction: float = 0.5,
    signal_mode: str = SIGNAL_MODE_ACC_NORM_DIFF,
) -> tuple[SyncModel, dict[str, Any]]:
    """Signal-only: SDA coarse offset + LIDA-style windowed drift fit."""
    ref_series = build_alignment_series(
        ref_df, sample_rate_hz=sample_rate_hz, signal_mode=signal_mode
    )
    tgt_series = build_alignment_series(
        tgt_df, sample_rate_hz=sample_rate_hz, signal_mode=signal_mode
    )
    if ref_series.signal.size == 0 or tgt_series.signal.size == 0:
        raise ValueError("Cannot sync from empty streams.")

    max_lag_samples = int(round(max_lag_seconds * sample_rate_hz))
    lag_samples, coarse_score = estimate_lag(
        ref_series.signal, tgt_series.signal, max_lag_samples=max_lag_samples,
    )
    lag_seconds = float(lag_samples) / sample_rate_hz
    coarse_offset_s = (
        float(ref_series.timestamps_seconds[0])
        - float(tgt_series.timestamps_seconds[0])
        + lag_seconds
    )
    tgt_origin_s = float(tgt_series.timestamps_seconds[0])

    target_times, offsets, scores, win_stats = windowed_lag_refinement(
        ref_series, tgt_series,
        coarse_lag_samples=lag_samples,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_valid_fraction=min_valid_fraction,
    )

    fit_r2 = 0.0
    if offsets.size == 0:
        offset_seconds = coarse_offset_s
        drift = 0.0
    else:
        weights = np.clip(scores, 0.0, None)
        offset_seconds, drift, fit_r2 = fit_offset_drift(
            target_times, offsets,
            target_origin_seconds=tgt_origin_s, weights=weights,
        )
        if fit_r2 < min_fit_r2:
            drift = 0.0

    model = make_sync_model(
        reference_name=reference_name,
        target_name=target_name,
        target_origin_seconds=tgt_origin_s,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
    )
    meta: dict[str, Any] = {
        "sync_method": "signal_only",
        "drift_source": "signal_windowed_fit" if drift != 0.0 else "none",
        "signal_mode": signal_mode,
        "sda_score": round(float(coarse_score), 4),
        "windowed": {
            "n_windows_accepted": int(win_stats["accepted_windows"]),
            "n_windows_rejected": int(win_stats["rejected_windows"]),
            "local_corr_mean": float(np.mean(scores)) if scores.size else None,
            "local_corr_median": float(np.median(scores)) if scores.size else None,
            "fit_r2": round(float(fit_r2), 4),
        },
    }
    return model, meta
