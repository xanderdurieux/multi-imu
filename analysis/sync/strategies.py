"""Synchronization strategy implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .activity import SIGNAL_MODE_ACC_NORM_DIFF, build_alignment_series
from .anchors import (
    CalibrationAnchor,
    calibration_anchor_to_dict,
    load_calibration_anchors,
)
from .model import SyncModel, make_sync_model
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

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0
DEFAULT_DRIFT_PPM = 300.0

_DEFAULT_SIGNAL_MODE = SIGNAL_MODE_ACC_NORM_DIFF
_DEFAULT_MIN_VALID_FRACTION = 0.5


def _target_origin_seconds(tgt_df: pd.DataFrame) -> float:
    return float(tgt_df["timestamp"].iloc[0]) / 1000.0


def _build_series_pair(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    sample_rate_hz: float,
):
    ref_series = build_alignment_series(
        ref_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=_DEFAULT_SIGNAL_MODE,
    )
    tgt_series = build_alignment_series(
        tgt_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=_DEFAULT_SIGNAL_MODE,
    )
    return ref_series, tgt_series


def _build_model(
    *,
    target_origin_seconds: float,
    offset_seconds: float,
    drift_seconds_per_second: float,
) -> SyncModel:
    return make_sync_model(
        target_origin_seconds=target_origin_seconds,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift_seconds_per_second,
    )


def _build_calibration_meta(
    anchors: list[CalibrationAnchor],
    *,
    fit_r2: float | None = None,
) -> dict[str, Any]:
    sorted_anchors = sorted(anchors, key=lambda a: a.tgt_ms)
    opening = sorted_anchors[0] if sorted_anchors else None
    closing = sorted_anchors[-1] if sorted_anchors else None

    calibration: dict[str, Any] = {
        "n_anchors": len(sorted_anchors),
        "anchor_span_s": (
            round(float((closing.tgt_ms - opening.tgt_ms) / 1000.0), 1)
            if opening is not None and closing is not None and len(sorted_anchors) >= 2
            else 0.0
        ),
        "anchors": [
            calibration_anchor_to_dict(anchor, index=i)
            for i, anchor in enumerate(sorted_anchors)
        ],
    }
    if fit_r2 is not None:
        calibration["fit_r2"] = round(float(fit_r2), 4)
    return calibration


def _fit_lida_drift(
    target_times: np.ndarray,
    offsets: np.ndarray,
    scores: np.ndarray,
    *,
    target_origin_seconds: float,
    fallback_offset_seconds: float,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
) -> tuple[float, float, float, bool]:
    """Run LIDA drift fit from local-window offsets.

    Returns ``(offset, drift, fit_r2, accepted)``.
    """
    if offsets.size == 0:
        return fallback_offset_seconds, 0.0, 0.0, False

    weights = np.clip(scores, 0.0, None)
    offset_seconds, drift, fit_r2 = fit_offset_drift(
        target_times,
        offsets,
        target_origin_seconds=target_origin_seconds,
        weights=weights,
    )
    if fit_r2 < min_fit_r2:
        return fallback_offset_seconds, 0.0, fit_r2, False
    return offset_seconds, drift, fit_r2, True


def _window_stats_payload(
    win_stats: dict[str, Any],
    *,
    fit_r2: float,
    scores: np.ndarray,
) -> dict[str, Any]:
    return {
        "accepted_windows": int(win_stats["accepted_windows"]),
        "rejected_windows": int(win_stats["rejected_windows"]),
        "fit_r2": round(float(fit_r2), 4),
        "local_corr_mean": float(np.mean(scores)) if scores.size else None,
        "local_corr_median": float(np.median(scores)) if scores.size else None,
    }


def estimate_multi_anchor(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str = "",
    reference_name: str = "",
    target_name: str = "",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    anchor_search_seconds: float = 5.0,
) -> tuple[SyncModel, dict[str, Any]]:
    """Fit offset and drift from all matched calibration anchors."""
    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
    )
    if len(anchors) < 2:
        raise ValueError(
            f"multi_anchor requires at least 2 anchors, found {len(anchors)}."
        )

    target_origin_s = _target_origin_seconds(tgt_df)
    target_times = np.asarray([a.tgt_ms / 1000.0 for a in anchors], dtype=float)
    offsets = np.asarray([a.offset_s for a in anchors], dtype=float)
    weights = np.ones(len(anchors), dtype=float)
    offset_seconds, drift, fit_r2 = fit_offset_drift(
        target_times,
        offsets,
        target_origin_seconds=target_origin_s,
        weights=weights,
    )

    model = _build_model(
        target_origin_seconds=target_origin_s,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift,
    )
    return model, {
        "sync_method": "multi_anchor",
        "drift_source": "anchor_fit",
        "signal_mode": SIGNAL_MODE_ACC_NORM_DIFF,
        "calibration": _build_calibration_meta(anchors, fit_r2=fit_r2),
        "hyperparameters": {
            "sample_rate_hz": float(sample_rate_hz),
            "max_lag_seconds": float(anchor_search_seconds + 1.0),
            "anchor_search_seconds": float(anchor_search_seconds),
        },
    }


def estimate_one_anchor_adaptive(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str = "",
    reference_name: str = "",
    target_name: str = "",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    anchor_search_seconds: float = 5.0,
) -> tuple[SyncModel, dict[str, Any]]:
    """One-anchor SDA initialization followed by causal LIDA drift fitting."""
    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
    )
    opening_anchor = anchors[0]
    ref_series, tgt_series = _build_series_pair(
        ref_df,
        tgt_df,
        sample_rate_hz=sample_rate_hz,
    )
    target_origin_s = _target_origin_seconds(tgt_df)

    target_times, offsets, scores, win_stats = adaptive_windowed_refinement(
        ref_series,
        tgt_series,
        initial_offset_seconds=opening_anchor.offset_s,
        initial_drift_seconds_per_second=0.0,
        target_origin_seconds=target_origin_s,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        window_step_seconds=DEFAULT_WINDOW_STEP_SECONDS,
        local_search_seconds=DEFAULT_LOCAL_SEARCH_SECONDS,
        min_window_score=DEFAULT_MIN_WINDOW_SCORE,
        min_valid_fraction=_DEFAULT_MIN_VALID_FRACTION,
        start_ref_time_seconds=(opening_anchor.ref_ms / 1000.0),
    )

    final_offset, final_drift, fit_r2, accepted = _fit_lida_drift(
        target_times,
        offsets,
        scores,
        target_origin_seconds=target_origin_s,
        fallback_offset_seconds=opening_anchor.offset_s,
    )

    model = _build_model(
        target_origin_seconds=target_origin_s,
        offset_seconds=final_offset,
        drift_seconds_per_second=final_drift,
    )
    return model, {
        "sync_method": "one_anchor_adaptive",
        "drift_source": "sda_lida" if accepted else "sda",
        "signal_mode": _DEFAULT_SIGNAL_MODE,
        "calibration": _build_calibration_meta(anchors),
        "adaptive": {
            "opening_anchor": calibration_anchor_to_dict(opening_anchor, index=0),
            **_window_stats_payload(win_stats, fit_r2=fit_r2, scores=scores),
        },
        "hyperparameters": {
            "sample_rate_hz": float(sample_rate_hz),
            "max_lag_seconds": float(anchor_search_seconds + 1.0),
            "anchor_search_seconds": float(anchor_search_seconds),
            "window_seconds": float(DEFAULT_WINDOW_SECONDS),
            "window_step_seconds": float(DEFAULT_WINDOW_STEP_SECONDS),
            "local_search_seconds": float(DEFAULT_LOCAL_SEARCH_SECONDS),
            "min_window_score": float(DEFAULT_MIN_WINDOW_SCORE),
        },
    }


def estimate_one_anchor_prior(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str = "",
    reference_name: str = "",
    target_name: str = "",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    drift_ppm: float = DEFAULT_DRIFT_PPM,
) -> tuple[SyncModel, dict[str, Any]]:
    """Use the opening anchor for offset and a fixed drift prior."""
    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
    )
    opening_anchor = anchors[0]

    target_origin_s = _target_origin_seconds(tgt_df)
    drift_s_per_s = float(drift_ppm) * 1e-6
    offset_at_origin = (
        opening_anchor.offset_s
        - drift_s_per_s * (opening_anchor.tgt_ms / 1000.0 - target_origin_s)
    )

    model = _build_model(
        target_origin_seconds=target_origin_s,
        offset_seconds=offset_at_origin,
        drift_seconds_per_second=drift_s_per_s,
    )
    return model, {
        "sync_method": "one_anchor_prior",
        "drift_source": "prior_ppm",
        "signal_mode": SIGNAL_MODE_ACC_NORM_DIFF,
        "drift_ppm_prior": float(drift_ppm),
        "calibration": _build_calibration_meta(anchors),
        "hyperparameters": {
            "sample_rate_hz": float(sample_rate_hz),
            "max_lag_seconds": 5.0,
            "drift_ppm_prior": float(drift_ppm),
        },
    }


def estimate_signal_only(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str = "",
    reference_name: str = "",
    target_name: str = "",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
) -> tuple[SyncModel, dict[str, Any]]:
    """Signal-only synchronization: SDA coarse lag + LIDA drift fitting."""
    ref_series, tgt_series = _build_series_pair(
        ref_df,
        tgt_df,
        sample_rate_hz=sample_rate_hz,
    )
    if ref_series.signal.size == 0 or tgt_series.signal.size == 0:
        raise ValueError("Cannot sync from empty streams.")

    max_lag_samples = int(round(max_lag_seconds * sample_rate_hz))
    lag_samples, coarse_score = estimate_lag(
        ref_series.signal,
        tgt_series.signal,
        max_lag_samples=max_lag_samples,
    )

    coarse_offset_s = (
        float(ref_series.timestamps_seconds[0])
        - float(tgt_series.timestamps_seconds[0])
        + float(lag_samples) / sample_rate_hz
    )
    target_origin_s = float(tgt_series.timestamps_seconds[0])

    target_times, offsets, scores, win_stats = windowed_lag_refinement(
        ref_series,
        tgt_series,
        coarse_lag_samples=lag_samples,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        window_step_seconds=DEFAULT_WINDOW_STEP_SECONDS,
        local_search_seconds=DEFAULT_LOCAL_SEARCH_SECONDS,
        min_window_score=DEFAULT_MIN_WINDOW_SCORE,
        min_valid_fraction=_DEFAULT_MIN_VALID_FRACTION,
    )

    offset_seconds, drift, fit_r2, accepted = _fit_lida_drift(
        target_times,
        offsets,
        scores,
        target_origin_seconds=target_origin_s,
        fallback_offset_seconds=coarse_offset_s,
    )

    model = _build_model(
        target_origin_seconds=target_origin_s,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift,
    )
    return model, {
        "sync_method": "signal_only",
        "drift_source": "sda_lida" if accepted else "sda",
        "signal_mode": _DEFAULT_SIGNAL_MODE,
        "sda_score": round(float(coarse_score), 4),
        "windowed": _window_stats_payload(win_stats, fit_r2=fit_r2, scores=scores),
        "hyperparameters": {
            "sample_rate_hz": float(sample_rate_hz),
            "max_lag_seconds": float(max_lag_seconds),
            "window_seconds": float(DEFAULT_WINDOW_SECONDS),
            "window_step_seconds": float(DEFAULT_WINDOW_STEP_SECONDS),
            "local_search_seconds": float(DEFAULT_LOCAL_SEARCH_SECONDS),
            "min_window_score": float(DEFAULT_MIN_WINDOW_SCORE),
        },
    }
