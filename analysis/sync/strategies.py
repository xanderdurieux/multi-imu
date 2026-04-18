"""Sync-strategy estimators.

Every estimator returns ``(SyncModel, meta)`` and shares one signature:

    estimate_<method>(ref_df, tgt_df, *, recording_name,
                      reference_sensor="sporsa", target_sensor="arduino",
                      config=None) -> (SyncModel, meta)

Tier hierarchy (strongest to weakest):
  1. multi_anchor        — all calibration anchors, affine fit
  2. one_anchor_adaptive — opening anchor + causal windowed refinement
  3. one_anchor_prior    — opening anchor + fixed drift prior (ppm)
  4. signal_only         — coarse xcorr + non-causal windowed refinement
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .anchors import (
    CalibrationAnchor,
    calibration_anchor_to_dict,
    load_calibration_anchors,
)
from .config import SyncConfig, default_sync_config
from .model import SyncModel
from .signals import build_resampled_activity_signal
from .xcorr import (
    collect_drift_observations_from_anchor,
    estimate_lag,
    fit_offset_drift,
    refine_drift_through_anchor,
    refine_drift_unconstrained,
    refine_offsets_from_coarse_offset,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _target_origin_seconds(tgt_df: pd.DataFrame) -> float:
    return float(tgt_df["timestamp"].iloc[0]) / 1000.0


def _build_signal_pair(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    config: SyncConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_ts, ref_sig = build_resampled_activity_signal(
        ref_df,
        sample_rate_hz=config.resample_rate_hz,
        signal_mode=config.signal_mode,
    )
    tgt_ts, tgt_sig = build_resampled_activity_signal(
        tgt_df,
        sample_rate_hz=config.resample_rate_hz,
        signal_mode=config.signal_mode,
    )
    return ref_ts, ref_sig, tgt_ts, tgt_sig


def _calibration_meta(
    anchors: list[CalibrationAnchor],
    *,
    fit_r2: float | None = None,
) -> dict[str, Any]:
    sorted_anchors = sorted(anchors, key=lambda a: a.tgt_ms)
    opening = sorted_anchors[0] if sorted_anchors else None
    closing = sorted_anchors[-1] if sorted_anchors else None

    meta: dict[str, Any] = {
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
        meta["fit_r2"] = round(float(fit_r2), 4)
    return meta


def _window_stats(
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


def _shared_hyperparameters(config: SyncConfig) -> dict[str, Any]:
    return {
        "resample_rate_hz": float(config.resample_rate_hz),
        "signal_mode": config.signal_mode,
        "min_valid_fraction": float(config.min_valid_fraction),
        "anchor_refinement": {
            "resample_rate_hz": float(config.anchor_refinement.resample_rate_hz),
            "search_seconds": float(config.anchor_refinement.search_seconds),
        },
    }


def _window_hyperparameters(config: SyncConfig) -> dict[str, Any]:
    w = config.window_refinement
    return {
        "window_seconds": float(w.window_seconds),
        "step_seconds": float(w.step_seconds),
        "local_search_seconds": float(w.local_search_seconds),
        "min_window_score": float(w.min_window_score),
        "min_fit_r2": float(w.min_fit_r2),
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def estimate_multi_anchor(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    config: SyncConfig | None = None,
) -> tuple[SyncModel, dict[str, Any]]:
    """Fit offset and drift from all matched calibration anchors."""
    config = config or default_sync_config()
    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
        resample_rate_hz=config.anchor_refinement.resample_rate_hz,
        search_seconds=config.anchor_refinement.search_seconds,
    )
    if len(anchors) < 2:
        raise ValueError(
            f"multi_anchor requires at least 2 anchors, found {len(anchors)}."
        )

    target_origin_s = _target_origin_seconds(tgt_df)
    target_times = np.asarray([a.tgt_ms / 1000.0 for a in anchors], dtype=float)
    offsets = np.asarray([a.offset_s for a in anchors], dtype=float)
    offset_seconds, drift, fit_r2 = fit_offset_drift(
        target_times,
        offsets,
        target_origin_seconds=target_origin_s,
        weights=np.ones(len(anchors), dtype=float),
    )

    model = SyncModel(
        target_time_origin_seconds=target_origin_s,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift,
    )
    return model, {
        "sync_method": "multi_anchor",
        "drift_source": "anchor_fit",
        "calibration": _calibration_meta(anchors, fit_r2=fit_r2),
        "hyperparameters": _shared_hyperparameters(config),
    }


def estimate_one_anchor_adaptive(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    config: SyncConfig | None = None,
) -> tuple[SyncModel, dict[str, Any]]:
    """Opening-anchor offset + causal windowed drift refinement."""
    config = config or default_sync_config()
    window = config.window_refinement

    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
        resample_rate_hz=config.anchor_refinement.resample_rate_hz,
        search_seconds=config.anchor_refinement.search_seconds,
    )
    opening = anchors[0]
    anchor_ref_s = opening.ref_ms / 1000.0
    anchor_tgt_s = opening.tgt_ms / 1000.0
    anchor_offset_s = opening.offset_s

    ref_ts, ref_sig, tgt_ts, tgt_sig = _build_signal_pair(ref_df, tgt_df, config)
    target_origin_s = _target_origin_seconds(tgt_df)

    target_times, offsets, scores, win_stats = collect_drift_observations_from_anchor(
        ref_ts, ref_sig, tgt_ts, tgt_sig,
        sample_rate_hz=config.resample_rate_hz,
        anchor_ref_time_seconds=anchor_ref_s,
        anchor_target_time_seconds=anchor_tgt_s,
        anchor_offset_seconds=anchor_offset_s,
        target_origin_seconds=target_origin_s,
        window_seconds=window.window_seconds,
        window_step_seconds=window.step_seconds,
        local_search_seconds=window.local_search_seconds,
        min_window_score=window.min_window_score,
        min_valid_fraction=config.min_valid_fraction,
    )

    final_offset, final_drift, fit_r2, accepted = refine_drift_through_anchor(
        target_times, offsets, scores,
        min_fit_r2=window.min_fit_r2,
        target_origin_seconds=target_origin_s,
        anchor_time_seconds=anchor_tgt_s,
        anchor_offset_seconds=anchor_offset_s,
        min_fit_ppm=window.min_fit_ppm,
        max_fit_ppm=window.max_fit_ppm,
    )

    model = SyncModel(
        target_time_origin_seconds=target_origin_s,
        offset_seconds=final_offset,
        drift_seconds_per_second=final_drift,
    )
    return model, {
        "sync_method": "one_anchor_adaptive",
        "drift_source": "causal_window_refinement" if accepted else "opening_anchor",
        "calibration": _calibration_meta(anchors),
        "window_refinement": {
            "causal": True,
            "opening_anchor": calibration_anchor_to_dict(opening, index=0),
            **_window_stats(win_stats, fit_r2=fit_r2, scores=scores),
        },
        "hyperparameters": {
            **_shared_hyperparameters(config),
            "window_refinement": _window_hyperparameters(config),
        },
    }


def estimate_one_anchor_prior(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    config: SyncConfig | None = None,
) -> tuple[SyncModel, dict[str, Any]]:
    """Opening anchor for offset plus a fixed drift prior."""
    config = config or default_sync_config()
    drift_ppm = config.one_anchor_prior_drift_ppm

    anchors = load_calibration_anchors(
        recording_name,
        ref_sensor=reference_sensor,
        tgt_sensor=target_sensor,
        resample_rate_hz=config.anchor_refinement.resample_rate_hz,
        search_seconds=config.anchor_refinement.search_seconds,
    )
    opening = anchors[0]

    target_origin_s = _target_origin_seconds(tgt_df)
    drift_s_per_s = drift_ppm * 1e-6
    offset_at_origin = (
        opening.offset_s
        - drift_s_per_s * (opening.tgt_ms / 1000.0 - target_origin_s)
    )

    model = SyncModel(
        target_time_origin_seconds=target_origin_s,
        offset_seconds=offset_at_origin,
        drift_seconds_per_second=drift_s_per_s,
    )
    return model, {
        "sync_method": "one_anchor_prior",
        "drift_source": "drift_prior",
        "calibration": _calibration_meta(anchors),
        "hyperparameters": {
            **_shared_hyperparameters(config),
            "drift_prior_ppm": float(drift_ppm),
        },
    }


def estimate_signal_only(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    recording_name: str = "",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    config: SyncConfig | None = None,
) -> tuple[SyncModel, dict[str, Any]]:
    """Signal-only coarse alignment followed by non-causal windowed refinement."""
    config = config or default_sync_config()
    window = config.window_refinement
    max_lag_seconds = config.signal_only_coarse_search_seconds

    ref_ts, ref_sig, tgt_ts, tgt_sig = _build_signal_pair(ref_df, tgt_df, config)
    if ref_sig.size == 0 or tgt_sig.size == 0:
        raise ValueError("Cannot sync from empty streams.")

    max_lag_samples = int(round(max_lag_seconds * config.resample_rate_hz))
    shift_samples, coarse_score = estimate_lag(
        ref_sig, tgt_sig, max_lag_samples=max_lag_samples,
    )

    coarse_offset_s = (
        float(ref_ts[0]) - float(tgt_ts[0])
        + float(shift_samples) / config.resample_rate_hz
    )
    target_origin_s = float(tgt_ts[0])

    target_times, offsets, scores, win_stats = refine_offsets_from_coarse_offset(
        ref_ts, ref_sig, tgt_ts, tgt_sig,
        sample_rate_hz=config.resample_rate_hz,
        coarse_offset_seconds=coarse_offset_s,
        window_seconds=window.window_seconds,
        window_step_seconds=window.step_seconds,
        local_search_seconds=window.local_search_seconds,
        min_window_score=window.min_window_score,
        min_valid_fraction=config.min_valid_fraction,
    )

    offset_seconds, drift, fit_r2, accepted = refine_drift_unconstrained(
        target_times, offsets, scores,
        min_fit_r2=window.min_fit_r2,
        target_origin_seconds=target_origin_s,
        fallback_offset_seconds=coarse_offset_s,
        min_fit_ppm=window.min_fit_ppm,
        max_fit_ppm=window.max_fit_ppm,
    )

    model = SyncModel(
        target_time_origin_seconds=target_origin_s,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift,
    )
    return model, {
        "sync_method": "signal_only",
        "drift_source": "window_refinement" if accepted else "coarse_signal_alignment",
        "coarse_alignment_score": round(float(coarse_score), 4),
        "window_refinement": {
            "causal": False,
            **_window_stats(win_stats, fit_r2=fit_r2, scores=scores),
        },
        "hyperparameters": {
            "resample_rate_hz": float(config.resample_rate_hz),
            "signal_mode": config.signal_mode,
            "min_valid_fraction": float(config.min_valid_fraction),
            "coarse_search_seconds": float(max_lag_seconds),
            "window_refinement": _window_hyperparameters(config),
        },
    }
