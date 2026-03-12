"""LIDA-style drift estimation built on top of SDA coarse alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .align_df import (
    AlignmentSeries,
    OffsetEstimate,
    build_alignment_series,
    estimate_offset_from_series,
)
from .common import apply_linear_time_transform, load_stream, resample_stream

DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_WINDOW_STEP_SECONDS = 10.0
DEFAULT_LOCAL_SEARCH_SECONDS = 0.5
DEFAULT_MIN_WINDOW_SCORE = 0.10
DEFAULT_MIN_FIT_R2 = 0.10


@dataclass(frozen=True)
class SyncModel:
    """Global offset + drift model with minimal metadata."""

    reference_csv: str
    target_csv: str
    target_time_origin_seconds: float
    offset_seconds: float
    drift_seconds_per_second: float
    sample_rate_hz: float
    max_lag_seconds: float
    created_at_utc: str


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation (Pearson r) between two equal-length arrays."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm_a = float(np.linalg.norm(a_c))
    norm_b = float(np.linalg.norm(b_c))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_c, b_c) / (norm_a * norm_b))


def _windowed_lag_refinement(
    reference_series: AlignmentSeries,
    target_series: AlignmentSeries,
    *,
    coarse_lag_samples: int,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_signal = reference_series.signal
    tgt_signal = target_series.signal
    ref_ts = reference_series.timestamps_seconds
    tgt_ts = target_series.timestamps_seconds
    sample_rate_hz = float(reference_series.sample_rate_hz)

    n_ref = ref_signal.size
    n_tgt = tgt_signal.size
    if n_ref == 0 or n_tgt == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    window_n = max(20, int(round(float(window_seconds) * sample_rate_hz)))
    step_n = max(5, int(round(float(window_step_seconds) * sample_rate_hz)))
    search_n = max(1, int(round(float(local_search_seconds) * sample_rate_hz)))
    half = window_n // 2
    if n_ref < window_n:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []
    lags: list[float] = []

    for center in range(half, n_ref - half, step_n):
        left = center - half
        right = center + half
        ref_win = ref_signal[left:right]
        if ref_win.size < 10:
            continue

        best_lag: int | None = None
        best_score = float("-inf")

        for delta in range(-search_n, search_n + 1):
            lag = int(coarse_lag_samples + delta)
            t_left = left - lag
            t_right = right - lag
            if t_left < 0 or t_right > n_tgt:
                continue

            tgt_win = tgt_signal[t_left:t_right]
            if tgt_win.size != ref_win.size:
                continue

            score = _ncc(ref_win, tgt_win)
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None or best_score < min_window_score:
            continue

        tgt_idx = center - best_lag
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        ref_time = float(ref_ts[center])
        tgt_time = float(tgt_ts[tgt_idx])
        target_times.append(tgt_time)
        offsets.append(ref_time - tgt_time)
        scores.append(best_score)
        lags.append(float(best_lag))

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        np.asarray(lags, dtype=float),
    )


def _fit_offset_drift(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    target_origin_seconds: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if target_times_sec.size == 0:
        return 0.0, 0.0, 0.0
    if target_times_sec.size == 1:
        return float(offsets_sec[0]), 0.0, 0.0

    x = np.asarray(target_times_sec, dtype=float) - float(target_origin_seconds)
    y = np.asarray(offsets_sec, dtype=float)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, a_min=0.0, a_max=None)
        if np.any(w > 0):
            slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        else:
            slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.polyfit(x, y, 1)

    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(intercept), float(slope), float(r2)


def estimate_sync_model(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    reference_name: str,
    target_name: str,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
) -> SyncModel:
    """
    Estimate complete synchronization model:
    coarse SDA lag + LIDA-style linear drift fit.
    """
    ref_series = build_alignment_series(
        reference_df,
        sample_rate_hz=sample_rate_hz,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    tgt_series = build_alignment_series(
        target_df,
        sample_rate_hz=sample_rate_hz,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )
    if ref_series.signal.size == 0 or tgt_series.signal.size == 0:
        raise ValueError("Cannot estimate sync model from empty streams.")

    coarse: OffsetEstimate = estimate_offset_from_series(
        reference_series=ref_series,
        target_series=tgt_series,
        max_lag_seconds=max_lag_seconds,
    )
    target_origin_seconds = float(tgt_series.timestamps_seconds[0])

    target_times, offsets, scores, _lags = _windowed_lag_refinement(
        reference_series=ref_series,
        target_series=tgt_series,
        coarse_lag_samples=coarse.lag_samples,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
    )

    if offsets.size == 0:
        offset_seconds = float(coarse.offset_seconds)
        drift_seconds_per_second = 0.0
    else:
        weights = np.clip(scores, a_min=0.0, a_max=None)
        offset_seconds, drift_seconds_per_second, fit_r2 = _fit_offset_drift(
            target_times,
            offsets,
            target_origin_seconds=target_origin_seconds,
            weights=weights,
        )
        if fit_r2 < min_fit_r2:
            drift_seconds_per_second = 0.0

    return SyncModel(
        reference_csv=str(reference_name),
        target_csv=str(target_name),
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset_seconds),
        drift_seconds_per_second=float(drift_seconds_per_second),
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def apply_sync_model(
    target_df: pd.DataFrame,
    model: SyncModel,
    *,
    replace_timestamp: bool = True,
) -> pd.DataFrame:
    """Apply offset+drift model to target timestamps."""
    out = target_df.copy()
    out["timestamp_orig"] = pd.to_numeric(out["timestamp"], errors="coerce")
    aligned = apply_linear_time_transform(
        out["timestamp_orig"],
        offset_seconds=model.offset_seconds,
        drift_seconds_per_second=model.drift_seconds_per_second,
        target_origin_seconds=model.target_time_origin_seconds,
    )
    out["timestamp_aligned"] = aligned
    if replace_timestamp:
        out["timestamp"] = out["timestamp_aligned"]
    return out


def resample_aligned_stream(
    aligned_df: pd.DataFrame,
    *,
    resample_rate_hz: float,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Uniformly resample an aligned stream."""
    return resample_stream(
        aligned_df,
        sample_rate_hz=resample_rate_hz,
        timestamp_col=timestamp_col,
    )


def save_sync_model(model: SyncModel, path: Path | str) -> Path:
    """Serialize sync model JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(model), indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_sync_model(path: Path | str) -> SyncModel:
    """Load sync model from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Ignore any unexpected keys from older, more detailed models.
    allowed_keys = {f.name for f in fields(SyncModel)}
    filtered = {k: v for k, v in data.items() if k in allowed_keys}
    return SyncModel(**filtered)


def fit_sync_from_paths(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
) -> SyncModel:
    """Load two CSV paths and estimate a sync model."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    return estimate_sync_model(
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
