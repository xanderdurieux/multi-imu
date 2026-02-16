"""
LIDA-style (Linear Interpolation Data Alignment) refined synchronization model.

This module implements refined time synchronization using windowed correlation
and linear drift estimation, improving upon SDA's discrete alignment precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .align_df import OffsetEstimate, build_alignment_signal, estimate_offset
from .common import apply_linear_time_transform, load_stream, resample_stream

DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_WINDOW_STEP_SECONDS = 10.0
DEFAULT_LOCAL_SEARCH_SECONDS = 2.0
DEFAULT_USE_ACC = True
DEFAULT_USE_GYRO = True
DEFAULT_USE_MAG = False


@dataclass(frozen=True)
class SyncModel:
    """
    Complete synchronization model with offset, drift, and quality metrics.

    Combines SDA coarse alignment with LIDA-style refined drift estimation
    for sub-sample precision alignment.
    """

    reference_csv: str
    target_csv: str
    target_time_origin_seconds: float
    offset_seconds: float
    drift_seconds_per_second: float
    scale: float
    sample_rate_hz: float
    coarse_lag_samples: int
    coarse_offset_seconds: float
    coarse_score: float
    window_seconds: float
    window_step_seconds: float
    local_search_seconds: float
    num_windows: int
    fit_r2: float
    median_window_score: float
    created_at_utc: str


def _windowed_lag_refinement(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    reference_start_sec: float,
    target_start_sec: float,
    sample_rate_hz: float,
    coarse_lag_samples: int,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine coarse lag estimate using windowed correlation (LIDA-style refinement).

    Slides windows over the reference signal and searches locally around the
    coarse lag to estimate time-varying offset, enabling drift estimation.
    """
    dt = 1.0 / sample_rate_hz
    n_ref = len(reference_signal)
    n_tgt = len(target_signal)

    window_n = max(20, int(round(window_seconds * sample_rate_hz)))
    step_n = max(5, int(round(window_step_seconds * sample_rate_hz)))
    search_n = max(1, int(round(local_search_seconds * sample_rate_hz)))
    half = window_n // 2

    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []

    for center in range(half, n_ref - half, step_n):
        left = center - half
        right = center + half
        ref_win = reference_signal[left:right]

        best_lag: int | None = None
        best_score = -np.inf

        for delta in range(-search_n, search_n + 1):
            lag = coarse_lag_samples + delta
            t_left = left - lag
            t_right = right - lag
            if t_left < 0 or t_right > n_tgt:
                continue

            tgt_win = target_signal[t_left:t_right]
            if len(tgt_win) != len(ref_win):
                continue

            score = float(np.dot(ref_win, tgt_win) / len(ref_win))
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None:
            continue

        ref_center_sec = reference_start_sec + center * dt
        tgt_center_sec = target_start_sec + (center - best_lag) * dt
        offset_sec = ref_center_sec - tgt_center_sec

        target_times.append(tgt_center_sec)
        offsets.append(offset_sec)
        scores.append(best_score)

    return np.asarray(target_times), np.asarray(offsets), np.asarray(scores)


def _fit_offset_drift(
    target_times_rel: np.ndarray,
    offsets: np.ndarray,
) -> tuple[float, float, float]:
    """Fit linear drift model (offset + drift_rate * time) via least squares."""
    if len(target_times_rel) < 2:
        return float(offsets[0]) if len(offsets) == 1 else 0.0, 0.0, 0.0

    slope, intercept = np.polyfit(target_times_rel, offsets, 1)
    predicted = slope * target_times_rel + intercept

    ss_res = float(np.sum((offsets - predicted) ** 2))
    ss_tot = float(np.sum((offsets - np.mean(offsets)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(intercept), float(slope), float(r2)


def estimate_sync_model(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    reference_name: str,
    target_name: str,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 20.0,
) -> SyncModel:
    """
    Estimate complete synchronization model (SDA coarse + LIDA refined drift).

    Combines SDA-style discrete lag estimation with windowed correlation refinement
    to estimate both time offset and clock drift for sub-sample precision alignment.
    """
    coarse: OffsetEstimate = estimate_offset(
        reference_df,
        target_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=DEFAULT_USE_ACC,
        use_gyro=DEFAULT_USE_GYRO,
        use_mag=DEFAULT_USE_MAG,
    )

    ref_rs = resample_stream(reference_df, sample_rate_hz)
    tgt_rs = resample_stream(target_df, sample_rate_hz)

    ref_signal = build_alignment_signal(
        ref_rs,
        use_acc=DEFAULT_USE_ACC,
        use_gyro=DEFAULT_USE_GYRO,
        use_mag=DEFAULT_USE_MAG,
    )
    tgt_signal = build_alignment_signal(
        tgt_rs,
        use_acc=DEFAULT_USE_ACC,
        use_gyro=DEFAULT_USE_GYRO,
        use_mag=DEFAULT_USE_MAG,
    )

    ref_start_sec = float(ref_rs["timestamp"].iloc[0]) / 1000.0
    tgt_start_sec = float(tgt_rs["timestamp"].iloc[0]) / 1000.0
    target_time_origin_seconds = tgt_start_sec

    target_times, offsets, scores = _windowed_lag_refinement(
        reference_signal=ref_signal,
        target_signal=tgt_signal,
        reference_start_sec=ref_start_sec,
        target_start_sec=tgt_start_sec,
        sample_rate_hz=sample_rate_hz,
        coarse_lag_samples=coarse.lag_samples,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        window_step_seconds=DEFAULT_WINDOW_STEP_SECONDS,
        local_search_seconds=DEFAULT_LOCAL_SEARCH_SECONDS,
    )

    if len(offsets) == 0:
        offset_seconds = coarse.offset_seconds
        drift_seconds_per_second = 0.0
        fit_r2 = 0.0
        median_score = coarse.score
    else:
        target_times_rel = target_times - target_time_origin_seconds
        offset_seconds, drift_seconds_per_second, fit_r2 = _fit_offset_drift(
            target_times_rel,
            offsets,
        )
        median_score = float(np.median(scores))

    scale = 1.0 + drift_seconds_per_second

    return SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=target_time_origin_seconds,
        offset_seconds=offset_seconds,
        drift_seconds_per_second=drift_seconds_per_second,
        scale=scale,
        sample_rate_hz=sample_rate_hz,
        coarse_lag_samples=coarse.lag_samples,
        coarse_offset_seconds=coarse.offset_seconds,
        coarse_score=coarse.score,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        window_step_seconds=DEFAULT_WINDOW_STEP_SECONDS,
        local_search_seconds=DEFAULT_LOCAL_SEARCH_SECONDS,
        num_windows=int(len(offsets)),
        fit_r2=fit_r2,
        median_window_score=median_score,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def apply_sync_model(
    target_df: pd.DataFrame,
    model: SyncModel,
    *,
    replace_timestamp: bool = True,
) -> pd.DataFrame:
    """Apply synchronization model to transform target stream timestamps to reference clock."""
    out = target_df.copy()
    out["timestamp_orig"] = pd.to_numeric(out["timestamp"], errors="coerce")
    aligned = apply_linear_time_transform(
        out["timestamp_orig"],
        offset_seconds=model.offset_seconds,
        drift_seconds_per_second=model.drift_seconds_per_second,
        target_origin_seconds=model.target_time_origin_seconds,
    )
    aligned_ms = np.rint(aligned)
    if np.isfinite(aligned_ms).all():
        out["timestamp_aligned"] = aligned_ms.astype(np.int64)
    else:
        out["timestamp_aligned"] = pd.Series(aligned_ms).astype("Int64")

    if replace_timestamp:
        out["timestamp"] = out["timestamp_aligned"]

    return out


def resample_aligned_stream(
    aligned_df: pd.DataFrame,
    *,
    rate_hz: float,
    timestamp_col: str = "timestamp",
    round_timestamp_ms: bool = True,
) -> pd.DataFrame:
    """Resample synchronized stream to uniform rate using linear interpolation (LIDA-style)."""
    out = resample_stream(aligned_df, rate_hz=rate_hz, timestamp_col=timestamp_col)
    if round_timestamp_ms:
        for col in (timestamp_col, "timestamp_orig", "timestamp_aligned"):
            if col not in out.columns:
                continue
            ts = np.rint(pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float))
            if np.isfinite(ts).all():
                out[col] = ts.astype(np.int64)
            else:
                out[col] = pd.Series(ts).astype("Int64")
    return out


def save_sync_model(model: SyncModel, path: Path | str) -> Path:
    """Save synchronization model to JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")
    return out


def load_sync_model(path: Path | str) -> SyncModel:
    """Load synchronization model from JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SyncModel(**data)


def fit_sync_from_paths(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 20.0,
) -> SyncModel:
    """Load reference and target CSV streams and estimate synchronization model."""
    ref_path = Path(reference_csv)
    tgt_path = Path(target_csv)

    reference_df = load_stream(ref_path)
    target_df = load_stream(tgt_path)

    return estimate_sync_model(
        reference_df,
        target_df,
        reference_name=str(ref_path),
        target_name=str(tgt_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
    )
