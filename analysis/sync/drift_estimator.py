"""Drift-aware synchronization model estimation and application."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .align_df import OffsetEstimate, build_alignment_signal, estimate_offset
from .common import apply_linear_time_transform, load_stream, resample_stream


@dataclass(frozen=True)
class SyncModel:
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
    ref_signal: np.ndarray,
    tgt_signal: np.ndarray,
    *,
    ref_start_sec: float,
    tgt_start_sec: float,
    sample_rate_hz: float,
    coarse_lag_samples: int,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / sample_rate_hz
    n_ref = len(ref_signal)
    n_tgt = len(tgt_signal)

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
        ref_win = ref_signal[left:right]

        best_lag: int | None = None
        best_score = -np.inf

        for delta in range(-search_n, search_n + 1):
            lag = coarse_lag_samples + delta
            t_left = left - lag
            t_right = right - lag
            if t_left < 0 or t_right > n_tgt:
                continue

            tgt_win = tgt_signal[t_left:t_right]
            if len(tgt_win) != len(ref_win):
                continue

            score = float(np.dot(ref_win, tgt_win) / len(ref_win))
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None:
            continue

        ref_center_sec = ref_start_sec + center * dt
        tgt_center_sec = tgt_start_sec + (center - best_lag) * dt
        offset_sec = ref_center_sec - tgt_center_sec

        target_times.append(tgt_center_sec)
        offsets.append(offset_sec)
        scores.append(best_score)

    return np.asarray(target_times), np.asarray(offsets), np.asarray(scores)


def _fit_offset_drift(
    target_times_rel: np.ndarray,
    offsets: np.ndarray,
) -> tuple[float, float, float]:
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
    window_seconds: float = 20.0,
    window_step_seconds: float = 10.0,
    local_search_seconds: float = 2.0,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
) -> SyncModel:
    """Estimate offset and linear drift from correlated IMU streams."""
    coarse: OffsetEstimate = estimate_offset(
        reference_df,
        target_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    ref_rs = resample_stream(reference_df, sample_rate_hz)
    tgt_rs = resample_stream(target_df, sample_rate_hz)

    ref_signal = build_alignment_signal(
        ref_rs,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    tgt_signal = build_alignment_signal(
        tgt_rs,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    ref_start_sec = float(ref_rs["timestamp"].iloc[0]) / 1000.0
    tgt_start_sec = float(tgt_rs["timestamp"].iloc[0]) / 1000.0
    target_time_origin_seconds = tgt_start_sec

    target_times, offsets, scores = _windowed_lag_refinement(
        ref_signal,
        tgt_signal,
        ref_start_sec=ref_start_sec,
        tgt_start_sec=tgt_start_sec,
        sample_rate_hz=sample_rate_hz,
        coarse_lag_samples=coarse.lag_samples,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
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
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        num_windows=int(len(offsets)),
        fit_r2=fit_r2,
        median_window_score=median_score,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def apply_sync_model(df: pd.DataFrame, model: SyncModel, *, replace_timestamp: bool = True) -> pd.DataFrame:
    """Apply a sync model to a target stream dataframe."""
    out = df.copy()
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
    df: pd.DataFrame,
    *,
    rate_hz: float,
    timestamp_col: str = "timestamp",
    round_timestamp_ms: bool = True,
) -> pd.DataFrame:
    """Resample a synchronized stream to a fixed sample rate."""
    out = resample_stream(df, rate_hz=rate_hz, timestamp_col=timestamp_col)
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
    """Persist sync model metadata as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")
    return out


def load_sync_model(path: Path | str) -> SyncModel:
    """Load a previously saved sync model JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SyncModel(**data)


def fit_sync_from_paths(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 20.0,
    window_seconds: float = 20.0,
    window_step_seconds: float = 10.0,
    local_search_seconds: float = 2.0,
) -> SyncModel:
    """Load two CSV streams and estimate a synchronization model."""
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
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
    )
