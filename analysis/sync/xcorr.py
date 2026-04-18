"""Cross-correlation primitives, windowed lag refinement, and drift fitting."""

from __future__ import annotations

import numpy as np

from .activity import AlignmentSeries
from .model import reference_to_target_seconds

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_WINDOW_STEP_SECONDS = 10.0
DEFAULT_LOCAL_SEARCH_SECONDS = 0.5
DEFAULT_MIN_WINDOW_SCORE = 0.10
DEFAULT_MIN_FIT_R2 = 0.10


def fft_correlate_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation via FFT (equivalent to np.correlate mode='full')."""
    ref = np.asarray(a, dtype=float)
    tgt = np.asarray(b, dtype=float)
    n = ref.size + tgt.size - 1
    if n <= 0:
        return np.asarray([], dtype=float)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(ref, nfft) * np.fft.rfft(tgt[::-1], nfft), nfft
    )
    return corr[:n]


def estimate_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int | None = None,
    min_overlap_samples: int = 10,
) -> tuple[int, float]:
    """Estimate the integer lag that maximises overlap-normalised correlation."""
    ref = np.asarray(reference_signal, dtype=float)
    tgt = np.asarray(target_signal, dtype=float)
    n_ref, n_tgt = ref.size, tgt.size
    if n_ref == 0 or n_tgt == 0:
        return 0, float("-inf")

    ref_clean = np.where(np.isfinite(ref), ref, 0.0)
    tgt_clean = np.where(np.isfinite(tgt), tgt, 0.0)
    ref_valid = np.isfinite(ref).astype(float)
    tgt_valid = np.isfinite(tgt).astype(float)

    corr = fft_correlate_full(ref_clean, tgt_clean)
    lags = np.arange(-(n_tgt - 1), n_ref, dtype=int)
    overlap = fft_correlate_full(ref_valid, tgt_valid)

    valid = overlap >= max(1, int(min_overlap_samples))
    if max_lag_samples is not None:
        valid &= np.abs(lags) <= int(max_lag_samples)
    if not valid.any():
        return 0, float("-inf")

    norm_score = np.full(corr.shape, -np.inf, dtype=float)
    norm_score[valid] = corr[valid] / overlap[valid]
    idx = int(np.argmax(norm_score))
    return int(lags[idx]), float(norm_score[idx])


def masked_ncc(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_valid_fraction: float = 0.5,
) -> tuple[float, float]:
    """Normalised cross-correlation (Pearson r) over the finite-overlap region."""
    if a.shape != b.shape:
        raise ValueError("Arrays must have equal length.")
    valid = np.isfinite(a) & np.isfinite(b)
    vfrac = float(valid.mean()) if valid.size else 0.0
    if valid.sum() < 3 or vfrac < min_valid_fraction:
        return float("-inf"), vfrac
    ac = a[valid] - a[valid].mean()
    bc = b[valid] - b[valid].mean()
    na, nb = float(np.linalg.norm(ac)), float(np.linalg.norm(bc))
    if na < 1e-10 or nb < 1e-10:
        return 0.0, vfrac
    return float(np.dot(ac, bc) / (na * nb)), vfrac


# ---------------------------------------------------------------------------
# Linear drift fitting
# ---------------------------------------------------------------------------


def fit_offset_drift(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    target_origin_seconds: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Weighted linear fit: offset = intercept + slope*(t - t0).

    Returns (intercept, slope, R^2).
    """
    if target_times_sec.size == 0:
        return 0.0, 0.0, 0.0
    if target_times_sec.size == 1:
        return float(offsets_sec[0]), 0.0, 0.0

    x = np.asarray(target_times_sec, dtype=float) - float(target_origin_seconds)
    y = np.asarray(offsets_sec, dtype=float)

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
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


# ---------------------------------------------------------------------------
# Non-causal windowed refinement (for signal_only tier)
# ---------------------------------------------------------------------------


def windowed_lag_refinement(
    ref_series: AlignmentSeries,
    tgt_series: AlignmentSeries,
    *,
    coarse_lag_samples: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Non-causal windowed lag refinement around a fixed coarse lag.

    Returns (target_times_sec, offsets_sec, scores, stats_dict).
    """
    ref_sig = ref_series.signal
    tgt_sig = tgt_series.signal
    ref_ts = ref_series.timestamps_seconds
    tgt_ts = tgt_series.timestamps_seconds
    sr = float(ref_series.sample_rate_hz)
    n_ref, n_tgt = ref_sig.size, tgt_sig.size

    empty = np.asarray([], dtype=float)
    empty_stats = {"accepted_windows": 0, "rejected_windows": 0}
    if n_ref == 0 or n_tgt == 0:
        return empty, empty, empty, empty_stats

    win_n = max(20, int(round(window_seconds * sr)))
    step_n = max(5, int(round(window_step_seconds * sr)))
    search_n = max(1, int(round(local_search_seconds * sr)))
    half = win_n // 2
    if n_ref < win_n:
        return empty, empty, empty, empty_stats

    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []
    rejected = 0

    for center in range(half, n_ref - half, step_n):
        left, right = center - half, center + half
        ref_win = ref_sig[left:right]
        if ref_win.size < 10:
            continue

        best_lag: int | None = None
        best_score = float("-inf")
        for delta in range(-search_n, search_n + 1):
            lag = int(coarse_lag_samples + delta)
            t_left, t_right = left - lag, right - lag
            if t_left < 0 or t_right > n_tgt:
                continue
            tgt_win = tgt_sig[t_left:t_right]
            if tgt_win.size != ref_win.size:
                continue
            score, _ = masked_ncc(
                ref_win, tgt_win, min_valid_fraction=min_valid_fraction
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None or best_score < min_window_score:
            rejected += 1
            continue

        tgt_idx = center - best_lag
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        target_times.append(float(tgt_ts[tgt_idx]))
        offsets.append(float(ref_ts[center]) - float(tgt_ts[tgt_idx]))
        scores.append(best_score)

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        {"accepted_windows": len(target_times), "rejected_windows": rejected},
    )


# ---------------------------------------------------------------------------
# Causal adaptive windowed refinement (for one_anchor_adaptive tier)
# ---------------------------------------------------------------------------


def adaptive_windowed_refinement(
    ref_series: AlignmentSeries,
    tgt_series: AlignmentSeries,
    *,
    initial_offset_seconds: float,
    initial_drift_seconds_per_second: float = 0.0,
    target_origin_seconds: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_points_for_drift: int = 3,
    min_valid_fraction: float = 0.5,
    start_ref_time_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Causal windowed refinement: updates running model after each window.

    Only past observations inform the search centre for future windows,
    so no future data leaks into the estimate.

    Returns (target_times_sec, offsets_sec, scores, stats_dict).
    """
    ref_sig = ref_series.signal
    tgt_sig = tgt_series.signal
    ref_ts = ref_series.timestamps_seconds
    tgt_ts = tgt_series.timestamps_seconds
    sr = float(ref_series.sample_rate_hz)
    n_ref, n_tgt = ref_sig.size, tgt_sig.size

    empty = np.asarray([], dtype=float)
    empty_stats = {"accepted_windows": 0, "rejected_windows": 0}
    if n_ref == 0 or n_tgt == 0:
        return empty, empty, empty, empty_stats

    win_n = max(20, int(round(window_seconds * sr)))
    step_n = max(5, int(round(window_step_seconds * sr)))
    search_n = max(1, int(round(local_search_seconds * sr)))
    half = win_n // 2
    if n_ref < win_n:
        return empty, empty, empty, empty_stats

    cur_offset = float(initial_offset_seconds)
    cur_drift = float(initial_drift_seconds_per_second)

    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []
    rejected = 0

    for center in range(half, n_ref - half, step_n):
        t_ref_center = float(ref_ts[center])
        if start_ref_time_seconds is not None and t_ref_center < float(start_ref_time_seconds):
            continue

        # Predict where this ref window maps in target time.
        t_tgt_pred = float(
            reference_to_target_seconds(
                t_ref_center,
                offset_seconds=cur_offset,
                drift_seconds_per_second=cur_drift,
                target_origin_seconds=target_origin_seconds,
            )
        )
        pred_tgt_idx = int(
            np.clip(np.searchsorted(tgt_ts, t_tgt_pred), 0, n_tgt - 1)
        )
        pred_lag = center - pred_tgt_idx

        left, right = center - half, center + half
        ref_win = ref_sig[left:right]
        if ref_win.size < 10:
            continue

        best_lag: int | None = None
        best_score = float("-inf")
        for delta in range(-search_n, search_n + 1):
            lag = pred_lag + delta
            t_left, t_right = left - lag, right - lag
            if t_left < 0 or t_right > n_tgt:
                continue
            tgt_win = tgt_sig[t_left:t_right]
            if tgt_win.size != ref_win.size:
                continue
            score, _ = masked_ncc(
                ref_win, tgt_win, min_valid_fraction=min_valid_fraction
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None or best_score < min_window_score:
            rejected += 1
            continue

        tgt_idx = center - best_lag
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        target_times.append(float(tgt_ts[tgt_idx]))
        offsets.append(float(ref_ts[center]) - float(tgt_ts[tgt_idx]))
        scores.append(best_score)

        # Update running model from all accumulated observations.
        n_pts = len(target_times)
        if n_pts >= min_points_for_drift:
            t_arr = np.asarray(target_times, dtype=float)
            o_arr = np.asarray(offsets, dtype=float)
            w_arr = np.clip(np.asarray(scores, dtype=float), 0.0, None)
            intercept, slope, _ = fit_offset_drift(
                t_arr, o_arr,
                target_origin_seconds=target_origin_seconds,
                weights=w_arr,
            )
            cur_offset = intercept
            cur_drift = slope
        elif n_pts >= 1:
            cur_offset = float(
                np.average(offsets, weights=np.clip(scores, 0.0, None))
            )

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        {"accepted_windows": len(target_times), "rejected_windows": rejected},
    )
