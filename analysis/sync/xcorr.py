"""Xcorr helpers for align arduino timestamps to the sporsa reference clock."""

from __future__ import annotations

import numpy as np

from .model import reference_to_target_seconds


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------


def _fft_correlate_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation via FFT (equivalent to ``np.correlate(mode='full')``)."""
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
    """Integer lag maximising overlap-normalised cross-correlation."""
    ref = np.asarray(reference_signal, dtype=float)
    tgt = np.asarray(target_signal, dtype=float)
    n_ref, n_tgt = ref.size, tgt.size
    if n_ref == 0 or n_tgt == 0:
        return 0, float("-inf")

    ref_clean = np.where(np.isfinite(ref), ref, 0.0)
    tgt_clean = np.where(np.isfinite(tgt), tgt, 0.0)
    ref_valid = np.isfinite(ref).astype(float)
    tgt_valid = np.isfinite(tgt).astype(float)

    corr = _fft_correlate_full(ref_clean, tgt_clean)
    lags = np.arange(-(n_tgt - 1), n_ref, dtype=int)
    overlap = _fft_correlate_full(ref_valid, tgt_valid)

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
    """Normalised cross-correlation (Pearson r) over finite-overlap region."""
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
# Drift fitting primitives
# ---------------------------------------------------------------------------


def fit_offset_drift(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    target_origin_seconds: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Fit offset drift."""
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


def fit_drift_through_point(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    anchor_time_seconds: float,
    anchor_offset_seconds: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit drift through point."""
    if target_times_sec.size == 0:
        return 0.0, 0.0

    dx = np.asarray(target_times_sec, dtype=float) - float(anchor_time_seconds)
    dy = np.asarray(offsets_sec, dtype=float) - float(anchor_offset_seconds)
    w = (
        np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if weights is not None
        else np.ones_like(dx)
    )

    denom = float(np.sum(w * dx * dx))
    if denom < 1e-12:
        return 0.0, 0.0
    slope = float(np.sum(w * dx * dy) / denom)

    w_sum = float(np.sum(w))
    mean_dy = float(np.sum(w * dy) / w_sum) if w_sum > 0 else 0.0
    ss_res = float(np.sum(w * (dy - slope * dx) ** 2))
    ss_tot = float(np.sum(w * (dy - mean_dy) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, float(r2)


# ---------------------------------------------------------------------------
# Offset refinement primitives
# ---------------------------------------------------------------------------


def _best_window_shift_samples(
    ref_sig: np.ndarray,
    tgt_sig: np.ndarray,
    *,
    left: int,
    right: int,
    base_shift_samples: int,
    search_radius_samples: int,
    min_valid_fraction: float,
) -> tuple[int | None, float]:
    """Search best sample shift for one reference window."""
    n_tgt = tgt_sig.size
    ref_win = ref_sig[left:right]
    if ref_win.size < 10:
        return None, float("-inf")

    best_shift: int | None = None
    best_score = float("-inf")
    for delta in range(-search_radius_samples, search_radius_samples + 1):
        shift = int(base_shift_samples + delta)
        t_left, t_right = left - shift, right - shift
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
            best_shift = shift
    return best_shift, best_score


def _collect_window_offsets(
    ref_timestamps_seconds: np.ndarray,
    ref_signal: np.ndarray,
    tgt_timestamps_seconds: np.ndarray,
    tgt_signal: np.ndarray,
    *,
    sample_rate_hz: float,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
    min_window_score: float,
    min_valid_fraction: float,
    start_ref_time_seconds: float | None = None,
    predict_target_time_seconds=None,
    on_accept=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collect per-window offset observations with shared lag-search logic."""
    ref_sig = np.asarray(ref_signal, dtype=float)
    tgt_sig = np.asarray(tgt_signal, dtype=float)
    ref_ts = np.asarray(ref_timestamps_seconds, dtype=float)
    tgt_ts = np.asarray(tgt_timestamps_seconds, dtype=float)
    sr = float(sample_rate_hz)
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
        t_ref_center = float(ref_ts[center])
        if (
            start_ref_time_seconds is not None
            and t_ref_center < float(start_ref_time_seconds)
        ):
            continue

        left, right = center - half, center + half
        t_tgt_pred = float(predict_target_time_seconds(center, t_ref_center))
        pred_tgt_idx = int(np.clip(np.searchsorted(tgt_ts, t_tgt_pred), 0, tgt_ts.size - 1))
        base_shift = center - pred_tgt_idx
        best_shift, best_score = _best_window_shift_samples(
            ref_sig,
            tgt_sig,
            left=left,
            right=right,
            base_shift_samples=base_shift,
            search_radius_samples=search_n,
            min_valid_fraction=min_valid_fraction,
        )

        if best_shift is None or best_score < min_window_score:
            rejected += 1
            continue

        tgt_idx = center - best_shift
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        target_times.append(float(tgt_ts[tgt_idx]))
        offsets.append(float(ref_ts[center]) - float(tgt_ts[tgt_idx]))
        scores.append(best_score)
        if on_accept is not None:
            on_accept(target_times, offsets, scores)

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        {"accepted_windows": len(target_times), "rejected_windows": rejected},
    )


def refine_offsets_from_coarse_offset(
    ref_timestamps_seconds: np.ndarray,
    ref_signal: np.ndarray,
    tgt_timestamps_seconds: np.ndarray,
    tgt_signal: np.ndarray,
    *,
    sample_rate_hz: float,
    coarse_offset_seconds: float,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
    min_window_score: float,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collect offsets from non-causal refinement around coarse time offset."""

    def _predict_target_time_seconds(_center: int, t_ref_center: float) -> float:
        """Return predict target time seconds."""
        return float(t_ref_center) - float(coarse_offset_seconds)

    return _collect_window_offsets(
        ref_timestamps_seconds,
        ref_signal,
        tgt_timestamps_seconds,
        tgt_signal,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_valid_fraction=min_valid_fraction,
        predict_target_time_seconds=_predict_target_time_seconds,
    )


def collect_drift_observations_from_anchor(
    ref_timestamps_seconds: np.ndarray,
    ref_signal: np.ndarray,
    tgt_timestamps_seconds: np.ndarray,
    tgt_signal: np.ndarray,
    *,
    sample_rate_hz: float,
    anchor_ref_time_seconds: float,
    anchor_target_time_seconds: float,
    anchor_offset_seconds: float,
    target_origin_seconds: float,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
    min_window_score: float,
    min_valid_fraction: float,
    min_points_for_drift: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Collect causal drift-fit observations constrained by the opening anchor."""
    tgt_ts = np.asarray(tgt_timestamps_seconds, dtype=float)

    cur_drift = 0.0
    cur_offset = float(anchor_offset_seconds) - cur_drift * (
        float(anchor_target_time_seconds) - float(target_origin_seconds)
    )

    def _predict_target_time_seconds(_center: int, t_ref_center: float) -> float:
        """Return predict target time seconds."""
        return float(
            reference_to_target_seconds(
                float(t_ref_center),
                offset_seconds=float(cur_offset),
                drift_seconds_per_second=float(cur_drift),
                target_origin_seconds=float(target_origin_seconds),
            )
        )

    def _on_accept(
        target_times: list[float],
        offsets: list[float],
        scores: list[float],
    ) -> None:
        """Return on accept."""
        nonlocal cur_drift, cur_offset
        if len(target_times) < int(min_points_for_drift):
            return
        slope, _ = fit_drift_through_point(
            np.asarray(target_times, dtype=float),
            np.asarray(offsets, dtype=float),
            anchor_time_seconds=float(anchor_target_time_seconds),
            anchor_offset_seconds=float(anchor_offset_seconds),
            weights=np.asarray(scores, dtype=float),
        )
        cur_drift = slope
        cur_offset = float(anchor_offset_seconds) - cur_drift * (
            float(anchor_target_time_seconds) - float(target_origin_seconds)
        )

    return _collect_window_offsets(
        ref_timestamps_seconds,
        ref_signal,
        tgt_timestamps_seconds,
        tgt_signal,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_valid_fraction=min_valid_fraction,
        start_ref_time_seconds=float(anchor_ref_time_seconds),
        predict_target_time_seconds=_predict_target_time_seconds,
        on_accept=_on_accept,
    )


def refine_drift_unconstrained(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    scores: np.ndarray,
    *,
    min_fit_r2: float,
    target_origin_seconds: float,
    fallback_offset_seconds: float,
    min_fit_ppm: float,
    max_fit_ppm: float,
) -> tuple[float, float, float, bool]:
    """Refine unconstrained offset+drift model from windowed offsets."""
    if offsets_sec.size == 0:
        return float(fallback_offset_seconds), 0.0, 0.0, False

    weights = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    offset_seconds, drift, fit_r2 = fit_offset_drift(
        target_times_sec,
        offsets_sec,
        target_origin_seconds=float(target_origin_seconds),
        weights=weights,
    )

    if fit_r2 < min_fit_r2 or drift < min_fit_ppm/1000.0 or drift > max_fit_ppm/1000.0:
        return float(fallback_offset_seconds), 0.0, fit_r2, False
    return float(offset_seconds), float(drift), float(fit_r2), True


def refine_drift_through_anchor(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    scores: np.ndarray,
    *,
    min_fit_r2: float,
    target_origin_seconds: float,
    anchor_time_seconds: float,
    anchor_offset_seconds: float,
    min_fit_ppm: float,
    max_fit_ppm: float,
) -> tuple[float, float, float, bool]:
    """Refine drift constrained to pass through calibration anchor."""
    if offsets_sec.size == 0:
        return float(anchor_offset_seconds), 0.0, 0.0, False

    weights = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    slope, fit_r2 = fit_drift_through_point(
        target_times_sec,
        offsets_sec,
        anchor_time_seconds=float(anchor_time_seconds),
        anchor_offset_seconds=float(anchor_offset_seconds),
        weights=weights,
    )

    if fit_r2 < min_fit_r2 or slope < min_fit_ppm/1000.0 or slope > max_fit_ppm/1000.0:
        return float(anchor_offset_seconds), 0.0, fit_r2, False
    offset_seconds = float(anchor_offset_seconds) - float(slope) * (
        float(anchor_time_seconds) - float(target_origin_seconds)
    )
    return float(offset_seconds), float(slope), float(fit_r2), True
