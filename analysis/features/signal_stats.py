"""Spectral and time-domain helpers for feature extraction."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import coherence

Array = np.ndarray


def safe_ratio(num: float, den: float, *, eps: float = 1e-12) -> float:
    """Return ``num / den`` when denominator is numerically stable, else NaN."""
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= eps:
        return float("nan")
    return float(num / den)


def signal_entropy(x: Array, *, n_bins: int = 32) -> float:
    """Shannon entropy of histogram of *x* (ignoring NaNs)."""
    v = x[np.isfinite(x)]
    if len(v) < 5:
        return float("nan")
    hist, _ = np.histogram(v, bins=n_bins, density=False)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def zero_crossing_rate(x: Array) -> float:
    """Fraction of sign changes on mean-centered *x*."""
    v = x[np.isfinite(x)]
    if len(v) < 3:
        return float("nan")
    c = v - np.nanmean(v)
    s = np.sign(c)
    s[s == 0] = np.nan
    changes = np.sum(np.abs(np.diff(s)) > 0)
    return float(changes / max(len(s) - 1, 1))


def dominant_frequency_hz(
    x: Array,
    fs_hz: float,
    *,
    f_min_hz: float = 0.5,
    f_max_hz: float = 25.0,
) -> float:
    """Strongest FFT bin magnitude in band (excludes DC)."""
    v = x[np.isfinite(x)]
    n = len(v)
    if n < 16 or fs_hz <= 0:
        return float("nan")
    v = v - np.nanmean(v)
    spec = np.abs(rfft(v))
    freqs = rfftfreq(n, d=1.0 / fs_hz)
    mask = (freqs >= f_min_hz) & (freqs <= f_max_hz)
    if not np.any(mask):
        return float("nan")
    i = np.argmax(spec[mask])
    idx = np.flatnonzero(mask)[i]
    return float(freqs[idx])


def band_energy_ratio(
    x: Array,
    fs_hz: float,
    *,
    low_band_hz: tuple[float, float] = (0.5, 3.0),
    high_band_hz: tuple[float, float] = (3.0, 15.0),
) -> tuple[float, float, float]:
    """Return ``(E_low, E_high, E_total)`` from periodogram-style FFT energy on *x*."""
    v = x[np.isfinite(x)]
    n = len(v)
    if n < 32 or fs_hz <= 0:
        nan = float("nan")
        return nan, nan, nan
    v = v - np.nanmean(v)
    spec = np.abs(rfft(v)) ** 2
    freqs = rfftfreq(n, d=1.0 / fs_hz)
    e_total = float(np.nansum(spec[1:]))  # skip DC
    lo = (freqs >= low_band_hz[0]) & (freqs <= low_band_hz[1])
    hi = (freqs >= high_band_hz[0]) & (freqs <= high_band_hz[1])
    e_lo = float(np.nansum(spec[lo]))
    e_hi = float(np.nansum(spec[hi]))
    return e_lo, e_hi, e_total


def crest_factor(x: Array) -> float:
    """Peak / RMS for finite *x*."""
    v = x[np.isfinite(x)]
    if len(v) < 2:
        return float("nan")
    rms = float(np.sqrt(np.nanmean(v * v)))
    peak = float(np.nanmax(np.abs(v)))
    return peak / rms if rms > 1e-12 else float("nan")


def mean_coherence_band(
    a: Array,
    b: Array,
    fs_hz: float,
    *,
    nperseg: int | None = None,
    f_min_hz: float = 0.5,
    f_max_hz: float = 15.0,
) -> float:
    """Mean magnitude-squared coherence in band (requires equal-length segments)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 32 or fs_hz <= 0:
        return float("nan")
    a = a[:n]
    b = b[:n]

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 32:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return float("nan")
    if nperseg is None:
        nperseg = max(16, min(256, len(a) // 4))
    nperseg = min(nperseg, len(a))
    if nperseg < 16:
        return float("nan")
    if nperseg // 2 >= nperseg:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            f, cxy = coherence(a, b, fs=fs_hz, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception:
        return float("nan")
    mask = (f >= f_min_hz) & (f <= f_max_hz)
    if not np.any(mask):
        return float("nan")
    band = cxy[mask]
    if band.size == 0 or not np.any(np.isfinite(band)):
        return float("nan")
    return float(np.nanmean(band))


def peak_time_difference_s(
    ts_a_s: Array,
    sig_a: Array,
    ts_b_s: Array,
    sig_b: Array,
) -> float:
    """Difference in time (seconds) between max-|signal| peaks on each timeline."""
    if len(sig_a) < 2 or len(sig_b) < 2:
        return float("nan")
    if not np.any(np.isfinite(sig_a)) or not np.any(np.isfinite(sig_b)):
        return float("nan")
    ia = int(np.nanargmax(np.abs(sig_a)))
    ib = int(np.nanargmax(np.abs(sig_b)))
    return float(ts_b_s[ib] - ts_a_s[ia])


def vec_disagreement_ms2(acc_a: Array, acc_b: Array) -> float:
    """Mean Euclidean norm of acc vector difference (aligned rows)."""
    n = min(len(acc_a), len(acc_b))
    if n < 2:
        return float("nan")
    d = acc_a[:n] - acc_b[:n]
    return float(np.nanmean(np.sqrt(np.sum(d * d, axis=1))))


def normalized_band_energy(x: Array, fs_hz: float, band_hz: tuple[float, float]) -> float:
    """Energy fraction in ``band_hz`` over non-DC total FFT energy."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 32 or fs_hz <= 0:
        return float("nan")
    v = v - np.nanmean(v)
    spec = np.abs(rfft(v)) ** 2
    freqs = rfftfreq(n, d=1.0 / fs_hz)
    e_total = float(np.nansum(spec[1:]))
    if e_total <= 1e-12:
        return float("nan")
    m = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    e_band = float(np.nansum(spec[m]))
    return float(e_band / e_total)
