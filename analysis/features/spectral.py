"""Spectral (rFFT) features over a single signal window."""

from __future__ import annotations

from typing import Any

import numpy as np

_EPSILON = 1e-9


def spectral_features(
    prefix: str,
    arr: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Return spectral features."""
    feats: dict[str, Any] = {
        f"{prefix}_spectral_centroid": float("nan"),
        f"{prefix}_spectral_energy_low": float("nan"),
        f"{prefix}_spectral_energy_mid": float("nan"),
        f"{prefix}_spectral_energy_high": float("nan"),
        f"{prefix}_dominant_freq": float("nan"),
    }
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n < 4:
        return feats

    window = np.hanning(n)
    windowed = clean * window
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    total_power = power.sum()
    if total_power < _EPSILON:
        return feats

    feats[f"{prefix}_spectral_centroid"] = float(np.sum(freqs * power) / total_power)

    low_mask = freqs < 2.0
    mid_mask = (freqs >= 2.0) & (freqs < 8.0)
    high_mask = freqs >= 8.0

    feats[f"{prefix}_spectral_energy_low"] = float(power[low_mask].sum())
    feats[f"{prefix}_spectral_energy_mid"] = float(power[mid_mask].sum())
    feats[f"{prefix}_spectral_energy_high"] = float(power[high_mask].sum())
    feats[f"{prefix}_dominant_freq"] = float(freqs[int(np.argmax(power))])

    return feats
