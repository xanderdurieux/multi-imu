"""Temporal shape features for IMU signal windows.

Flat aggregates (mean, std, etc.) discard the ordering of samples within a
window.  These features encode *how* a signal evolves over time, which is
essential for events whose signature is front- or back-loaded:

* Hard braking  – large impulse at the start that decays toward the end.
* Cornering     – sustained angular velocity with consistent sign throughout.
* Speed bump    – sharp transient near the middle, then quiet.

Apply to signals that carry temporal shape information: acc_norm, jerk_norm,
gyro_norm (always ≥ 0) and yaw_rate (signed – sign_consistency is key here).
"""

from __future__ import annotations

import numpy as np

_NaN = float("nan")

_TEMPORAL_KEYS = (
    "early_mean",
    "late_mean",
    "decay_ratio",
    "slope",
    "peak_position",
    "energy_centroid",
    "autocorr_lag1",
    "sign_consistency",
    "entropy",
)


def temporal_features(prefix: str, arr: np.ndarray) -> dict[str, float]:
    """Return temporal shape features for a 1-D signal window.

    Parameters
    ----------
    prefix:
        Column prefix, e.g. ``"bike_acc_norm"``.
    arr:
        1-D signal array (may contain NaN; only finite values are used).

    Returns
    -------
    dict with keys ``{prefix}_temporal_{key}`` for each key in
    :data:`_TEMPORAL_KEYS`.  All values are NaN when fewer than 6 finite
    samples are available.
    """
    out: dict[str, float] = {f"{prefix}_temporal_{k}": _NaN for k in _TEMPORAL_KEYS}
    x = arr[np.isfinite(arr)]
    n = len(x)
    if n < 6:
        return out

    # --- Sub-window thirds: front/back loading --------------------------
    # decay_ratio > 1  →  signal is stronger at the start  (hard braking)
    # decay_ratio < 1  →  signal grows toward the end      (corner entry)
    thirds = np.array_split(x, 3)
    early = float(np.mean(thirds[0])) if len(thirds[0]) else _NaN
    late = float(np.mean(thirds[2])) if len(thirds[2]) else _NaN
    out[f"{prefix}_temporal_early_mean"] = early
    out[f"{prefix}_temporal_late_mean"] = late
    if np.isfinite(early) and np.isfinite(late):
        out[f"{prefix}_temporal_decay_ratio"] = early / (abs(late) + 1e-9)

    # --- Linear trend slope (direction and magnitude of change) ---------
    t = np.arange(n, dtype=float)
    out[f"{prefix}_temporal_slope"] = float(np.polyfit(t, x, 1)[0])

    # --- Peak timing: where within the window does the maximum occur ----
    # 0.0 = very start, 1.0 = very end.
    # Hard braking peak_position ≈ 0.0–0.2; corner apex ≈ 0.4–0.6.
    out[f"{prefix}_temporal_peak_position"] = float(np.argmax(x)) / max(n - 1, 1)

    # --- Energy-weighted centroid (normalised to [0, 1]) ----------------
    total = float(np.sum(x))
    if abs(total) > 1e-12:
        out[f"{prefix}_temporal_energy_centroid"] = (
            float(np.dot(x, t)) / (total * max(n - 1, 1))
        )
    else:
        out[f"{prefix}_temporal_energy_centroid"] = 0.5

    # --- Autocorrelation at lag-1: temporal persistence -----------------
    # High (near 1)  →  smooth, sustained signal  (cornering)
    # Low (near 0)   →  noisy or abrupt           (impulse events)
    s = x - x.mean()
    var = float(np.var(s))
    if var > 1e-12:
        ac = float(np.correlate(s[1:], s[:-1])[0]) / (var * n)
        out[f"{prefix}_temporal_autocorr_lag1"] = float(np.clip(ac, -1.0, 1.0))

    # --- Sign consistency: sustained direction vs oscillation -----------
    # Useful for yaw_rate and signed gyro axes:
    # High (~1.0)  →  turn is in one direction throughout (real corner)
    # Low (~0.5)   →  signal flips sign repeatedly       (wobble/noise)
    mean_val = float(x.mean())
    if abs(mean_val) > 1e-12:
        out[f"{prefix}_temporal_sign_consistency"] = float(
            np.mean(np.sign(x) == np.sign(mean_val))
        )
    else:
        out[f"{prefix}_temporal_sign_consistency"] = 0.5

    # --- Shannon entropy of 10-bin histogram (signal complexity) --------
    # Low entropy  →  concentrated signal (clean event)
    # High entropy →  spread across many amplitude levels (complex/noisy)
    hist, _ = np.histogram(x, bins=10)
    h = hist.astype(float) + 1e-10
    h /= h.sum()
    out[f"{prefix}_temporal_entropy"] = float(-np.sum(h * np.log(h)))

    return out
