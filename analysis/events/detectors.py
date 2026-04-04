"""Event detectors for dual-IMU cycling data."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from events.config import EventConfig


@dataclass
class EventCandidate:
    event_type: str       # "bump", "brake", "swerve", "disagree", "fall"
    confidence: float     # 0..1
    start_ms: float
    end_ms: float
    peak_ms: float
    peak_value: float
    sensor: str           # "sporsa", "arduino", or "cross"
    signal_name: str      # e.g. "acc_hf", "gyro_norm"
    ambiguous: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ambiguous"] = bool(d["ambiguous"])
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ms_col(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a millisecond timestamp series from the DataFrame if available."""
    for name in ("timestamp_ms", "time_ms", "ms", "t_ms"):
        if name in df.columns:
            return df[name]
    # Fall back to index if it looks like ms timestamps (large values)
    if df.index.dtype in (np.float64, np.int64):
        return pd.Series(df.index, index=df.index)
    return None


def _get_ms_array(df: pd.DataFrame) -> np.ndarray:
    """Return a numpy array of millisecond timestamps, or synthesised row indices."""
    ts = _ms_col(df)
    if ts is not None:
        return ts.to_numpy(dtype=float)
    # Synthesise: assume 100 Hz -> 10 ms per sample
    return np.arange(len(df), dtype=float) * 10.0


def _col(df: pd.DataFrame, *names: str) -> Optional[np.ndarray]:
    """Return the first matching column as a float numpy array, or None."""
    for n in names:
        if n in df.columns:
            arr = pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float)
            return arr
    return None


def _detect_threshold_events(
    signal: np.ndarray,
    times_ms: np.ndarray,
    threshold: float,
    min_duration_s: float,
    max_duration_s: float,
    event_type: str,
    sensor: str,
    signal_name: str,
    confidence_scale: float,
    min_confidence: float = 0.0,
    negate: bool = False,
) -> list[EventCandidate]:
    """
    Generic peak-based event detector using scipy find_peaks.

    Parameters
    ----------
    signal:
        1-D signal array (already absolute-valued or signed depending on *negate*).
    times_ms:
        Corresponding millisecond timestamps.
    threshold:
        Minimum peak height.
    min_duration_s / max_duration_s:
        Allowed event duration (peak width at half-prominence).
    negate:
        If True, negate the signal before peak detection (for braking – negative acc).
    confidence_scale:
        Peak value is divided by this to get confidence (clipped to [0, 1]).
    min_confidence:
        Discard candidates below this confidence level.
    """
    if len(signal) == 0:
        return []

    work = -signal if negate else signal.copy()
    # Replace NaNs with 0 so peak finder doesn't crash
    work = np.nan_to_num(work, nan=0.0)

    sample_rate_hz = 1000.0 / np.median(np.diff(times_ms)) if len(times_ms) > 1 else 100.0
    min_width_samples = max(1, int(min_duration_s * sample_rate_hz))
    max_width_samples = max(min_width_samples + 1, int(max_duration_s * sample_rate_hz))
    # distance: at least min_width_samples between peaks
    distance = max(1, min_width_samples)

    peaks, props = find_peaks(
        work,
        height=threshold,
        distance=distance,
    )

    if len(peaks) == 0:
        return []

    # Compute widths at half the peak height (relative to 0 baseline)
    widths, _, left_ips, right_ips = peak_widths(work, peaks, rel_height=0.5)

    candidates: list[EventCandidate] = []
    for idx, peak_idx in enumerate(peaks):
        w_samples = widths[idx]
        if w_samples < min_width_samples or w_samples > max_width_samples:
            continue

        peak_val = float(work[peak_idx])
        confidence = min(1.0, peak_val / confidence_scale)
        if confidence < min_confidence:
            continue

        # Convert sample indices to ms
        left_sample = max(0, int(np.floor(left_ips[idx])))
        right_sample = min(len(times_ms) - 1, int(np.ceil(right_ips[idx])))
        start_ms = float(times_ms[left_sample])
        end_ms = float(times_ms[right_sample])
        peak_ms = float(times_ms[peak_idx])

        # For negate mode the actual peak value should be negative (deceleration)
        actual_peak_val = float(signal[peak_idx])

        candidates.append(EventCandidate(
            event_type=event_type,
            confidence=round(confidence, 4),
            start_ms=start_ms,
            end_ms=end_ms,
            peak_ms=peak_ms,
            peak_value=round(actual_peak_val, 4),
            sensor=sensor,
            signal_name=signal_name,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_bumps(
    df: pd.DataFrame,
    config: EventConfig,
    sensor: str,
) -> list[EventCandidate]:
    """Detect bumps from acc_hf or acc_deviation column."""
    times_ms = _get_ms_array(df)
    candidates: list[EventCandidate] = []

    # Prefer acc_hf, fall back to acc_deviation
    signal = _col(df, "acc_hf", "acc_deviation")
    if signal is None:
        return candidates

    # Work with absolute magnitude
    signal_abs = np.abs(signal)
    confidence_scale = config.bump_acc_threshold_ms2 * 3.0

    candidates.extend(_detect_threshold_events(
        signal=signal_abs,
        times_ms=times_ms,
        threshold=config.bump_acc_threshold_ms2,
        min_duration_s=config.bump_min_duration_s,
        max_duration_s=config.bump_max_duration_s,
        event_type="bump",
        sensor=sensor,
        signal_name="acc_hf" if "acc_hf" in df.columns else "acc_deviation",
        confidence_scale=confidence_scale,
        min_confidence=config.bump_min_confidence,
    ))

    return candidates


def detect_braking(
    df: pd.DataFrame,
    config: EventConfig,
    sensor: str,
) -> list[EventCandidate]:
    """Detect braking events from negative longitudinal acceleration."""
    times_ms = _get_ms_array(df)
    candidates: list[EventCandidate] = []

    # Try longitudinal / forward-axis signals; fall back to acc_lf
    signal = _col(df, "acc_longitudinal", "acc_forward", "acc_lf", "acc_vertical")
    if signal is None:
        return candidates

    # Braking = large negative value; negate=True so we find peaks in -signal
    confidence_scale = config.brake_acc_threshold_ms2 * 3.0

    candidates.extend(_detect_threshold_events(
        signal=signal,
        times_ms=times_ms,
        threshold=config.brake_acc_threshold_ms2,
        min_duration_s=config.brake_min_duration_s,
        max_duration_s=config.brake_max_duration_s,
        event_type="brake",
        sensor=sensor,
        signal_name="acc_lf" if "acc_lf" in df.columns else "acc_longitudinal",
        confidence_scale=confidence_scale,
        negate=True,
    ))

    return candidates


def detect_swerves(
    df: pd.DataFrame,
    config: EventConfig,
    sensor: str,
) -> list[EventCandidate]:
    """Detect swerves/corners from gyro_norm or gyro_hf spikes."""
    times_ms = _get_ms_array(df)
    candidates: list[EventCandidate] = []

    signal = _col(df, "gyro_norm", "gyro_hf")
    if signal is None:
        # Try to compute gyro norm from raw gyro components
        gx = _col(df, "gx")
        gy = _col(df, "gy")
        gz = _col(df, "gz")
        if gx is not None and gy is not None and gz is not None:
            signal = np.sqrt(gx**2 + gy**2 + gz**2)
        else:
            return candidates

    signal_abs = np.abs(signal)
    confidence_scale = config.swerve_gyro_threshold_dps * 3.0
    signal_name = "gyro_norm" if "gyro_norm" in df.columns else (
        "gyro_hf" if "gyro_hf" in df.columns else "gyro_computed_norm"
    )

    candidates.extend(_detect_threshold_events(
        signal=signal_abs,
        times_ms=times_ms,
        threshold=config.swerve_gyro_threshold_dps,
        min_duration_s=config.swerve_min_duration_s,
        max_duration_s=config.swerve_max_duration_s,
        event_type="swerve",
        sensor=sensor,
        signal_name=signal_name,
        confidence_scale=confidence_scale,
    ))

    return candidates


def detect_disagreement(
    cross_df: pd.DataFrame,
    config: EventConfig,
) -> list[EventCandidate]:
    """Detect sensor disagreement episodes from disagree_score column."""
    candidates: list[EventCandidate] = []

    signal = _col(cross_df, "disagree_score", "disagreement_score", "disagreement")
    if signal is None:
        return candidates

    times_ms = _get_ms_array(cross_df)
    confidence_scale = config.disagree_threshold * 3.0

    candidates.extend(_detect_threshold_events(
        signal=signal,
        times_ms=times_ms,
        threshold=config.disagree_threshold,
        min_duration_s=config.disagree_min_duration_s,
        max_duration_s=60.0,  # no hard upper bound for disagreement
        event_type="disagree",
        sensor="cross",
        signal_name="disagree_score",
        confidence_scale=confidence_scale,
    ))

    return candidates


def detect_falls(
    df: pd.DataFrame,
    cross_df: pd.DataFrame,
    config: EventConfig,
) -> list[EventCandidate]:
    """Detect falls: very large acc spike + large gyro on both sensors.

    Strategy:
    - Find time windows where acc magnitude exceeds fall_acc_threshold_ms2.
    - Check that a large gyro spike (> fall_gyro_threshold_dps) is also present
      within a short window of those acc spikes.
    - Use cross_df to verify the event appears in both sensors when possible.
    """
    candidates: list[EventCandidate] = []
    times_ms = _get_ms_array(df)

    # Acc signal – prefer acc_norm or compute from components
    acc = _col(df, "acc_norm", "acc_magnitude")
    if acc is None:
        ax = _col(df, "ax")
        ay = _col(df, "ay")
        az = _col(df, "az")
        if ax is not None and ay is not None and az is not None:
            acc = np.sqrt(ax**2 + ay**2 + az**2)
        else:
            acc = _col(df, "acc_hf", "acc_deviation")
            if acc is not None:
                acc = np.abs(acc)

    if acc is None:
        return candidates

    # Gyro signal
    gyro = _col(df, "gyro_norm", "gyro_hf")
    if gyro is None:
        gx = _col(df, "gx")
        gy = _col(df, "gy")
        gz = _col(df, "gz")
        if gx is not None and gy is not None and gz is not None:
            gyro = np.sqrt(gx**2 + gy**2 + gz**2)

    acc_abs = np.abs(np.nan_to_num(acc, nan=0.0))

    # Find large acc peaks
    sample_rate_hz = 1000.0 / np.median(np.diff(times_ms)) if len(times_ms) > 1 else 100.0
    min_dist = max(1, int(0.1 * sample_rate_hz))

    peaks, _ = find_peaks(acc_abs, height=config.fall_acc_threshold_ms2, distance=min_dist)

    for peak_idx in peaks:
        # Check gyro at same region (±0.5 s window)
        window_half = int(0.5 * sample_rate_hz)
        lo = max(0, peak_idx - window_half)
        hi = min(len(acc_abs) - 1, peak_idx + window_half)

        gyro_ok = False
        if gyro is not None:
            gyro_abs = np.abs(np.nan_to_num(gyro, nan=0.0))
            if gyro_abs[lo:hi + 1].max() >= config.fall_gyro_threshold_dps:
                gyro_ok = True

        # For a fall we require both signals; if gyro not available, mark ambiguous
        ambiguous = not gyro_ok

        peak_val = float(acc_abs[peak_idx])
        confidence = min(1.0, peak_val / (config.fall_acc_threshold_ms2 * 2.0))

        # Duration: use a fixed 0.3 s window around the peak
        half_dur_ms = 150.0
        start_ms = max(float(times_ms[0]), float(times_ms[peak_idx]) - half_dur_ms)
        end_ms = min(float(times_ms[-1]), float(times_ms[peak_idx]) + half_dur_ms)

        notes = "" if gyro_ok else "gyro below threshold – check manually"

        candidates.append(EventCandidate(
            event_type="fall",
            confidence=round(confidence, 4),
            start_ms=start_ms,
            end_ms=end_ms,
            peak_ms=float(times_ms[peak_idx]),
            peak_value=round(float(acc[peak_idx]), 4),
            sensor="unknown",  # caller (detect_events) sets the correct sensor name
            signal_name="acc_norm",
            ambiguous=ambiguous,
            notes=notes,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_events(
    events: list[EventCandidate],
    gap_s: float,
) -> list[EventCandidate]:
    """Merge overlapping or nearby events of the same type (same event_type + sensor).

    Two events are merged when the gap between them is <= gap_s seconds.
    The merged event keeps the higher-confidence candidate's metadata and
    takes the union of the time window.
    """
    if not events:
        return []

    gap_ms = gap_s * 1000.0

    # Group by (event_type, sensor)
    from collections import defaultdict
    groups: dict[tuple[str, str], list[EventCandidate]] = defaultdict(list)
    for ev in events:
        groups[(ev.event_type, ev.sensor)].append(ev)

    merged: list[EventCandidate] = []
    for _key, group in groups.items():
        # Sort by start time
        group.sort(key=lambda e: e.start_ms)
        current = group[0]
        for nxt in group[1:]:
            gap = nxt.start_ms - current.end_ms
            if gap <= gap_ms:
                # Merge: extend window, keep best confidence
                if nxt.confidence > current.confidence:
                    best = nxt
                else:
                    best = current
                current = EventCandidate(
                    event_type=best.event_type,
                    confidence=max(current.confidence, nxt.confidence),
                    start_ms=min(current.start_ms, nxt.start_ms),
                    end_ms=max(current.end_ms, nxt.end_ms),
                    peak_ms=best.peak_ms,
                    peak_value=best.peak_value,
                    sensor=best.sensor,
                    signal_name=best.signal_name,
                    ambiguous=current.ambiguous or nxt.ambiguous,
                    notes="; ".join(filter(None, [current.notes, nxt.notes])),
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# Top-level detector
# ---------------------------------------------------------------------------

def detect_events(
    sporsa_df: pd.DataFrame,
    arduino_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    config: EventConfig,
) -> list[EventCandidate]:
    """Run all detectors, merge, and return sorted list of EventCandidates."""
    all_events: list[EventCandidate] = []

    # --- Per-sensor detectors ---
    for df, sensor_name in ((sporsa_df, "sporsa"), (arduino_df, "arduino")):
        if df is None or len(df) == 0:
            continue
        all_events.extend(detect_bumps(df, config, sensor_name))
        all_events.extend(detect_braking(df, config, sensor_name))
        all_events.extend(detect_swerves(df, config, sensor_name))
        fall_events = detect_falls(df, cross_df, config)
        for ev in fall_events:
            ev.sensor = sensor_name
        all_events.extend(fall_events)

    # --- Cross-sensor detectors ---
    if cross_df is not None and len(cross_df) > 0:
        all_events.extend(detect_disagreement(cross_df, config))

    # --- Merge nearby events of the same type+sensor ---
    all_events = merge_events(all_events, config.merge_gap_s)

    # Sort by start time
    all_events.sort(key=lambda e: e.start_ms)

    return all_events
