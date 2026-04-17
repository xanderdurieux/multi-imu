"""Calibration-sequence detection: ~5 s static → 4–6 sharp taps → ~5 s static.

All timestamps in milliseconds throughout. No sample-index arithmetic is
exposed in the public API; parameter windows are computed from the median
inter-sample interval estimated from the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pickletools import float8
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common.paths import (
    load_calibration_segments_config_data,
    read_json_file,
    recording_stage_dir,
    write_json_file,
)
from common.signals import smooth_moving_average

log = logging.getLogger(__name__)

_G = 9.81
_SMOOTH_MS = 100.0
_MIN_PEAK_DISTANCE_MS = 150.0
_PROMINENCE_WLEN_MS = 1200.0


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSegment:
    """One detected calibration sequence; all times in milliseconds."""

    start_ms: float
    end_ms: float
    peak_ms: list[float]
    peak_strength: float = 0.0

    # Downstream code uses these names; keep as aliases.
    @property
    def start_timestamp(self) -> float:
        return self.start_ms

    @property
    def end_timestamp(self) -> float:
        return self.end_ms

    @property
    def peak_timestamps(self) -> list[float]:
        return self.peak_ms


@dataclass(frozen=True)
class CalSegParams:
    """Per-sensor detection parameters (all durations in milliseconds)."""

    static_threshold: float
    peak_min_height: float
    peak_min_count: int
    peak_max_count: int
    static_gap_max_ms: float
    static_overlap_max_ms: float
    static_min_start_fraction: float
    static_min_pre_ms: float
    static_min_post_ms: float
    burst_max_span_ms: float
    peak_pair_max_gap_ms: float
    transient_baseline_ms: float
    peak_prominence: float


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------


def load_cal_seg_params(sensor: str, **overrides: Any) -> CalSegParams:
    """Load detection parameters for *sensor* from the config file."""
    config = load_calibration_segments_config_data()
    block = config.get(sensor.strip().lower())
    if not isinstance(block, dict):
        raise KeyError(f"No calibration-segment params for sensor {sensor!r}")
    if overrides:
        block = {**block, **overrides}
    return CalSegParams(
        static_threshold=float(block["static_threshold"]),
        peak_min_height=float(block["peak_min_height"]),
        peak_min_count=int(block["peak_min_count"]),
        peak_max_count=int(block["peak_max_count"]),
        static_gap_max_ms=float(block["static_gap_max_s"]) * 1000.0,
        static_overlap_max_ms=float(block["static_overlap_max_s"]) * 1000.0,
        static_min_start_fraction=float(block["static_min_start_fraction"]),
        static_min_pre_ms=float(block["static_min_pre_s"]) * 1000.0,
        static_min_post_ms=float(block["static_min_post_s"]) * 1000.0,
        burst_max_span_ms=float(block["burst_max_span_s"]) * 1000.0,
        peak_pair_max_gap_ms=float(block["peak_pair_max_gap_s"]) * 1000.0,
        transient_baseline_ms=float(block["transient_baseline_s"]) * 1000.0,
        peak_prominence=float(block["peak_prominence"]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_dt(ts_ms: np.ndarray) -> float:
    dt = np.diff(ts_ms)
    pos = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.median(pos)) if pos.size > 0 else 10.0


def _to_samples(ms: float, dt: float) -> int:
    return max(1, round(ms / dt))


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _find_candidate_peaks(
    ts_ms: np.ndarray,
    acc: np.ndarray,
    params: CalSegParams,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return (peak_ms, dynamic_smooth, is_static) for the given signal."""
    finite = np.isfinite(acc)
    if not finite.any():
        z = np.zeros(len(acc))
        return [], z, np.ones(len(acc), dtype=bool)

    x = acc.copy()
    if not finite.all():
        idx = np.arange(len(x), dtype=float)
        x = np.interp(idx, idx[finite], x[finite])

    g = float(np.nanmedian(x))
    if not 8.0 <= g <= 12.5:
        g = _G

    dt = _median_dt(ts_ms)
    is_static = np.abs(x - g) < params.static_threshold

    smooth_win = _to_samples(_SMOOTH_MS, dt)
    dynamic_smooth = np.abs(smooth_moving_average(x, smooth_win) - g)

    baseline_win = _to_samples(params.transient_baseline_ms, dt)
    transient = np.maximum(
        dynamic_smooth - smooth_moving_average(dynamic_smooth, baseline_win), 0.0
    )

    peak_idx, _ = find_peaks(
        transient,
        distance=_to_samples(_MIN_PEAK_DISTANCE_MS, dt),
        prominence=max(params.peak_prominence, 1e-9),
        wlen=max(7, _to_samples(_PROMINENCE_WLEN_MS, dt)),
    )

    keep = np.isfinite(dynamic_smooth[peak_idx]) & (
        dynamic_smooth[peak_idx] >= params.peak_min_height * 0.5
    )
    return [float(ts_ms[i]) for i in peak_idx[keep]], dynamic_smooth, is_static


def _cluster_peaks(peak_ms: list[float], params: CalSegParams) -> list[list[float]]:
    """Split peaks into bursts where every consecutive gap ≤ peak_pair_max_gap_ms."""
    if not peak_ms:
        return []
    clusters: list[list[float]] = []
    cur = [peak_ms[0]]
    for p in peak_ms[1:]:
        if p - cur[-1] <= params.peak_pair_max_gap_ms:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    return clusters


def _filter_valid_clusters(
    clusters: list[list[float]], params: CalSegParams
) -> list[list[float]]:
    """Keep clusters with valid peak count and burst span."""
    return [
        c
        for c in clusters
        if params.peak_min_count <= len(c) <= params.peak_max_count
        and c[-1] - c[0] <= params.burst_max_span_ms
    ]


def _detect_static_periods(
    ts_ms: np.ndarray, is_static: np.ndarray
) -> list[tuple[float, float]]:
    """Return (start_ms, end_ms) of contiguous static windows."""
    periods: list[tuple[float, float]] = []
    run_start: float | None = None
    prev_t = float(ts_ms[0])
    for t_raw, flag in zip(ts_ms, is_static):
        t = float(t_raw)
        if flag and run_start is None:
            run_start = t
        elif not flag and run_start is not None:
            periods.append((run_start, prev_t))
            run_start = None
        prev_t = t
    if run_start is not None:
        periods.append((run_start, float(ts_ms[-1])))
    return periods


def _segment_for_cluster(
    cluster_ms: list[float],
    static_periods: list[tuple[float, float]],
    ts_ms: np.ndarray,
    dynamic_smooth: np.ndarray,
    params: CalSegParams,
) -> CalibrationSegment | None:
    """Find static flanks around *cluster_ms* and return a segment, or None."""
    c_start, c_end = cluster_ms[0], cluster_ms[-1]
    first_t = float(ts_ms[0])

    pre_run: tuple[float, float] | None = None
    for s_start, s_end in reversed(static_periods):
        if s_end > c_start + params.static_overlap_max_ms:
            continue
        if c_start - s_end > params.static_gap_max_ms:
            break
        min_pre = params.static_min_pre_ms
        if s_start <= first_t:
            min_pre *= params.static_min_start_fraction
        if s_end - s_start >= min_pre:
            pre_run = (s_start, s_end)
            break

    if pre_run is None:
        return None

    post_run: tuple[float, float] | None = None
    for s_start, s_end in static_periods:
        if s_end < c_end:
            continue
        if s_start - c_end > params.static_gap_max_ms:
            break
        if c_end - s_start > params.static_overlap_max_ms:
            continue
        if s_end - s_start >= params.static_min_post_ms:
            post_run = (s_start, s_end)
            break

    if post_run is None:
        return None

    strength = 0.0
    for p_ms in cluster_ms:
        i = min(int(np.searchsorted(ts_ms, p_ms)), len(dynamic_smooth) - 1)
        if np.isfinite(dynamic_smooth[i]):
            strength += float(dynamic_smooth[i])

    return CalibrationSegment(
        start_ms=pre_run[0],
        end_ms=post_run[1],
        peak_ms=cluster_ms,
        peak_strength=strength,
    )


def _deduplicate_segments(
    segments: list[CalibrationSegment],
) -> list[CalibrationSegment]:
    """Remove duplicate detections sharing the same tap burst; keep strongest."""
    if len(segments) <= 1:
        return list(segments)

    segs = sorted(segments, key=lambda s: s.peak_ms[0] if s.peak_ms else s.start_ms)
    n = len(segs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = segs[i], segs[j]
            if not (a.peak_ms and b.peak_ms):
                continue
            if b.peak_ms[0] > a.peak_ms[-1]:
                break
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    buckets: dict[int, list[CalibrationSegment]] = {}
    for i, seg in enumerate(segs):
        buckets.setdefault(find(i), []).append(seg)

    kept = [max(g, key=lambda s: s.peak_strength) for g in buckets.values()]
    return sorted(kept, key=lambda s: s.start_ms)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_calibration_segments(
    df: pd.DataFrame,
    *,
    sensor: str,
    **overrides: Any,
) -> list[CalibrationSegment]:
    """Detect calibration sequences in *df* for *sensor*.

    Returns segments sorted chronologically.
    """
    params = load_cal_seg_params(sensor, **overrides)

    norm = df["acc_norm"].to_numpy(dtype=float8)
    if norm is None or len(norm) == 0:
        return []

    ts_ms = df["timestamp"].to_numpy(dtype=float)

    peak_ms, dynamic_smooth, is_static = _find_candidate_peaks(ts_ms, norm, params)
    if len(peak_ms) < params.peak_min_count:
        return []

    clusters = _filter_valid_clusters(_cluster_peaks(peak_ms, params), params)
    if not clusters:
        return []

    static_periods = _detect_static_periods(ts_ms, is_static)

    segments = []
    for cluster in clusters:
        seg = _segment_for_cluster(cluster, static_periods, ts_ms, dynamic_smooth, params)
        if seg is not None:
            segments.append(seg)

    return _deduplicate_segments(segments)


def export_calibration_segments_json(
    recording_name: str,
    sensor: str,
    segments: list[CalibrationSegment],
    params: CalSegParams,
    *,
    existing: dict[str, Any] | None = None,
) -> None:
    """Write or merge the *sensor* block into calibration_segments.json."""
    path = recording_stage_dir(recording_name, "parsed") / "calibration_segments.json"
    payload: dict[str, Any] = existing or {
        "recording_name": recording_name,
        "sensors": {},
    }
    total_s = sum((s.end_ms - s.start_ms) / 1000.0 for s in segments)
    payload["sensors"][sensor] = {
        "detection_params": {
            "static_threshold": params.static_threshold,
            "peak_min_height": params.peak_min_height,
            "peak_min_count": params.peak_min_count,
            "peak_max_count": params.peak_max_count,
            "static_gap_max_s": params.static_gap_max_ms / 1000.0,
            "static_overlap_max_s": params.static_overlap_max_ms / 1000.0,
            "static_min_start_fraction": params.static_min_start_fraction,
            "static_min_pre_s": params.static_min_pre_ms / 1000.0,
            "static_min_post_s": params.static_min_post_ms / 1000.0,
            "burst_max_span_s": params.burst_max_span_ms / 1000.0,
            "peak_pair_max_gap_s": params.peak_pair_max_gap_ms / 1000.0,
            "transient_baseline_s": params.transient_baseline_ms / 1000.0,
            "peak_prominence": params.peak_prominence,
        },
        "num_segments": len(segments),
        "total_duration_s": round(total_s, 3),
        "segments": [
            {
                "segment_index": i,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "n_peaks": len(seg.peak_ms),
                "peak_ms": seg.peak_ms,
            }
            for i, seg in enumerate(segments)
        ],
    }
    write_json_file(path, payload, indent=2, sort_keys=True)
    log.info(
        "Wrote calibration segments for %s/%s → %s (%d segment(s))",
        recording_name, sensor, path, len(segments),
    )


def load_calibration_segments_from_json(
    recording_name: str,
    sensor: str,
) -> list[CalibrationSegment]:
    """Load pre-detected calibration segments for *sensor* from the recording JSON.

    Supports both the current format (``start_ms``/``end_ms``/``peak_ms``) and
    the legacy format (``start_timestamp``/``end_timestamp``/``peak_timestamp``).
    """
    path = recording_stage_dir(recording_name, "parsed") / "calibration_segments.json"
    data = read_json_file(path)
    sensor_data = data["sensors"][sensor]
    segments: list[CalibrationSegment] = []
    for rec in sensor_data.get("segments", []):
        start = float(rec.get("start_ms", rec.get("start_timestamp", 0.0)))
        end = float(rec.get("end_ms", rec.get("end_timestamp", 0.0)))
        peaks_raw = rec.get("peak_ms", rec.get("peak_timestamp", []))
        segments.append(
            CalibrationSegment(
                start_ms=start,
                end_ms=end,
                peak_ms=[float(p) for p in peaks_raw],
            )
        )
    return segments
