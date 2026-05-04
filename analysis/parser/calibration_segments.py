"""Calibration segments helpers for parse raw sensor logs and split recordings into sections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
_DEDUP_SEGMENT_GAP_MS = 1500.0


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
    static_pre_ms: float = 0.0
    static_post_ms: float = 0.0

@dataclass(frozen=True)
class CalSegParams:
    """Per-sensor detection parameters (all durations in milliseconds)."""

    static_threshold: float
    peak_min_height: float
    peak_min_count: int
    peak_max_count: int
    static_gap_max_ms: float
    static_flank_gap_max_ms: float
    static_overlap_max_ms: float
    static_min_start_fraction: float
    static_min_pre_ms: float
    static_min_post_ms: float
    recording_gap_min_ms: float
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
        static_flank_gap_max_ms=float(block["static_flank_gap_max_s"]) * 1000.0,
        static_overlap_max_ms=float(block["static_overlap_max_s"]) * 1000.0,
        static_min_start_fraction=float(block["static_min_start_fraction"]),
        static_min_pre_ms=float(block["static_min_pre_s"]) * 1000.0,
        static_min_post_ms=float(block["static_min_post_s"]) * 1000.0,
        recording_gap_min_ms=float(block["recording_gap_min_s"]) * 1000.0,
        burst_max_span_ms=float(block["burst_max_span_s"]) * 1000.0,
        peak_pair_max_gap_ms=float(block["peak_pair_max_gap_s"]) * 1000.0,
        transient_baseline_ms=float(block["transient_baseline_s"]) * 1000.0,
        peak_prominence=float(block["peak_prominence"]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median_dt(ts_ms: np.ndarray) -> float:
    """Return median dt."""
    dt = np.diff(ts_ms)
    pos = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.median(pos)) if pos.size > 0 else 10.0


def _to_samples(ms: float, dt: float) -> int:
    """Convert samples."""
    return max(1, round(ms / dt))


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _find_candidate_peaks(
    ts_ms: np.ndarray,
    acc: np.ndarray,
    params: CalSegParams,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate peaks and intermediate detection signals."""
    finite = np.isfinite(acc)
    if not finite.any():
        z = np.zeros(len(acc))
        return [], z, z, np.ones(len(acc), dtype=bool)

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

    kept_peak_ms: list[float] = []
    min_dynamic_height = params.peak_min_height * 0.5
    for peak_i in peak_idx:
        peak_t = float(ts_ms[peak_i])
        peak_dynamic = float(dynamic_smooth[peak_i]) if np.isfinite(dynamic_smooth[peak_i]) else float("nan")
        if not np.isfinite(dynamic_smooth[peak_i]):
            continue
        if peak_dynamic < min_dynamic_height:
            continue
        kept_peak_ms.append(peak_t)
    return kept_peak_ms, dynamic_smooth, transient, is_static


def _cluster_peaks(peak_ms: list[float], params: CalSegParams) -> list[list[float]]:
    """Split peaks into nearby bursts."""
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


def _cluster_is_near_recording_gap(
    cluster_ms: list[float],
    ts_ms: np.ndarray,
    *,
    nearby_window_ms: float,
    min_gap_ms: float,
) -> bool:
    """Return cluster is near recording gap."""
    if len(cluster_ms) == 0 or len(ts_ms) < 2:
        return False

    cluster_start = cluster_ms[0]
    cluster_end = cluster_ms[-1]
    dts = np.diff(ts_ms)
    for left_t, right_t, gap_ms in zip(ts_ms[:-1], ts_ms[1:], dts):
        if not np.isfinite(gap_ms) or gap_ms < min_gap_ms:
            continue
        if (cluster_start - nearby_window_ms) <= float(right_t) <= (cluster_end + nearby_window_ms):
            return True
        if (cluster_start - nearby_window_ms) <= float(left_t) <= (cluster_end + nearby_window_ms):
            return True
    return False


def _filter_valid_clusters(
    clusters: list[list[float]],
    params: CalSegParams,
    ts_ms: np.ndarray,
    dynamic_smooth: np.ndarray,
) -> list[list[float]]:
    """Keep clusters with valid peak count and burst span."""
    valid: list[list[float]] = []
    for cluster in clusters:
        working = list(cluster)
        n_peaks = len(working)

        # If there are too many peaks in one burst, keep only the strongest
        # ones (by dynamic amplitude) rather than dropping the whole segment.
        # This avoids missing true calibration bursts with a few extra ring-down peaks.
        if n_peaks > params.peak_max_count:
            scored: list[tuple[float, float]] = []
            for p_ms in working:
                i = min(int(np.searchsorted(ts_ms, p_ms)), len(dynamic_smooth) - 1)
                score = float(dynamic_smooth[i]) if np.isfinite(dynamic_smooth[i]) else float("-inf")
                scored.append((p_ms, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            strongest = [p_ms for p_ms, _score in scored[: params.peak_max_count]]
            working = sorted(strongest)
            n_peaks = len(working)

        span_ms = working[-1] - working[0]
        if span_ms > params.burst_max_span_ms:
            continue

        # Relax the minimum peak count for clusters that are near a recording gap.
        min_count = params.peak_min_count
        if (_cluster_is_near_recording_gap(working, ts_ms, nearby_window_ms=params.static_gap_max_ms, min_gap_ms=params.recording_gap_min_ms)):
            min_count = max(2, params.peak_min_count - 2)

        # Reject clusters that have too few peaks.
        if n_peaks < min_count:
            continue

        # Pass clusters that are valid.
        valid.append(working)
    return valid


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
    return [period for period in periods if period[1] > period[0]]


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

    pre_gap_ms = c_start - pre_run[1]
    if (
        pre_gap_ms > params.static_flank_gap_max_ms
        and not _recording_gap_exists_between(ts_ms, pre_run[1], c_start, min_gap_ms=params.recording_gap_min_ms)
    ):
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

    post_gap_ms = post_run[0] - c_end
    if ( 
        post_gap_ms > params.static_flank_gap_max_ms
        and not _recording_gap_exists_between(ts_ms, c_end, post_run[0], min_gap_ms=params.recording_gap_min_ms)
    ):
        return None

    strength = 0.0
    for p_ms in cluster_ms:
        i = min(int(np.searchsorted(ts_ms, p_ms)), len(dynamic_smooth) - 1)
        if np.isfinite(dynamic_smooth[i]):
            strength += float(dynamic_smooth[i])

    segment = CalibrationSegment(
        start_ms=pre_run[0],
        end_ms=post_run[1],
        peak_ms=cluster_ms,
        peak_strength=strength,
        static_pre_ms=max(0.0, pre_run[1] - pre_run[0]),
        static_post_ms=max(0.0, post_run[1] - post_run[0]),
    )
    return segment


# ---------------------------------------------------------------------------
# Heuristic helers
# ---------------------------------------------------------------------------


def _cluster_is_near_recording_gap(
    cluster_ms: list[float],
    ts_ms: np.ndarray,
    *,
    nearby_window_ms: float,
    min_gap_ms: float,
) -> bool:
    """Return cluster is near recording gap."""
    if len(cluster_ms) == 0 or len(ts_ms) < 2:
        return False

    cluster_start = cluster_ms[0]
    cluster_end = cluster_ms[-1]
    dts = np.diff(ts_ms)
    for left_t, right_t, gap_ms in zip(ts_ms[:-1], ts_ms[1:], dts):
        if not np.isfinite(gap_ms) or gap_ms < min_gap_ms:
            continue
        if (cluster_start - nearby_window_ms) <= float(right_t) <= (cluster_end + nearby_window_ms):
            return True
        if (cluster_start - nearby_window_ms) <= float(left_t) <= (cluster_end + nearby_window_ms):
            return True
    return False


def _recording_gap_exists_between(
    ts_ms: np.ndarray,
    start_ms: float,
    end_ms: float,
    *,
    min_gap_ms: float,
) -> bool:
    """Return whether a large timestamp gap exists within [start_ms, end_ms]."""
    if len(ts_ms) < 2 or end_ms <= start_ms:
        return False
    dts = np.diff(ts_ms)
    for left_t, right_t, gap_ms in zip(ts_ms[:-1], ts_ms[1:], dts):
        if not np.isfinite(gap_ms) or gap_ms < min_gap_ms:
            continue
        if float(left_t) >= start_ms and float(right_t) <= end_ms:
            return True
    return False


def _deduplicate_segments(
    segments: list[CalibrationSegment],
) -> list[CalibrationSegment]:
    """Remove duplicate detections that overlap or nearly overlap; keep longest static periods."""
    if len(segments) <= 1:
        return list(segments)

    segs = sorted(segments, key=lambda s: s.start_ms)
    n = len(segs)
    parent = list(range(n))

    def find(i: int) -> int:
        """Return find."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    for i in range(n):
        for j in range(i + 1, n):
            a, b = segs[i], segs[j]
            if b.start_ms > a.end_ms + _DEDUP_SEGMENT_GAP_MS:
                break
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    buckets: dict[int, list[CalibrationSegment]] = {}
    for i, seg in enumerate(segs):
        buckets.setdefault(find(i), []).append(seg)

    kept = [max(g, key=lambda s: s.static_pre_ms + s.static_post_ms) for g in buckets.values()]
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
    """Return find calibration segments."""
    params = load_cal_seg_params(sensor, **overrides)

    norm = df["acc_norm"].to_numpy(dtype=float)
    if norm is None or len(norm) == 0:
        return []

    ts_ms = df["timestamp"].to_numpy(dtype=float)
    peak_ms, dynamic_smooth, _transient, is_static = _find_candidate_peaks(ts_ms, norm, params)
    if len(peak_ms) < params.peak_min_count:
        return []

    all_clusters = _cluster_peaks(peak_ms, params)
    clusters = _filter_valid_clusters(all_clusters, params, ts_ms, dynamic_smooth)
    if not clusters:
        return []

    static_periods = _detect_static_periods(ts_ms, is_static)

    segments = []
    for cluster in clusters:
        seg = _segment_for_cluster(
            cluster,
            static_periods,
            ts_ms,
            dynamic_smooth,
            params,
        )
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
            "static_flank_gap_max_s": params.static_flank_gap_max_ms / 1000.0,
            "static_overlap_max_s": params.static_overlap_max_ms / 1000.0,
            "static_min_start_fraction": params.static_min_start_fraction,
            "static_min_pre_s": params.static_min_pre_ms / 1000.0,
            "static_min_post_s": params.static_min_post_ms / 1000.0,
            "recording_gap_min_s": params.recording_gap_min_ms / 1000.0,
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
                "static_pre_ms": seg.static_pre_ms,
                "static_post_ms": seg.static_post_ms,
                "peak_strength": seg.peak_strength,
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
    """Load pre-detected calibration segments for *sensor* from the recording JSON."""
    path = recording_stage_dir(recording_name, "parsed") / "calibration_segments.json"
    data = read_json_file(path)
    return _parse_segments(data["sensors"][sensor].get("segments", []))


def _parse_segments(records: list[dict]) -> list[CalibrationSegment]:
    """Parse segments."""
    return [
        CalibrationSegment(
            start_ms=float(r.get("start_ms", 0.0)),
            end_ms=float(r.get("end_ms", 0.0)),
            peak_ms=[float(p) for p in r.get("peak_ms", [])],
            peak_strength=float(r.get("peak_strength", 0.0)),
            static_pre_ms=float(r.get("static_pre_ms", r.get("pre_static_ms", 0.0))),
            static_post_ms=float(r.get("static_post_ms", r.get("post_static_ms", 0.0))),
        )
        for r in records
    ]


def _segments_to_records(segments: list[CalibrationSegment]) -> list[dict]:
    """Return segments to records."""
    return [
        {
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "peak_ms": seg.peak_ms,
            "peak_strength": seg.peak_strength,
            "static_pre_ms": seg.static_pre_ms,
            "static_post_ms": seg.static_post_ms,
        }
        for seg in segments
    ]


def clip_segments_to_range(
    segs: list[CalibrationSegment],
    t_min: float,
    t_max: float,
) -> list[CalibrationSegment]:
    """Clip segment boundaries to [t_min, t_max], adjusting static-flank durations."""
    out: list[CalibrationSegment] = []
    for seg in segs:
        if seg.end_ms < t_min or seg.start_ms > t_max:
            continue
        new_start = max(seg.start_ms, t_min)
        new_end = min(seg.end_ms, t_max)
        pre_cut = new_start - seg.start_ms
        post_cut = seg.end_ms - new_end
        out.append(CalibrationSegment(
            start_ms=new_start,
            end_ms=new_end,
            peak_ms=[p for p in seg.peak_ms if new_start <= p <= new_end],
            peak_strength=seg.peak_strength,
            static_pre_ms=max(0.0, seg.static_pre_ms - pre_cut),
            static_post_ms=max(0.0, seg.static_post_ms - post_cut),
        ))
    return out


def write_section_calibration_segments(
    section_path: Path,
    segments_by_sensor: dict[str, list[CalibrationSegment]],
) -> None:
    """Write per-section calibration segments (synced timestamps) to section directory."""
    payload = {
        "section": section_path.name,
        "sensors": {sensor: _segments_to_records(segs) for sensor, segs in segments_by_sensor.items()},
    }
    path = section_path / "calibration_segments.json"
    write_json_file(path, payload, indent=2)
    log.debug("Wrote section calibration segments → %s", path)


def load_section_calibration_segments(
    section_path: Path,
    sensor: str,
) -> list[CalibrationSegment]:
    """Load per-section calibration segments (synced timestamps) for *sensor*."""
    path = section_path / "calibration_segments.json"
    data = read_json_file(path)
    return _parse_segments(data["sensors"].get(sensor, []))
