"""Calibration-sequence detection shared across the analysis pipeline.

This module implements detection of calibration sequences in a single IMU
recording. A calibration sequence has the pattern::

    ~5 s static  →  4–6 short, sharp acceleration peaks (a tight burst)
    →  ~5 s static

Detection is **peak-first**: candidate taps are found on a transient emphasis
of the smoothed |acc−g| envelope, split into bursts where consecutive peaks
are separated by at most *peak_pair_max_gap_s*, and each burst must fit
within *burst_max_span_s*.  Pre- and post-static requirements are **asymmetric**
(*static_min_pre_s* / *static_min_post_s*) so the Arduino stream can admit a
shorter “static” lead-in while the helmet is handled, while Sporsa can demand
longer true stillness.

The detection logic is used by:

- :mod:`parser.split_sections` to form calibration-bounded recording sections.
- :mod:`sync.methods` to locate calibration anchors for
  calibration-based time synchronisation.
- :mod:`visualization.plot_calibration_segments` for diagnostic figures.

Centralising this logic keeps the definition of a "calibration sequence"
consistent across all stages of the thesis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common.paths import cal_segments_config_path, read_json_file
from common.signals import smooth_moving_average, vector_norm

_NOMINAL_GRAVITY_MS2 = 9.81

# Keys required in each sensor block of the cal-segments config JSON.
_CAL_SEGMENT_PARAM_KEYS: frozenset[str] = frozenset({
    "sample_rate_hz",
    "static_threshold",
    "peak_min_height",
    "peak_min_count",
    "peak_max_count",
    "static_gap_max_s",
    "static_overlap_max_s",
    "static_min_start_fraction",
    "static_min_pre_s",
    "static_min_post_s",
    "burst_max_span_s",
    "peak_pair_max_gap_s",
    "transient_baseline_s",
    "peak_prominence",
})


@dataclass(frozen=True)
class CalibrationSegment:
    """Indices (inclusive) of one calibration sequence in the DataFrame."""

    start_idx: int
    end_idx: int
    peak_indices: list[int]
    #: Sum of smoothed |acc−g| at *peak_indices*; used to break overlapping duplicates.
    peak_strength: float = 0.0


@dataclass(frozen=True)
class _DetectionArrays:
    """Intermediate signals shared by detection and diagnostic plots."""

    norm: np.ndarray
    g: float
    dynamic_raw: np.ndarray
    is_static: np.ndarray
    dynamic_smooth: np.ndarray


def _acc_norm_from_df(df: pd.DataFrame) -> np.ndarray | None:
    if "acc_norm" in df.columns:
        return df["acc_norm"].to_numpy(dtype=float)
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols:
        return None
    return vector_norm(df, acc_cols)


def _prepare_detection_arrays(
    df: pd.DataFrame,
    *,
    static_threshold: float,
    sample_rate_hz: float,
) -> _DetectionArrays | None:
    """Match :func:`_find_calibration_segments_impl` norm → smoothed |acc−g| pipeline."""
    norm = _acc_norm_from_df(df)
    if norm is None or len(norm) == 0:
        return None

    if not np.all(np.isfinite(norm)):
        good = np.isfinite(norm)
        if not np.any(good):
            return None
        idx = np.arange(len(norm), dtype=float)
        norm = np.interp(idx, idx[good], norm[good])

    g_med = float(np.nanmedian(norm))
    if 8.0 <= g_med <= 12.5:
        g = g_med
    else:
        g = _NOMINAL_GRAVITY_MS2

    dynamic_raw = np.abs(norm - g)
    is_dropout = norm < 0.1 * g
    is_static = (dynamic_raw < static_threshold) | is_dropout

    norm_for_smooth = norm.copy()
    norm_for_smooth[is_dropout] = g

    smooth_win = max(3, int(sample_rate_hz * 0.1))
    dynamic_smooth = np.abs(smooth_moving_average(norm_for_smooth, smooth_win) - g)

    return _DetectionArrays(
        norm=norm,
        g=float(g),
        dynamic_raw=dynamic_raw,
        is_static=is_static,
        dynamic_smooth=dynamic_smooth,
    )


def prepare_calibration_detection_arrays(
    df: pd.DataFrame,
    *,
    sensor: str,
    **overrides: Any,
) -> _DetectionArrays | None:
    """Same preprocessing as :func:`find_calibration_segments` (norm, static mask, smoothed |acc−g|)."""
    params = cal_segment_kwargs_for_sensor(sensor)
    params.update(overrides)
    return _prepare_detection_arrays(
        df,
        static_threshold=float(params["static_threshold"]),
        sample_rate_hz=float(params["sample_rate_hz"]),
    )


def peak_detection_dynamic_smooth(
    df: pd.DataFrame,
    *,
    sensor: str,
    **overrides: Any,
) -> np.ndarray | None:
    """Smoothed |acc−g| used for tap peak picking (same as :func:`find_calibration_segments`)."""
    prep = prepare_calibration_detection_arrays(df, sensor=sensor, **overrides)
    return None if prep is None else prep.dynamic_smooth


def describe_calibration_segments(
    df: pd.DataFrame,
    segments: list[CalibrationSegment],
    *,
    sample_rate_hz: float = 100.0,
) -> pd.DataFrame:
    """Return a summary table with timing information for each calibration segment.

    The returned DataFrame contains one row per segment with:

    - ``segment_index``: zero-based segment counter.
    - ``start_idx`` / ``end_idx``: inclusive sample indices into *df*.
    - ``start_time_s`` / ``end_time_s``: times in seconds relative to the first
      sample in *df* (or relative to index 0 when no timestamps are present).
    - ``n_peaks``: number of detected peaks in the segment.
    - ``peak_indices``: list of peak sample indices.
    - ``peak_times_s``: list of peak times in seconds (same reference as the
      start/end times).
    """
    if not segments:
        return pd.DataFrame(
            columns=[
                "segment_index",
                "start_idx",
                "end_idx",
                "start_time_s",
                "end_time_s",
                "n_peaks",
                "peak_indices",
                "peak_times_s",
            ]
        )

    n_samples = len(df)
    if n_samples == 0:
        raise ValueError("Input DataFrame is empty; cannot describe calibration segments.")

    if "timestamp" in df.columns and n_samples > 0:
        ts = df["timestamp"].astype(float).to_numpy()
        t0 = ts[0]
        time_s = (ts - t0) / 1000.0
    else:
        time_s = np.arange(n_samples, dtype=float) / float(sample_rate_hz)

    records: list[dict] = []
    for i, seg in enumerate(segments):
        start_idx = int(seg.start_idx)
        end_idx = int(seg.end_idx)
        peak_indices = [int(p) for p in seg.peak_indices]

        # Guard against stale indices that fall outside the DataFrame.
        if start_idx < 0 or end_idx >= n_samples:
            continue

        peak_indices_clipped = [p for p in peak_indices if 0 <= p < n_samples]
        peak_times_s = [float(time_s[p]) for p in peak_indices_clipped]

        records.append(
            {
                "segment_index": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time_s": float(time_s[start_idx]),
                "end_time_s": float(time_s[end_idx]),
                "n_peaks": len(peak_indices_clipped),
                "peak_indices": peak_indices_clipped,
                "peak_times_s": peak_times_s,
            }
        )

    return pd.DataFrame.from_records(records)


def _refine_peaks_to_raw_max(
    peak_indices: list[int] | np.ndarray,
    signal: np.ndarray,
    *,
    search_radius: int,
    min_spacing: int,
) -> list[int]:
    """Snap candidate indices onto local maxima of *signal* (default: |acc| magnitude).

    Peak picking runs on a smoothed |acc−g|-derived envelope; refinement uses
    the **raw acceleration magnitude** so stored indices match the crests of the
    |acc| trace in plots and in timing that should follow physical impacts.

    Neighbouring taps use midpoint-bounded search intervals so two candidates
    do not latch onto the same sample.
    """
    if len(peak_indices) == 0:
        return []

    x = np.asarray(signal, dtype=float)
    pks = sorted(int(v) for v in np.asarray(peak_indices, dtype=int).ravel())
    refined: list[int] = []
    for i, p in enumerate(pks):
        if i == 0:
            lo_bound = 0
        else:
            lo_bound = (pks[i - 1] + p) // 2
        if i + 1 == len(pks):
            hi_bound = len(x)
        else:
            hi_bound = (p + pks[i + 1]) // 2
        lo = max(0, p - search_radius, lo_bound)
        hi = min(len(x), p + search_radius + 1, hi_bound)
        if hi <= lo:
            lo = max(0, p - search_radius)
            hi = min(len(x), p + search_radius + 1)
        seg = x[lo:hi]
        if not np.any(np.isfinite(seg)):
            refined.append(p)
            continue
        local_offset = int(np.nanargmax(seg))
        refined.append(lo + local_offset)

    refined = sorted(refined)
    merge_if_closer = max(1, min(2, min_spacing // 4))
    deduped: list[int] = []
    for idx in refined:
        if not deduped:
            deduped.append(idx)
            continue
        if idx - deduped[-1] < merge_if_closer:
            if x[idx] > x[deduped[-1]]:
                deduped[-1] = idx
        else:
            deduped.append(idx)
    return deduped


def _peak_passes_strength_gate(
    p: int,
    *,
    dynamic_smooth: np.ndarray,
    dynamic_raw: np.ndarray,
    transient: np.ndarray,
    peak_min_height: float,
    peak_prominence: float,
) -> bool:
    """A tap is *strong* if any of the parallel detector channels says so.

    ``dynamic_smooth`` can be near zero on |acc| crests that are narrow relative
    to the 100 ms smoothing window; ``dynamic_raw`` and ``transient`` keep
    legitimate bursts from being discarded by the quality filter.
    """
    p = int(p)
    smooth_thr = float(peak_min_height) * 0.72
    raw_thr = float(peak_min_height) * 0.32
    trans_thr = float(peak_prominence) * 2.0
    return bool(
        (np.isfinite(dynamic_smooth[p]) and dynamic_smooth[p] >= smooth_thr)
        or (np.isfinite(dynamic_raw[p]) and dynamic_raw[p] >= raw_thr)
        or (np.isfinite(transient[p]) and transient[p] >= trans_thr)
    )


def _is_spike_dominated_false_burst(
    refined_cluster: list[int],
    dynamic_smooth: np.ndarray,
    *,
    peak_min_height: float,
) -> bool:
    """Reject motion where most taps are weak on the smoothed envelope but one lobe dominates."""
    if len(refined_cluster) < 4:
        return False
    sm = np.asarray(
        [float(dynamic_smooth[int(p)]) for p in refined_cluster],
        dtype=float,
    )
    med = float(np.median(sm))
    mx = float(np.max(sm))
    pmh = float(peak_min_height)
    n_soft = int(np.sum(sm < pmh * 0.52))
    if n_soft >= 2 and mx / max(med, 1e-9) > 4.2:
        return True
    return med < pmh * 0.57 and mx > pmh * 2.35


def _collapse_nearby_peaks_on_norm(
    peaks: list[int],
    norm: np.ndarray,
    min_gap_samples: int,
) -> list[int]:
    """One peak per physical impact when |acc| lobes sit closer than *min_gap_samples*."""
    pks = sorted({int(p) for p in peaks})
    if len(pks) < 2:
        return pks
    x = np.asarray(norm, dtype=float)
    out: list[int] = []
    i = 0
    n = len(pks)
    gap = max(1, int(min_gap_samples))
    while i < n:
        j = i + 1
        while j < n and pks[j] - pks[i] < gap:
            j += 1
        chunk = pks[i:j]
        best = max(chunk, key=lambda p: x[p] if np.isfinite(x[p]) else -np.inf)
        out.append(int(best))
        i = j
    return out


def _transient_envelope(
    dynamic_smooth: np.ndarray,
    *,
    sample_rate_hz: float,
    transient_baseline_s: float,
) -> np.ndarray:
    """Emphasise short taps by subtracting a slower baseline from |acc−g| (smoothed)."""
    x = np.asarray(dynamic_smooth, dtype=float)
    tb = float(transient_baseline_s)
    if tb <= 0.0:
        return x
    win = max(3, int(sample_rate_hz * tb))
    baseline = smooth_moving_average(x, win)
    return np.maximum(x - baseline, 0.0)


def _pre_filter_transient_valleys(
    raw_burst: list[int],
    dynamic_smooth: np.ndarray,
    *,
    max_sm_ratio: float = 0.20,
    neighbor_ratio: float = 0.25,
) -> list[int]:
    """Remove obvious inter-tap echo peaks from *raw_burst* before burst selection.

    Some IMU sensors (e.g. Sporsa) produce ring-down oscillations between real
    calibration taps.  These appear as alternating strong / weak peaks on the
    smoothed |acc−g| envelope.  A peak is treated as a transient echo if it
    satisfies BOTH:

    1. ``sm[p] < max_sm_ratio × max(sm over burst)`` — it is small relative to
       the strongest peak in the burst, AND
    2. ``sm[p] < neighbor_ratio × max(sm of immediate neighbors)`` — it is much
       weaker than the adjacent peaks, making it a local valley.

    The filter iterates until no more peaks are removed (removing a peak changes
    the neighbourhood of adjacent peaks).

    Edge peaks (first / last) are held to the same threshold using their single
    available neighbour.
    """
    if len(raw_burst) < 2:
        return list(raw_burst)

    sm = np.asarray(dynamic_smooth, dtype=float)
    burst_arr = [int(p) for p in raw_burst]
    burst_max_sm = float(np.nanmax(sm[np.array(burst_arr, dtype=int)]))
    if burst_max_sm <= 0.0:
        return burst_arr

    sm_threshold = float(max_sm_ratio) * burst_max_sm

    changed = True
    peaks = burst_arr
    while changed:
        changed = False
        new_peaks: list[int] = []
        for i, p in enumerate(peaks):
            sm_p = float(sm[p])
            if sm_p < sm_threshold:
                neighbors: list[float] = []
                if i > 0:
                    neighbors.append(float(sm[peaks[i - 1]]))
                if i < len(peaks) - 1:
                    neighbors.append(float(sm[peaks[i + 1]]))
                if neighbors and sm_p < float(neighbor_ratio) * max(neighbors):
                    changed = True
                    continue
            new_peaks.append(p)
        peaks = new_peaks

    return peaks


def _split_peaks_by_pair_gap(peaks: np.ndarray, pair_max_samples: int) -> list[list[int]]:
    """Partition *peaks* so every consecutive pair in a part is ≤ *pair_max_samples*."""
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return []
    parts: list[list[int]] = []
    cur: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        if p - cur[-1] <= pair_max_samples:
            cur.append(p)
        else:
            parts.append(cur)
            cur = [p]
    parts.append(cur)
    return parts


def _best_contiguous_burst(
    peaks: list[int],
    *,
    peak_min_count: int,
    peak_max_count: int,
    pair_max_samples: int,
    burst_max_samples: int,
    dynamic_smooth: np.ndarray,
    dynamic_raw: np.ndarray | None = None,
    peak_min_height: float = 1.0,
) -> list[int] | None:
    """Choose a length-*peak_min_count*…*peak_max_count* sub-run with valid gaps and span.

    Scoring blends *dynamic_smooth* with capped *dynamic_raw* so the chosen
    window keeps **trailing** and **leading** taps that are clear in |acc−g|
    but smeared on the 100 ms-smoothed curve.  Among equal width we maximise
    that score, then prefer a **later** burst end (captures the last tap), then
    a tighter span, then an earlier onset.
    """
    n = len(peaks)
    if n < peak_min_count:
        return None
    x = np.asarray(dynamic_smooth, dtype=float)
    dr = (
        np.asarray(dynamic_raw, dtype=float)
        if dynamic_raw is not None
        else np.zeros_like(x)
    )
    cap = max(float(peak_min_height) * 2.75, 1.0)
    best: list[int] | None = None
    best_key: tuple | None = None
    max_w = min(peak_max_count, n)
    for width in range(peak_min_count, max_w + 1):
        for i in range(n - width + 1):
            sub = peaks[i : i + width]
            if sub[-1] - sub[0] > burst_max_samples:
                continue
            if any(sub[j + 1] - sub[j] > pair_max_samples for j in range(width - 1)):
                continue
            score = float(
                np.nansum(x[sub] + 0.52 * np.minimum(dr[sub], cap))
            )
            span = int(sub[-1] - sub[0])
            key = (-abs(width - 5), score, int(sub[-1]), -span, -int(sub[0]))
            if best_key is None or key > best_key:
                best_key = key
                best = list(sub)
    return best


def _median_peak_index(seg: CalibrationSegment) -> float:
    if not seg.peak_indices:
        return float(seg.start_idx)
    return float(np.median(np.asarray(seg.peak_indices, dtype=float)))


def _segments_overlap(
    a: CalibrationSegment,
    b: CalibrationSegment,
    max_gap_samples: int,
) -> bool:
    return a.start_idx <= b.end_idx + max_gap_samples and a.end_idx >= b.start_idx - max_gap_samples


def _peak_spans_duplicate_hypothesis(
    a: CalibrationSegment,
    b: CalibrationSegment,
    *,
    margin_samples: int,
    min_iou: float = 0.36,
) -> bool:
    """True if *a* and *b* are likely duplicate detections of the same tap burst.

    Segment index ranges often overlap for adjacent calibrations (shared flank
    stillness), so deduplication must **not** use :func:`_segments_overlap`
    alone.  We only merge when the *peak index* spans substantially intersect
    after a small margin (shifted duplicate hypotheses).
    """
    if not a.peak_indices or not b.peak_indices:
        return False
    ma = int(margin_samples)
    a0 = min(a.peak_indices) - ma
    a1 = max(a.peak_indices) + ma
    b0 = min(b.peak_indices) - ma
    b1 = max(b.peak_indices) + ma
    inter = min(a1, b1) - max(a0, b0) + 1
    if inter <= 0:
        return False
    la = a1 - a0 + 1
    lb = b1 - b0 + 1
    return float(inter) / float(min(la, lb)) >= min_iou


def _pick_overlap_group_winner(
    group: list[CalibrationSegment],
    *,
    strength_close_frac: float = 0.88,
) -> CalibrationSegment:
    """Choose one segment from a mutually overlapping set.

    Prefer clearly higher *peak_strength*.  When two (or more) candidates are
    within *strength_close_frac* of the group's maximum strength, treat the
    group as ambiguous (precursor vs real tap train) and keep the **later**
    burst (higher median peak index).
    """
    if len(group) == 1:
        return group[0]
    mx = max(s.peak_strength for s in group)
    if mx <= 0.0:
        return max(group, key=_median_peak_index)
    close = [s for s in group if s.peak_strength >= mx * strength_close_frac]
    if len(close) >= 2:
        return max(close, key=_median_peak_index)
    return max(group, key=lambda s: s.peak_strength)


def _deduplicate_segments(
    segments: list[CalibrationSegment],
    *,
    peak_merge_margin_samples: int,
) -> list[CalibrationSegment]:
    """Drop duplicate hypotheses that share the same tap burst (peak span).

    Adjacent real calibrations often have **overlapping** ``start_idx…end_idx``
    ranges because the post-static of one run meets the pre-static of the
    next.  Unioning on segment bounds therefore deletes genuine segments.
    We only merge when :func:`_peak_spans_duplicate_hypothesis` says the peak
    trains substantially coincide.
    """
    if len(segments) <= 1:
        return list(segments)

    n = len(segments)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    margin = int(peak_merge_margin_samples)
    for i in range(n):
        for j in range(i + 1, n):
            if _peak_spans_duplicate_hypothesis(segments[i], segments[j], margin_samples=margin):
                union(i, j)

    buckets: dict[int, list[CalibrationSegment]] = {}
    for i, seg in enumerate(segments):
        r = find(i)
        buckets.setdefault(r, []).append(seg)

    kept = [_pick_overlap_group_winner(g) for g in buckets.values()]
    return sorted(kept, key=lambda s: s.start_idx)


def _prune_tight_weak_neighbors(
    segments: list[CalibrationSegment],
    *,
    sample_rate_hz: float,
    min_peak_time_s: float = 10.0,
    strength_ratio: float = 0.72,
) -> list[CalibrationSegment]:
    """Drop the weaker of two segments whose tap bursts are very close in time.

    Duplicate hypotheses on the same motion, or a weak artefact shortly after a
    real calibration, often sit within ~10 s (median peak time) of another
    segment.  Distinct protocol calibrations are usually farther apart.
    """
    if len(segments) <= 1:
        return segments
    order = sorted(segments, key=_median_peak_index)
    med = [_median_peak_index(s) for s in order]
    # ~10.15 s at the nominal rate: catches ~590-sample gaps at 59 Hz (duplicate
    # hypotheses) while leaving ~602+ samples (~10.2 s) as distinct runs.
    eff_s = max(float(min_peak_time_s), 10.15)
    min_sep = max(1, int(float(sample_rate_hz) * eff_s) + 1)
    drop = [False] * len(order)
    for i in range(len(order) - 1):
        if drop[i] or drop[i + 1]:
            continue
        if med[i + 1] - med[i] >= min_sep:
            continue
        a, b = order[i], order[i + 1]
        sa, sb = a.peak_strength, b.peak_strength
        if sa <= 0.0 and sb <= 0.0:
            continue
        if sb < sa * strength_ratio:
            drop[i + 1] = True
        elif sa < sb * strength_ratio:
            drop[i] = True
    kept = [s for s, d in zip(order, drop) if not d]
    return sorted(kept, key=lambda s: s.start_idx)


def _prune_duplicate_calibration_retry(
    segments: list[CalibrationSegment],
    *,
    sample_rate_hz: float,
    gap_lo_s: float = 11.5,
    gap_hi_s: float = 36.0,
    strength_ratio: float = 0.92,
) -> list[CalibrationSegment]:
    """When the same calibration is performed twice in quick succession, keep the later run.

    The operator may repeat a burst after insufficient flank stillness; those
    attempts sit a few tens of seconds apart with similar *peak_strength*.
    """
    if len(segments) <= 1:
        return segments
    order = sorted(segments, key=_median_peak_index)
    med = [_median_peak_index(s) for s in order]
    gap_lo = max(1, int(float(sample_rate_hz) * float(gap_lo_s)))
    gap_hi = max(gap_lo + 1, int(float(sample_rate_hz) * float(gap_hi_s)))
    drop = [False] * len(order)
    for i in range(len(order) - 1):
        if drop[i] or drop[i + 1]:
            continue
        d = med[i + 1] - med[i]
        if d < gap_lo or d > gap_hi:
            continue
        sa, sb = order[i].peak_strength, order[i + 1].peak_strength
        mx = max(sa, sb)
        if mx <= 0.0:
            continue
        if min(sa, sb) / mx < float(strength_ratio):
            continue
        drop[i] = True
    kept = [s for s, d in zip(order, drop) if not d]
    return sorted(kept, key=lambda s: s.start_idx)


def find_calibration_segments(
    df: pd.DataFrame,
    *,
    sensor: str,
    **overrides: Any,
) -> list[CalibrationSegment]:
    """Find calibration sequences using ``data/_configs/cal_segments*.json`` for *sensor*.

    Numeric tuning lives only in the config file (per-sensor blocks). Optional
    *overrides* (e.g. ``sample_rate_hz=…``) replace loaded values for this call.
    """
    params = cal_segment_kwargs_for_sensor(sensor)
    params.update(overrides)
    return _find_calibration_segments_impl(df, params)


def _find_calibration_segments_impl(
    df: pd.DataFrame,
    params: dict[str, Any],
) -> list[CalibrationSegment]:
    """Core detector: *params* must contain all keys in ``_CAL_SEGMENT_PARAM_KEYS``."""
    sample_rate_hz = float(params["sample_rate_hz"])
    static_threshold = float(params["static_threshold"])
    peak_min_height = float(params["peak_min_height"])
    peak_min_count = int(params["peak_min_count"])
    peak_max_count: int | None = params["peak_max_count"]
    if peak_max_count is not None:
        peak_max_count = int(peak_max_count)
    static_gap_max_s = float(params["static_gap_max_s"])
    static_overlap_max_s = float(params["static_overlap_max_s"])
    static_min_start_fraction = float(params["static_min_start_fraction"])
    static_min_pre_s = float(params["static_min_pre_s"])
    static_min_post_s = float(params["static_min_post_s"])
    burst_max_span_s = float(params["burst_max_span_s"])
    peak_pair_max_gap_s = float(params["peak_pair_max_gap_s"])
    transient_baseline_s = float(params["transient_baseline_s"])
    peak_prominence = float(params["peak_prominence"])

    prep = _prepare_detection_arrays(
        df,
        static_threshold=static_threshold,
        sample_rate_hz=sample_rate_hz,
    )
    if prep is None:
        return []

    is_static = prep.is_static
    dynamic_smooth = prep.dynamic_smooth
    dynamic_raw = prep.dynamic_raw
    smooth_win = max(3, int(sample_rate_hz * 0.1))

    transient = _transient_envelope(
        dynamic_smooth,
        sample_rate_hz=sample_rate_hz,
        transient_baseline_s=transient_baseline_s,
    )
    min_peak_distance = max(1, int(sample_rate_hz * 0.15))
    wlen = max(7, int(sample_rate_hz * 1.2))
    peak_idx, _ = find_peaks(
        transient,
        distance=min_peak_distance,
        prominence=max(peak_prominence, 1e-9),
        wlen=wlen,
    )
    peak_idx = peak_idx[np.isfinite(dynamic_smooth[peak_idx])]
    # Prominence is computed on *transient*; very short impacts can fall below
    # *peak_min_height* on the 100 ms-smoothed envelope while still being real taps.
    transient_alt = max(2.0 * peak_prominence, 0.28 * peak_min_height)
    smooth_floor = float(peak_min_height) * (0.52 if peak_min_count <= 4 else 0.48)
    keep = (dynamic_smooth[peak_idx] >= peak_min_height) | (
        (transient[peak_idx] >= transient_alt)
        & (dynamic_raw[peak_idx] >= 0.38 * float(peak_min_height))
        & (dynamic_smooth[peak_idx] >= smooth_floor)
    )
    peak_idx = peak_idx[keep]

    if peak_idx.size < peak_min_count:
        return []

    pair_max_samples = max(1, int(sample_rate_hz * peak_pair_max_gap_s))
    burst_max_samples = max(1, int(sample_rate_hz * burst_max_span_s))
    raw_bursts = _split_peaks_by_pair_gap(peak_idx, pair_max_samples)

    burst_peak_max = int(peak_max_count) if peak_max_count is not None else max(peak_min_count, 12)

    clusters: list[list[int]] = []
    for raw in raw_bursts:
        # Remove inter-tap echo peaks before selecting the best window; echo
        # peaks alternate with real taps on sensors with ring-down oscillations.
        filtered = _pre_filter_transient_valleys(raw, dynamic_smooth)
        candidate = filtered if len(filtered) >= peak_min_count else raw
        chosen = _best_contiguous_burst(
            candidate,
            peak_min_count=peak_min_count,
            peak_max_count=burst_peak_max,
            pair_max_samples=pair_max_samples,
            burst_max_samples=burst_max_samples,
            dynamic_smooth=dynamic_smooth,
            dynamic_raw=dynamic_raw,
            peak_min_height=peak_min_height,
        )
        if chosen is not None:
            clusters.append(chosen)

    if not clusters:
        return []

    static_min_pre_samples = int(sample_rate_hz * static_min_pre_s)
    static_min_post_samples = int(sample_rate_hz * static_min_post_s)
    static_gap_max_samples = int(sample_rate_hz * static_gap_max_s)
    static_overlap_max_samples = int(sample_rate_hz * static_overlap_max_s)
    static_min_start_samples = max(1, int(static_min_pre_samples * static_min_start_fraction))
    refine_radius = max(1, smooth_win // 2 + 1)
    peak_min_spacing = max(1, int(sample_rate_hz * 0.15))
    peak_dedup_margin = max(8, int(sample_rate_hz * 0.2))

    static_runs: list[tuple[int, int, int]] = []
    run_start: int | None = None
    for i, flag in enumerate(is_static):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            static_runs.append((run_start, i - 1, i - run_start))
            run_start = None
    if run_start is not None:
        static_runs.append((run_start, len(is_static) - 1, len(is_static) - run_start))

    segments: list[CalibrationSegment] = []
    for cluster in clusters:
        refined_cluster = _refine_peaks_to_raw_max(
            cluster,
            prep.norm,
            search_radius=refine_radius,
            min_spacing=peak_min_spacing,
        )
        collapse_gap = max(5, int(sample_rate_hz * 0.10))
        collapsed = _collapse_nearby_peaks_on_norm(
            refined_cluster,
            prep.norm,
            collapse_gap,
        )
        if len(collapsed) >= peak_min_count and (
            peak_max_count is None or len(collapsed) <= peak_max_count
        ):
            refined_cluster = collapsed

        if len(refined_cluster) < peak_min_count:
            continue
        if peak_max_count is not None and len(refined_cluster) > peak_max_count:
            continue

        if _is_spike_dominated_false_burst(
            refined_cluster,
            dynamic_smooth,
            peak_min_height=peak_min_height,
        ):
            continue

        n_strong = sum(
            _peak_passes_strength_gate(
                int(p),
                dynamic_smooth=dynamic_smooth,
                dynamic_raw=dynamic_raw,
                transient=transient,
                peak_min_height=peak_min_height,
                peak_prominence=peak_prominence,
            )
            for p in refined_cluster
        )
        if n_strong < peak_min_count - 1:
            continue

        med_raw = float(
            np.median([float(dynamic_raw[int(p)]) for p in refined_cluster])
        )
        med_raw_min = float(peak_min_height) * (1.78 if peak_min_count >= 5 else 1.5)
        if med_raw < med_raw_min:
            continue

        drs = [float(dynamic_raw[int(p)]) for p in refined_cluster]
        min_raw = min(drs)
        max_raw = max(drs)
        pmh_f = float(peak_min_height)
        # Reject bursts where several taps are soft on |acc−g| (transient lobes);
        # keep sequences with a single unusually soft flank if other taps are strong (r3).
        n_soft_raw = sum(1 for v in drs if v < pmh_f * 0.42)
        if n_soft_raw >= 2:
            continue
        if min_raw < pmh_f * 0.31 and max_raw < pmh_f * 2.5:
            continue

        c_start = refined_cluster[0]
        c_end = refined_cluster[-1]

        pre_run: tuple[int, int] | None = None
        for rs, re, rl in reversed(static_runs):
            min_pre = static_min_pre_samples
            if rs == 0:
                min_pre = static_min_start_samples
            if rl < min_pre:
                continue
            if (c_start - re) > static_gap_max_samples:
                break
            # Allow a small overlap where static classification still labels
            # a few samples as static even though peak detection already
            # started.
            if (re - c_start) > static_overlap_max_samples:
                continue
            pre_run = (rs, re)
            break

        if pre_run is None:
            continue

        n_samples = len(df)
        tail_remaining = max(0, (n_samples - 1) - c_end)
        min_post_eff = static_min_post_samples
        if tail_remaining < static_min_post_samples:
            min_post_eff = max(
                int(static_min_post_samples * 0.55),
                tail_remaining - 30,
            )

        post_candidates: list[tuple[int, int, int]] = []
        for rs, re, rl in static_runs:
            if rl < min_post_eff:
                continue
            if re < c_end:
                continue
            if rs > c_end + static_gap_max_samples:
                break
            if (c_end - rs) > static_overlap_max_samples:
                continue
            post_candidates.append((rs, re, rl))

        post_run: tuple[int, int] | None = None
        if post_candidates:
            rs, re, _ = max(post_candidates, key=lambda t: t[2])
            post_run = (rs, re)

        if post_run is None:
            continue

        peak_strength = float(np.nansum(dynamic_smooth[np.asarray(refined_cluster, dtype=int)]))
        segments.append(
            CalibrationSegment(
                start_idx=pre_run[0],
                end_idx=post_run[1],
                peak_indices=refined_cluster,
                peak_strength=peak_strength,
            )
        )

    out = _deduplicate_segments(segments, peak_merge_margin_samples=peak_dedup_margin)
    out = _prune_tight_weak_neighbors(out, sample_rate_hz=sample_rate_hz)
    return _prune_duplicate_calibration_retry(out, sample_rate_hz=sample_rate_hz)


def load_cal_segments_config() -> dict[str, Any]:
    """Load ``cal_segments_args.json`` (per-sensor detection parameter blocks)."""
    return read_json_file(cal_segments_config_path())


def cal_segment_kwargs_for_sensor(sensor: str) -> dict[str, Any]:
    """Return detection parameters for *sensor* from the cal-segments config file.

    Each sensor block must define exactly the keys in ``_CAL_SEGMENT_PARAM_KEYS``.
    """
    raw = load_cal_segments_config()
    if not isinstance(raw, dict):
        raise TypeError("Cal-segments config must be a JSON object.")
    key = sensor.strip().lower()
    block = raw.get(key)
    if not isinstance(block, dict):
        raise KeyError(
            f"Cal-segments config has no parameter object for sensor {sensor!r} (key {key!r})."
        )
    missing = _CAL_SEGMENT_PARAM_KEYS - block.keys()
    if missing:
        raise KeyError(
            f"Cal-segments config for {sensor!r} missing keys: {sorted(missing)}"
        )
    extra = set(block.keys()) - _CAL_SEGMENT_PARAM_KEYS
    if extra:
        raise ValueError(
            f"Cal-segments config for {sensor!r} has unknown keys: {sorted(extra)}"
        )
    return dict(block)
