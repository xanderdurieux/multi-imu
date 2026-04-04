"""Calibration-sequence detection shared across the analysis pipeline.

This module implements detection of calibration sequences in a single IMU
recording. A calibration sequence has the pattern::

    ~5 s static  →  ~5 acceleration peaks  →  ~5 s static

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
from typing import List

import numpy as np
import pandas as pd

from common.signals import acc_norm_from_imu_df, find_peaks_simple, smooth_moving_average


@dataclass(frozen=True)
class CalibrationSegment:
    """Indices (inclusive) of one calibration sequence in the DataFrame."""

    start_idx: int
    end_idx: int
    peak_indices: list[int]


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
    dynamic_raw: np.ndarray,
    *,
    search_radius: int,
    min_spacing: int,
) -> list[int]:
    """Move smoothed peak indices onto the nearest strong raw impact maxima.

    The shared detector uses a lightly smoothed acceleration envelope to find
    tap clusters robustly. For sharp taps, the smoothed local maximum lands a
    few samples after the true impact. This refinement step keeps the cluster
    structure but snaps each candidate back to the strongest raw excursion in a
    local neighbourhood so downstream protocol plots and opening-routine timing
    align with the visible taps.
    """
    if len(peak_indices) == 0:
        return []

    x = np.asarray(dynamic_raw, dtype=float)
    refined: list[int] = []
    for peak_idx in np.asarray(peak_indices, dtype=int):
        lo = max(0, int(peak_idx) - search_radius)
        hi = min(len(x), int(peak_idx) + search_radius + 1)
        if hi <= lo:
            refined.append(int(peak_idx))
            continue
        local_offset = int(np.nanargmax(x[lo:hi]))
        refined.append(lo + local_offset)

    refined = sorted(refined)
    deduped: list[int] = []
    for idx in refined:
        if not deduped:
            deduped.append(idx)
            continue
        if idx - deduped[-1] < min_spacing:
            if x[idx] > x[deduped[-1]]:
                deduped[-1] = idx
        else:
            deduped.append(idx)
    return deduped


def _segment_quality(seg: CalibrationSegment) -> tuple:
    """Return a sortable quality tuple for *seg* (higher = better).

    Priority (most to least important):

    1. **Peak count, capped at the expected calibration count** — extra peaks
       beyond the target do not improve quality, and suppress clusters from
       prolonged non-calibration activity (e.g. sustained cycling) from
       out-competing genuine calibration sequences.
    2. **Pre-static length** (samples from segment start to first peak) — the
       primary discriminator between a genuine calibration and incidental
       activity that happens to match the pattern.  A real calibration is
       always preceded by an intentional "at rest" period; a false positive
       created by, e.g., donning equipment mid-recording will have a very short
       "pre-static" that is merely the tail of the preceding genuine post-static.
    3. **Post-static length** (samples from last peak to segment end) — a
       genuine calibration is also followed by an intentional rest.  Incidental
       peak clusters that happen to precede another burst of activity will have
       a short post-static (only the gap to the next peaks).
    4. **Total duration** — longer segment as final tiebreaker.
    """
    n_peaks = len(seg.peak_indices)
    pre_static = seg.peak_indices[0] - seg.start_idx if seg.peak_indices else 0
    post_static = seg.end_idx - seg.peak_indices[-1] if seg.peak_indices else 0
    duration = seg.end_idx - seg.start_idx
    return (n_peaks, pre_static, post_static, duration)


def _deduplicate_segments(
    segments: list[CalibrationSegment],
    max_gap_samples: int = 0,
) -> list[CalibrationSegment]:
    """Remove overlapping or near-adjacent duplicate segments.

    When a calibration sequence contains a brief mid-sequence pause, the peak
    detector may split it into two sub-clusters that each match the shared
    "mid-static" region as their opposing flank.  This produces two overlapping
    (or barely adjacent) segments for one real calibration.

    Additionally, incidental activity that resembles a calibration (e.g.
    donning equipment after a genuine calibration) may be detected as a second
    segment whose "pre-static" is actually the tail of the preceding genuine
    post-static — and is therefore much shorter than the pre-static of the real
    calibration.

    Segments are ranked by :func:`_segment_quality` and accepted greedily: a
    candidate is kept only if it does not overlap with any already-accepted
    segment.  The result is returned in chronological order.

    *max_gap_samples* widens the overlap check by this many samples on each
    side, catching pairs that are adjacent rather than strictly overlapping.
    Passing ``static_min_samples`` is safe: two genuinely distinct calibrations
    are always separated by more than one full ``static_min_s`` flanking window.
    """
    if len(segments) <= 1:
        return list(segments)

    by_quality = sorted(segments, key=_segment_quality, reverse=True)

    kept: list[CalibrationSegment] = []
    for seg in by_quality:
        overlaps = any(
            seg.start_idx <= k.end_idx + max_gap_samples
            and seg.end_idx >= k.start_idx - max_gap_samples
            for k in kept
        )
        if not overlaps:
            kept.append(seg)

    return sorted(kept, key=lambda s: s.start_idx)


def find_calibration_segments(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_count: int = 6,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    static_overlap_max_s: float = 1.0,
    static_min_start_fraction: float = 0.75,
) -> list[CalibrationSegment]:
    """Find calibration sequences in an IMU DataFrame.

    A calibration sequence is detected as:

    1. A run of ``static_min_s`` seconds where each sample's instantaneous
       ``|acc_norm - g|`` is below ``static_threshold`` (m/s²).
    2. Followed by a cluster of at least ``peak_min_count`` (and at most
       ``peak_max_count``) peaks where the lightly smoothed
       ``|acc_norm - g|`` exceeds ``peak_min_height`` (m/s²), with
       consecutive peaks no farther apart than ``peak_max_gap_s`` seconds.
    3. Followed by another run of ``static_min_s`` seconds.

    The flanking static regions must also lie within ``static_gap_max_s``
    seconds of the first/last peak, which prevents exercise-activity clusters
    from being matched to a static run that is many minutes away.

    ``static_overlap_max_s`` allows a small overlap between the flanking
    static windows and the first/last detected peak indices when the
    static classifier and peak detector disagree slightly around threshold
    crossings (a common issue near recording boundaries).

    ``static_min_start_fraction`` relaxes the required pre-static duration
    when the flanking static window begins exactly at the first sample of
    the recording (static classifier noise can otherwise truncate the initial
    static period just below ``static_min_s``).

    *peak_max_count* caps the number of peaks per cluster.  Clusters with
    more peaks than this are skipped.  This is useful to reject continuous
    exercise activity (e.g. cycling) which can produce very large peak counts
    and otherwise dominates the quality ranking.  ``None`` (default) means no
    upper limit.
    """
    norm = acc_norm_from_imu_df(df)
    if len(norm) == 0:
        return []

    # Estimate |g| as the global median of acc_norm. This works well for
    # recordings that contain substantial static periods (e.g. the Sporsa
    # handlebar sensor).  More specialised handling for sensors that are
    # mostly dynamic (e.g. helmet-mounted Arduino) should be implemented
    # upstream, without changing this shared definition.
    g = float(np.nanmedian(norm))

    dynamic_raw = np.abs(norm - g)
    is_dropout = norm < 0.1 * g
    is_static = (dynamic_raw < static_threshold) | is_dropout

    # Impute dropout positions with g before smoothing so that near-zero
    # packets (common on the Arduino BLE sensor) don't create spurious
    # spikes in the smoothed dynamic signal and cause false peak detections.
    norm_for_smooth = norm.copy()
    norm_for_smooth[is_dropout] = g

    smooth_win = max(3, int(sample_rate_hz * 0.1))
    dynamic_smooth = np.abs(smooth_moving_average(norm_for_smooth, smooth_win) - g)

    peaks = find_peaks_simple(
        dynamic_smooth,
        height=peak_min_height,
        distance=max(1, int(sample_rate_hz * 0.3)),
    )

    if len(peaks) < peak_min_count:
        return []

    max_gap_samples = int(sample_rate_hz * peak_max_gap_s)
    clusters: list[list[int]] = []
    current: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - current[-1] <= max_gap_samples:
            current.append(int(p))
        else:
            if len(current) >= peak_min_count:
                clusters.append(current)
            current = [int(p)]
    if len(current) >= peak_min_count:
        clusters.append(current)

    if not clusters:
        return []

    if peak_max_count is not None:
        clusters = [c for c in clusters if len(c) <= peak_max_count]
    if not clusters:
        return []

    static_min_samples = int(sample_rate_hz * static_min_s)
    static_gap_max_samples = int(sample_rate_hz * static_gap_max_s)
    static_overlap_max_samples = int(sample_rate_hz * static_overlap_max_s)
    static_min_start_samples = max(1, int(static_min_samples * static_min_start_fraction))
    refine_radius = max(1, smooth_win // 2 + 1)
    peak_min_spacing = max(1, int(sample_rate_hz * 0.15))

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
            dynamic_raw,
            search_radius=refine_radius,
            min_spacing=peak_min_spacing,
        )
        if len(refined_cluster) < peak_min_count:
            continue

        c_start = refined_cluster[0]
        c_end = refined_cluster[-1]

        pre_run: tuple[int, int] | None = None
        for rs, re, rl in reversed(static_runs):
            min_pre = static_min_samples
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

        post_run: tuple[int, int] | None = None
        for rs, re, rl in static_runs:
            if rl < static_min_samples:
                continue
            if (rs - c_end) > static_gap_max_samples:
                break
            # Allow a small overlap where static classification extends
            # slightly into the last-peak region.
            if (c_end - rs) > static_overlap_max_samples:
                continue
            post_run = (rs, re)
            break

        if post_run is None:
            continue

        segments.append(
            CalibrationSegment(
                start_idx=pre_run[0],
                end_idx=post_run[1],
                peak_indices=refined_cluster,
            )
        )

    return _deduplicate_segments(segments, max_gap_samples=static_min_samples)
