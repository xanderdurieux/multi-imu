"""Extract static window DataFrames from calibration sequences.

A calibration sequence is:  ~5 s static → 5 acceleration peaks → ~5 s static

This module slices out the flanking static sub-DataFrames from each detected
calibration segment and exposes them for downstream bias/orientation estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from parser.split_sections import CalibrationSegment, find_calibration_segments

_GRAVITY_MS2 = 9.81


def _remove_dropouts(df: pd.DataFrame, *, epsilon_fraction: float = 0.1) -> pd.DataFrame:
    """Remove near-zero acceleration rows (sensor dropout packets).

    Some sensors (e.g. the Arduino BLE device) occasionally emit zero-valued
    packets.  Their acc_norm ≈ 0 looks like a huge deviation from gravity to
    the calibration detector, causing false peak detections and corrupting the
    static window classification.
    """
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(acc ** 2, axis=1))
    g_approx = float(np.nanmedian(norm))
    if g_approx <= 0:
        return df
    threshold = epsilon_fraction * g_approx
    valid = norm >= threshold
    return df.loc[valid].reset_index(drop=True)

log = logging.getLogger(__name__)


@dataclass
class StaticWindows:
    """Static window samples extracted from calibration sequences.

    Attributes
    ----------
    segments:
        One ``(pre_static, post_static)`` DataFrame pair per detected
        calibration segment.  Either element may be empty if the corresponding
        flanking static region contained fewer samples than the guard buffer.
    combined:
        All static samples from all segments concatenated.  Empty if no
        calibration segments were found.
    n_calibration_segments:
        Number of calibration segments that were detected.
    """

    segments: list[tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=list)
    combined: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_calibration_segments: int = 0

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_static_samples(self) -> int:
        return len(self.combined)

    @property
    def n_mag_samples(self) -> int:
        if self.combined.empty or "mx" not in self.combined.columns:
            return 0
        return int(self.combined["mx"].notna().sum())

    def mag_subset(self, *, min_samples: int = 5) -> pd.DataFrame:
        """Return rows of *combined* that have valid magnetometer readings.

        Falls back to scanning only the post-static windows from each segment
        if the overall mag-sample count is below *min_samples* (covers the
        Arduino BMM150 ~1.5 s startup delay that may blank the opening
        pre-static window).
        """
        if self.combined.empty or "mx" not in self.combined.columns:
            return pd.DataFrame(columns=self.combined.columns)

        all_mag = self.combined[self.combined["mx"].notna()]
        if len(all_mag) >= min_samples:
            return all_mag

        # Fallback: collect only from post-static windows
        log.warning(
            "Only %d mag samples in combined static windows "
            "(threshold=%d). Collecting from post-static windows only.",
            len(all_mag), min_samples,
        )
        post_frames = [post for _, post in self.segments if not post.empty]
        if not post_frames:
            return all_mag

        post_combined = pd.concat(post_frames, ignore_index=True)
        post_mag = post_combined[post_combined["mx"].notna()]
        if len(post_mag) >= min_samples:
            return post_mag

        log.warning(
            "Still only %d mag samples from post-static windows. "
            "Mag-based calibration may be unreliable.",
            len(post_mag),
        )
        return post_mag


def extract_static_windows(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    buffer_samples: int = 10,
    filter_dropouts: bool = True,
    dropout_epsilon: float = 0.1,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
) -> StaticWindows:
    """Extract static window DataFrames from a sensor recording.

    Runs :func:`parser.split_sections.find_calibration_segments` on *df* and
    for each detected segment slices out:

    - **pre-static**: rows from ``seg.start_idx`` up to
      ``seg.peak_indices[0] - buffer_samples`` (exclusive).
    - **post-static**: rows from ``seg.peak_indices[-1] + buffer_samples``
      (exclusive) up to ``seg.end_idx``.

    The *buffer_samples* guard keeps peak-transition samples (which contain
    high-acceleration transients) out of the static windows.

    Parameters
    ----------
    df:
        Parsed IMU DataFrame (``timestamp, ax, ay, az, gx, gy, gz, …``),
        sorted by timestamp.
    sample_rate_hz:
        Approximate sampling rate, forwarded to the segment detector.
    buffer_samples:
        Number of samples to exclude before the first peak and after the last
        peak when cutting the static windows.
    filter_dropouts:
        When ``True`` (default), remove near-zero acceleration rows (sensor
        dropout packets) before segment detection.  This is important for the
        Arduino sensor which emits zero-valued BLE packets that corrupt the
        calibration detector.  Dropout removal is applied to the working copy
        of *df*; windows are sliced from the filtered data.
    dropout_epsilon:
        Fraction of estimated gravity below which a row is classified as a
        dropout (default: 0.1).  Only used when *filter_dropouts* is ``True``.
    static_min_s, static_threshold, peak_min_height, peak_min_count,
    peak_max_gap_s, static_gap_max_s:
        Forwarded to :func:`parser.split_sections.find_calibration_segments`.

    Returns
    -------
    StaticWindows
        Aggregated static data.  ``n_calibration_segments == 0`` when no
        calibration sequences are found.

    Raises
    ------
    ValueError
        If *df* is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot extract static windows.")

    # Optionally remove dropout packets before calibration detection.
    # We always detect segments on the (possibly filtered) working copy and
    # return windows sliced from that same copy.
    working_df = _remove_dropouts(df, epsilon_fraction=dropout_epsilon) if filter_dropouts else df
    if filter_dropouts and len(working_df) < len(df):
        n_dropped = len(df) - len(working_df)
        log.info(
            "Removed %d dropout packets (%.1f%%) before calibration detection.",
            n_dropped, 100.0 * n_dropped / len(df),
        )

    # Auto-detect sample rate from the (filtered) data's median timestamp
    # interval.  This is important when the actual rate differs from the
    # nominal value (e.g. Arduino BLE at ~55 Hz after dropout removal vs the
    # nominal 100 Hz — a 2× error that causes static windows to be classified
    # as too short).
    if "timestamp" in working_df.columns and len(working_df) > 1:
        ts = working_df["timestamp"].to_numpy(dtype=float)
        dt_ms = float(np.median(np.diff(ts)))
        if dt_ms > 0:
            detected_hz = 1000.0 / dt_ms
            if abs(detected_hz - sample_rate_hz) / sample_rate_hz > 0.15:
                log.info(
                    "Auto-detected sample rate %.1f Hz (nominal %.1f Hz); "
                    "using detected rate for calibration segment detection.",
                    detected_hz, sample_rate_hz,
                )
            sample_rate_hz = detected_hz

    cal_segments: list[CalibrationSegment] = find_calibration_segments(
        working_df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )

    if not cal_segments:
        log.warning("No calibration segments found in the recording.")
        return StaticWindows(n_calibration_segments=0, combined=pd.DataFrame(columns=working_df.columns))

    log.info("Found %d calibration segment(s).", len(cal_segments))

    segment_pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    all_frames: list[pd.DataFrame] = []

    for i, seg in enumerate(cal_segments):
        # Pre-static: from segment start up to (but not including) the guard
        # zone before the first peak.
        pre_end_excl = max(seg.start_idx, seg.peak_indices[0] - buffer_samples)
        pre_static = working_df.iloc[seg.start_idx:pre_end_excl].reset_index(drop=True)

        # Post-static: from the guard zone after the last peak to segment end.
        post_start_incl = min(seg.end_idx + 1, seg.peak_indices[-1] + buffer_samples + 1)
        post_static = working_df.iloc[post_start_incl:seg.end_idx + 1].reset_index(drop=True)

        segment_pairs.append((pre_static, post_static))

        n_pre = len(pre_static)
        n_post = len(post_static)
        log.debug(
            "Segment %d/%d: pre_static=%d samples, post_static=%d samples.",
            i + 1, len(cal_segments), n_pre, n_post,
        )

        if not pre_static.empty:
            all_frames.append(pre_static)
        if not post_static.empty:
            all_frames.append(post_static)

    combined = (
        pd.concat(all_frames, ignore_index=True)
        if all_frames
        else pd.DataFrame(columns=working_df.columns)
    )

    return StaticWindows(
        segments=segment_pairs,
        combined=combined,
        n_calibration_segments=len(cal_segments),
    )
