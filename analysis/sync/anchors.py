"""Calibration-anchor extraction for synchronisation.

The synchronisation model needs one time anchor per calibration sequence: a
point where the reference and target clocks observed the same physical event.
The recording protocol guarantees that both sensors are tapped simultaneously
during each calibration sequence, so the **median tap timestamp** in each
sensor stream is a direct coarse anchor.

Anchor extraction:

1. Load the calibration-segment JSON for both sensors (written by the parser).
2. Match segments 1-to-1 in chronological order (equal counts required).
3. For each pair compute a coarse offset from median peak timestamps::

       coarse_offset_s = (median(ref_peak_ms) − median(tgt_peak_ms)) / 1000

4. Refine each anchor by resampling the ``acc_norm`` signal within the
   peak window at a common rate and maximising cross-correlation (via
   :func:`sync.xcorr.estimate_lag`) within the configured local search span.
   The best sample shift from correlation converts directly to an offset
   correction in seconds.

The result feeds directly into the drift-fitting step in
:mod:`sync.strategies`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.paths import recording_stage_dir
from parser.calibration_segments import (
    CalibrationSegment,
    load_calibration_segments_from_json,
)
from .config import default_sync_config
from .signals import load_stream, resample_stream
from .xcorr import estimate_lag

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class CalibrationAnchor:
    """A matched calibration-sequence pair with a derived time offset.

    Timestamps are in **milliseconds** (raw sensor time); offset is in
    **seconds** (consistent with the sync model).
    """

    ref_ms: float   # median peak timestamp in reference stream
    tgt_ms: float   # inferred target timestamp (ref_ms − offset_s*1000)
    offset_s: float           # (ref_ms − tgt_ms) / 1000, refined by xcorr
    score: float = 0.0        # overlap-normalised xcorr score (NaN if unavailable)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refine_offset_xcorr(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    ref_seg: CalibrationSegment,
    coarse_offset_s: float,
    *,
    resample_rate_hz: float,
    search_seconds: float,
) -> tuple[float, float]:
    """Refine *coarse_offset_s* by maximising acc_norm cross-correlation.

    Uses the full segment span (not just detected peaks) so that partial
    detections near recording gaps do not bias the window placement.
    Searches within ±``search_seconds`` around zero sample shift.
    """
    ref_start = ref_seg.start_ms
    ref_end = ref_seg.end_ms

    # Corresponding tgt window according to coarse offset.
    tgt_start = ref_start - coarse_offset_s * 1000.0
    tgt_end = ref_end - coarse_offset_s * 1000.0

    ref_win = resample_stream(
        ref_df, resample_rate_hz,
        start_ms=ref_start, end_ms=ref_end, columns=["acc_norm"],
    )
    tgt_win = resample_stream(
        tgt_df, resample_rate_hz,
        start_ms=tgt_start, end_ms=tgt_end, columns=["acc_norm"],
    )

    if "acc_norm" not in ref_win.columns or "acc_norm" not in tgt_win.columns:
        return coarse_offset_s, float("nan")

    ref_sig = ref_win["acc_norm"].to_numpy(dtype=float)
    tgt_sig = tgt_win["acc_norm"].to_numpy(dtype=float)
    if ref_sig.size < 10 or tgt_sig.size < 10:
        return coarse_offset_s, float("nan")

    search_n = max(1, int(round(search_seconds * resample_rate_hz)))
    shift_samples, score = estimate_lag(ref_sig, tgt_sig, max_lag_samples=search_n)

    if not np.isfinite(score) or score <= 0:
        return coarse_offset_s, float("nan")

    # Positive shift: ref is ahead of tgt by `shift_samples` samples.
    refined_offset_s = coarse_offset_s + shift_samples / resample_rate_hz
    return refined_offset_s, float(score)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def load_calibration_anchors(
    recording_name: str,
    *,
    ref_sensor: str = "sporsa",
    tgt_sensor: str = "arduino",
    resample_rate_hz: float | None = None,
    search_seconds: float | None = None,
) -> list[CalibrationAnchor]:
    """Load calibration segments and produce one refined anchor per matched pair."""
    config = default_sync_config()
    if resample_rate_hz is None:
        resample_rate_hz = config.anchor_refinement.resample_rate_hz
    if search_seconds is None:
        search_seconds = config.anchor_refinement.search_seconds

    ref_segs = load_calibration_segments_from_json(recording_name, ref_sensor)
    tgt_segs = load_calibration_segments_from_json(recording_name, tgt_sensor)

    if not ref_segs:
        raise ValueError(
            f"No calibration segments found for reference sensor {ref_sensor!r} "
            f"in recording {recording_name!r}."
        )
    if not tgt_segs:
        raise ValueError(
            f"No calibration segments found for target sensor {tgt_sensor!r} "
            f"in recording {recording_name!r}."
        )
    if len(ref_segs) != len(tgt_segs):
        raise ValueError(
            f"Calibration segment count mismatch for {recording_name!r}: "
            f"{ref_sensor}={len(ref_segs)}, {tgt_sensor}={len(tgt_segs)}. "
            "Segments must be detected 1-to-1 for anchor matching."
        )

    parsed_dir = recording_stage_dir(recording_name, "parsed")
    ref_df = load_stream(parsed_dir / f"{ref_sensor}.csv")
    tgt_df = load_stream(parsed_dir / f"{tgt_sensor}.csv")

    anchors: list[CalibrationAnchor] = []
    for i, (ref_seg, tgt_seg) in enumerate(zip(ref_segs, tgt_segs)):
        if not ref_seg.peak_ms:
            log.warning("Anchor %d: reference segment has no peaks — skipping.", i)
            continue
        if not tgt_seg.peak_ms:
            log.warning("Anchor %d: target segment has no peaks — skipping.", i)
            continue

        ref_center_ms = np.median(ref_seg.peak_ms)
        tgt_center_ms = np.median(tgt_seg.peak_ms)
        coarse_offset_s = (ref_center_ms - tgt_center_ms) / 1000.0

        refined_offset_s, score = _refine_offset_xcorr(
            ref_df,
            tgt_df,
            ref_seg,
            coarse_offset_s,
            resample_rate_hz=float(resample_rate_hz),
            search_seconds=float(search_seconds),
        )

        tgt_center_ms = ref_center_ms - refined_offset_s * 1000.0
        anchors.append(
            CalibrationAnchor(
                ref_ms=ref_center_ms,
                tgt_ms=tgt_center_ms,
                offset_s=refined_offset_s,
                score=score,
            )
        )
        log.debug(
            "Anchor %d: ref=%.1f ms  tgt=%.1f ms  coarse=%.4f s  refined=%.4f s  score=%.3f",
            i, ref_center_ms, tgt_center_ms, coarse_offset_s, refined_offset_s,
            score if np.isfinite(score) else -1.0,
        )

    if not anchors:
        raise ValueError(
            f"No valid anchors could be extracted for {recording_name!r} "
            f"({ref_sensor} / {tgt_sensor})."
        )

    return sorted(anchors, key=lambda a: a.tgt_ms)


def calibration_anchor_to_dict(
    anchor: CalibrationAnchor,
    *,
    index: int | None = None,
) -> dict[str, Any]:
    """Serialise one anchor for sync metadata."""
    data: dict[str, Any] = {
        "offset_s": round(float(anchor.offset_s), 6),
        "ref_ms": round(float(anchor.ref_ms), 1),
        "tgt_ms": round(float(anchor.tgt_ms), 1),
        "score": round(float(anchor.score), 4) if np.isfinite(anchor.score) else None,
    }
    if index is not None:
        data["index"] = int(index)
    return data
