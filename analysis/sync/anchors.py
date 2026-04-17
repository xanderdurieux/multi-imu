"""Calibration-anchor extraction for synchronisation.

The synchronisation model needs one time anchor per calibration sequence: a
point where the reference and target clocks observed the same physical event.
The recording protocol guarantees that both sensors are tapped simultaneously
during each calibration sequence, so the **median tap timestamp** in each
sensor stream is a direct anchor.

Anchor extraction is therefore purely timestamp-based:

1. Load the calibration-segment JSON for both sensors (written by the parser).
2. Match segments 1-to-1 in chronological order (equal counts required).
3. For each pair compute::

       offset_s = (median(ref_peak_timestamps_ms) − median(tgt_peak_timestamps_ms)) / 1000

The result feeds directly into the drift-fitting step in
:mod:`sync.strategies`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from parser.calibration_segments import (
    CalibrationSegment,
    load_calibration_segments_from_json,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationAnchor:
    """A matched calibration-sequence pair with a derived time offset.

    Timestamps are in **milliseconds** (raw sensor time); offset is in
    **seconds** (consistent with the sync model).
    """

    ref_timestamp_ms: float   # median peak timestamp in reference stream
    tgt_timestamp_ms: float   # median peak timestamp in target stream (raw)
    offset_s: float           # (ref_ms − tgt_ms) / 1000


def load_calibration_anchors(
    recording_name: str,
    *,
    ref_sensor: str = "sporsa",
    tgt_sensor: str = "arduino",
) -> list[CalibrationAnchor]:
    """Load calibration segments and produce one anchor per matched pair."""
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

    anchors: list[CalibrationAnchor] = []
    for i, (ref_seg, tgt_seg) in enumerate(zip(ref_segs, tgt_segs)):
        if not ref_seg.peak_timestamps:
            log.warning("Anchor %d: reference segment has no peak timestamps — skipping.", i)
            continue
        if not tgt_seg.peak_timestamps:
            log.warning("Anchor %d: target segment has no peak timestamps — skipping.", i)
            continue

        ref_med_ms = float(np.median(ref_seg.peak_timestamps))
        tgt_med_ms = float(np.median(tgt_seg.peak_timestamps))
        offset_s = (ref_med_ms - tgt_med_ms) / 1000.0

        anchors.append(
            CalibrationAnchor(
                ref_timestamp_ms=ref_med_ms,
                tgt_timestamp_ms=tgt_med_ms,
                offset_s=offset_s,
            )
        )
        log.debug(
            "Anchor %d: ref=%.1f ms  tgt=%.1f ms  offset=%.4f s",
            i, ref_med_ms, tgt_med_ms, offset_s,
        )

    if not anchors:
        raise ValueError(
            f"No valid anchors could be extracted for {recording_name!r} "
            f"({ref_sensor} / {tgt_sensor})."
        )

    return sorted(anchors, key=lambda a: a.tgt_timestamp_ms)


def calibration_anchor_to_dict(
    anchor: CalibrationAnchor,
    *,
    index: int | None = None,
) -> dict[str, Any]:
    """Serialise one anchor for sync metadata."""
    data: dict[str, Any] = {
        "offset_s": round(float(anchor.offset_s), 6),
        "ref_timestamp_ms": round(float(anchor.ref_timestamp_ms), 1),
        "tgt_timestamp_ms": round(float(anchor.tgt_timestamp_ms), 1),
    }
    if index is not None:
        data["index"] = int(index)
    return data
