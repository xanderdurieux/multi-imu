"""Calibration anchor matching"""

from dataclasses import dataclass
import numpy as np

from common.paths import calibration_segments_json_path, read_json_file


@dataclass(frozen=True)
class CalibrationSegment:
    """A calibration segment is a segment of the recording that contains a calibration sequence."""
    segment_index: int
    start_idx: int
    end_idx: int
    start_timestamp: float
    end_timestamp: float
    n_peaks: int
    peak_indices: list[int]
    peak_timestamp: list[float]


@dataclass(frozen=True)
class CalibrationAnchor:
    """A calibration anchor is a point in time where the reference and target streams are aligned."""
    reference_timestamp: float
    target_timestamp: float
    offset: float
    score: float


def load_calibration_segments_from_json(
    recording_name: str,
    sensor: str,
) -> list[CalibrationSegment]:
    """Load calibration segments from JSON file."""
    path = calibration_segments_json_path(recording_name)
    data = read_json_file(path)
    sensor_data = data["sensors"][sensor]
    return [
        CalibrationSegment(
            segment_index=int(seg["segment_index"]),
            start_idx=int(seg["start_idx"]),
            end_idx=int(seg["end_idx"]),
            start_timestamp=float(seg["start_timestamp"]),
            end_timestamp=float(seg["end_timestamp"]),
            n_peaks=int(seg["n_peaks"]),
            peak_indices=[int(p) for p in seg["peak_indices"]],
            peak_timestamp=[float(p) for p in seg["peak_timestamp"]],
        ) 
        for seg in sensor_data.get("segments", [])
    ]

def estimate_calibration_anchors(
    recording_name: str,
) -> list[CalibrationAnchor]:
    """
    Loads calibration segments for a given recording and sensor,
    checks that all segments are present and matches them 1:1 as anchors.

    Returns:
        List of CalibrationAnchor objects with reference and target timestamps aligned.
    """
    # Load segments from JSON
    ref_segments = load_calibration_segments_from_json(recording_name, "sporsa")
    tgt_segments = load_calibration_segments_from_json(recording_name, "arduino")

    if not ref_segments or not tgt_segments:
        raise ValueError(f"No calibration segments found for {recording_name}")

    if len(ref_segments) != len(tgt_segments):
        raise ValueError(f"Mismatch between number of reference and target calibration segments for {recording_name}")

    anchors: list[CalibrationAnchor] = []
    for ref_seg, tgt_seg in zip(ref_segments, tgt_segments):
        ref_peaks_med = np.median(ref_seg.peak_timestamp)
        tgt_peaks_med = np.median(tgt_seg.peak_timestamp)

        offset = ref_peaks_med - tgt_peaks_med

        anchors.append(CalibrationAnchor(
            reference_timestamp=ref_peaks_med,
            target_timestamp=tgt_peaks_med,
            offset=offset,
            score=1
        ))

    return anchors


if __name__ == "__main__":
    anchors = estimate_calibration_anchors(recording_name="2026-02-26_r3")
    for anchor in anchors:
        print(getattr,)
        print(getattr(anchor, "target_timestamp"))
        print(getattr(anchor, "offset"))
        print(getattr(anchor, "score"))