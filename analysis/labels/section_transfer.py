"""Transfer recording-level interval labels to section-level label files.

When a recording is split into sections, any time-interval labels that overlap
with a section are clipped to that section's time range and written to
``<section_dir>/labels.csv``.

The expected label file in the recording stage directory is named
``labels.csv`` or ``labels.json`` and uses the standard :class:`LabelRow`
schema (columns: start_ms, end_ms, label, scenario_label, …).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .parser import LabelRow, load_labels, write_labels

log = logging.getLogger(__name__)


def load_recording_interval_rows_for_transfer(
    recording_stage_dir: Path,
    recording_name: str,
) -> list[LabelRow]:
    """Load label rows from a recording stage directory for section transfer.

    Looks for ``labels.csv`` or ``labels.json`` inside *recording_stage_dir*.
    Falls back to ``data/sessions/<session>/labels.csv`` if none found in the
    stage directory.

    Returns an empty list if no label file is found.
    """
    candidates = [
        recording_stage_dir / "labels.csv",
        recording_stage_dir / "labels.json",
    ]
    # Also look in recording root (sibling of the stage dir).
    rec_root = recording_stage_dir.parent
    candidates += [
        rec_root / "labels.csv",
        rec_root / "labels.json",
    ]

    for path in candidates:
        if path.exists():
            rows = load_labels(path)
            if rows:
                log.debug(
                    "Loaded %d label row(s) for %s from %s",
                    len(rows), recording_name, path,
                )
                return rows

    return []


def write_section_labels_from_recording_intervals(
    *,
    recording_name: str,
    section_idx: int,
    section_dir: Path,
    recording_origin_ms: float,
    section_abs_start_ms: float,
    section_abs_end_ms: float,
    sporsa_section_df: pd.DataFrame,
    intervals: list[LabelRow],
) -> None:
    """Clip and write recording-level interval labels to a section directory.

    Parameters
    ----------
    recording_name:
        Name of the recording (for logging).
    section_idx:
        Section index (for logging).
    section_dir:
        Directory where ``labels.csv`` will be written.
    recording_origin_ms:
        Absolute timestamp (ms) of the recording's first sample — used to
        convert any relative timestamps if needed (not currently used, but
        reserved for future relative-time label formats).
    section_abs_start_ms:
        Absolute start of this section in ms.
    section_abs_end_ms:
        Absolute end of this section in ms.
    sporsa_section_df:
        The reference sensor's DataFrame for this section (used to get the
        exact actual time range if needed).
    intervals:
        Recording-level :class:`LabelRow` objects to transfer.
    """
    if not intervals:
        return

    clipped: list[LabelRow] = []
    for row in intervals:
        # Keep only intervals that overlap [section_abs_start_ms, section_abs_end_ms].
        if row.end_ms <= section_abs_start_ms or row.start_ms >= section_abs_end_ms:
            continue
        clipped_row = LabelRow(
            start_ms=max(row.start_ms, section_abs_start_ms),
            end_ms=min(row.end_ms, section_abs_end_ms),
            label=row.label,
            scenario_label=row.scenario_label,
            scope=row.scope,
            annotator=row.annotator,
            confidence=row.confidence,
            ambiguous=row.ambiguous,
            notes=row.notes,
        )
        clipped.append(clipped_row)

    if not clipped:
        log.debug(
            "No labels overlap section %s s%d [%.0f, %.0f]",
            recording_name, section_idx, section_abs_start_ms, section_abs_end_ms,
        )
        return

    out_path = section_dir / "labels.csv"
    write_labels(clipped, out_path)
    log.info(
        "Wrote %d label row(s) to %s",
        len(clipped), out_path,
    )
