"""Transfer recording-level interval labels to section-level label files.

When a recording is split into sections, any time-interval labels that overlap
with a section are clipped to that section's time range and written to
``<section_dir>/labels/labels.csv``.

The expected label file in the recording stage directory is named
``labels.csv`` or ``labels.json`` and uses the standard :class:`LabelRow`
schema (columns: start_ms, end_ms, label, scenario_label, …).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    parse_section_folder_name,
    project_relative_path,
    read_csv,
    recording_labels_csv,
    section_labels_csv,
    sensor_csv,
)
from .parser import LabelRow, load_labels, write_labels

log = logging.getLogger(__name__)


def load_recording_interval_rows_for_transfer(
    recording_name: str,
    *,
    recording_origin_ms: float | None = None,
) -> list[LabelRow]:
    """Load recording-level label rows for section transfer.

    Reads from the canonical path ``data/_labels/labels_intervals_<recording>.csv``.
    Returns an empty list if the file does not exist or contains no rows.
    """
    path = recording_labels_csv(recording_name)
    if not path.exists():
        return []
    rows = load_labels(path, time_origin_ms=recording_origin_ms)
    if rows:
        log.debug("Loaded %d label row(s) for %s from %s", len(rows), recording_name, path)
    return rows


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
        Absolute timestamp (ms) of the recording's first sample.
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
    section_origin_ms = float(section_abs_start_ms)
    for row in intervals:
        # Keep only intervals that overlap [section_abs_start_ms, section_abs_end_ms].
        if row.end_ms <= section_abs_start_ms or row.start_ms >= section_abs_end_ms:
            continue
        clipped_start_ms = max(row.start_ms, section_abs_start_ms)
        clipped_end_ms = min(row.end_ms, section_abs_end_ms)
        clipped_row = LabelRow(
            start_ms=clipped_start_ms,
            end_ms=clipped_end_ms,
            start_s=(clipped_start_ms - section_origin_ms) / 1000.0,
            end_s=(clipped_end_ms - section_origin_ms) / 1000.0,
            label=row.label or row.scenario_label,
            scenario_label=row.scenario_label,
            scope=row.scope,
            recording_id=recording_name,
            section_id=section_dir.name,
            label_source=row.label_source,
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

    out_path = section_labels_csv(section_dir)
    write_labels(clipped, out_path)
    log.info(
        "Wrote %d label row(s) to %s",
        len(clipped), project_relative_path(out_path),
    )


def transfer_labels_to_sections(recording_name: str) -> None:
    """Transfer recording-level labels from ``data/_labels/`` to all existing sections.

    This is a convenience function for the common workflow where the event
    labeler is used at **recording** level (section_id left empty) and the
    labels need to be propagated to each section directory afterwards.

    For each section found under ``data/sections/<recording_name>s*/``:

    1. The section's time range is read from ``calibrated/sporsa.csv`` (or
       ``sporsa.csv`` as a fallback).
    2. Recording-level label rows are clipped to that range.
    3. Clipped rows are written to ``<section_dir>/labels/labels.csv``.

    Parameters
    ----------
    recording_name:
        Recording name (e.g. ``"2026-02-26_r2"``).
    """
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'.", recording_name)
        return

    # Read origin timestamp from synced stage if present, else first section.
    recording_origin_ms: float | None = None
    try:
        df_ref = read_csv(sensor_csv(f"{recording_name}/synced", "sporsa"))
        ts = pd.to_numeric(df_ref["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            recording_origin_ms = float(ts.iloc[0])
    except Exception:
        pass

    if recording_origin_ms is None:
        for sec_dir in section_dirs:
            try:
                df_ref = read_csv(sensor_csv(sec_dir.name, "sporsa"))
                ts = pd.to_numeric(df_ref["timestamp"], errors="coerce").dropna()
                if not ts.empty:
                    recording_origin_ms = float(ts.iloc[0])
                    break
            except Exception:
                continue

    intervals = load_recording_interval_rows_for_transfer(
        recording_name,
        recording_origin_ms=recording_origin_ms,
    )

    if not intervals:
        log.warning(
            "No recording-level labels found, expected file: data/_labels/labels_intervals_%s.csv",
            recording_name,
        )
        return

    log.info(
        "Loaded %d recording-level label row(s) for '%s'.",
        len(intervals), recording_name,
    )

    for sec_dir in section_dirs:
        try:
            sporsa_path = sensor_csv(sec_dir.name, "sporsa")
        except FileNotFoundError:
            log.warning("No sporsa.csv found in %s — skipping.", sec_dir.name)
            continue

        try:
            df_sec = read_csv(sporsa_path)
            df_sec = df_sec.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df_sec.empty:
                log.warning("Empty sporsa.csv in %s — skipping.", sec_dir.name)
                continue
            ts_start = float(df_sec["timestamp"].iloc[0])
            ts_end = float(df_sec["timestamp"].iloc[-1])
        except Exception as exc:
            log.warning("Could not read %s: %s", sporsa_path, exc)
            continue

        try:
            _, sec_idx = parse_section_folder_name(sec_dir.name)
        except ValueError:
            sec_idx = 0

        write_section_labels_from_recording_intervals(
            recording_name=recording_name,
            section_idx=sec_idx,
            section_dir=sec_dir,
            recording_origin_ms=recording_origin_ms or ts_start,
            section_abs_start_ms=ts_start,
            section_abs_end_ms=ts_end,
            sporsa_section_df=df_sec,
            intervals=intervals,
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for standalone label transfer.

    Usage::

        python -m labels.section_transfer 2026-02-26_r2
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m labels.section_transfer",
        description=(
            "Transfer recording-level interval labels to all sections of a recording. "
            "Labels are read from data/_labels/labels_intervals_<recording>.csv "
            "and written (clipped) to data/sections/<recording>s<n>/labels/labels.csv."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording name (e.g. 2026-02-26_r2).",
    )
    args = parser.parse_args(argv)
    transfer_labels_to_sections(args.recording_name)


if __name__ == "__main__":
    main()
