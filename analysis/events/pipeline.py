"""Event detection pipeline: per-section and per-recording entry points."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from common.paths import iter_sections_for_recording, read_csv, sections_root, write_csv
from events.config import EventConfig
from events.detectors import EventCandidate, detect_events

log = logging.getLogger(__name__)

# Derived CSV filenames expected under <section_dir>/derived/
_SPORSA_CSV = "sporsa_signals.csv"
_ARDUINO_CSV = "arduino_signals.csv"
_CROSS_CSV = "cross_sensor_signals.csv"

# Per-section override config filename
_EVENT_CONFIG_JSON = "event_config.json"


def _load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Load a derived CSV; return None and warn if missing or unreadable."""
    if not path.exists():
        log.warning("Derived CSV not found, skipping: %s (%s)", path, label)
        return None
    try:
        df = read_csv(path)
        log.debug("Loaded %s: %d rows, columns=%s", label, len(df), list(df.columns))
        return df
    except Exception as exc:
        log.warning("Failed to read %s (%s): %s", path, label, exc)
        return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _section_duration_s(
    sporsa_df: Optional[pd.DataFrame],
    arduino_df: Optional[pd.DataFrame],
    cross_df: Optional[pd.DataFrame],
) -> float:
    """Estimate section duration in seconds from the longest available DataFrame."""
    durations: list[float] = []
    for df, label in ((sporsa_df, "sporsa"), (arduino_df, "arduino"), (cross_df, "cross")):
        if df is None or len(df) == 0:
            continue
        for col in ("timestamp_ms", "time_ms", "ms", "t_ms"):
            if col in df.columns:
                ts = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(ts) >= 2:
                    durations.append((ts.iloc[-1] - ts.iloc[0]) / 1000.0)
                break
        else:
            # No timestamp column – estimate from row count at 100 Hz
            durations.append(len(df) / 100.0)
    return max(durations) if durations else 0.0


def _build_summary(
    events: list[EventCandidate],
    section_duration_s: float,
) -> dict:
    """Build the event_summary.json content."""
    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1

    return {
        "total_events": len(events),
        "by_type": by_type,
        "section_duration_s": round(section_duration_s, 3),
    }


def _candidates_to_df(events: list[EventCandidate]) -> pd.DataFrame:
    """Convert a list of EventCandidates to a DataFrame."""
    if not events:
        return pd.DataFrame(columns=[
            "event_type", "confidence", "start_ms", "end_ms",
            "peak_ms", "peak_value", "sensor", "signal_name",
            "ambiguous", "notes",
        ])
    rows = [ev.to_dict() for ev in events]
    df = pd.DataFrame(rows, columns=[
        "event_type", "confidence", "start_ms", "end_ms",
        "peak_ms", "peak_value", "sensor", "signal_name",
        "ambiguous", "notes",
    ])
    return df


def process_section_events(
    section_dir: Path,
    *,
    config: Optional[EventConfig] = None,
    force: bool = False,
) -> list[EventCandidate]:
    """Detect events for one section.

    Reads derived signals from ``<section_dir>/derived/``, writes results to
    ``<section_dir>/events/``.

    Parameters
    ----------
    section_dir:
        Path to the section directory (e.g. ``data/sections/2026-02-26_r1s1``).
    config:
        Optional pre-built config; if None, loads from
        ``<section_dir>/events/event_config.json`` or uses defaults.
    force:
        If True, overwrite existing output files.
    """
    events_dir = section_dir / "events"
    out_csv = events_dir / "event_candidates.csv"
    out_json = events_dir / "event_summary.json"

    if out_csv.exists() and out_json.exists() and not force:
        log.info(
            "Events already detected for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        # Return existing candidates
        try:
            df = read_csv(out_csv)
            events: list[EventCandidate] = []
            for _, row in df.iterrows():
                events.append(EventCandidate(
                    event_type=str(row["event_type"]),
                    confidence=float(row["confidence"]),
                    start_ms=float(row["start_ms"]),
                    end_ms=float(row["end_ms"]),
                    peak_ms=float(row["peak_ms"]),
                    peak_value=float(row["peak_value"]),
                    sensor=str(row["sensor"]),
                    signal_name=str(row["signal_name"]),
                    ambiguous=bool(row.get("ambiguous", False)),
                    notes=str(row.get("notes", "")),
                ))
            return events
        except Exception as exc:
            log.warning("Could not reload existing events from %s: %s", out_csv, exc)
            return []

    # Load per-section config override if no config provided
    if config is None:
        config_path = events_dir / _EVENT_CONFIG_JSON
        config = EventConfig.load(config_path)

    derived_dir = section_dir / "derived"

    sporsa_df = _load_csv(derived_dir / _SPORSA_CSV, "sporsa_signals")
    arduino_df = _load_csv(derived_dir / _ARDUINO_CSV, "arduino_signals")
    cross_df = _load_csv(derived_dir / _CROSS_CSV, "cross_sensor_signals")

    # At least one signal must be available
    if sporsa_df is None and arduino_df is None and cross_df is None:
        log.warning(
            "No derived CSVs found in %s/derived/ — no events detected.", section_dir.name
        )
        return []

    sporsa_df = sporsa_df if sporsa_df is not None else _empty_df()
    arduino_df = arduino_df if arduino_df is not None else _empty_df()
    cross_df = cross_df if cross_df is not None else _empty_df()

    log.info("Running event detection for %s ...", section_dir.name)
    events = detect_events(sporsa_df, arduino_df, cross_df, config)
    log.info("Detected %d event(s) in %s", len(events), section_dir.name)

    # Write outputs
    events_dir.mkdir(parents=True, exist_ok=True)

    candidates_df = _candidates_to_df(events)
    write_csv(candidates_df, out_csv)
    log.info("Wrote event candidates → %s (%d rows)", out_csv, len(candidates_df))

    duration_s = _section_duration_s(
        sporsa_df if len(sporsa_df) else None,
        arduino_df if len(arduino_df) else None,
        cross_df if len(cross_df) else None,
    )
    summary = _build_summary(events, duration_s)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote event summary → %s", out_json)

    return events


def process_recording_events(
    recording_name: str,
    *,
    config: Optional[EventConfig] = None,
    force: bool = False,
) -> dict[str, list[EventCandidate]]:
    """Detect events for all sections of a recording.

    Parameters
    ----------
    recording_name:
        Name of the recording (e.g. ``2026-02-26_r1``).
    config:
        Shared config for all sections.  If None, each section loads its own
        ``events/event_config.json`` or falls back to defaults.
    force:
        If True, overwrite existing output files in each section.

    Returns
    -------
    dict mapping section folder name -> list of EventCandidate
    """
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'", recording_name)
        return {}

    results: dict[str, list[EventCandidate]] = {}
    for sec_dir in section_dirs:
        log.info("Processing section %s ...", sec_dir.name)
        try:
            candidates = process_section_events(sec_dir, config=config, force=force)
            results[sec_dir.name] = candidates
        except Exception as exc:
            log.error("Failed to process events for %s: %s", sec_dir.name, exc)
            results[sec_dir.name] = []

    total = sum(len(v) for v in results.values())
    log.info(
        "Recording '%s': %d section(s) processed, %d total event(s)",
        recording_name, len(results), total,
    )
    return results
