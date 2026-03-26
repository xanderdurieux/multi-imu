"""Copy recording-level interval labels into section folders when splitting.

Labels exported from the event labeler under ``<recording>/<stage>/`` use
``window_start_s`` / ``window_end_s`` as seconds from the **first sporsa
timestamp in that stage folder** (same origin as the labeler). When
``parser.split_sections`` cuts the recording into sections, this module
rewrites overlapping intervals in **section-relative** seconds (first sporsa
timestamp in the section CSV) and sets ``section_id`` to ``section_<n>``.

Source files: any ``labels*.csv`` in the stage directory (e.g.
``labels_intervals_*.csv``). Only rows with ``scope=interval`` and matching
``recording_id`` are considered. Empty ``section_id`` in the source file is
allowed (recording-wide intervals).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

LABEL_CSV_PREFIX_GLOB = "labels*.csv"

# Same columns as ``labels.event_labeler.LABEL_CSV_HEADER``.
INTERVAL_CSV_COLUMNS = [
    "scope",
    "recording_id",
    "section_id",
    "window_start_s",
    "window_end_s",
    "scenario_label",
    "label_source",
]


def _recording_id_aliases(recording_id: str) -> set[str]:
    """Return acceptable label CSV recording_id spellings.

    The project historically used both:
    - ``2026-02-26_r2`` (current pipeline naming)
    - ``2026-02-26_2``  (legacy naming from earlier exports)

    When transferring labels into sections we accept either spelling.
    """
    rid = recording_id.strip()
    out = {rid}
    m = re.match(r"^(.+)_r(\d+)$", rid)
    if m:
        out.add(f"{m.group(1)}_{m.group(2)}")
    m2 = re.match(r"^(.+)_([0-9]+)$", rid)
    if m2:
        out.add(f"{m2.group(1)}_r{m2.group(2)}")
    return out


def _recording_slug(recording_id: str) -> str:
    s = recording_id.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "labels"


def discover_label_csvs(stage_dir: Path) -> list[Path]:
    """Return sorted unique ``labels*.csv`` paths directly under ``stage_dir``."""
    if not stage_dir.is_dir():
        return []
    paths = sorted({p.resolve() for p in stage_dir.glob(LABEL_CSV_PREFIX_GLOB) if p.is_file()})
    return paths


def load_recording_interval_rows_for_transfer(
    stage_dir: Path,
    recording_id: str,
) -> list[dict[str, Any]]:
    """Parse interval rows that belong to ``recording_id`` (recording-relative windows)."""
    need = (
        "scope",
        "recording_id",
        "window_start_s",
        "window_end_s",
        "scenario_label",
    )
    out: list[dict[str, Any]] = []
    aliases = _recording_id_aliases(recording_id)
    for path in discover_label_csvs(stage_dir):
        try:
            df = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            log.warning("Skip label file %s: %s", path.name, exc)
            continue
        if not all(c in df.columns for c in need):
            log.debug("Skip %s: need columns %s", path.name, need)
            continue
        for _, row in df.iterrows():
            if str(row["scope"]).strip().lower() != "interval":
                continue
            rid = str(row["recording_id"]).strip()
            if rid not in aliases:
                continue
            try:
                t0 = float(row["window_start_s"])
                t1 = float(row["window_end_s"])
            except (TypeError, ValueError):
                continue
            lab = str(row["scenario_label"]).strip()
            if not lab:
                continue
            src = row.get("label_source")
            if src is None or (isinstance(src, float) and pd.isna(src)):
                src_s = "manual"
            else:
                src_s = str(src).strip() or "manual"
            out.append(
                {
                    "window_start_s": t0,
                    "window_end_s": t1,
                    "scenario_label": lab,
                    "label_source": src_s,
                }
            )
    return out


def write_section_labels_from_recording_intervals(
    *,
    recording_name: str,
    section_idx: int,
    section_dir: Path,
    recording_origin_ms: float,
    section_abs_start_ms: float,
    section_abs_end_ms: float,
    sporsa_section_df: pd.DataFrame,
    intervals: list[dict[str, Any]],
) -> Path | None:
    """Write ``labels_intervals_<slug>s<idx>.csv`` under ``section_dir``.

    Clips each interval to ``[section_abs_start_ms, section_abs_end_ms]`` in
    absolute host time, then expresses bounds in section-relative seconds
    (origin = first sporsa timestamp in ``sporsa_section_df``).
    """
    if not intervals or sporsa_section_df.empty or "timestamp" not in sporsa_section_df.columns:
        return None
    section_origin_ms = float(sporsa_section_df["timestamp"].iloc[0])
    section_id = f"section_{section_idx}"
    rows_out: list[dict[str, Any]] = []

    for it in intervals:
        w0 = float(it["window_start_s"])
        w1 = float(it["window_end_s"])
        if w1 <= w0:
            continue
        abs_0 = recording_origin_ms + w0 * 1000.0
        abs_1 = recording_origin_ms + w1 * 1000.0
        if abs_1 < section_abs_start_ms or abs_0 > section_abs_end_ms:
            continue
        c0 = max(abs_0, section_abs_start_ms)
        c1 = min(abs_1, section_abs_end_ms)
        if c1 <= c0:
            continue
        ws = (c0 - section_origin_ms) / 1000.0
        we = (c1 - section_origin_ms) / 1000.0
        rows_out.append(
            {
                "scope": "interval",
                "recording_id": recording_name,
                "section_id": section_id,
                "window_start_s": ws,
                "window_end_s": we,
                "scenario_label": it["scenario_label"],
                "label_source": it["label_source"],
            }
        )

    if not rows_out:
        log.debug(
            "No interval labels overlap %s %s in absolute time; skip label CSV",
            recording_name,
            section_id,
        )
        return None

    out_path = section_dir / f"labels_intervals_{_recording_slug(recording_name)}s{section_idx}.csv"
    pd.DataFrame(rows_out, columns=INTERVAL_CSV_COLUMNS).to_csv(out_path, index=False)
    log.info(
        "Wrote %d interval row(s) for %s → %s",
        len(rows_out),
        section_id,
        out_path.name,
    )
    return out_path
