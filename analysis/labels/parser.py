"""Label file parsing.

Canonical label CSV format
--------------------------
Columns:
    start_ms        float   Absolute timestamp (ms) of interval start.
    end_ms          float   Absolute timestamp (ms) of interval end.
    start_s         float   Optional seconds from the folder origin used while labeling.
    end_s           float   Optional seconds from the folder origin used while labeling.
    label           str     Short event/scenario label (e.g. "bump", "sprint").
    scenario_label  str     Broader scenario label (e.g. "road_section_1").
    scope           str     One of "section", "interval", "event".
    recording_id    str     Recording identifier (optional).
    section_id      str     Section identifier (optional).
    label_source    str     Source of the label rows (optional).
    annotator       str     Annotator identifier (optional).
    confidence      float   Annotation confidence in [0, 1] (optional, default 1.0).
    ambiguous       bool    True if the annotation is uncertain (optional).
    notes           str     Free-form notes (optional).

Legacy event-labeler CSVs that store ``window_start_s`` / ``window_end_s`` are
also accepted. When ``time_origin_ms`` is provided to :func:`load_labels`, those
relative seconds are converted into absolute ``start_ms`` / ``end_ms`` values.

Label JSON format
-----------------
A JSON file with a top-level key ``"labels"`` containing a list of objects
with the same field names as the CSV columns above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import read_csv, write_csv


@dataclass
class LabelRow:
    """A single time-interval label annotation."""

    start_ms: float
    end_ms: float
    label: str
    scenario_label: str = ""
    scope: str = "interval"
    start_s: float | None = None
    end_s: float | None = None
    recording_id: str = ""
    section_id: str = ""
    label_source: str = ""
    annotator: str = ""
    confidence: float = 1.0
    ambiguous: bool = False
    notes: str = ""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        time_origin_ms: float | None = None,
    ) -> "LabelRow":
        start_s = _maybe_float(
            d.get("start_s", d.get("window_start_s"))
        )
        end_s = _maybe_float(
            d.get("end_s", d.get("window_end_s"))
        )
        start_ms = _maybe_float(d.get("start_ms"))
        end_ms = _maybe_float(d.get("end_ms"))

        if start_ms is None and start_s is not None:
            start_ms = start_s * 1000.0 + (time_origin_ms or 0.0)
        if end_ms is None and end_s is not None:
            end_ms = end_s * 1000.0 + (time_origin_ms or 0.0)

        label = _clean_str(d.get("label"))
        scenario_label = _clean_str(d.get("scenario_label"))
        if not label and scenario_label:
            label = scenario_label

        return cls(
            start_ms=float(start_ms or 0.0),
            end_ms=float(end_ms or 0.0),
            label=label,
            scenario_label=scenario_label,
            scope=_clean_str(d.get("scope")) or "interval",
            start_s=start_s,
            end_s=end_s,
            recording_id=_clean_str(d.get("recording_id")),
            section_id=_clean_str(d.get("section_id")),
            label_source=_clean_str(d.get("label_source")),
            annotator=_clean_str(d.get("annotator")),
            confidence=float(_maybe_float(d.get("confidence")) or 1.0),
            ambiguous=_parse_bool(d.get("ambiguous", False)),
            notes=_clean_str(d.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "label": self.label,
            "scenario_label": self.scenario_label,
            "scope": self.scope,
            "recording_id": self.recording_id,
            "section_id": self.section_id,
            "label_source": self.label_source,
            "annotator": self.annotator,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "notes": self.notes,
        }


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(value)


def load_labels(path: Path | str, *, time_origin_ms: float | None = None) -> list[LabelRow]:
    """Load labels from a CSV or JSON file.

    Returns an empty list if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".json":
        return _load_json_labels(p, time_origin_ms=time_origin_ms)
    return _load_csv_labels(p, time_origin_ms=time_origin_ms)


def _load_csv_labels(path: Path, *, time_origin_ms: float | None = None) -> list[LabelRow]:
    try:
        df = read_csv(path)
    except Exception:
        return []
    rows: list[LabelRow] = []
    for _, row in df.iterrows():
        try:
            rows.append(LabelRow.from_dict(row.to_dict(), time_origin_ms=time_origin_ms))
        except Exception:
            continue
    return rows


def _load_json_labels(path: Path, *, time_origin_ms: float | None = None) -> list[LabelRow]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("labels", [])
    else:
        return []

    rows: list[LabelRow] = []
    for item in items:
        try:
            rows.append(LabelRow.from_dict(item, time_origin_ms=time_origin_ms))
        except Exception:
            continue
    return rows


def write_labels(labels: list[LabelRow], path: Path | str) -> None:
    """Write labels to a CSV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        df = pd.DataFrame(
            columns=[
                "start_ms",
                "end_ms",
                "start_s",
                "end_s",
                "label",
                "scenario_label",
                "scope",
                "recording_id",
                "section_id",
                "label_source",
                "annotator",
                "confidence",
                "ambiguous",
                "notes",
            ]
        )
    else:
        df = pd.DataFrame([r.to_dict() for r in labels])
    write_csv(df, p)
