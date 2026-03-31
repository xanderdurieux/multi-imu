"""Label file parsing.

Label CSV format
----------------
Columns:
    start_ms        float   Absolute timestamp (ms) of interval start.
    end_ms          float   Absolute timestamp (ms) of interval end.
    label           str     Short event/scenario label (e.g. "bump", "sprint").
    scenario_label  str     Broader scenario label (e.g. "road_section_1").
    scope           str     One of "section", "interval", "event".
    annotator       str     Annotator identifier (optional).
    confidence      float   Annotation confidence in [0, 1] (optional, default 1.0).
    ambiguous       bool    True if the annotation is uncertain (optional).
    notes           str     Free-form notes (optional).

Label JSON format
-----------------
A JSON file with a top-level key ``"labels"`` containing a list of objects
with the same field names as the CSV columns above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LabelRow:
    """A single time-interval label annotation."""

    start_ms: float
    end_ms: float
    label: str
    scenario_label: str = ""
    scope: str = "interval"
    annotator: str = ""
    confidence: float = 1.0
    ambiguous: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelRow":
        return cls(
            start_ms=float(d.get("start_ms", 0)),
            end_ms=float(d.get("end_ms", 0)),
            label=str(d.get("label", "")),
            scenario_label=str(d.get("scenario_label", "")),
            scope=str(d.get("scope", "interval")),
            annotator=str(d.get("annotator", "")),
            confidence=float(d.get("confidence", 1.0)),
            ambiguous=bool(d.get("ambiguous", False)),
            notes=str(d.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "label": self.label,
            "scenario_label": self.scenario_label,
            "scope": self.scope,
            "annotator": self.annotator,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "notes": self.notes,
        }


def load_labels(path: Path | str) -> list[LabelRow]:
    """Load labels from a CSV or JSON file.

    Returns an empty list if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".json":
        return _load_json_labels(p)
    return _load_csv_labels(p)


def _load_csv_labels(path: Path) -> list[LabelRow]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    rows: list[LabelRow] = []
    for _, row in df.iterrows():
        try:
            rows.append(LabelRow.from_dict(row.to_dict()))
        except Exception:
            continue
    return rows


def _load_json_labels(path: Path) -> list[LabelRow]:
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
            rows.append(LabelRow.from_dict(item))
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
                "start_ms", "end_ms", "label", "scenario_label",
                "scope", "annotator", "confidence", "ambiguous", "notes",
            ]
        )
    else:
        df = pd.DataFrame([r.to_dict() for r in labels])
    df.to_csv(p, index=False)
