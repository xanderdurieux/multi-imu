"""Parser helpers for parse, inspect, and transfer manual interval labels."""

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
        """Create an instance from a dictionary."""
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
        """Return a JSON-ready dictionary."""
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
    """Return clean str."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _maybe_float(value: Any) -> float | None:
    """Return maybe float."""
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
    """Parse bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(value)


def load_labels(path: Path | str, *, time_origin_ms: float | None = None) -> list[LabelRow]:
    """Load labels."""
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".json":
        return _load_json_labels(p, time_origin_ms=time_origin_ms)
    return _load_csv_labels(p, time_origin_ms=time_origin_ms)


def _load_csv_labels(path: Path, *, time_origin_ms: float | None = None) -> list[LabelRow]:
    """Load csv labels."""
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
    """Load json labels."""
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
