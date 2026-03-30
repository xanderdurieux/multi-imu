"""Load manual labels from CSV/JSON and resolve them onto windows/events.

Resolution precedence for feature windows (most specific wins):
``interval`` → ``section`` → ``recording``.

CSV/JSON fields (required unless noted):

- ``scope``: ``recording`` | ``section`` | ``interval`` | ``event``
- ``recording_id``
- ``section_id`` (required for ``section``/``interval``/``event``)
- ``window_start_s``/``window_end_s`` (required for ``interval``)
- ``event_id`` (preferred for ``event`` when matching events)
- ``event_type``/``event_time_s`` (fallback event matching keys)
- ``scenario_label``
- ``label_source`` (optional)

Provenance metadata (all optional):
- ``labeler``
- ``labeled_at_utc`` (ISO 8601 string)
- ``label_confidence`` (0..1)
- ``label_notes``
- ``label_status`` (for example: ``confirmed``, ``unknown``, ``ambiguous``, ``mixed``)
- ``label_schema_version``
- ``suggestion_source``
- ``suggestion_rank``

JSON format is a list of objects with the same fields.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Scope = Literal["recording", "section", "interval", "event"]

PROVENANCE_FIELDS = [
    "labeler",
    "labeler_role",
    "labeled_at_utc",
    "label_confidence",
    "label_notes",
    "label_status",
    "ambiguity_label",
    "ambiguity_notes",
    "provenance_source",
    "source_artifact",
    "annotation_template_id",
    "annotation_example_id",
    "annotation_id",
    "annotation_batch_id",
    "double_label_group",
    "adjudication_status",
    "label_schema_version",
    "suggestion_source",
    "suggestion_rank",
]


@dataclass
class LabelRule:
    """One label rule from file."""

    scope: Scope
    recording_id: str
    section_id: str | None
    window_start_s: float | None
    window_end_s: float | None
    event_id: str | None
    event_type: str | None
    event_time_s: float | None
    scenario_label: str
    label_source: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelIndex:
    """Fast lookup for (recording, section, window interval) + event tables."""

    rules: list[LabelRule] = field(default_factory=list)

    def resolve(
        self,
        recording_id: str,
        section_id: str,
        window_start_s: float,
        window_end_s: float,
    ) -> tuple[str, str]:
        """Backward-compatible tuple return: ``(scenario_label, label_source)``."""
        meta = self.resolve_window_metadata(
            recording_id=recording_id,
            section_id=section_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )
        return str(meta["scenario_label"]), str(meta["label_source"])

    def resolve_window_metadata(
        self,
        *,
        recording_id: str,
        section_id: str,
        window_start_s: float,
        window_end_s: float,
    ) -> dict[str, Any]:
        """Resolve label+provenance for one feature window."""
        best: LabelRule | None = None
        best_rank = -1

        for r in self.rules:
            if r.recording_id != recording_id:
                continue
            rank = 0
            if r.scope == "recording" and r.section_id is None:
                rank = 1
            elif r.scope == "section" and r.section_id == section_id:
                rank = 2
            elif r.scope == "interval" and r.section_id == section_id:
                if r.window_start_s is None or r.window_end_s is None:
                    continue
                if window_end_s <= r.window_start_s or window_start_s >= r.window_end_s:
                    continue
                rank = 3
            else:
                continue

            if rank > best_rank:
                best_rank = rank
                best = r

        if best is None:
            return _blank_metadata()
        return _metadata_from_rule(best)

    def apply_to_events(
        self,
        events_df: pd.DataFrame,
        *,
        event_match_tolerance_s: float = 0.5,
    ) -> pd.DataFrame:
        """Attach labels/provenance to an event table.

        Event matching precedence:
        1. ``event_id`` exact match (if present in both rule and row).
        2. ``event_type`` + ``event_time_s`` within tolerance (same section+recording).
        """
        if events_df.empty:
            out = events_df.copy()
            _ensure_label_columns(out)
            return out

        out = events_df.copy()
        _ensure_label_columns(out)

        for idx, row in out.iterrows():
            rec = str(row.get("recording_id", "")).strip()
            sec = str(row.get("section_id", "")).strip()
            if not rec:
                continue
            event_id = str(row.get("event_id", "")).strip() or None
            event_type = str(row.get("event_type", "")).strip() or None
            event_time_s = _as_float(row.get("time_s"))
            if event_time_s is None:
                event_time_s = _as_float(row.get("event_time_s"))

            best: LabelRule | None = None
            best_rank = -1
            for r in self.rules:
                if r.scope != "event":
                    continue
                if r.recording_id != rec:
                    continue
                if r.section_id and sec and r.section_id != sec:
                    continue

                rank = 0
                if event_id and r.event_id and event_id == r.event_id:
                    rank = 3
                elif event_type and r.event_type and event_type == r.event_type:
                    if event_time_s is not None and r.event_time_s is not None:
                        if abs(event_time_s - r.event_time_s) <= event_match_tolerance_s:
                            rank = 2
                if rank > best_rank:
                    best_rank = rank
                    best = r

            if best is None:
                continue
            meta = _metadata_from_rule(best)
            for k, v in meta.items():
                out.at[idx, k] = v

        return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _blank_metadata() -> dict[str, Any]:
    base = {
        "scenario_label": "",
        "label_source": "none",
        "label_scope": "none",
    }
    for f in PROVENANCE_FIELDS:
        base[f] = "" if f != "label_confidence" else np.nan
    return base


def _metadata_from_rule(rule: LabelRule) -> dict[str, Any]:
    out = _blank_metadata()
    out["scenario_label"] = rule.scenario_label
    out["label_source"] = rule.label_source
    out["label_scope"] = rule.scope
    for f in PROVENANCE_FIELDS:
        if f in rule.provenance:
            out[f] = rule.provenance[f]
    return out


def _ensure_label_columns(df: pd.DataFrame) -> None:
    base_cols = ["scenario_label", "label_source", "label_scope", *PROVENANCE_FIELDS]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan if c == "label_confidence" else ""


def _row_to_rule(row: dict[str, Any]) -> LabelRule:
    scope = str(row.get("scope", "")).strip().lower()
    if scope not in ("recording", "section", "interval", "event"):
        raise ValueError(f"Invalid scope {scope!r} in row {row!r}")

    rec = str(row.get("recording_id", "")).strip()
    if not rec:
        raise ValueError(f"Missing recording_id in row {row!r}")

    sec_s = _clean_optional_text(row.get("section_id"))
    wsf = _as_float(row.get("window_start_s"))
    wef = _as_float(row.get("window_end_s"))

    event_id = _clean_optional_text(row.get("event_id"))
    event_type = _clean_optional_text(row.get("event_type"))
    event_time_s = _as_float(row.get("event_time_s"))

    label = str(row.get("scenario_label", "")).strip()
    if not label:
        raise ValueError(f"Missing scenario_label in row {row!r}")

    src = str(row.get("label_source", "manual")).strip() or "manual"

    if scope in ("section", "interval", "event") and not sec_s:
        raise ValueError(f"section_id required for scope={scope} in row {row!r}")
    if scope == "interval" and (wsf is None or wef is None):
        raise ValueError(f"window_start_s/window_end_s required for interval in row {row!r}")
    if scope == "event" and not event_id and not (event_type and event_time_s is not None):
        raise ValueError(
            "event scope requires event_id or (event_type + event_time_s) "
            f"in row {row!r}"
        )

    prov: dict[str, Any] = {}
    for f in PROVENANCE_FIELDS:
        if f not in row:
            continue
        val = row.get(f)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if f == "label_confidence":
            fv = _as_float(val)
            if fv is not None:
                prov[f] = float(np.clip(fv, 0.0, 1.0))
        elif f == "suggestion_rank":
            try:
                prov[f] = int(val)
            except (TypeError, ValueError):
                continue
        else:
            sval = str(val).strip()
            if sval:
                prov[f] = sval

    return LabelRule(
        scope=scope,  # type: ignore[arg-type]
        recording_id=rec,
        section_id=sec_s,
        window_start_s=wsf,
        window_end_s=wef,
        event_id=event_id,
        event_type=event_type,
        event_time_s=event_time_s,
        scenario_label=label,
        label_source=src,
        provenance=prov,
    )


def load_labels_from_path(path: Path | str) -> LabelIndex:
    """Load labels from a ``.csv`` or ``.json`` path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON labels must be a list of objects")
        rows = data
    else:
        raise ValueError(f"Unsupported label format: {suffix} (use .csv or .json)")

    skipped = 0
    rules: list[LabelRule] = []
    for r in rows:
        raw = dict(r)
        lab = raw.get("scenario_label")
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            skipped += 1
            continue
        if str(lab).strip() == "":
            skipped += 1
            continue
        rules.append(_row_to_rule(raw))
    if skipped:
        log.info("Skipped %d row(s) with empty scenario_label in %s", skipped, path)
    log.info("Loaded %d label rule(s) from %s", len(rules), path)
    return LabelIndex(rules=rules)


def warn_unlabeled_windows(
    df: pd.DataFrame,
    *,
    recording_col: str = "recording_id",
    section_col: str = "section_id",
) -> int:
    """Log a warning for rows with empty ``scenario_label``; return count."""
    _ = recording_col, section_col
    if "scenario_label" not in df.columns:
        return 0
    empty = df["scenario_label"].isna() | (df["scenario_label"].astype(str).str.strip() == "")
    n = int(empty.sum())
    if n:
        log.warning(
            "%d feature window(s) have no scenario_label (of %d total)",
            n,
            len(df),
        )
    return n
