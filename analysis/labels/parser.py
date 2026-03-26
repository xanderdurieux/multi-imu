"""Load manual labels from CSV or JSON and resolve them onto feature windows.

Resolution order (most specific wins): interval → section → recording.

CSV columns (header required):

- ``scope``: ``recording`` | ``section`` | ``interval``
- ``recording_id``: e.g. ``2026-02-26_r5``
- ``section_id``: e.g. ``2026-02-26_r5s1`` (required for ``section`` and ``interval``)
- ``window_start_s``, ``window_end_s``: section-relative seconds for ``interval`` rows
- ``scenario_label``: class name / scenario string
- ``label_source`` (optional): free text, e.g. ``manual_v1``

JSON: a list of objects with the same field names.

Interval bounds are treated as half-open ``[start, end)`` in section-relative seconds,
consistent with feature ``window_start_s`` / ``window_end_s`` (inclusive-exclusive
matching uses overlap: window overlaps interval if not (win_end <= int_start or win_start >= int_end)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

log = logging.getLogger(__name__)

Scope = Literal["recording", "section", "interval"]


@dataclass
class LabelRule:
    """One label rule from file."""

    scope: Scope
    recording_id: str
    section_id: str | None
    window_start_s: float | None
    window_end_s: float | None
    scenario_label: str
    label_source: str


@dataclass
class LabelIndex:
    """Fast lookup for (recording, section, window interval)."""

    rules: list[LabelRule] = field(default_factory=list)

    def resolve(
        self,
        recording_id: str,
        section_id: str,
        window_start_s: float,
        window_end_s: float,
    ) -> tuple[str, str]:
        """Return ``(scenario_label, label_source)``; empty label if unknown."""
        best_label = ""
        best_source = "none"
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
                # overlap [ws, we) style: treat window as [start, end]
                if window_end_s <= r.window_start_s or window_start_s >= r.window_end_s:
                    continue
                rank = 3
            else:
                continue

            if rank > best_rank:
                best_rank = rank
                best_label = r.scenario_label
                best_source = r.label_source

        return best_label, best_source


def _row_to_rule(row: dict[str, Any]) -> LabelRule:
    scope = str(row.get("scope", "")).strip().lower()
    if scope not in ("recording", "section", "interval"):
        raise ValueError(f"Invalid scope {scope!r} in row {row!r}")

    rec = str(row.get("recording_id", "")).strip()
    if not rec:
        raise ValueError(f"Missing recording_id in row {row!r}")

    sec = row.get("section_id")
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        sec_s = None
    else:
        sec_s = str(sec).strip() or None

    ws = row.get("window_start_s")
    we = row.get("window_end_s")
    wsf = float(ws) if ws is not None and str(ws).strip() != "" else None
    wef = float(we) if we is not None and str(we).strip() != "" else None

    label = str(row.get("scenario_label", "")).strip()
    if not label:
        raise ValueError(f"Missing scenario_label in row {row!r}")

    src = str(row.get("label_source", "manual")).strip() or "manual"

    if scope in ("section", "interval") and not sec_s:
        raise ValueError(f"section_id required for scope={scope} in row {row!r}")
    if scope == "interval" and (wsf is None or wef is None):
        raise ValueError(f"window_start_s/window_end_s required for interval in row {row!r}")

    return LabelRule(
        scope=scope,  # type: ignore[arg-type]
        recording_id=rec,
        section_id=sec_s,
        window_start_s=wsf,
        window_end_s=wef,
        scenario_label=label,
        label_source=src,
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
