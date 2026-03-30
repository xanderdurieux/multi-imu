"""Lightweight labeling workflow for section/window/event annotations.

Commands (run from ``analysis/``):

1) Section-level scaffold:
   ``uv run python -m labels.workflow scaffold-sections 2026-02-26_r1 --out labels/sections.csv``

2) Event suggestions from detected candidates:
   ``uv run python -m labels.workflow suggest-events data/sections/2026-02-26_r1s1 --out labels/events_suggested.csv``

3) Join event labels back to event_candidates table:
   ``uv run python -m labels.workflow apply-event-labels data/sections/2026-02-26_r1s1 --labels labels/events_labeled.csv``
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    parse_section_folder_name,
    recording_dir,
    sections_root,
)
from labels.parser import load_labels_from_path

DEFAULT_TAXONOMY = Path(__file__).resolve().parent / "taxonomy_thesis_scenarios.json"

EVENT_TO_SCENARIO_DEFAULT = {
    "bump_shock_candidate": "surface_disturbance",
    "braking_burst": "hard_braking",
    "swerve_roll_rate_candidate": "rapid_swerve",
    "rider_bicycle_divergence": "rider_bicycle_mismatch",
    "fall_or_bicycle_drop_candidate": "fall_or_drop",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_sections(target: str) -> list[Path]:
    t = Path(target)
    if t.is_dir() and (t / "derived").is_dir():
        return [t.resolve()]

    # section folder short name
    sec_candidate = sections_root() / target
    if sec_candidate.is_dir():
        return [sec_candidate.resolve()]

    # recording name
    rec_dir = recording_dir(target)
    if rec_dir.is_dir():
        return list(iter_sections_for_recording(target, stage="calibrated"))

    raise FileNotFoundError(f"Could not resolve section/recording target: {target}")


def scaffold_sections(target: str, out_path: Path, *, labeler: str = "", source: str = "manual") -> Path:
    rows: list[dict[str, Any]] = []
    for sec_path in _resolve_sections(target):
        sec_id = sec_path.name
        rec_id, _ = parse_section_folder_name(sec_id)
        rows.append(
            {
                "scope": "section",
                "recording_id": rec_id,
                "section_id": sec_id,
                "window_start_s": "",
                "window_end_s": "",
                "event_id": "",
                "event_type": "",
                "event_time_s": "",
                "scenario_label": "unknown",
                "label_source": source,
                "labeler": labeler,
                "labeler_role": "primary",
                "labeled_at_utc": "",
                "label_confidence": 0.5,
                "label_notes": "",
                "label_status": "unknown",
                "ambiguity_label": "clear",
                "ambiguity_notes": "",
                "provenance_source": "manual_review",
                "source_artifact": "",
                "annotation_template_id": "thesis_section_v1",
                "annotation_example_id": "",
                "annotation_id": f"{sec_id}__section",
                "annotation_batch_id": "",
                "double_label_group": "single",
                "adjudication_status": "not_required",
                "label_schema_version": "thesis_v1",
                "suggestion_source": "",
                "suggestion_rank": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def suggest_event_labels(
    target: str,
    out_path: Path,
    *,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
    min_confidence: float = 0.0,
    labeler: str = "",
    source: str = "event_candidate_suggestion",
) -> Path:
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8")) if taxonomy_path.is_file() else {}
    custom_map = taxonomy.get("event_candidate_defaults", {}) if isinstance(taxonomy, dict) else {}
    event_map = {**EVENT_TO_SCENARIO_DEFAULT, **custom_map}

    rows: list[dict[str, Any]] = []
    for sec_path in _resolve_sections(target):
        event_csv = sec_path / "events" / "event_candidates.csv"
        if not event_csv.is_file():
            continue
        df = pd.read_csv(event_csv)
        if df.empty:
            continue
        if "confidence" in df.columns:
            df = df[pd.to_numeric(df["confidence"], errors="coerce") >= min_confidence]

        for rank, (_, r) in enumerate(df.sort_values("confidence", ascending=False).iterrows(), start=1):
            event_type = str(r.get("event_type", "")).strip()
            scenario = event_map.get(event_type, "ambiguous")
            conf = float(pd.to_numeric(r.get("confidence"), errors="coerce") or 0.0)
            rows.append(
                {
                    "scope": "event",
                    "recording_id": str(r.get("recording_id", "")).strip(),
                    "section_id": str(r.get("section_id", "")).strip(),
                    "window_start_s": "",
                    "window_end_s": "",
                    "event_id": str(r.get("event_id", "")).strip(),
                    "event_type": event_type,
                    "event_time_s": float(pd.to_numeric(r.get("time_s"), errors="coerce")),
                    "scenario_label": scenario,
                    "label_source": source,
                    "labeler": labeler,
                    "labeler_role": "primary",
                    "labeled_at_utc": "",
                    "label_confidence": round(max(0.2, min(conf, 0.95)), 3),
                    "label_notes": "candidate suggestion; human review required",
                    "label_status": "suggested",
                    "ambiguity_label": "review_required",
                    "ambiguity_notes": "",
                    "provenance_source": "event_candidate",
                    "source_artifact": "events/event_candidates.csv",
                    "annotation_template_id": "thesis_event_v1",
                    "annotation_example_id": "",
                    "annotation_id": str(r.get("event_id", "")).strip() or f"{str(r.get('section_id', '')).strip()}__{rank}",
                    "annotation_batch_id": "",
                    "double_label_group": "single",
                    "adjudication_status": "not_required",
                    "label_schema_version": "thesis_v1",
                    "suggestion_source": "events.event_candidates",
                    "suggestion_rank": rank,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def apply_event_labels(section_path: Path | str, labels_path: Path | str) -> Path:
    sec = Path(section_path)
    event_csv = sec / "events" / "event_candidates.csv"
    if not event_csv.is_file():
        raise FileNotFoundError(f"event_candidates.csv missing at {event_csv}")
    events_df = pd.read_csv(event_csv)
    labels = load_labels_from_path(labels_path)
    out_df = labels.apply_to_events(events_df)
    out_path = sec / "events" / "event_candidates_labeled.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def build_double_label_assignments(
    annotations_path: Path | str,
    out_path: Path,
    *,
    reviewer_b: str,
    fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """Create explicit second-rater assignment rows for a thesis-scale subset."""
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0.0, 1.0]")
    df = pd.read_csv(annotations_path)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    rnd = random.Random(seed)
    keys = df.get("annotation_id")
    if keys is None:
        keys = df.apply(lambda r: f"{r.get('scope','')}::{r.get('recording_id','')}::{r.get('section_id','')}", axis=1)
    candidates = sorted(str(x) for x in keys.dropna().astype(str).unique().tolist() if x.strip())
    n_pick = max(1, int(round(len(candidates) * fraction))) if candidates else 0
    picked = set(rnd.sample(candidates, k=min(n_pick, len(candidates)))) if n_pick else set()

    out_df = df.copy()
    out_df["double_label_group"] = out_df.apply(
        lambda r: "double" if str(r.get("annotation_id", "")).strip() in picked else "single",
        axis=1,
    )
    out_df["adjudication_status"] = out_df["double_label_group"].map(
        lambda g: "pending_second_label" if g == "double" else "not_required"
    )
    out_df["second_labeler"] = out_df["double_label_group"].map(lambda g: reviewer_b if g == "double" else "")
    out_df["assignment_generated_at_utc"] = _utc_now_iso()
    out_df["assignment_seed"] = seed
    out_df["assignment_fraction"] = fraction
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Section/window/event labeling workflow utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sec = sub.add_parser("scaffold-sections", help="Generate section-level label scaffold CSV")
    p_sec.add_argument("target", type=str, help="Recording id, section id, or section path")
    p_sec.add_argument("--out", type=Path, required=True)
    p_sec.add_argument("--labeler", type=str, default="")
    p_sec.add_argument("--source", type=str, default="manual")

    p_evt = sub.add_parser("suggest-events", help="Generate event-label suggestions from event candidates")
    p_evt.add_argument("target", type=str, help="Recording id, section id, or section path")
    p_evt.add_argument("--out", type=Path, required=True)
    p_evt.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    p_evt.add_argument("--min-confidence", type=float, default=0.0)
    p_evt.add_argument("--labeler", type=str, default="")
    p_evt.add_argument("--source", type=str, default="event_candidate_suggestion")

    p_apply = sub.add_parser("apply-event-labels", help="Apply event-level labels to event_candidates.csv")
    p_apply.add_argument("section", type=Path, help="Section directory")
    p_apply.add_argument("--labels", type=Path, required=True)

    p_double = sub.add_parser(
        "double-label-subset",
        help="Select and export a small explicit second-rater subset for inter-rater agreement.",
    )
    p_double.add_argument("--annotations", type=Path, required=True, help="Primary annotation CSV")
    p_double.add_argument("--out", type=Path, required=True, help="Output assignment CSV")
    p_double.add_argument("--reviewer-b", type=str, required=True, help="Second reviewer id")
    p_double.add_argument("--fraction", type=float, default=0.2, help="Subset fraction [0,1], default 0.2")
    p_double.add_argument("--seed", type=int, default=42)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "scaffold-sections":
        out = scaffold_sections(args.target, args.out, labeler=args.labeler, source=args.source)
        print(f"wrote {out}")
    elif args.cmd == "suggest-events":
        out = suggest_event_labels(
            args.target,
            args.out,
            taxonomy_path=args.taxonomy,
            min_confidence=args.min_confidence,
            labeler=args.labeler,
            source=args.source,
        )
        print(f"wrote {out}")
    elif args.cmd == "apply-event-labels":
        out = apply_event_labels(args.section, args.labels)
        print(f"wrote {out}")
    elif args.cmd == "double-label-subset":
        out = build_double_label_assignments(
            args.annotations,
            args.out,
            reviewer_b=args.reviewer_b,
            fraction=args.fraction,
            seed=args.seed,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
