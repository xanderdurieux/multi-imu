"""End-to-end pipeline: parse → sync → sections → calibrate → orientation → features → exports → QC.

Run from the ``analysis`` directory::

    uv run python -m pipeline --session 2026-02-26 --sync-method calibration \\
        --orientation-filter complementary_orientation --labels path/to/labels.csv

Omit ``--session`` when ``parsed/`` already exists for all recordings under
``--recordings-dir``. Use ``--force`` to re-run steps even when outputs exist.

Example (recordings only, no raw parse)::

    uv run python -m pipeline --sync-method lida --orientation-filter complementary_orientation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


@dataclass
class SectionStatus:
    """Per-section step outcomes."""

    section_id: str
    steps: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RecordingStatus:
    """Per-recording pipeline status."""

    recording_id: str
    steps: dict[str, str] = field(default_factory=dict)
    sections: list[SectionStatus] = field(default_factory=list)
    error: str | None = None


def _recordings_list(recordings_root: Path) -> list[str]:
    return sorted(
        d.name
        for d in recordings_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

def run_pipeline(
    *,
    session: str | None = None,
    only_recordings: frozenset[str] | None = None,
    force: bool = False,
    sync_mode: Literal["best", "sda", "lida", "calibration", "online"] = "calibration",
    split_stage: str = "synced",
    no_plots: bool = False,
    orientation_filter: str = "complementary_orientation",
    labels_path: Path | None = None,
    frame_alignment: str = "gravity_only",
    skip_exports: bool = False,
) -> list[RecordingStatus]:
    """Execute pipeline steps for all recordings. Returns status objects (also written to JSON/CSV).

    If *only_recordings* is set, only those folder names under the recordings root are processed.
    """
    from common.paths import recordings_root
    from common.paths import iter_sections_for_recording, parse_section_folder_name
    from common.paths import recording_name_prefix_for_session, sections_root as _sections_root

    root = recordings_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Recordings root not found: {root}")

    label_index = None
    if labels_path is not None:
        from labels.parser import load_labels_from_path

        label_index = load_labels_from_path(Path(labels_path))

    # --- parse (optional, whole session) ---
    if session:
        from parser.session import process_session

        need_parse = force
        if not need_parse:
            session_prefix = recording_name_prefix_for_session(session)
            for name in _recordings_list(root):
                if not name.startswith(session_prefix):
                    continue
                if not (root / name / "parsed" / "sporsa.csv").is_file():
                    need_parse = True
                    break
        if need_parse:
            log.info("Parsing session %s", session)
            process_session(session)
        else:
            log.info("Skipping parse (outputs exist, use --force to redo)")

    recordings = _recordings_list(root)
    if only_recordings:
        want = frozenset(only_recordings)
        not_found = sorted(want - set(recordings))
        if not_found:
            log.warning("Requested --recording not found under recordings root: %s", not_found)
        recordings = [r for r in recordings if r in want]
    statuses: list[RecordingStatus] = []

    from calibration.calibrate import calibrate_section
    from derived.compute import derive_section_signals
    from features.extract import extract_section
    from orientation.estimate import estimate_section
    from parser.split_sections import split_recording
    from sync.pipeline import (
        synchronize_recording_all_methods,
        synchronize_recording_chosen_method,
    )
    from validation.comprehensive import write_section_qc

    for rec_id in recordings:
        rs = RecordingStatus(recording_id=rec_id)
        rpath = root / rec_id
        try:
            # sync
            sync_done = (rpath / "synced" / "sync_info.json").is_file()
            if force or not sync_done:
                log.info("[%s] Synchronizing (%s)", rec_id, sync_mode)
                if sync_mode == "best":
                    synchronize_recording_all_methods(rec_id)
                else:
                    synchronize_recording_chosen_method(rec_id, sync_mode, quiet=True)
                rs.steps["sync"] = "ok"
            else:
                rs.steps["sync"] = "skipped"

            # split sections
            sec_root = _sections_root()
            has_sections = any(
                d.is_dir() and d.name.startswith(f"{rec_id}s")
                for d in sec_root.iterdir()
            ) if sec_root.exists() else False
            if force or not has_sections:
                log.info("[%s] Splitting sections from stage %s", rec_id, split_stage)
                split_recording(
                    rec_id,
                    stage=split_stage,
                    plot=not no_plots,
                    sync=True,
                )
                rs.steps["split"] = "ok"
            else:
                rs.steps["split"] = "skipped"

            sync_method = _read_selected_sync_method(rpath)

            for sdir in iter_sections_for_recording(rec_id):
                _rec, sec_idx = parse_section_folder_name(sdir.name)
                section_id = sdir.name
                ss = SectionStatus(section_id=section_id)
                try:
                    # calibrate
                    cal_marker = sdir / "calibrated" / "calibration.json"
                    if force or not cal_marker.is_file():
                        calibrate_section(
                            sdir.resolve(),
                            write_plots=not no_plots,
                            frame_alignment=frame_alignment,
                        )
                        ss.steps["calibrate"] = "ok"
                    else:
                        ss.steps["calibrate"] = "skipped"

                    # orientation
                    orient_marker = sdir / "orientation" / "orientation_stats.json"
                    if force or not orient_marker.is_file():
                        estimate_section(
                            sdir.resolve(),
                            write_plots=not no_plots,
                            variants=[orientation_filter],
                        )
                        ss.steps["orientation"] = "ok"
                    else:
                        ss.steps["orientation"] = "skipped"

                    # derived signals
                    derived_meta = sdir / "derived" / "derived_signals_meta.json"
                    if force or not derived_meta.is_file():
                        derive_section_signals(
                            sdir.resolve(),
                            orientation_variant=orientation_filter,
                            include_normalized=True,
                        )
                        ss.steps["derived"] = "ok"
                    else:
                        ss.steps["derived"] = "skipped"

                    # features
                    feat_csv = sdir / "features" / "features.csv"
                    if force or not feat_csv.is_file():
                        extract_section(
                            sdir,
                            section_id,
                            write_plots=not no_plots,
                            orientation_variant=orientation_filter,
                            label_index=label_index,
                            sync_method=sync_method,
                            recording_id=rec_id,
                            section_id=section_id,
                        )
                        from labels.parser import warn_unlabeled_windows

                        import pandas as pd

                        warn_unlabeled_windows(pd.read_csv(feat_csv))
                        ss.steps["features"] = "ok"
                    else:
                        ss.steps["features"] = "skipped"

                    write_section_qc(
                        sdir.resolve(),
                        orientation_variant=orientation_filter,
                    )
                    ss.steps["qc"] = "ok"
                except Exception as exc:
                    ss.error = str(exc)
                    log.exception("[%s/%s] step failed", rec_id, sdir.name)
                rs.sections.append(ss)

        except Exception as exc:
            rs.error = str(exc)
            log.exception("[%s] recording-level failure", rec_id)

        statuses.append(rs)

    if not skip_exports:
        from features.exports import export_qc_summaries, export_thesis_feature_tables

        export_thesis_feature_tables(recordings_root_path=root)
        export_qc_summaries(recordings_root_path=root)

    _write_run_summary(root, statuses)
    return statuses


def _read_selected_sync_method(rpath: Path) -> str:
    p = rpath / "synced" / "all_methods.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return str(data.get("selected_method", "") or "")
    except Exception:
        return ""


def _write_run_summary(recordings_root: Path, statuses: list[RecordingStatus]) -> None:
    out_json = recordings_root.parent / "pipeline_run_summary.json"
    out_csv = recordings_root.parent / "pipeline_run_summary_sections.csv"

    payload: dict[str, Any] = {
        "recordings": [],
    }
    csv_rows: list[dict[str, Any]] = []

    for rs in statuses:
        rec_entry: dict[str, Any] = {
            "recording_id": rs.recording_id,
            "steps": rs.steps,
            "error": rs.error,
            "sections": [],
        }
        for ss in rs.sections:
            rec_entry["sections"].append(
                {
                    "section_id": ss.section_id,
                    "steps": ss.steps,
                    "error": ss.error,
                }
            )
            csv_rows.append(
                {
                    "recording_id": rs.recording_id,
                    "section_id": ss.section_id,
                    "recording_error": rs.error or "",
                    "section_error": ss.error or "",
                    **{f"step_{k}": v for k, v in ss.steps.items()},
                }
            )
        payload["recordings"].append(rec_entry)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if csv_rows:
        keys = sorted({k for row in csv_rows for k in row})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    log.info("Wrote %s and %s", out_json, out_csv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Dual-IMU thesis preprocessing pipeline (parse → sync → … → features → exports).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session date folder under data/sessions/ to parse before sync (optional).",
    )
    parser.add_argument(
        "--recording",
        action="append",
        dest="recordings",
        metavar="ID",
        help=(
            "Process only this recording folder (repeat flag for several), "
            "e.g. --recording 2026-02-26_r2"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist.")
    parser.add_argument(
        "--sync-method",
        choices=("best", "sda", "lida", "calibration", "online"),
        default="calibration",
        help="Synchronization method, or 'best' to run all four and auto-select.",
    )
    parser.add_argument(
        "--split-stage",
        default="synced",
        help="Recording stage passed to split_sections (default: synced).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable diagnostic plot generation.")
    parser.add_argument(
        "--orientation-filter",
        default="complementary_orientation",
        help="Single orientation algorithm to run (default: complementary_orientation).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels CSV/JSON (see labels/labels_template.csv).",
    )
    parser.add_argument(
        "--frame-alignment",
        choices=("gravity_only", "gravity_plus_forward", "section_horizontal_frame"),
        default="gravity_only",
        help="Ride-level frame: gravity only, per-sensor forward yaw, or section-level reference frame.",
    )
    parser.add_argument("--skip-exports", action="store_true", help="Skip consolidated export CSVs.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    only = frozenset(args.recordings) if args.recordings else None
    run_pipeline(
        session=args.session,
        only_recordings=only,
        force=args.force,
        sync_mode=args.sync_method,
        split_stage=args.split_stage,
        no_plots=args.no_plots,
        orientation_filter=args.orientation_filter,
        labels_path=args.labels,
        frame_alignment=args.frame_alignment,
        skip_exports=args.skip_exports,
    )


if __name__ == "__main__":
    main()
