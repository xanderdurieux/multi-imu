"""CLI for running individual pipeline stages.

Usage::

    python -m pipeline <stage> [target] [options]

Stages: calibration, orientation, derived, events, features, exports

Examples::

    python -m pipeline calibration 2026-02-26_r1s1
    python -m pipeline calibration --recording 2026-02-26_r1
    python -m pipeline calibration --recording 2026-02-26_r1 --frame gravity_plus_forward
    python -m pipeline orientation 2026-02-26_r1s1 --filter madgwick
    python -m pipeline derived --recording 2026-02-26_r1
    python -m pipeline events 2026-02-26_r1s1
    python -m pipeline features --recording 2026-02-26_r1 --window 2.0 --hop 1.0
    python -m pipeline exports --quality good
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import iter_sections_for_recording, parse_section_folder_name, section_dir
from visualization.stage_plots import plot_section_pipeline_stage


def _section_dir(name: str) -> Path:
    try:
        parse_section_folder_name(name)
    except ValueError:
        print(f"Invalid section name: {name}", file=sys.stderr)
        sys.exit(1)
    d = section_dir(name)
    if not d.exists():
        print(f"Section not found: {d}", file=sys.stderr)
        sys.exit(1)
    return d


def _run_stage_plots_for_sections(recording: str, stage: str) -> None:
    """Generate stage plots for every section in a recording."""
    for section_dir in iter_sections_for_recording(recording):
        plot_section_pipeline_stage(section_dir, stage)


def _run_calibration(args: argparse.Namespace) -> None:
    from calibration.pipeline import calibrate_section, calibrate_recording_sections
    if args.recording:
        cals = calibrate_recording_sections(
            args.recording,
            frame_alignment=args.frame,
            force=args.force,
        )
        print(f"Calibrated {len(cals)} section(s).")
        if not args.no_plots:
            _run_stage_plots_for_sections(args.recording, "calibration")
    else:
        d = _section_dir(args.target)
        cal = calibrate_section(d, frame_alignment=args.frame, force=args.force)
        print(f"Quality: {cal.calibration_quality}")
        if not args.no_plots:
            plot_section_pipeline_stage(d, "calibration")


def _run_orientation(args: argparse.Namespace) -> None:
    from orientation.pipeline import process_section_orientation, process_recording_orientation

    if args.recording:
        results = process_recording_orientation(args.recording, force=args.force)
        print(f"Processed {len(results)} section(s).")
        if not args.no_plots:
            _run_stage_plots_for_sections(args.recording, "orientation")
    else:
        d = _section_dir(args.target)
        process_section_orientation(d, force=args.force)
        print("Done.")
        if not args.no_plots:
            plot_section_pipeline_stage(d, "orientation")


def _run_derived(args: argparse.Namespace) -> None:
    from derived.pipeline import process_section_derived, process_recording_derived
    if args.recording:
        ok_list = process_recording_derived(args.recording, force=args.force)
        print(f"Processed {sum(ok_list)}/{len(ok_list)} section(s).")
    else:
        d = _section_dir(args.target)
        ok = process_section_derived(d, force=args.force)
        print("OK" if ok else "FAILED")


def _run_events(args: argparse.Namespace) -> None:
    from events.pipeline import process_section_events, process_recording_events
    from events.config import EventConfig
    cfg = EventConfig.load(Path(args.event_config)) if args.event_config else None
    if args.recording:
        process_recording_events(args.recording, config=cfg, force=args.force)
        print("Done.")
    else:
        d = _section_dir(args.target)
        events = process_section_events(d, config=cfg, force=args.force)
        print(f"Detected {len(events)} event(s).")


def _run_features(args: argparse.Namespace) -> None:
    from features.pipeline import process_section_features, process_recording_features
    if args.recording:
        df = process_recording_features(
            args.recording,
            window_s=args.window,
            hop_s=args.hop,
            force=args.force,
        )
        print(f"Extracted {len(df)} feature window(s).")
        if not args.no_plots:
            _run_stage_plots_for_sections(args.recording, "features")
    else:
        d = _section_dir(args.target)
        df = process_section_features(
            d, window_s=args.window, hop_s=args.hop, force=args.force
        )
        print(f"Extracted {len(df)} feature window(s).")
        if not args.no_plots:
            plot_section_pipeline_stage(d, "features")


def _run_exports(args: argparse.Namespace) -> None:
    from exports.pipeline import run_exports
    recordings = [args.recording] if args.recording else None
    paths = run_exports(recordings, min_quality_label=args.quality, force=args.force)
    for name, p in paths.items():
        print(f"  {name}: {p}")


_STAGE_RUNNERS = {
    "calibration": _run_calibration,
    "orientation": _run_orientation,
    "derived": _run_derived,
    "events": _run_events,
    "features": _run_features,
    "exports": _run_exports,
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Run individual pipeline stages.",
    )
    parser.add_argument(
        "stage",
        choices=list(_STAGE_RUNNERS.keys()),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="Section name (e.g. 2026-02-26_r1s1) for single-section mode.",
    )
    parser.add_argument("--recording", help="Recording name to process all its sections.")
    parser.add_argument(
        "--frame",
        default="gravity_only",
        choices=["gravity_only", "gravity_plus_forward"],
        help="[calibration] Frame alignment mode.",
    )
    parser.add_argument(
        "--event-config",
        default="",
        help="[events] Path to event_config.json.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="[features] Window length in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=1.0,
        help="[features] Hop length in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--quality",
        default="marginal",
        choices=["good", "marginal", "poor"],
        help="[exports] Minimum quality tier (default: marginal).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip stage visualization plots.",
    )

    args = parser.parse_args(argv)

    if not args.target and not args.recording and args.stage != "exports":
        print("Provide a section name or --recording.", file=sys.stderr)
        sys.exit(1)

    runner = _STAGE_RUNNERS[args.stage]
    try:
        runner(args)
    except SystemExit:
        raise
    except Exception as exc:
        logging.exception("Stage %s failed: %s", args.stage, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
