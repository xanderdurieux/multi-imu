"""CLI entry point for the event detection stage.

Usage::

    python -m events <section_name>
    python -m events <section_name> --force
    python -m events --recording <recording_name>
    python -m events --recording <recording_name> --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import parse_section_folder_name, section_dir
from events.config import EventConfig
from events.pipeline import process_section_events, process_recording_events


def _print_summary(section_name: str, events: list) -> None:
    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1
    print(f"Section '{section_name}': {len(events)} event(s) detected.")
    if by_type:
        for etype, count in sorted(by_type.items()):
            print(f"  {etype}: {count}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m events",
        description="Detect cycling events (bumps, braking, swerves, falls) from derived IMU signals.",
    )
    parser.add_argument(
        "section_name",
        nargs="?",
        help="Section folder name (e.g. 2026-02-26_r1s1).",
    )
    parser.add_argument(
        "--recording",
        help="Recording name to process all its sections (e.g. 2026-02-26_r1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing event outputs.",
    )
    parser.add_argument(
        "--config",
        metavar="JSON_PATH",
        help="Path to a custom event_config.json to use instead of section defaults.",
    )
    args = parser.parse_args(argv)

    # Load shared config if supplied
    shared_config: EventConfig | None = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        shared_config = EventConfig.load(config_path)
        logging.getLogger(__name__).info("Loaded custom config from %s", config_path)

    if args.recording:
        results = process_recording_events(
            args.recording,
            config=shared_config,
            force=args.force,
        )
        if not results:
            print(f"No sections found for recording '{args.recording}'.", file=sys.stderr)
            sys.exit(1)
        total = sum(len(v) for v in results.values())
        print(
            f"Recording '{args.recording}': "
            f"{len(results)} section(s), {total} total event(s)."
        )
        for section_name, events in sorted(results.items()):
            _print_summary(section_name, events)

    elif args.section_name:
        try:
            recording_name, section_idx = parse_section_folder_name(args.section_name)
            section_path = section_dir(recording_name, section_idx)
        except ValueError:
            print(f"Invalid section name: {args.section_name}", file=sys.stderr)
            sys.exit(1)
        if not section_path.exists():
            print(f"Section not found: {section_path}", file=sys.stderr)
            sys.exit(1)
        events = process_section_events(section_path, config=shared_config, force=args.force)
        _print_summary(args.section_name, events)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
