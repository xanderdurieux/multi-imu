"""CLI entry point for the protocol-aware calibration stage.

Usage::

    python -m calibration <section_name>
    python -m calibration --recording 2026-02-26_r1
    python -m calibration --recording 2026-02-26_r1 --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.paths import parse_section_folder_name, project_relative_path, section_dir
from .pipeline import calibrate_section, calibrate_recording_sections


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m calibration",
        description=(
            "Protocol-aware IMU calibration: detects the opening routine "
            "(static + taps + static), estimates intrinsics from the pre-tap "
            "static window, and estimates alignment from the first post-mount "
            "stable window (Arduino) or the opening static window (Sporsa)."
        ),
    )
    parser.add_argument(
        "section_name",
        nargs="?",
        help="Section folder name (e.g. 2026-02-26_r1s1).",
    )
    parser.add_argument(
        "--recording",
        help="Recording name to calibrate all its sections (e.g. 2026-02-26_r1).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Approximate sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing calibration outputs.",
    )
    parser.add_argument(
        "--static-cal",
        dest="static_cal",
        default=None,
        help=(
            "Path to a static hardware calibration JSON file for the Arduino "
            "sensor (accelerometer bias/scale + gyroscope bias).  When omitted "
            "the pipeline looks for data/_calibrations/arduino_imu_calibration.json."
        ),
    )
    args = parser.parse_args(argv)

    static_cal_path = None
    if args.static_cal:
        from pathlib import Path
        static_cal_path = Path(args.static_cal)
        if not static_cal_path.exists():
            print(f"Static calibration file not found: {static_cal_path}", file=sys.stderr)
            sys.exit(1)

    if args.recording:
        results = calibrate_recording_sections(
            args.recording,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
            static_calibration_path=static_cal_path,
        )
        print(f"Calibrated {len(results)} section(s) for recording '{args.recording}'.")
    elif args.section_name:
        try:
            parse_section_folder_name(args.section_name)
        except ValueError:
            print(f"Invalid section name: {args.section_name}", file=sys.stderr)
            sys.exit(1)
        section_path = section_dir(args.section_name)
        if not section_path.exists():
            print(f"Section not found: {project_relative_path(section_path)}", file=sys.stderr)
            sys.exit(1)
        cal = calibrate_section(
            section_path,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
            static_calibration_path=static_cal_path,
        )
        q = cal.quality.get("overall", "?")
        print(f"Protocol detected: {cal.protocol_detected}")
        print(f"Overall quality: {q}")
        tags = cal.quality.get("tags", [])
        if tags:
            print(f"Tags: {', '.join(tags)}")
        for sensor, alignment in cal.alignment.items():
            print(
                f"  {sensor}: yaw_source={alignment.yaw_source}"
                f"  confidence={alignment.yaw_confidence:.3f}"
                f"  gravity_residual={alignment.gravity_residual_ms2:.3f} m/s²"
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
