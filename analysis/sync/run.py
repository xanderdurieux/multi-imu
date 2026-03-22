"""Command-line entry for the sync package: run methods, compare, and select."""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import (
    run_selection_only,
    run_selection_session,
    synchronize_recording_all_methods,
    synchronize_session,
)

_METHOD_CMDS = frozenset({
    "sda",
    "sda_sync",
    "lida",
    "lida_sync",
    "cal",
    "calibration_sync",
    "online",
    "online_sync",
})


def _normalize_method_cmd(cmd: str) -> str:
    return {
        "sda": "sda_sync",
        "sda_sync": "sda_sync",
        "lida": "lida_sync",
        "lida_sync": "lida_sync",
        "cal": "calibration_sync",
        "calibration_sync": "calibration_sync",
        "online": "online_sync",
        "online_sync": "online_sync",
    }[cmd]


def _parse_recording_stage(s: str) -> tuple[str, str]:
    parts = s.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("Expected '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').")
    return parts[0], parts[1]


def _run_method_subcommand(argv: list[str]) -> None:
    cmd = _normalize_method_cmd(argv[0])
    rest = argv[1:]
    if cmd == "sda_sync":
        p = argparse.ArgumentParser(prog="python -m sync sda_sync", add_help=True)
        p.add_argument("recording_stage", help="'<recording>/<stage>' e.g. '2026-02-26_5/parsed'")
        p.add_argument("--reference-sensor", default="sporsa")
        p.add_argument("--target-sensor", default="arduino")
        p.add_argument("--max-lag-seconds", type=float, default=60.0)
        p.add_argument("--sample-rate-hz", type=float, default=100.0)
        args = p.parse_args(rest)
        from .sda_sync import synchronize_recording_sda

        rec, stage = _parse_recording_stage(args.recording_stage)
        synchronize_recording_sda(
            rec,
            stage,
            reference_sensor=args.reference_sensor,
            target_sensor=args.target_sensor,
            max_lag_seconds=args.max_lag_seconds,
            sample_rate_hz=args.sample_rate_hz,
        )
        return

    if cmd == "lida_sync":
        p = argparse.ArgumentParser(prog="python -m sync lida_sync")
        p.add_argument("recording_stage", help="'<recording>/<stage>'")
        p.add_argument("--reference-sensor", default="sporsa")
        p.add_argument("--target-sensor", default="arduino")
        p.add_argument("--max-lag-seconds", type=float, default=60.0)
        p.add_argument("--sample-rate-hz", type=float, default=100.0)
        p.add_argument("--resample-rate-hz", type=float, default=None)
        args = p.parse_args(rest)
        from .lida_sync import synchronize_recording

        rec, stage = _parse_recording_stage(args.recording_stage)
        synchronize_recording(
            rec,
            stage,
            reference_sensor=args.reference_sensor,
            target_sensor=args.target_sensor,
            max_lag_seconds=args.max_lag_seconds,
            sample_rate_hz=args.sample_rate_hz,
            resample_rate_hz=args.resample_rate_hz,
        )
        return

    if cmd == "calibration_sync":
        p = argparse.ArgumentParser(prog="python -m sync calibration_sync")
        p.add_argument("recording_stage", help="'<recording>/<stage>'")
        p.add_argument("--reference-sensor", default="sporsa")
        p.add_argument("--target-sensor", default="arduino")
        p.add_argument("--sample-rate-hz", type=float, default=100.0)
        p.add_argument("--no-plot", action="store_true")
        args = p.parse_args(rest)
        from .calibration_sync import synchronize_recording_from_calibration

        rec, stage = _parse_recording_stage(args.recording_stage)
        synchronize_recording_from_calibration(
            rec,
            stage,
            reference_sensor=args.reference_sensor,
            target_sensor=args.target_sensor,
            sample_rate_hz=args.sample_rate_hz,
            plot=not args.no_plot,
        )
        return

    if cmd == "online_sync":
        p = argparse.ArgumentParser(prog="python -m sync online_sync")
        p.add_argument("recording_stage", help="'<recording>/<stage>'")
        p.add_argument("--drift-ppm", type=float, default=None)
        p.add_argument("--reference-sensor", default="sporsa")
        p.add_argument("--target-sensor", default="arduino")
        p.add_argument("--sample-rate-hz", type=float, default=100.0)
        p.add_argument("--plot", action="store_true")
        args = p.parse_args(rest)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        from .online_sync import synchronize_recording_online

        rec, stage = _parse_recording_stage(args.recording_stage)
        synchronize_recording_online(
            rec,
            stage,
            reference_sensor=args.reference_sensor,
            target_sensor=args.target_sensor,
            drift_ppm=args.drift_ppm,
            sample_rate_hz=args.sample_rate_hz,
            plot=args.plot,
        )
        return

    raise SystemExit(f"Unknown subcommand: {cmd}")


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        raise SystemExit(
            "Usage: python -m sync <recording> [options] | "
            "python -m sync <sda_sync|lida_sync|calibration_sync|online_sync> …\n"
            "Try: python -m sync --help"
        )

    if argv[0] in _METHOD_CMDS:
        _run_method_subcommand(argv)
        return

    parser = argparse.ArgumentParser(
        prog="python -m sync",
        description=(
            "Run SDA, LIDA, calibration, and online sync on a recording (or full session), "
            "then pick the best method and optionally copy it to synced/. "
            "Use --select-only to compare existing outputs without re-running sync. "
            "Per-method CLI: python -m sync sda_sync|lida_sync|calibration_sync|online_sync <rec>/<stage> "
            "(short aliases: sda, lida, cal, online)."
        ),
    )
    parser.add_argument(
        "name",
        help="Recording name (e.g. '2026-02-26_5') or session prefix with --all (e.g. '2026-02-26').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process all recordings whose name starts with NAME followed by '_'.",
    )
    parser.add_argument(
        "--stage",
        default="parsed",
        metavar="STAGE",
        help="Input stage directory when running sync methods (default: parsed).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["sda", "lida", "calibration", "online"],
        default=None,
        metavar="METHOD",
        help="Sync methods to run (default: all four).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the selected method's outputs to synced/.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate per-method plots and comparison overlays where applicable.",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Skip running sync methods; only load existing outputs, compare, and select.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.select_only:
        if args.all_recordings:
            run_selection_session(args.name, apply=args.apply, plot=args.plot)
        else:
            run_selection_only(args.name, apply=args.apply, plot=args.plot)
        return

    if args.all_recordings:
        synchronize_session(
            session_name=args.name,
            stage_in=args.stage,
            methods=args.methods,
            apply=args.apply,
            plot=args.plot,
        )
    else:
        synchronize_recording_all_methods(
            recording_name=args.name,
            stage_in=args.stage,
            methods=args.methods,
            apply=args.apply,
            plot=args.plot,
        )
