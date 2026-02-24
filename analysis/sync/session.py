"""Session-level synchronization pipeline.

CLI:
    uv run -m sync.session <session_name>

This command:
- Locates reference/target IMU CSVs in the parsed stage for the session.
- Runs SDA + LIDA synchronization using ``sync_streams.synchronize``.
- Writes synced (and optionally resampled) CSVs into the ``synced/`` stage
  for the same session.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from common import parsed_session_dir, synced_session_dir, find_sensor_csv

from .sync_streams import DEFAULT_MAX_LAG_SECONDS, DEFAULT_SAMPLE_RATE_HZ, synchronize


def synchronize_session(
    session_name: str,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    resample_rate_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """Run synchronization for a parsed session and write outputs to the synced stage."""
    reference_csv = find_sensor_csv(session_name, "parsed", "sporsa")
    target_csv = find_sensor_csv(session_name, "parsed", "arduino")
    out_dir = synced_session_dir(session_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference CSV into the synced stage for convenience.
    shutil.copy2(reference_csv, out_dir / reference_csv.name)

    return synchronize(
        reference_csv=reference_csv,
        target_csv=target_csv,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        resample_rate_hz=resample_rate_hz,
        output_dir=out_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for session-level synchronization."""
    parser = argparse.ArgumentParser(
        prog="python -m sync.session",
        description="Synchronize Arduino (target) to Sporsa (reference) for one session.",
    )
    parser.add_argument(
        "session_name",
        help="Session name; expects parsed CSVs under data/<session_name>/parsed/.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"Resampling rate used during alignment (default: {DEFAULT_SAMPLE_RATE_HZ}).",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=DEFAULT_MAX_LAG_SECONDS,
        help=f"Maximum absolute lag to search (default: {DEFAULT_MAX_LAG_SECONDS}).",
    )
    parser.add_argument(
        "--resample-rate-hz",
        type=float,
        default=None,
        help="Optional output rate for a uniformly resampled synced CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    synchronize_session(
        session_name=args.session_name,
        sample_rate_hz=args.sample_rate_hz,
        max_lag_seconds=args.max_lag_seconds,
        resample_rate_hz=args.resample_rate_hz,
    )


if __name__ == "__main__":
    main()

