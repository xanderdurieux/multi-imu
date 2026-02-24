"""
CLI entrypoint for synchronizing two processed IMU streams.

Implements SDA + LIDA synchronization pipeline: coarse alignment followed by
refined drift estimation for sub-sample precision time synchronization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import write_dataframe
from .common import load_stream, resample_to_reference_timestamps
from .drift_estimator import (
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
)

DEFAULT_SAMPLE_RATE_HZ = 50.0
DEFAULT_MAX_LAG_SECONDS = 30.0


def synchronize(reference_csv: Path, target_csv: Path, output_dir: Path):
    """
    Synchronize target stream to reference using SDA + LIDA pipeline.

    Estimates synchronization model and writes aligned target stream CSV.
    Optionally outputs uniformly resampled synchronized stream.

    Returns:
        Tuple of (sync_json_path, synced_csv_path).
    """
    ref_df = load_stream(reference_csv)
    tgt_df = load_stream(target_csv)

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(reference_csv),
        target_name=str(target_csv),
    )

    sync_json_path = output_dir / f"{target_csv.stem}_to_{reference_csv.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    # Apply synchronization model to target and then resample the aligned target
    # onto the reference timestamp series so both outputs share the same time grid.
    aligned_df = apply_sync_model(target_df=tgt_df, model=model, replace_timestamp=True)
    synced_df = resample_to_reference_timestamps(
        target_df=aligned_df,
        reference_df=ref_df,
        timestamp_col="timestamp",
    )

    write_dataframe(synced_df, output_dir / f"{target_csv.stem}_synced.csv")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for two-stream synchronization."""
    parser = argparse.ArgumentParser(
        prog="python -m sync.sync_streams",
        description="Synchronize a target IMU CSV to a reference IMU CSV.",
    )
    parser.add_argument("reference_csv", type=Path, help="Reference stream CSV (e.g. Sporsa).")
    parser.add_argument("target_csv", type=Path, help="Target stream CSV (e.g. Arduino).")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    synchronize(
        reference_csv=args.reference_csv,
        target_csv=args.target_csv,
    )


if __name__ == "__main__":
    main()
