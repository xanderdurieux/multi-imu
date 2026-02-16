"""Run a full synchronization-and-plot pipeline for one processed session."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import processed_session_dir
from common.csv_schema import load_dataframe
from plot.compare_streams import plot_stream_comparison
from plot.plot_device import plot_dataframe
from sync.sync_streams import synchronize

DEFAULT_SYNC_SAMPLE_RATE_HZ = 50.0
DEFAULT_MAX_LAG_SECONDS = 20.0
DEFAULT_RESAMPLE_RATE_HZ = 100.0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the session pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.session_pipeline",
        description=(
            "Run synchronization and generate comparison plots for one processed session."
        ),
    )
    parser.add_argument(
        "session_name",
        help="Session folder name under data/processed containing arduino.csv and sporsa.csv.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SYNC_SAMPLE_RATE_HZ,
        help=f"Alignment resampling rate (default: {DEFAULT_SYNC_SAMPLE_RATE_HZ}).",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=DEFAULT_MAX_LAG_SECONDS,
        help=f"Maximum lag search window (default: {DEFAULT_MAX_LAG_SECONDS}).",
    )
    parser.add_argument(
        "--resample-rate-hz",
        type=float,
        default=DEFAULT_RESAMPLE_RATE_HZ,
        help=(
            "Output resampling rate for synced target CSV. "
            f"Set <= 0 to disable (default: {DEFAULT_RESAMPLE_RATE_HZ})."
        ),
    )
    return parser


def run_session_pipeline(
    session_name: str,
    *,
    sample_rate_hz: float = DEFAULT_SYNC_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    resample_rate_hz: float = DEFAULT_RESAMPLE_RATE_HZ,
) -> dict[str, Path]:
    """
    Execute synchronization and plot generation for one session.

    Expected inputs in `data/processed/<session_name>/`:
    - `sporsa.csv` (reference)
    - `arduino.csv` (target)

    Generated artifacts are written into the same folder.
    """
    session_dir = processed_session_dir(session_name)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Processed session folder not found: {session_dir}")

    reference_csv = session_dir / "sporsa.csv"
    target_csv = session_dir / "arduino.csv"
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Reference stream not found: {reference_csv}")
    if not target_csv.is_file():
        raise FileNotFoundError(f"Target stream not found: {target_csv}")

    write_resampled_hz = resample_rate_hz if resample_rate_hz > 0 else None

    sync_json, synced_csv, resampled_csv = synchronize(
        reference_csv=reference_csv,
        target_csv=target_csv,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        write_resampled_hz=write_resampled_hz,
    )

    sporsa_df = load_dataframe(reference_csv)
    arduino_df = load_dataframe(target_csv)
    arduino_synced_df = load_dataframe(synced_csv)

    # Single-stream summary plots.
    sporsa_plot = session_dir / "sporsa_pipeline_overview.png"
    arduino_plot = session_dir / "arduino_pipeline_overview.png"
    arduino_synced_plot = session_dir / "arduino_synced_pipeline_overview.png"
    plot_dataframe(sporsa_df, title="Sporsa", output=sporsa_plot, magnitudes=False)
    plot_dataframe(arduino_df, title="Arduino", output=arduino_plot, magnitudes=False)
    plot_dataframe(
        arduino_synced_df,
        title="Arduino Synced",
        output=arduino_synced_plot,
        magnitudes=False,
    )

    # Comparison plots for alignment evaluation.
    raw_comparison = session_dir / "sporsa_vs_arduino_raw.png"
    synced_comparison = session_dir / "sporsa_vs_arduino_synced.png"
    synced_magnitudes = session_dir / "sporsa_vs_arduino_synced_magnitudes.png"

    plot_stream_comparison(
        sporsa_df,
        arduino_df,
        label_a="sporsa",
        label_b="arduino",
        title="Sporsa vs Arduino (Raw)",
        output=raw_comparison,
        relative_time=False,
        split_axes=False,
        magnitudes=False,
    )
    plot_stream_comparison(
        sporsa_df,
        arduino_synced_df,
        label_a="sporsa",
        label_b="arduino_synced",
        title="Sporsa vs Arduino (Synced)",
        output=synced_comparison,
        relative_time=False,
        split_axes=False,
        magnitudes=False,
    )
    plot_stream_comparison(
        sporsa_df,
        arduino_synced_df,
        label_a="sporsa",
        label_b="arduino_synced",
        title="Sporsa vs Arduino (Synced Magnitudes)",
        output=synced_magnitudes,
        relative_time=False,
        split_axes=False,
        magnitudes=True,
    )

    outputs: dict[str, Path] = {
        "sync_json": sync_json,
        "synced_csv": synced_csv,
        "sporsa_plot": sporsa_plot,
        "arduino_plot": arduino_plot,
        "arduino_synced_plot": arduino_synced_plot,
        "raw_comparison": raw_comparison,
        "synced_comparison": synced_comparison,
        "synced_magnitudes": synced_magnitudes,
    }

    if resampled_csv is not None:
        outputs["resampled_csv"] = resampled_csv

    return outputs


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the session pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    outputs = run_session_pipeline(
        session_name=args.session_name,
        sample_rate_hz=args.sample_rate_hz,
        max_lag_seconds=args.max_lag_seconds,
        resample_rate_hz=args.resample_rate_hz,
    )

    print(f"session={args.session_name}")
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
