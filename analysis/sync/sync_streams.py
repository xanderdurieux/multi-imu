"""CLI entrypoint to synchronize two processed IMU streams."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import processed_session_dir
from .common import load_stream
from .drift_estimator import (
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
)

DEFAULT_SAMPLE_RATE_HZ = 50.0
DEFAULT_MAX_LAG_SECONDS = 20.0


def _split_combined_sensor_csv(session_dir: Path) -> tuple[Path, Path] | None:
    """Split `sensor.csv` into reference/target files when both devices are mixed."""
    combined = session_dir / "sensor.csv"
    if not combined.is_file():
        return None

    df = pd.read_csv(combined)
    sensor_col = next(
        (col for col in ("sensor", "source", "device", "stream", "imu") if col in df.columns),
        None,
    )
    if sensor_col is None:
        return None

    sensor = df[sensor_col].astype(str).str.lower()

    ref_mask = (
        sensor.str.contains("sporsa")
        | sensor.str.contains("handlebar")
        | sensor.str.contains("ref")
        | sensor.str.contains("reference")
    )
    tgt_mask = (
        sensor.str.contains("arduino")
        | sensor.str.contains("rider")
        | sensor.str.contains("tgt")
        | sensor.str.contains("target")
    )

    if not ref_mask.any() or not tgt_mask.any():
        return None

    ref_csv = session_dir / "sensor_sporsa.csv"
    tgt_csv = session_dir / "sensor_arduino.csv"

    df.loc[ref_mask].copy().to_csv(ref_csv, index=False)
    df.loc[tgt_mask].copy().to_csv(tgt_csv, index=False)

    return ref_csv, tgt_csv


def _pick_session_streams(session_name: str) -> tuple[Path, Path]:
    """Resolve reference and target CSV files for a processed session."""
    session_dir = processed_session_dir(session_name)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"processed session not found: {session_dir}")

    def _prefer_csv(name: str) -> Path | None:
        exact = session_dir / f"{name}.csv"
        if exact.is_file():
            return exact

        candidates = sorted(session_dir.glob(f"*{name}*.csv"))
        for c in candidates:
            stem = c.stem.lower()
            if any(token in stem for token in ("sync", "aligned", "resampled", "synced")):
                continue
            return c
        return None

    ref = _prefer_csv("sporsa")
    tgt = _prefer_csv("arduino")

    if ref is None or tgt is None:
        split = _split_combined_sensor_csv(session_dir)
        if split is not None:
            return split
        raise FileNotFoundError(
            f"could not infer sporsa/arduino CSVs under {session_dir}; "
            "also failed to split sensor.csv automatically; "
            "provide explicit files in manual mode"
        )

    return ref, tgt


def synchronize(
    reference_csv: Path,
    target_csv: Path,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    write_resampled_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """
    Estimate synchronization and write artifacts next to the target CSV.

    Outputs:
    - `<target>_to_<reference>_sync.json`
    - `<target>_synced.csv`
    - optional resampled CSV
    """
    ref_df = load_stream(reference_csv)
    tgt_df = load_stream(target_csv)

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(reference_csv),
        target_name=str(target_csv),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
    )

    base = target_csv.parent
    sync_json = base / f"{target_csv.stem}_to_{reference_csv.stem}_sync.json"
    synced_csv = base / f"{target_csv.stem}_synced.csv"

    save_sync_model(model, sync_json)

    synced_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    synced_df.to_csv(synced_csv, index=False)

    resampled_csv: Path | None = None
    if write_resampled_hz is not None:
        resampled_df = resample_aligned_stream(
            synced_df,
            rate_hz=write_resampled_hz,
            timestamp_col="timestamp",
        )
        resampled_csv = base / f"{target_csv.stem}_synced_resampled_{write_resampled_hz:g}hz.csv"
        resampled_df.to_csv(resampled_csv, index=False)

    print(f"reference_csv={reference_csv}")
    print(f"target_csv={target_csv}")
    print(f"sync_json={sync_json}")
    print(f"synced_csv={synced_csv}")
    if resampled_csv is not None:
        print(f"resampled_csv={resampled_csv}")

    print(
        "model="
        f"offset_seconds={model.offset_seconds:.6f}, "
        f"drift_seconds_per_second={model.drift_seconds_per_second:.9f}, "
        f"scale={model.scale:.9f}, "
        f"fit_r2={model.fit_r2:.4f}, windows={model.num_windows}"
    )

    return sync_json, synced_csv, resampled_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for stream synchronization."""
    parser = argparse.ArgumentParser(
        prog="python -m sync.sync_streams",
        description="Synchronize Arduino (target) to Sporsa (reference).",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    session_parser = subparsers.add_parser(
        "session",
        help="Automatic mode: infer sporsa/arduino files from a session name.",
    )
    session_parser.add_argument("session_name", help="Session folder under data/processed.")

    pair_parser = subparsers.add_parser(
        "pair",
        help="Manual mode: provide explicit reference and target CSV paths.",
    )
    pair_parser.add_argument("reference_csv", type=Path, help="Reference stream CSV (Sporsa).")
    pair_parser.add_argument("target_csv", type=Path, help="Target stream CSV (Arduino).")

    for sub in (session_parser, pair_parser):
        sub.add_argument(
            "--sample-rate-hz",
            type=float,
            default=DEFAULT_SAMPLE_RATE_HZ,
            help=f"Resampling rate used during alignment (default: {DEFAULT_SAMPLE_RATE_HZ}).",
        )
        sub.add_argument(
            "--max-lag-seconds",
            type=float,
            default=DEFAULT_MAX_LAG_SECONDS,
            help=f"Maximum absolute lag to search (default: {DEFAULT_MAX_LAG_SECONDS}).",
        )
        sub.add_argument(
            "--resample-rate-hz",
            type=float,
            default=None,
            help="Optional output rate for a uniformly resampled synced CSV.",
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "session":
        reference_csv, target_csv = _pick_session_streams(args.session_name)
    else:
        reference_csv = args.reference_csv
        target_csv = args.target_csv

    synchronize(
        reference_csv=reference_csv,
        target_csv=target_csv,
        sample_rate_hz=args.sample_rate_hz,
        max_lag_seconds=args.max_lag_seconds,
        write_resampled_hz=args.resample_rate_hz,
    )


if __name__ == "__main__":
    main()
