"""CLI and helpers for synchronizing one target stream to one reference stream."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import write_dataframe

from .common import load_stream, resample_to_reference_timestamps
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
)

DEFAULT_SAMPLE_RATE_HZ = 50.0
DEFAULT_MAX_LAG_SECONDS = 30.0


def synchronize(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
) -> tuple[Path, Path, Path | None]:
    """
    Synchronize target stream to reference stream and write outputs.

    Returns:
      (sync_json_path, target_synced_csv_path, optional_uniform_resampled_csv_path)
    """
    ref_path = Path(reference_csv)
    tgt_path = Path(target_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = load_stream(ref_path)
    tgt_df = load_stream(tgt_path)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(ref_path),
        target_name=str(tgt_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    sync_json_path = out_dir / f"{tgt_path.stem}_to_{ref_path.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    # Drop intermediate alignment timestamp columns so that exported CSVs
    # keep a single, consistent timestamp column.
    drop_cols = [c for c in ("timestamp_orig", "timestamp_aligned") if c in aligned_df.columns]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    synced_df = resample_to_reference_timestamps(aligned_df, ref_df, timestamp_col="timestamp")
    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_dataframe(synced_df, synced_csv_path)

    uniform_csv_path: Path | None = None
    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(
            aligned_df,
            resample_rate_hz=float(resample_rate_hz),
            timestamp_col="timestamp",
        )
        uniform_csv_path = out_dir / f"{tgt_path.stem}_synced_uniform.csv"
        write_dataframe(uniform_df, uniform_csv_path)

    return sync_json_path, synced_csv_path, uniform_csv_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.sync_streams",
        description="Synchronize a target IMU CSV to a reference IMU CSV using the default configuration.",
    )
    parser.add_argument("reference_csv", type=Path, help="Reference stream CSV.")
    parser.add_argument("target_csv", type=Path, help="Target stream CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: <target_csv_dir>/synced).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or (args.target_csv.parent / "synced")
    sync_json, synced_csv, uniform_csv = synchronize(
        reference_csv=args.reference_csv,
        target_csv=args.target_csv,
        output_dir=output_dir,
    )
    print(f"sync_model: {sync_json}")
    print(f"target_synced_csv: {synced_csv}")
    if uniform_csv is not None:
        print(f"target_synced_uniform_csv: {uniform_csv}")


if __name__ == "__main__":
    main()
