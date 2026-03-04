"""CLI and helpers for synchronizing one target stream to one reference stream."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_dataframe

from .common import add_vector_norms, load_stream, resample_stream
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
    SyncModel,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


def _acc_norm_correlation(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    sample_rate_hz: float,
) -> float | None:
    """Pearson r of acc_norm between two streams over their overlapping time window."""
    ref_r = add_vector_norms(resample_stream(ref_df, sample_rate_hz))
    tgt_r = add_vector_norms(resample_stream(tgt_df, sample_rate_hz))

    ref_ts = ref_r["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_r["timestamp"].to_numpy(dtype=float)
    if ref_ts.size == 0 or tgt_ts.size == 0:
        return None

    lo = max(float(ref_ts[0]), float(tgt_ts[0]))
    hi = min(float(ref_ts[-1]), float(tgt_ts[-1]))
    if lo >= hi:
        return None

    ref_acc = ref_r.loc[(ref_ts >= lo) & (ref_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    tgt_acc = tgt_r.loc[(tgt_ts >= lo) & (tgt_ts <= hi), "acc_norm"].to_numpy(dtype=float)

    n = min(len(ref_acc), len(tgt_acc))
    if n < 10:
        return None

    x, y = ref_acc[:n], tgt_acc[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None

    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _compute_sync_correlations(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    model: SyncModel,
    *,
    sample_rate_hz: float,
) -> dict:
    """Correlation of acc_norm before (offset only) and after (offset + drift) sync."""
    # Before drift correction: apply only the offset (drift=0)
    offset_only_model = replace(model, drift_seconds_per_second=0.0)
    offset_only_df = apply_sync_model(tgt_df, offset_only_model, replace_timestamp=True)

    # After full sync: offset + drift
    full_model_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    corr_offset = _acc_norm_correlation(ref_df, offset_only_df, sample_rate_hz=sample_rate_hz)
    corr_full = _acc_norm_correlation(ref_df, full_model_df, sample_rate_hz=sample_rate_hz)

    return {
        "offset_only": round(corr_offset, 4) if corr_offset is not None else None,
        "offset_and_drift": round(corr_full, 4) if corr_full is not None else None,
        "signal": "acc_norm",
        "sample_rate_hz": sample_rate_hz,
    }


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
    """Synchronize target stream to reference stream and write outputs.

    Applies offset + drift correction to the target's timestamps so they align
    with the reference clock.  The target CSV is written with corrected timestamps
    and its original sensor values unchanged.

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

    # Save model JSON and augment it with correlation stats.
    sync_json_path = out_dir / f"{tgt_path.stem}_to_{ref_path.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    correlations = _compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    # Apply full sync model to target timestamps.
    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    # Drop intermediate and metadata columns that are no longer meaningful
    # after timestamp correction.
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_dataframe(aligned_df, synced_csv_path)

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
