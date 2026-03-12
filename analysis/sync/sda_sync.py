"""SDA-only (offset-only) recording-level synchronization.

Method 1 (of 4): estimates a coarse clock offset using Signal-Density Alignment
(SDA) across the full recording but applies no drift correction.

The simplest and fastest offline sync method in the pipeline — a useful baseline
and a fallback when recording duration is too short for reliable drift estimation.

Writes aligned outputs to ``synced/sda/``::

    synced/sda/
        sporsa.csv          ← reference copy
        arduino.csv         ← target with offset-corrected timestamps
        sync_info.json      ← fitted offset model (drift = 0)

CLI::

    python -m sync.sda_sync <recording_name>/<stage>
    python -m sync.sda_sync 2026-02-26_5/parsed
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from common import find_sensor_csv, recording_stage_dir, write_dataframe

from .align_df import estimate_offset
from .common import load_stream
from .drift_estimator import SyncModel, apply_sync_model, save_sync_model
from .metrics import compute_sync_correlations

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


def synchronize_recording_sda(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    plot: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronize two sensor streams using SDA (offset-only, no drift correction).

    Reads CSVs from ``<stage_in>/``, writes clean-named outputs to ``synced/sda/``:

    - ``synced/sda/<reference_sensor>.csv``  — reference copy
    - ``synced/sda/<target_sensor>.csv``     — target with offset-corrected timestamps
    - ``synced/sda/sync_info.json``          — offset model (drift = 0)

    Returns ``(reference_csv, synced_target_csv, sync_info_json)``.
    """
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)
    out_dir = recording_stage_dir(recording_name, "synced/sda")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced/sda] {reference_sensor} (ref) ← {ref_csv.name}")
    print(f"[{recording_name}/synced/sda] {target_sensor} (target) ← {tgt_csv.name}")

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)

    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    offset = estimate_offset(
        ref_df,
        tgt_df,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    model = SyncModel(
        reference_csv=str(ref_csv),
        target_csv=str(tgt_csv),
        target_time_origin_seconds=tgt_origin_s,
        offset_seconds=float(offset.offset_seconds),
        drift_seconds_per_second=0.0,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )

    sync_json_path = out_dir / "sync_info.json"
    save_sync_model(model, sync_json_path)

    correlations = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "sda_offset_only"
    sync_data["sda_score"] = round(float(offset.score), 4)
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    ref_out = out_dir / f"{reference_sensor}.csv"
    tgt_out = out_dir / f"{target_sensor}.csv"
    shutil.copy2(ref_csv, ref_out)
    write_dataframe(aligned_df, tgt_out)

    print(f"[{recording_name}/synced/sda] {ref_out.name}")
    print(f"[{recording_name}/synced/sda] {tgt_out.name}")
    print(f"[{recording_name}/synced/sda] {sync_json_path.name}")

    if plot:
        from visualization import plot_comparison
        stage_ref = f"{recording_name}/synced/sda"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return ref_out, tgt_out, sync_json_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.sda_sync",
        description=(
            "Synchronize two IMU streams using SDA offset-only alignment. "
            "No drift correction is applied. Writes to synced/sda/."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help="Recording name and input stage as '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').",
    )
    parser.add_argument(
        "--reference-sensor",
        default="sporsa",
        help="Reference sensor name token (default: sporsa).",
    )
    parser.add_argument(
        "--target-sensor",
        default="arduino",
        help="Target sensor name token (default: arduino).",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=DEFAULT_MAX_LAG_SECONDS,
        help=f"Maximum SDA lag search range in seconds (default: {DEFAULT_MAX_LAG_SECONDS}).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"Resampling rate for alignment signal (default: {DEFAULT_SAMPLE_RATE_HZ}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")
    recording_name, stage_in = parts
    synchronize_recording_sda(
        recording_name=recording_name,
        stage_in=stage_in,
        reference_sensor=args.reference_sensor,
        target_sensor=args.target_sensor,
        max_lag_seconds=args.max_lag_seconds,
        sample_rate_hz=args.sample_rate_hz,
    )


if __name__ == "__main__":
    main()
