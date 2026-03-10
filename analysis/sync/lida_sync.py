"""SDA + LIDA recording-level synchronization.

Method 2 (of 4): estimates clock offset and drift by combining a global
Signal-Density Alignment (SDA) coarse pass with a Local Instance-based Drift
Analysis (LIDA) refinement pass that slides a window over the recording.

Reads ``<stage_in>/sporsa.csv`` (reference) and ``<stage_in>/arduino.csv``
(target), writes aligned outputs to ``synced_lida/``::

    synced_lida/
        sporsa.csv          ← reference copy
        arduino.csv         ← target with corrected timestamps
        sync_info.json      ← fitted offset + drift model

CLI::

    python -m sync.lida_sync <recording_name>/<stage>
    python -m sync.lida_sync 2026-02-26_5/parsed
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common import find_sensor_csv, recording_stage_dir, write_dataframe

from .common import load_stream
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    SyncModel,
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
)
from .metrics import compute_sync_correlations

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


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
    lowpass_cutoff_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """Synchronize target stream to reference stream using SDA + LIDA.

    Applies offset + drift correction to the target's timestamps so they align
    with the reference clock.  The target CSV is written with corrected timestamps
    and its original sensor values unchanged.

    Returns:
      ``(sync_json_path, target_synced_csv_path, optional_uniform_resampled_csv_path)``
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
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    sync_json_path = out_dir / f"{tgt_path.stem}_to_{ref_path.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    correlations = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "sda_lida"
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
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


def synchronize_recording(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    plot: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronize two sensor streams for one recording using SDA + LIDA.

    Reads CSVs from ``<stage_in>/``, writes clean-named outputs to ``synced_lida/``:

    - ``synced_lida/<reference_sensor>.csv``  — reference copy
    - ``synced_lida/<target_sensor>.csv``     — target with corrected timestamps
    - ``synced_lida/sync_info.json``          — offset + drift model

    Returns ``(reference_csv, synced_target_csv, sync_info_json)``.
    """
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)

    out_dir = recording_stage_dir(recording_name, "synced_lida")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced_lida] {reference_sensor} (ref) ← {ref_csv.name}")
    print(f"[{recording_name}/synced_lida] {target_sensor} (target) ← {tgt_csv.name}")

    tmp_dir = out_dir / "_tmp"
    try:
        sync_json_raw, synced_csv_raw, uniform_csv_raw = synchronize(
            reference_csv=ref_csv,
            target_csv=tgt_csv,
            output_dir=tmp_dir,
            sample_rate_hz=sample_rate_hz,
            max_lag_seconds=max_lag_seconds,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
            local_search_seconds=local_search_seconds,
            resample_rate_hz=resample_rate_hz,
            use_acc=use_acc,
            use_gyro=use_gyro,
            use_mag=use_mag,
        )

        ref_out = out_dir / f"{reference_sensor}.csv"
        tgt_out = out_dir / f"{target_sensor}.csv"
        sync_json_out = out_dir / "sync_info.json"

        shutil.copy2(ref_csv, ref_out)
        shutil.move(str(synced_csv_raw), tgt_out)
        shutil.move(str(sync_json_raw), sync_json_out)

        if uniform_csv_raw is not None:
            uniform_out = out_dir / f"{target_sensor}_uniform.csv"
            shutil.move(str(uniform_csv_raw), uniform_out)
            print(f"[{recording_name}/synced_lida] {uniform_out.name}")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    print(f"[{recording_name}/synced_lida] {ref_out.name}")
    print(f"[{recording_name}/synced_lida] {tgt_out.name}")
    print(f"[{recording_name}/synced_lida] {sync_json_out.name}")

    if plot:
        from visualization import plot_comparison
        stage_ref = f"{recording_name}/synced_lida"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return ref_out, tgt_out, sync_json_out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.lida_sync",
        description=(
            "Synchronize two IMU streams for one recording using SDA + LIDA. "
            "Default: sporsa as reference, arduino as target."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help="Recording name and input stage as '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').",
    )
    parser.add_argument(
        "--reference-sensor",
        default="sporsa",
        help="Sensor name token for the reference stream (default: sporsa).",
    )
    parser.add_argument(
        "--target-sensor",
        default="arduino",
        help="Sensor name token for the target stream (default: arduino).",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=DEFAULT_MAX_LAG_SECONDS,
        help=f"Maximum coarse lag search range in seconds (default: {DEFAULT_MAX_LAG_SECONDS}).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"Resampling rate for alignment signal (default: {DEFAULT_SAMPLE_RATE_HZ}).",
    )
    parser.add_argument(
        "--resample-rate-hz",
        type=float,
        default=None,
        help="If set, also write a uniformly resampled synced target CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")

    recording_name, stage_in = parts
    synchronize_recording(
        recording_name=recording_name,
        stage_in=stage_in,
        reference_sensor=args.reference_sensor,
        target_sensor=args.target_sensor,
        max_lag_seconds=args.max_lag_seconds,
        sample_rate_hz=args.sample_rate_hz,
        resample_rate_hz=args.resample_rate_hz,
    )


if __name__ == "__main__":
    main()
