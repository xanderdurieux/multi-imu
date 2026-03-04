"""Recording-level synchronization using the SDA + LIDA method.

Reads ``<stage_in>/sporsa.csv`` (reference) and ``<stage_in>/arduino.csv`` (target),
writes aligned outputs to ``synced_lida/``::

    synced_lida/
        sporsa.csv          ← reference copy
        arduino.csv         ← target with corrected timestamps
        sync_info.json      ← fitted offset + drift model

See also ``sync.calibration_sync`` for the calibration-sequence based variant
that writes to ``synced_cal/``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import find_sensor_csv, recording_stage_dir
from visualization.plot_session import plot_recording

from .sync_streams import (
    DEFAULT_MAX_LAG_SECONDS,
    DEFAULT_SAMPLE_RATE_HZ,
    synchronize,
)
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
)


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

    # Run synchronization into a temporary output dir, then rename to clean names.
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

    plot_recording(recording_name, stage_filter="synced_lida")

    return ref_out, tgt_out, sync_json_out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.session",
        description=(
            "Synchronize two IMU streams for one recording. "
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
