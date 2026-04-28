"""Split synced IMU recordings into per-section directories.

Sections are defined by pre-detected calibration sequences on the reference
sensor.  This stage only reads saved outputs from the upstream pipeline:

- Sensor streams from ``<recording>/<stage>/`` (default: ``synced``, where
  both sensors already share the reference timeline).
- Calibration segments from ``<recording>/parsed/calibration_segments.json``.

No signals are re-processed and no sync is re-run here.  Each section spans
from the start of one calibration to the end of the next, so both the
opening and closing calibrations are included.

Outputs land under::

    data/sections/<recording_name>s<section_idx>/
        sporsa.csv
        arduino.csv

CLI usage::

    python -m parser.split_sections 2026-02-26_r5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from common.paths import read_csv, recording_stage_dir, section_dir, sensor_csv, write_csv
from labels.section_transfer import (
    load_recording_interval_rows_for_transfer,
    write_section_labels_from_recording_intervals,
)
from parser.calibration_segments import (
    CalibrationSegment,
    clip_segments_to_range,
    load_calibration_segments_from_json,
    write_section_calibration_segments,
)
from sync.model import apply_linear_time_transform

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration segment helpers
# ---------------------------------------------------------------------------


def _load_sync_params(recording_name: str) -> dict[str, float] | None:
    """Load Arduino→synced linear transform params from sync_info.json."""
    path = recording_stage_dir(recording_name, "synced") / "sync_info.json"
    if not path.exists():
        return None
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "offset_seconds": float(data["offset_seconds"]),
            "drift_seconds_per_second": float(data["drift_seconds_per_second"]),
            "target_origin_seconds": float(data["target_time_origin_seconds"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("Could not parse sync_info.json for %s: %s", recording_name, exc)
        return None


def _transform_segments(
    segs: list[CalibrationSegment],
    sync_params: dict[str, float],
) -> list[CalibrationSegment]:
    """Map segment timestamps from raw Arduino millis to the synced frame."""
    if not segs:
        return segs
    import numpy as np
    starts = apply_linear_time_transform(
        [s.start_ms for s in segs], **sync_params
    )
    ends = apply_linear_time_transform(
        [s.end_ms for s in segs], **sync_params
    )
    drift_factor = 1.0 + float(sync_params["drift_seconds_per_second"])
    out = []
    for seg, s_ms, e_ms in zip(segs, starts, ends):
        peaks = apply_linear_time_transform(seg.peak_ms, **sync_params).tolist() if seg.peak_ms else []
        out.append(CalibrationSegment(
            start_ms=float(s_ms),
            end_ms=float(e_ms),
            peak_ms=peaks,
            peak_strength=seg.peak_strength,
            static_pre_ms=seg.static_pre_ms * drift_factor,
            static_post_ms=seg.static_post_ms * drift_factor,
        ))
    return out


def _build_section_calibration_segments(
    recording_name: str,
    sensors: list[str],
    reference_sensor: str,
    reference_segs_open: CalibrationSegment | None,
    reference_segs_close: CalibrationSegment | None,
    sync_params: dict[str, float] | None,
    t_min: float,
    t_max: float,
) -> dict[str, list[CalibrationSegment]]:
    """Build per-section calibration segments for all sensors.

    The reference sensor segments are the two bounding calibrations (already
    on the synced timeline). Non-reference sensors have their recording-level
    segments loaded, sync-transformed where needed, and clipped to [t_min, t_max].
    Pass None for either reference segment when no calibration protocol exists.
    """
    result: dict[str, list[CalibrationSegment]] = {}

    for sensor in sensors:
        if sensor == reference_sensor:
            result[sensor] = [s for s in (reference_segs_open, reference_segs_close) if s is not None]
            continue

        try:
            segs = load_calibration_segments_from_json(recording_name, sensor)
        except Exception as exc:
            log.warning("Could not load calibration segments for %s/%s: %s", recording_name, sensor, exc)
            result[sensor] = []
            continue

        if sync_params is not None:
            segs = _transform_segments(segs, sync_params)
        else:
            log.warning("%s/%s: no sync params — calibration segments kept in raw timestamps", recording_name, sensor)

        result[sensor] = clip_segments_to_range(segs, t_min, t_max)

    return result


# ---------------------------------------------------------------------------
# Section plotting
# ---------------------------------------------------------------------------


def _plot_section(section_path: Path, sensors: list[str]) -> None:
    """Generate sensor and comparison plots for one section directory."""
    from visualization.plot_comparison import plot_comparison_data
    from visualization.plot_sensor import plot_sensor_data

    for sensor_name in sensors:
        csv_path = section_path / f"{sensor_name}.csv"
        if not csv_path.exists():
            continue
        try:
            plot_sensor_data(csv_path)
            plot_sensor_data(
                csv_path,
                norm_only=True,
                acc_only=True,
                output_path=section_path / f"{sensor_name}_norm_acc.png",
            )
        except Exception as exc:
            log.warning("Plotting failed for %s/%s: %s", section_path.name, sensor_name, exc)
    try:
        plot_comparison_data(section_path)
    except Exception as exc:
        log.warning("Comparison plotting failed for %s: %s", section_path.name, exc)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _load_sensor_streams(
    recording_name: str,
    stage: str,
    sensors: list[str],
) -> dict[str, pd.DataFrame]:
    """Read each sensor's CSV from ``<recording>/<stage>/``.

    Missing sensors are logged and skipped.  Rows with NaN timestamps are
    dropped and the result is sorted chronologically.
    """
    streams: dict[str, pd.DataFrame] = {}
    for sensor in sensors:
        try:
            csv_path = sensor_csv(f"{recording_name}/{stage}", sensor)
        except FileNotFoundError:
            log.warning("Sensor '%s' not found in %s/%s — skipping", sensor, recording_name, stage)
            continue
        df = read_csv(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        streams[sensor] = df
    return streams


def split_recording(
    recording_name: str,
    stage: str = "synced",
    sensors: list[str] | None = None,
    *,
    reference_sensor: str = "sporsa",
    label_set: str | None = None,
    plot: bool = True,
) -> list[Path]:
    """Split a recording into sections using saved sync and parse outputs.

    Parameters
    ----------
    recording_name:
        Recording folder name (e.g. ``"2026-02-26_r5"``).
    stage:
        Input stage to read sensor CSVs from.  Default ``"synced"``; both
        sensors there share the reference timeline.
    sensors:
        Sensors to split.  Defaults to ``["sporsa", "arduino"]``.  The
        reference sensor is prepended automatically if missing.
    reference_sensor:
        Sensor whose saved calibration segments define section boundaries.
    label_set:
        Label-set directory under ``data/_labels``. Defaults to the active
        label set from ``MULTI_IMU_LABEL_SET`` or ``v1``.
    plot:
        If ``True``, generate sensor and comparison plots for every section.

    Returns the list of written CSV paths in chronological order.
    """
    if sensors is None:
        sensors = ["sporsa", "arduino"]
    if reference_sensor not in sensors:
        sensors = [reference_sensor] + sensors

    sensor_dfs = _load_sensor_streams(recording_name, stage, sensors)
    if reference_sensor not in sensor_dfs:
        raise FileNotFoundError(
            f"Reference sensor '{reference_sensor}' missing in {recording_name}/{stage}"
        )

    available_sensors = list(sensor_dfs)
    ref_df = sensor_dfs[reference_sensor]

    calibrations = load_calibration_segments_from_json(recording_name, reference_sensor)
    sync_params = _load_sync_params(recording_name)

    recording_origin_ms = float(ref_df["timestamp"].iloc[0])
    source_interval_rows = load_recording_interval_rows_for_transfer(
        recording_name,
        recording_origin_ms=recording_origin_ms,
        label_set=label_set,
    )
    if source_interval_rows:
        log.info(
            "Label transfer: loaded %d recording-level interval row(s) for %s",
            len(source_interval_rows),
            recording_name,
        )

    def _write_section(
        section_idx: int,
        sensor_slices: dict[str, pd.DataFrame],
        ts_start: float,
        ts_end: float,
        c_open: CalibrationSegment,
        c_close: CalibrationSegment,
    ) -> list[Path]:
        section_path = section_dir(f"{recording_name}s{section_idx}")
        section_path.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for sensor, df in sensor_slices.items():
            out_path = section_path / f"{sensor}.csv"
            write_csv(df, out_path)
            paths.append(out_path)
            log.info("Wrote %s/%s (%d rows)", section_path.name, out_path.name, len(df))

        cal_segs = _build_section_calibration_segments(
            recording_name=recording_name,
            sensors=available_sensors,
            reference_sensor=reference_sensor,
            reference_segs_open=c_open,
            reference_segs_close=c_close,
            sync_params=sync_params,
            t_min=ts_start,
            t_max=ts_end,
        )
        write_section_calibration_segments(section_path, cal_segs)

        if source_interval_rows and reference_sensor in sensor_slices:
            write_section_labels_from_recording_intervals(
                recording_name=recording_name,
                section_idx=section_idx,
                section_dir=section_path,
                recording_origin_ms=recording_origin_ms,
                section_abs_start_ms=ts_start,
                section_abs_end_ms=ts_end,
                sporsa_section_df=sensor_slices[reference_sensor],
                intervals=source_interval_rows,
            )
        if plot:
            _plot_section(section_path, available_sensors)
        return paths

    if len(calibrations) < 2:
        log.warning(
            "Fewer than 2 calibration sequences found for %s/%s "
            "(%d found) — saving full streams as first section",
            recording_name, reference_sensor, len(calibrations),
        )
        ts_full_0 = float(ref_df["timestamp"].iloc[0])
        ts_full_1 = float(ref_df["timestamp"].iloc[-1])
        return _write_section(1, sensor_dfs, ts_full_0, ts_full_1, None, None)

    written: list[Path] = []
    for i, (c_open, c_close) in enumerate(zip(calibrations, calibrations[1:]), start=1):
        ts_start = c_open.start_ms
        ts_end = c_close.end_ms

        slices: dict[str, pd.DataFrame] = {}
        for sensor, df in sensor_dfs.items():
            mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            slices[sensor] = df.loc[mask].reset_index(drop=True)

        written.extend(_write_section(i, slices, ts_start, ts_end, c_open, c_close))

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.split_sections",
        description=(
            "Split a synced recording into per-section directories using the "
            "calibration segments saved under parsed/.  Each section spans "
            "from the start of one calibration to the end of the next and is "
            "saved under data/sections/<recording_name>s<section_idx>/."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording name (e.g. '2026-02-26_r5').",
    )
    parser.add_argument(
        "--stage",
        default="synced",
        help="Input stage to read sensor CSVs from (default: synced).",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["sporsa", "arduino"],
        metavar="SENSOR",
        help="Sensor names to split (default: sporsa arduino).",
    )
    parser.add_argument(
        "--reference-sensor",
        default="sporsa",
        help="Sensor whose saved calibration segments define sections (default: sporsa).",
    )
    parser.add_argument(
        "--label-set",
        default=None,
        help="Label set directory under data/_labels (default: MULTI_IMU_LABEL_SET or v1).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots for each section.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    written = split_recording(
        recording_name=args.recording_name,
        stage=args.stage,
        sensors=args.sensors,
        reference_sensor=args.reference_sensor,
        label_set=args.label_set,
        plot=not args.no_plot,
    )

    print(f"\n{len(written)} file(s) written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
