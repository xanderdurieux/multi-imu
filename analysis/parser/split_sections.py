"""Detect calibration sequences and split IMU recordings into sections.

A calibration sequence has the pattern:
  ~5 s static  →  ~5 acceleration peaks  →  ~5 s static

This module finds those sequences in the reference sensor, then splits all
configured sensors into matching time windows saved under::

    data/sections/
        <recording_name>s<section_idx>/
            sporsa.csv
            arduino.csv
            …

CLI usage::

    python -m parser.split_sections 2026-02-26_r5/synced
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import read_csv, section_dir, sensor_csv, write_csv
from parser.calibration_segments import CalibrationSegment, find_calibration_segments
from labels.section_transfer import (
    load_recording_interval_rows_for_transfer,
    write_section_labels_from_recording_intervals,
)

log = logging.getLogger(__name__)

_GRAVITY_MS2 = 9.81


# ---------------------------------------------------------------------------
# Section plotting and splitting
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


def split_recording(
    recording_name: str,
    stage: str = "synced",
    sensors: list[str] | None = None,
    *,
    reference_sensor: str = "sporsa",
    sample_rate_hz: float = 100.0,
    plot: bool = True,
    sync: bool = True,
) -> list[Path]:
    """Split a recording into sections for all sensors, grouped per section.

    Calibration sequences are detected in ``reference_sensor``.  Each section
    spans from the start of one calibration to the end of the next, so both
    the opening and closing calibrations are included.

    Outputs are saved to::

        data/sections/<recording_name>s<section_idx>/
            sporsa.csv
            arduino.csv
            *.png  (if plot=True)

    If fewer than two calibration sequences are found a warning is logged and
    the full streams are saved as the first section.

    Parameters
    ----------
    sensors:
        Sensor names to split.  Defaults to ``["sporsa", "arduino"]``.
    reference_sensor:
        The sensor used to detect calibration segments.  If not already in
        ``sensors``, it is prepended automatically.
    plot:
        If ``True``, generate sensor and comparison plots for every section.
    sync:
        If ``True`` (default), run calibration-based sync on each section
        after splitting so that the per-section timestamps are precisely
        aligned.

    Returns a list of written CSV paths in chronological order.
    """
    if sensors is None:
        sensors = ["sporsa", "arduino"]
    if reference_sensor not in sensors:
        sensors = [reference_sensor] + sensors

    # Load reference sensor and detect calibration segments.
    ref_csv = sensor_csv(f"{recording_name}/{stage}", reference_sensor)
    ref_df = read_csv(ref_csv)
    ref_df = ref_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    calibrations = find_calibration_segments(ref_df, sensor=reference_sensor)

    # Load all sensor DataFrames up-front (skip missing ones with a warning).
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in sensors:
        try:
            csv_path = sensor_csv(f"{recording_name}/{stage}", sensor)
        except FileNotFoundError:
            log.warning("Sensor '%s' not found in %s/%s — skipping", sensor, recording_name, stage)
            continue
        df = read_csv(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        sensor_dfs[sensor] = df

    available_sensors = list(sensor_dfs)

    recording_origin_ms = float(ref_df["timestamp"].iloc[0])
    source_interval_rows = load_recording_interval_rows_for_transfer(
        recording_name,
        recording_origin_ms=recording_origin_ms,
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
    ) -> list[Path]:
        section_path = section_dir(f"{recording_name}s{section_idx}")
        section_path.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for sensor, df in sensor_slices.items():
            out_path = section_path / f"{sensor}.csv"
            write_csv(df, out_path)
            paths.append(out_path)
            log.info("Wrote %s/%s (%d rows)", section_path.name, out_path.name, len(df))
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
            "Fewer than 2 calibration sequences found in %s/%s/%s "
            "(%d found) — saving full streams as first section",
            recording_name, stage, reference_sensor, len(calibrations),
        )
        ts_full_0 = float(ref_df["timestamp"].iloc[0])
        ts_full_1 = float(ref_df["timestamp"].iloc[-1])
        return _write_section(1, sensor_dfs, ts_full_0, ts_full_1)

    written: list[Path] = []
    for i, (c_open, c_close) in enumerate(zip(calibrations, calibrations[1:]), start=1):
        ts_start = float(ref_df.iloc[c_open.start_idx]["timestamp"])
        ts_end = float(ref_df.iloc[c_close.end_idx]["timestamp"])

        slices: dict[str, pd.DataFrame] = {}
        for sensor, df in sensor_dfs.items():
            mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            slices[sensor] = df.loc[mask].reset_index(drop=True)

        written.extend(_write_section(i, slices, ts_start, ts_end))

    if sync and reference_sensor in sensor_dfs:
        target_sensors = [s for s in available_sensors if s != reference_sensor]
        for tgt in target_sensors:
            sync_sections(
                recording_name,
                reference_sensor=reference_sensor,
                target_sensor=tgt,
                sample_rate_hz=sample_rate_hz,
                plot=plot,
            )

    return written


def sync_sections(
    recording_name: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 100.0,
    coarse_max_lag_s: float = 10.0,
    cal_search_s: float = 3.0,
    plot: bool = True,
) -> None:
    """Run calibration-based sync on each section of a recording, in-place.

    Each section must contain ≥ 2 calibration sequences (opening + closing).
    The target sensor's timestamps are corrected and its CSV is replaced
    in-place.  A ``sync_info.json`` is also written to each section directory.

    Parameters
    ----------
    recording_name:
        Name of the recording (e.g. ``"2026-02-26_r5"``).
    reference_sensor:
        Sensor used as the time reference.
    target_sensor:
        Sensor whose timestamps are corrected.
    sample_rate_hz:
        Sampling rate for peak detection and cross-correlation.
    coarse_max_lag_s:
        Max lag (seconds) for the coarse SDA search.  Sections are already
        roughly synced, so a small value (e.g. 10 s) is sufficient.
    cal_search_s:
        Search window (±s) for each calibration window alignment.
    plot:
        If ``True``, regenerate sensor and comparison plots after syncing.
    """
    from sync.pipeline import synchronize_from_calibration

    from common.paths import iter_sections_for_recording

    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No section directories found for %s", recording_name)
        return

    for section_dir in section_dirs:
        ref_csv = section_dir / f"{reference_sensor}.csv"
        tgt_csv = section_dir / f"{target_sensor}.csv"

        if not ref_csv.exists() or not tgt_csv.exists():
            log.warning(
                "Skipping %s/%s: missing %s or %s",
                recording_name, section_dir.name, ref_csv.name, tgt_csv.name,
            )
            continue

        log.info("Syncing %s/%s ...", recording_name, section_dir.name)
        tmp_dir = section_dir / "_tmp_sync"
        try:
            sync_json_raw, synced_csv_raw, _ = synchronize_from_calibration(
                reference_csv=ref_csv,
                target_csv=tgt_csv,
                output_dir=tmp_dir,
                reference_sensor=reference_sensor,
                sample_rate_hz=sample_rate_hz,
                coarse_max_lag_s=coarse_max_lag_s,
                cal_search_s=cal_search_s,
            )

            shutil.move(str(synced_csv_raw), tgt_csv)
            shutil.move(str(sync_json_raw), section_dir / "sync_info.json")
            log.info("  → wrote %s/sync_info.json", section_dir.name)

        except Exception as exc:
            log.error(
                "Failed to sync %s/%s: %s",
                recording_name, section_dir.name, exc,
            )
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        if plot:
            _plot_section(section_dir, [reference_sensor, target_sensor])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.split_sections",
        description=(
            "Detect calibration sequences in a recording and split it into "
            "sections.  Each section spans from the start of one calibration "
            "to the end of the next and is saved under "
            "data/sections/<recording_name>s<section_idx>/."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help=(
            "Recording name and input stage as '<recording_name>/<stage>' "
            "(e.g. '2026-02-26_r5/synced_cal')."
        ),
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
        help="Sensor used to detect calibration sequences (default: sporsa).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Nominal sampling rate (Hz) for per-section sync (default: 100).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots for each section.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip running the calibration sync on each section after splitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")
    recording_name, stage = parts

    written = split_recording(
        recording_name=recording_name,
        stage=stage,
        sensors=args.sensors,
        reference_sensor=args.reference_sensor,
        sample_rate_hz=args.sample_rate_hz,
        plot=not args.no_plot,
        sync=not args.no_sync,
    )

    print(f"\n{len(written)} file(s) written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
