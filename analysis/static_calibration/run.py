"""Run static calibration: parse raw logs to CSV, estimate parameters, write JSON and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.paths import read_csv

from .imu_static import (
    calibration_data_dir,
    calibration_sensor_dir,
    default_calibration_raw_logs,
    estimate_calibration_from_summaries,
    summarize_stationary_recording,
    write_parsed_csvs,
)
from .plotting import (
    plot_calibration_parameters,
    plot_recording_details,
    plot_recordings_overview,
)

DEFAULT_TRIM_FRACTION = 0.05
_DEFAULT_SENSORS = ("arduino", "sporsa")


def run_calibration_pipeline(
    sensor: str = "arduino",
    raw_log_paths: list[Path] | None = None,
    *,
    parsed_dir: Path | None = None,
    output_json: Path | None = None,
    plots_dir: Path | None = None,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    write_plots: bool = True,
    parsed_only: bool = False,
) -> dict[str, Any]:
    """Parse sensor raw logs, estimate IMU calibration, write JSON and optional plots."""

    root = calibration_data_dir()
    sensor_root = calibration_sensor_dir(sensor)
    parsed_dir = parsed_dir or (sensor_root / "parsed")
    output_json = output_json or (sensor_root / "imu_calibration.json")
    plots_dir = plots_dir or (sensor_root / "plots")
    if parsed_only:
        existing_csvs = sorted(parsed_dir.glob("*.csv"))
        if not existing_csvs:
            raise FileNotFoundError(
                "No parsed calibration CSVs found under "
                f"data/_calibrations/{sensor}/parsed/."
            )
        stem_to_csv = {p.stem: p for p in existing_csvs}
        source_logs: list[str] = []
    else:
        paths = (
            list(raw_log_paths)
            if raw_log_paths is not None
            else default_calibration_raw_logs(sensor)
        )
        if not paths:
            raise FileNotFoundError(
                "No calibration logs found. Place *.txt under "
                f"data/_calibrations/{sensor}/raw/ or run with parsed_only=True."
            )
        stem_to_csv = write_parsed_csvs(paths, parsed_dir, sensor=sensor, trim_fraction=trim_fraction)
        source_logs = [str(p) for p in sorted(Path(p) for p in paths)]

    summaries = []
    for stem in sorted(stem_to_csv.keys()):
        df = read_csv(stem_to_csv[stem])
        summaries.append(summarize_stationary_recording(df, stem, trim_fraction=trim_fraction))

    calibration = estimate_calibration_from_summaries(summaries)
    calibration["sensor"] = sensor
    calibration["calibration_root"] = str(root)
    calibration["source_logs"] = source_logs
    calibration["source_parsed_csvs"] = [str(p) for p in sorted(stem_to_csv.values())]
    calibration["per_recording"] = summaries

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)

    plot_outputs: dict[str, Path | list[Path]] = {}
    if write_plots:
        parsed_by_stem = {stem: read_csv(csv_path) for stem, csv_path in stem_to_csv.items()}
        per_recording = calibration["per_recording"]
        plot_outputs["overview"] = plot_recordings_overview(
            per_recording,
            parsed_by_stem,
            plots_dir / "recordings_overview.png",
        )
        plot_outputs["details"] = plot_recording_details(
            per_recording,
            parsed_by_stem,
            plots_dir / "recordings",
        )
        plot_outputs["parameters"] = plot_calibration_parameters(
            per_recording,
            calibration,
            plots_dir / "calibration_parameters.png",
            parsed_by_stem=parsed_by_stem,
        )

    return {
        "calibration": calibration,
        "calibration_json": output_json,
        "parsed_csv_by_stem": stem_to_csv,
        "plots": plot_outputs,
    }


def run_calibration_pipeline_all_sensors(
    *,
    sensors: tuple[str, ...] = _DEFAULT_SENSORS,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    write_plots: bool = True,
    parsed_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run static calibration for all requested sensors."""
    results: dict[str, dict[str, Any]] = {}
    for sensor in sensors:
        try:
            results[sensor] = run_calibration_pipeline(
                sensor=sensor,
                trim_fraction=trim_fraction,
                write_plots=write_plots,
                parsed_only=parsed_only,
            )
        except FileNotFoundError:
            continue
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m static_calibration",
        description="Run static calibration for one or both sensors.",
    )
    parser.add_argument(
        "--sensor",
        choices=["arduino", "sporsa", "all"],
        default="all",
        help="Sensor to calibrate (default: all).",
    )
    parser.add_argument(
        "--parsed-only",
        action="store_true",
        help="Use existing parsed CSVs and skip parsing raw logs.",
    )
    return parser


def main() -> None:
    """Run static calibration for both sensors."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    sensors = _DEFAULT_SENSORS if args.sensor == "all" else (args.sensor,)
    results = run_calibration_pipeline_all_sensors(
        sensors=sensors,
        parsed_only=bool(args.parsed_only),
    )
    if not results:
        print("No calibration logs found under data/_calibrations/<sensor>/raw/")
        return

    for sensor, result in results.items():
        cal = result["calibration"]
        print(f"[{sensor}] Wrote {result['calibration_json']}")
        print(f"[{sensor}] Accelerometer bias: {cal['accelerometer']['bias']}")
        print(f"[{sensor}] Accelerometer scale: {cal['accelerometer']['scale']}")
        print(f"[{sensor}] Gyroscope bias: {cal['gyroscope']['bias_deg_s']}")
        if result["plots"]:
            print(f"[{sensor}] Wrote overview plot to {result['plots']['overview']}")
            print(f"[{sensor}] Wrote calibration parameters plot to {result['plots']['parameters']}")
        if cal["warnings"]:
            print(f"[{sensor}] Warnings:")
            for warning in cal["warnings"]:
                print(f"- {warning}")
