"""Run static calibration: parse raw logs to CSV, estimate parameters, write JSON and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.paths import read_csv

from .imu_static import (
    calibration_data_dir,
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


def run_calibration_pipeline(
    raw_log_paths: list[Path] | None = None,
    *,
    parsed_dir: Path | None = None,
    output_json: Path | None = None,
    plots_dir: Path | None = None,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Parse each raw ``*.txt`` to ``parsed_dir``, estimate IMU calibration, write JSON and optional plots."""

    paths = list(raw_log_paths) if raw_log_paths is not None else default_calibration_raw_logs()
    if not paths:
        raise FileNotFoundError(
            "No calibration logs found. Place *.txt under data/_calibrations/raw/ or pass raw_log_paths."
        )

    root = calibration_data_dir()
    parsed_dir = parsed_dir or (root / "parsed")
    output_json = output_json or (root / "arduino_imu_calibration.json")
    plots_dir = plots_dir or (root / "plots")

    stem_to_csv = write_parsed_csvs(paths, parsed_dir, trim_fraction=trim_fraction)

    summaries = []
    for stem in sorted(stem_to_csv.keys()):
        df = read_csv(stem_to_csv[stem])
        summaries.append(summarize_stationary_recording(df, stem, trim_fraction=trim_fraction))

    calibration = estimate_calibration_from_summaries(summaries)
    calibration["source_logs"] = [str(p) for p in sorted(Path(p) for p in paths)]
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


def main() -> None:
    """Run the command-line interface."""
    result = run_calibration_pipeline()
    cal = result["calibration"]
    print(f"Wrote {result['calibration_json']}")
    print("Accelerometer bias:", cal["accelerometer"]["bias"])
    print("Accelerometer scale:", cal["accelerometer"]["scale"])
    print("Gyroscope bias:", cal["gyroscope"]["bias_deg_s"])
    if result["plots"]:
        print(f"Wrote overview plot to {result['plots']['overview']}")
        print(f"Wrote calibration parameters plot to {result['plots']['parameters']}")
    if cal["warnings"]:
        print("Warnings:")
        for warning in cal["warnings"]:
            print(f"- {warning}")
