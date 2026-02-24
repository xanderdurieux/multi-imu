"""Plot sensor data from CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, session_stage_dir
from .labels import SENSOR_COMPONENTS, SENSOR_LABELS

def prepare_sensor_axes(num_series: int, num_columns: int = 1, *, sharex: bool = True):
    """Create a grid of axes and normalize return type to a list."""
    fig, axes = plt.subplots(
        num_series,
        num_columns,
        figsize=(10 * num_columns, 3 * num_series),
        sharex=sharex,
        constrained_layout=True,
    )

    if num_series == 1 and num_columns == 1:
        axes = [[axes]]
    elif num_series == 1:
        axes = [axes]
    elif num_columns == 1:
        axes = [[ax] for ax in axes]

    return fig, axes


def sensor_norm(df: pd.DataFrame, sensor_type: str) -> np.ndarray:
    """Compute vector norm for one sensor type."""
    cols = SENSOR_COMPONENTS[sensor_type]
    return np.sqrt(sum(df[col].values ** 2 for col in cols))


def _plot_sensor_data(ax: plt.Axes, df: pd.DataFrame, time_seconds: np.ndarray, sensor_type: str, norm: bool = False) -> None:
    """Plot sensor data on a single subplot."""
    data = df[list(SENSOR_COMPONENTS[sensor_type])]
    mask = data.notna().all(axis=1)

    if norm:
        data = sensor_norm(data, sensor_type)
    
    ax.plot(time_seconds[mask], data[mask], label=SENSOR_COMPONENTS[sensor_type])

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(time_seconds.iloc[0], time_seconds.iloc[-1])
    ax.set_ylabel(SENSOR_LABELS[sensor_type][1])
    ax.set_title(SENSOR_LABELS[sensor_type][0])


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Plot sensor data from CSV files.")
    parser.add_argument("session_name_stage", type=str, help="Session name and stage.")
    parser.add_argument("sensor_name", type=str, help="Sensor name.")
    parser.add_argument(
        "--norm",
        dest="norm",
        action="store_true",
        help="Plot vector norms instead of axes components.",
    )
    parser.add_argument(
        "--acc",
        action="store_true",
        help="Plot only the accelerometer data.",
    )
    parser.add_argument(
        "--gyro",
        action="store_true",
        help="Plot only the gyroscope data.",
    )
    parser.add_argument(
        "--mag",
        action="store_true",
        help="Plot only the magnetometer data.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    parts = args.session_name_stage.split("/", 1)
    if len(parts) != 2:
        parser.error("session_name_stage must be in format 'session_name/stage'")
    session_name, stage = parts
    session_dir = session_stage_dir(session_name, stage)

    # Find sensor CSV file
    try:
        csv_path = find_sensor_csv(session_name, stage, args.sensor_name)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # Load sensor data into DataFrame
    df = load_dataframe(csv_path)
    if df.empty:
        parser.error(f"CSV file is empty: {csv_path}")

    # Use relative time in seconds for clearer x-axis units
    time_seconds = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0

    # List of sensor types to plot on separate images
    sensor_types = [["acc", "gyro", "mag"]]
    if args.acc:
        sensor_types = [["acc"]]
    if args.gyro:
        sensor_types = [["gyro"]]
    if args.mag:
        sensor_types = [["mag"]]

    for st in sensor_types:
        fig, ax_grid = prepare_sensor_axes(len(st))

        # Plot each sensor type on a separate subplot of the same image
        for i, sensor_type in enumerate(st):
            ax = ax_grid[i][0]
            
            _plot_sensor_data(ax, df, time_seconds, sensor_type, args.norm)

        fig.suptitle(f"{session_name} / {stage} - {args.sensor_name}")

        # Build output filename and save image
        filename = "".join([
            f"{csv_path.stem}",
            f"_{st[0]}" if len(st) == 1 else "",
            f"_norm" if args.norm else "",
            f".png",
        ])
        fig.savefig(session_dir / filename, bbox_inches="tight")
        print(f"Saved plot: {filename}")

if __name__ == "__main__":
    main()
