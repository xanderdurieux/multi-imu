"""Plot sensor data from a recording stage CSV."""

from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir
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


def _plot_sensor_data(
    ax: plt.Axes,
    df: pd.DataFrame,
    time_seconds: np.ndarray,
    sensor_type: str,
    norm: bool = False,
) -> None:
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
    parser = argparse.ArgumentParser(description="Plot sensor data from a recording stage CSV.")
    parser.add_argument(
        "recording_name_stage",
        type=str,
        help="Recording name and stage as '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').",
    )
    parser.add_argument("sensor_name", type=str, help="Sensor name (e.g. sporsa, arduino).")
    parser.add_argument("--norm", action="store_true", help="Plot vector norms instead of axes components.")
    parser.add_argument("--acc", action="store_true", help="Plot only the accelerometer data.")
    parser.add_argument("--gyro", action="store_true", help="Plot only the gyroscope data.")
    parser.add_argument("--mag", action="store_true", help="Plot only the magnetometer data.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        parser.error("recording_name_stage must be in format '<recording_name>/<stage>'")
    recording_name, stage = parts
    stage_dir = recording_stage_dir(recording_name, stage)

    try:
        csv_path = find_sensor_csv(recording_name, stage, args.sensor_name)
    except FileNotFoundError:
        print(f"[{recording_name}/{stage}] skipping {args.sensor_name}: no CSV found")
        return
    except ValueError as exc:
        parser.error(str(exc))

    df = load_dataframe(csv_path)
    if df.empty:
        print(f"[{recording_name}/{stage}] skipping {args.sensor_name}: CSV is empty")
        return

    time_seconds = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0

    sensor_types = [["acc", "gyro", "mag"]]
    if args.acc:
        sensor_types = [["acc"]]
    if args.gyro:
        sensor_types = [["gyro"]]
    if args.mag:
        sensor_types = [["mag"]]

    for st in sensor_types:
        fig, ax_grid = prepare_sensor_axes(len(st))

        for i, sensor_type in enumerate(st):
            _plot_sensor_data(ax_grid[i][0], df, time_seconds, sensor_type, args.norm)

        fig.suptitle(f"{recording_name} / {stage} — {args.sensor_name}")

        filename = "".join([
            csv_path.stem,
            f"_{st[0]}" if len(st) == 1 else "",
            "_norm" if args.norm else "",
            ".png",
        ])
        fig.savefig(stage_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"[{recording_name}/{stage}] {filename}")


if __name__ == "__main__":
    main()
