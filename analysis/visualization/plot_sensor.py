"""Plot sensor data from a recording stage CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir
from .labels import SENSOR_COMPONENTS, SENSOR_LABELS
from ._utils import mask_dropout_packets as _mask_dropout_packets
from ._utils import mask_valid_plot_x


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
    tx = time_seconds.to_numpy(dtype=float, copy=False)
    mask = mask.to_numpy(dtype=bool, copy=False) & mask_valid_plot_x(tx)

    if norm:
        data = sensor_norm(data, sensor_type)
        label = f"{sensor_type}_norm"
    else:
        label = ", ".join(SENSOR_COMPONENTS[sensor_type])

    dy = data.to_numpy(dtype=float, copy=False) if hasattr(data, "to_numpy") else np.asarray(data, dtype=float)
    ax.plot(tx[mask], dy[mask], label=label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")
    if mask.any():
        ax.set_xlim(float(tx[mask].min()), float(tx[mask].max()))
    ax.set_ylabel(SENSOR_LABELS[sensor_type][1])
    ax.set_title(SENSOR_LABELS[sensor_type][0])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot sensor data from a recording stage CSV.")
    parser.add_argument(
        "source",
        type=str,
        help=(
            "Either '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed') "
            "or a direct CSV path."
        ),
    )
    parser.add_argument(
        "sensor_name",
        nargs="?",
        default=None,
        help="Sensor name when source is '<recording_name>/<stage>' (e.g. sporsa, arduino).",
    )
    parser.add_argument("--norm", action="store_true", help="Plot vector norms instead of axes components.")
    parser.add_argument("--acc", action="store_true", help="Plot only the accelerometer data.")
    parser.add_argument("--gyro", action="store_true", help="Plot only the gyroscope data.")
    parser.add_argument("--mag", action="store_true", help="Plot only the magnetometer data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the PNG should be written when plotting from a direct CSV path.",
    )
    return parser


def _resolve_source(
    source: str,
    sensor_name: str | None,
    output_dir: Path | None,
    parser: argparse.ArgumentParser,
) -> tuple[Path, Path, str, str]:
    """Resolve either a stage reference or a direct CSV path."""

    csv_candidate = Path(source)
    if csv_candidate.suffix.lower() == ".csv" or csv_candidate.is_file():
        if not csv_candidate.exists():
            parser.error(f"CSV path does not exist: {csv_candidate}")
        csv_path = csv_candidate
        stage_dir = output_dir or csv_path.parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        label = str(csv_path)
        resolved_sensor_name = sensor_name or csv_path.stem
        return csv_path, stage_dir, label, resolved_sensor_name

    parts = source.split("/", 1)
    if len(parts) != 2:
        parser.error("source must be either '<recording_name>/<stage>' or a direct CSV path")
    if sensor_name is None:
        parser.error("sensor_name is required when source is '<recording_name>/<stage>'")

    recording_name, stage = parts

    # New layout: section stage strings map to top-level data/sections/<rec>s<idx>/.
    if stage.startswith("sections/"):
        from common.paths import section_dir as _section_dir

        sec_s = stage.split("/", 1)[1]
        if not sec_s.startswith("section_"):
            parser.error(f"Unrecognized section stage id: {stage!r}")
        sec_idx = int(sec_s.split("_", 1)[1])
        stage_dir = _section_dir(recording_name, sec_idx)
        csv_path = stage_dir / f"{sensor_name}.csv"
        if not csv_path.exists():
            print(f"[{recording_name}/{stage}] skipping {sensor_name}: missing {csv_path}")
            raise SystemExit(0)
        return csv_path, stage_dir, f"{recording_name} / {stage}", sensor_name

    try:
        stage_dir = recording_stage_dir(recording_name, stage)
        csv_path = find_sensor_csv(recording_name, stage, sensor_name)
    except FileNotFoundError:
        print(f"[{recording_name}/{stage}] skipping {sensor_name}: no CSV found")
        raise SystemExit(0)
    except ValueError as exc:
        parser.error(str(exc))

    return csv_path, stage_dir, f"{recording_name} / {stage}", sensor_name


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    csv_path, stage_dir, plot_label, sensor_name = _resolve_source(
        source=args.source,
        sensor_name=args.sensor_name,
        output_dir=args.output_dir,
        parser=parser,
    )

    df = _mask_dropout_packets(load_dataframe(csv_path))
    if df.empty:
        print(f"[{plot_label}] skipping {sensor_name}: CSV is empty")
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

        fig.suptitle(f"{plot_label} — {sensor_name}")

        filename = "".join([
            csv_path.stem,
            f"_{st[0]}" if len(st) == 1 else "",
            "_norm" if args.norm else "",
            ".png",
        ])
        fig.savefig(stage_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"[{plot_label}] {filename}")


if __name__ == "__main__":
    main()
