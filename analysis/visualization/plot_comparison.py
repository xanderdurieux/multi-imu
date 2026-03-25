"""Plot comparison of two sensor streams from a recording stage."""

from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir
from .labels import SENSOR_COMPONENTS, SENSOR_LABELS
from .plot_sensor import prepare_sensor_axes, sensor_norm
from ._utils import mask_dropout_packets as _mask_dropout_packets
from ._utils import mask_valid_plot_x


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot comparison of two sensor streams.")
    parser.add_argument(
        "recording_name_stage",
        type=str,
        help="Recording name and stage as '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').",
    )
    parser.add_argument("sensor_name_a", type=str, default="sporsa", nargs="?", help="Sensor A name.")
    parser.add_argument("sensor_name_b", type=str, default="arduino", nargs="?", help="Sensor B name.")
    parser.add_argument("--norm", action="store_true", help="Plot vector norms instead of axes components.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        parser.error("recording_name_stage must be in format '<recording_name>/<stage>'")
    recording_name, stage = parts

    # New layout: section stage strings map to top-level data/sections/<rec>s<idx>/.
    if stage.startswith("sections/"):
        from common.paths import section_dir as _section_dir

        sec_s = stage.split("/", 1)[1]
        if not sec_s.startswith("section_"):
            parser.error(f"Unrecognized section stage id: {stage!r}")
        sec_idx = int(sec_s.split("_", 1)[1])
        stage_dir = _section_dir(recording_name, sec_idx)

        csv_path_a = stage_dir / f"{args.sensor_name_a}.csv"
        csv_path_b = stage_dir / f"{args.sensor_name_b}.csv"
    else:
        stage_dir = recording_stage_dir(recording_name, stage)
        try:
            csv_path_a = find_sensor_csv(recording_name, stage, args.sensor_name_a)
            csv_path_b = find_sensor_csv(recording_name, stage, args.sensor_name_b)
        except FileNotFoundError as exc:
            print(f"[{recording_name}/{stage}] skipping comparison: {exc}")
            return
        except ValueError as exc:
            parser.error(str(exc))

    if not csv_path_a.is_file() or not csv_path_b.is_file():
        print(f"[{recording_name}/{stage}] skipping comparison: missing CSV(s)")
        return

    df_a = _mask_dropout_packets(load_dataframe(csv_path_a))
    df_b = _mask_dropout_packets(load_dataframe(csv_path_b))

    if df_a.empty or df_b.empty:
        print(f"[{recording_name}/{stage}] skipping comparison: one or both CSVs are empty")
        return

    # Use a shared time reference if both streams overlap on the same clock
    # (e.g. after sync), otherwise normalize each stream to its own start
    # (e.g. parsed, where arduino has boot-time and sporsa has epoch-time).
    ts_a = df_a["timestamp"].astype(float)
    ts_b = df_b["timestamp"].astype(float)
    overlap = min(ts_a.max(), ts_b.max()) - max(ts_a.min(), ts_b.min())
    if overlap > 0:
        t_ref = min(ts_a.min(), ts_b.min())
        time_seconds_a = (ts_a - t_ref) / 1000.0
        time_seconds_b = (ts_b - t_ref) / 1000.0
    else:
        time_seconds_a = (ts_a - ts_a.min()) / 1000.0
        time_seconds_b = (ts_b - ts_b.min()) / 1000.0

    ta = time_seconds_a.to_numpy(dtype=float)
    tb = time_seconds_b.to_numpy(dtype=float)
    mx_a = mask_valid_plot_x(ta)
    mx_b = mask_valid_plot_x(tb)

    num_cols = 1 if args.norm else 3
    sensor_types = ["acc", "gyro", "mag"]
    fig, ax_grid = prepare_sensor_axes(len(sensor_types), num_cols)

    for i, sensor_type in enumerate(sensor_types):
        data_a = df_a[list(SENSOR_COMPONENTS[sensor_type])]
        data_b = df_b[list(SENSOR_COMPONENTS[sensor_type])]
        mask_a = data_a.notna().all(axis=1).to_numpy(dtype=bool) & mx_a
        mask_b = data_b.notna().all(axis=1).to_numpy(dtype=bool) & mx_b

        if args.norm:
            data_a = sensor_norm(data_a, sensor_type)
            data_b = sensor_norm(data_b, sensor_type)

        for j in range(num_cols):
            col_data_a = data_a if args.norm else data_a[SENSOR_COMPONENTS[sensor_type][j]]
            col_data_b = data_b if args.norm else data_b[SENSOR_COMPONENTS[sensor_type][j]]

            ax = ax_grid[i][j]
            ax.plot(ta[mask_a], col_data_a[mask_a], label=args.sensor_name_a, alpha=0.8)
            ax.plot(tb[mask_b], col_data_b[mask_b], label=args.sensor_name_b, alpha=0.8)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time [s]")
            x_parts = []
            if mask_a.any():
                x_parts.append(ta[mask_a])
            if mask_b.any():
                x_parts.append(tb[mask_b])
            if x_parts:
                xc = np.concatenate(x_parts)
                ax.set_xlim(float(xc.min()), float(xc.max()))
            ax.set_ylabel(SENSOR_LABELS[sensor_type][1])
            ax.set_title(SENSOR_LABELS[sensor_type][0])

    fig.suptitle(f"{recording_name} / {stage} — {args.sensor_name_a} vs {args.sensor_name_b}")

    filename = "".join([
        csv_path_a.stem,
        "_vs_",
        csv_path_b.stem,
        "_norm" if args.norm else "",
        ".png",
    ])
    fig.savefig(stage_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {filename}")


if __name__ == "__main__":
    main()
