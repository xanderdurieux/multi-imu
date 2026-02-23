"""Plot comparison of two sensor streams."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe, session_stage_dir
from .labels import SENSOR_COMPONENTS, SENSOR_LABELS
from .plot_sensor import prepare_sensor_axes, find_sensor_csv, sensor_norm



def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Plot comparison of two sensor streams.")
	parser.add_argument("session_name_stage", type=str, help="Session name and stage.")
	parser.add_argument("sensor_name_a", type=str, help="Sensor name A.", default="sporsa", nargs="?")
	parser.add_argument("sensor_name_b", type=str, help="Sensor name B.", default="arduino", nargs="?")
	parser.add_argument(
		"--norm", 
		action="store_true", 
		help="Plot vector norms instead of axes components.",
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

	try:
		csv_path_a = find_sensor_csv(session_dir, args.sensor_name_a)
		csv_path_b = find_sensor_csv(session_dir, args.sensor_name_b)
	except (FileNotFoundError, ValueError) as exc:
		parser.error(str(exc))

	df_a = load_dataframe(csv_path_a)
	df_b = load_dataframe(csv_path_b)

	if df_a.empty or df_b.empty:
		parser.error("One or both CSV files are empty")

	time_seconds_a = (df_a["timestamp"].astype(float) - df_a["timestamp"].min()) / 1000.0
	time_seconds_b = (df_b["timestamp"].astype(float) - df_b["timestamp"].min()) / 1000.0

	num_cols = 1 if args.norm else 3
	sensor_types = ["acc", "gyro", "mag"]
	fig, ax_grid = prepare_sensor_axes(len(sensor_types), num_cols)

	for i, sensor_type in enumerate(sensor_types):

		data_a = df_a[list(SENSOR_COMPONENTS[sensor_type])]
		data_b = df_b[list(SENSOR_COMPONENTS[sensor_type])]
		mask_a = data_a.notna().all(axis=1)
		mask_b = data_b.notna().all(axis=1)

		if args.norm:
			data_a = sensor_norm(data_a, sensor_type)
			data_b = sensor_norm(data_b, sensor_type)

		for j in range(num_cols):
			if not args.norm:
				col_data_a = data_a[SENSOR_COMPONENTS[sensor_type][j]]
				col_data_b = data_b[SENSOR_COMPONENTS[sensor_type][j]]
			else:
				col_data_a = data_a
				col_data_b = data_b
	
			ax = ax_grid[i][j]
			ax.plot(time_seconds_a[mask_a], col_data_a[mask_a], label=args.sensor_name_a, alpha=0.8)
			ax.plot(time_seconds_b[mask_b], col_data_b[mask_b], label=args.sensor_name_b, alpha=0.8)

			ax.legend(loc="upper right")
			ax.grid(True, alpha=0.3)
			ax.set_xlabel("Time [s]")
			ax.set_xlim(
				min(time_seconds_a.min(), time_seconds_b.min()), 
				max(time_seconds_a.max(), time_seconds_b.max())
			)
			ax.set_ylabel(SENSOR_LABELS[sensor_type][1])
			ax.set_title(SENSOR_LABELS[sensor_type][0])

	fig.suptitle(f"{session_name} / {stage} — {args.sensor_name_a} vs {args.sensor_name_b}")


	filename = "".join([
		f"{args.sensor_name_a}",
		f"_vs_",
		f"{args.sensor_name_b}",
		f"_norm" if args.norm else "",
		f".png",
	])

	fig.savefig(session_dir / filename, bbox_inches="tight")
	print(f"Saved plot: {filename}")

if __name__ == "__main__":
	main()