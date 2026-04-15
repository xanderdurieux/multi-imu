"""Plot raw or processed IMU data for a single sensor.

CLI usage::

    python -m visualization.plot_sensor <recording>/<stage> <sensor> [--norm] [--acc]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import read_csv, sensor_csv
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

def _ts_seconds(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if ts.size > 0:
        finite = ts[np.isfinite(ts)]
        t0 = finite[0] if finite.size > 0 else 0.0
        ts = (ts - t0) / 1000.0  # relative seconds
    return ts


def _prepare_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep plotting rows aligned on valid, monotonic timestamps."""
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def plot_sensor_data(
    csv_path: Path,
    *,
    norm_only: bool = False,
    acc_only: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Load a sensor CSV and save a plot.

    Returns the path of the saved figure.
    """
    df = read_csv(csv_path)
    for col in ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _prepare_sensor_df(df)

    ts = _ts_seconds(df)

    if output_path is None:
        stem = csv_path.stem
        if norm_only:
            stem += "_norm"
        if acc_only:
            stem += "_acc"
        output_path = csv_path.parent / f"{stem}.png"

    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]
    mag_cols = [c for c in ["mx", "my", "mz"] if c in df.columns]
    has_mag = len(mag_cols) == 3

    if norm_only:
        fig, ax = plt.subplots(figsize=(12, 3))
        if acc_cols:
            acc_norm = strict_vector_norm(df, acc_cols)
            x, y = filter_valid_plot_xy(ts, acc_norm)
            ax.plot(x, y, lw=0.8, label="|acc|")
        if not acc_only and gyro_cols:
            gyro_norm = strict_vector_norm(df, gyro_cols)
            x, y = filter_valid_plot_xy(ts, gyro_norm)
            ax.plot(x, y, lw=0.8, label="|gyro|")
        if not acc_only and has_mag:
            mag_norm = strict_vector_norm(df, mag_cols)
            x, y = filter_valid_plot_xy(ts, mag_norm)
            ax.plot(x, y, lw=0.8, label="|mag|")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm")
        ax.set_title(f"{csv_path.parent.name}/{csv_path.name} — norms")
        ax.legend(loc="upper right", fontsize=7)
    else:
        n_rows = 1
        if not acc_only:
            if gyro_cols:
                n_rows += 1
            if has_mag:
                n_rows += 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        row = 0
        if acc_cols:
            for col in acc_cols:
                y = df[col].to_numpy(dtype=float)
                x_plot, y_plot = filter_valid_plot_xy(ts, y)
                axes[row].plot(x_plot, y_plot, lw=0.7, label=col)
            axes[row].set_ylabel("Acc (m/s²)")
            axes[row].legend(loc="upper right", fontsize=7)
            row += 1

        if not acc_only and gyro_cols and row < n_rows:
            for col in gyro_cols:
                y = df[col].to_numpy(dtype=float)
                x_plot, y_plot = filter_valid_plot_xy(ts, y)
                axes[row].plot(x_plot, y_plot, lw=0.7, label=col)
            axes[row].set_ylabel("Gyro (°/s or rad/s)")
            axes[row].legend(loc="upper right", fontsize=7)
            row += 1

        if not acc_only and has_mag and row < n_rows:
            mag_colors = {"mx": "#d62728", "my": "#2ca02c", "mz": "#1f77b4"}
            for col in mag_cols:
                y = df[col].to_numpy(dtype=float)
                x_plot, y_plot = filter_valid_plot_xy(ts, y)
                axes[row].plot(x_plot, y_plot, lw=0.7, label=col, color=mag_colors.get(col))
            axes[row].set_ylabel("Mag (µT)")
            axes[row].legend(loc="upper right", fontsize=7)
            axes[row].grid(alpha=0.2, lw=0.4)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{csv_path.parent.name}/{csv_path.name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Saved sensor plot → %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_sensor")
    parser.add_argument("stage_ref", help="<recording>/<stage> or section folder name")
    parser.add_argument("sensor_name", help="Sensor name (e.g. sporsa, arduino)")
    parser.add_argument("--norm", action="store_true", help="Plot norms only")
    parser.add_argument("--acc", action="store_true", help="Accelerometer only")
    parser.add_argument("-o", "--output", help="Output PNG path (auto-derived if omitted)")
    args = parser.parse_args(argv)

    try:
        csv_path = sensor_csv(args.stage_ref, args.sensor_name)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Could not find CSV for %s/%s: %s", args.stage_ref, args.sensor_name, exc)
        return
    out = Path(args.output) if args.output else None
    try:
        saved = plot_sensor_data(csv_path, norm_only=args.norm, acc_only=args.acc, output_path=out)
        print(f"Saved → {saved}")
    except Exception as exc:
        log.error("Failed to plot %s: %s", csv_path, exc)


if __name__ == "__main__":
    main()
