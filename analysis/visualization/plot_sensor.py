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

from common.paths import analysis_root, sections_root, recording_stage_dir

log = logging.getLogger(__name__)


def _resolve_csv(stage_ref: str, sensor_name: str) -> Path | None:
    """Try to find <sensor>.csv under sections/ or recordings/ for the given stage_ref."""
    # stage_ref examples:
    #   "2026-02-26_r2/synced"                      (recording stage)
    #   "2026-02-26_r2s1"                            (section shorthand)
    data_root = analysis_root() / "data"

    # Try sections root first
    sec_dir = sections_root() / stage_ref
    if not sec_dir.exists():
        # Try as recording/stage
        parts = stage_ref.split("/", 1)
        if len(parts) == 2:
            rec, stage = parts
            # Check if it looks like a section folder name
            if "s" in parts[1] and not parts[1].startswith("s"):
                sec_dir = sections_root() / parts[1]
            else:
                sec_dir = recording_stage_dir(rec, stage)
        else:
            sec_dir = data_root / stage_ref

    csv = sec_dir / f"{sensor_name}.csv"
    if csv.exists():
        return csv

    # Try globbing
    matches = list(sec_dir.glob(f"*{sensor_name}*.csv"))
    if matches:
        return matches[0]

    return None


def _ts_seconds(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if ts.size > 0:
        ts = (ts - ts[0]) / 1000.0  # relative seconds
    return ts


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
    df = pd.read_csv(csv_path)
    for col in ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

    if norm_only:
        fig, ax = plt.subplots(figsize=(12, 3))
        if acc_cols:
            acc_arr = df[acc_cols].to_numpy(dtype=float)
            acc_norm = np.sqrt(np.nansum(acc_arr ** 2, axis=1))
            ax.plot(ts, acc_norm, lw=0.8, label="|acc|")
        if not acc_only and gyro_cols:
            gyro_arr = df[gyro_cols].to_numpy(dtype=float)
            gyro_norm = np.sqrt(np.nansum(gyro_arr ** 2, axis=1))
            ax.plot(ts, gyro_norm, lw=0.8, label="|gyro|")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm")
        ax.set_title(f"{csv_path.parent.name}/{csv_path.name} — norms")
        ax.legend(loc="upper right", fontsize=7)
    else:
        rows = (1 if acc_only else 2) if (acc_cols or gyro_cols) else 1
        fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=True)
        if rows == 1:
            axes = [axes]

        if acc_cols:
            for col in acc_cols:
                axes[0].plot(ts, df[col].to_numpy(dtype=float), lw=0.7, label=col)
            axes[0].set_ylabel("Acc (m/s²)")
            axes[0].legend(loc="upper right", fontsize=7)

        if not acc_only and gyro_cols and rows > 1:
            for col in gyro_cols:
                axes[1].plot(ts, df[col].to_numpy(dtype=float), lw=0.7, label=col)
            axes[1].set_ylabel("Gyro (°/s or rad/s)")
            axes[1].legend(loc="upper right", fontsize=7)

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

    csv_path = _resolve_csv(args.stage_ref, args.sensor_name)
    if csv_path is None:
        log.warning("Could not find CSV for %s/%s", args.stage_ref, args.sensor_name)
        return

    out = Path(args.output) if args.output else None
    try:
        saved = plot_sensor_data(csv_path, norm_only=args.norm, acc_only=args.acc, output_path=out)
        print(f"Saved → {saved}")
    except Exception as exc:
        log.error("Failed to plot %s: %s", csv_path, exc)


if __name__ == "__main__":
    main()
