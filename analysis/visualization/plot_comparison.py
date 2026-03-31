"""Plot overlaid comparison of bike and rider IMU signals.

CLI usage::

    python -m visualization.plot_comparison <recording>/<stage> [--norm]
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

from common.paths import project_relative_path, read_csv, resolve_data_dir
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

SENSORS = ["sporsa", "arduino"]
COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}


def _prepare_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep plotting rows aligned on valid, monotonic timestamps."""
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def plot_comparison_data(
    stage_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Plot accelerometer norms (and optionally gyro) for all sensors in stage_dir."""
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in SENSORS:
        csv = stage_dir / f"{sensor}.csv"
        if csv.exists():
            df = read_csv(csv)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            sensor_dfs[sensor] = _prepare_sensor_df(df)

    if not sensor_dfs:
        log.warning("No sensor CSVs found in %s", stage_dir)
        return stage_dir / "comparison.png"

    if output_path is None:
        output_path = stage_dir / f"comparison.png"

    # Compute a single t0 across all sensors so overlaid signals stay aligned.
    all_ts_arrays = [
        pd.to_numeric(df.get("timestamp", pd.Series()), errors="coerce").to_numpy(dtype=float)
        for df in sensor_dfs.values()
    ]
    valid_starts = [ts[np.isfinite(ts)][0] for ts in all_ts_arrays if np.isfinite(ts).any()]
    t0_global = min(valid_starts) if valid_starts else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for sensor, df in sensor_dfs.items():
        ts = pd.to_numeric(df.get("timestamp", pd.Series()), errors="coerce").to_numpy(dtype=float)
        if ts.size == 0:
            continue
        ts_s = (ts - t0_global) / 1000.0
        color = COLORS.get(sensor, None)

        acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
        gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]

        if acc_cols:
            acc_norm = strict_vector_norm(df, acc_cols)
            x, y = filter_valid_plot_xy(ts_s, acc_norm)
            axes[0].plot(x, y, lw=0.8, color=color, label=f"{sensor} |acc|", alpha=0.8)
        if gyro_cols:
            gyro_norm = strict_vector_norm(df, gyro_cols)
            x, y = filter_valid_plot_xy(ts_s, gyro_norm)
            axes[1].plot(x, y, lw=0.8, color=color, label=f"{sensor} |gyro|", alpha=0.8)

    axes[0].set_ylabel("|acc| (m/s²)")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"{stage_dir.parent.name}/{stage_dir.name} — sensor comparison")

    axes[1].set_ylabel("|gyro|")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Saved comparison plot → %s", output_path)
    return output_path


def plot_stage_data(
    stage_ref: str | Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Generate per-sensor and comparison plots for one stage directory."""
    from visualization.plot_sensor import plot_sensor_data

    stage_dir = resolve_data_dir(stage_ref)
    selected_sensors = sensors if sensors is not None else list(SENSORS)
    saved: list[Path] = []

    for sensor in selected_sensors:
        csv_path = stage_dir / f"{sensor}.csv"
        if not csv_path.exists():
            continue
        out = stage_dir / f"{sensor}.png"
        try:
            saved_path = plot_sensor_data(csv_path, output_path=out)
            saved.append(saved_path)
            log.info("Plot written: %s", project_relative_path(saved_path))
        except Exception as exc:
            log.warning("Failed to plot %s/%s: %s", stage_dir.name, sensor, exc)

    try:
        saved_path = plot_comparison_data(stage_dir, output_path=stage_dir / "comparison.png")
        saved.append(saved_path)
        log.info("Plot written: %s", project_relative_path(saved_path))
    except Exception as exc:
        log.warning("Failed comparison plot for %s: %s", stage_dir, exc)

    return saved


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_comparison")
    parser.add_argument("stage_ref", help="<recording>/<stage> or section folder name")
    parser.add_argument("-o", "--output", help="Output PNG path (auto-derived if omitted)")
    args = parser.parse_args(argv)

    try:
        stage_dir = resolve_data_dir(args.stage_ref)
    except FileNotFoundError as exc:
        log.error("Failed to resolve %s: %s", args.stage_ref, exc)
        return
    out = Path(args.output) if args.output else None
    try:
        saved = plot_comparison_data(stage_dir, output_path=out)
        print(f"Saved → {saved}")
    except Exception as exc:
        log.error("Failed to plot comparison: %s", exc)


if __name__ == "__main__":
    main()
