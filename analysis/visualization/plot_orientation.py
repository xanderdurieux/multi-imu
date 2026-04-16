"""Plot orientation outputs for one section.

Generates:
- ``orientation_overlay.png``            selected method, sensors overlaid
- ``orientation_methods_comparison.png`` per-method quality comparison
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

from common.paths import read_csv, resolve_data_dir
from visualization._utils import (
    SENSOR_COLORS,
    SENSORS,
    filter_valid_plot_xy,
    load_json,
    save_figure,
    timestamps_to_relative_seconds,
)

log = logging.getLogger(__name__)

_ANGLES = ("yaw_deg", "pitch_deg", "roll_deg")
_METHOD_COLORS = {"madgwick": "#2ca02c", "complementary": "#d62728"}


def _prepare_wrapped_angle_plot_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter invalid rows and break the line at wraparound jumps."""
    x_plot, y_plot = filter_valid_plot_xy(x, y)
    if x_plot.size < 2:
        return x_plot, y_plot

    jump_idx = np.flatnonzero(np.abs(np.diff(y_plot)) > 180.0)
    if jump_idx.size == 0:
        return x_plot, y_plot

    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    start = 0
    for idx in jump_idx:
        stop = idx + 1
        parts_x.append(x_plot[start:stop])
        parts_y.append(y_plot[start:stop])
        parts_x.append(np.array([np.nan]))
        parts_y.append(np.array([np.nan]))
        start = stop
    parts_x.append(x_plot[start:])
    parts_y.append(y_plot[start:])
    return np.concatenate(parts_x), np.concatenate(parts_y)


def _resolve_orientation_dir(target: str | Path) -> Path:
    base = resolve_data_dir(target)
    if base.name == "orientation":
        return base
    orient = base / "orientation"
    if orient.is_dir():
        return orient
    raise FileNotFoundError(f"No orientation directory found for: {target}")


def _load_selected_sensor_csvs(orient_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sensor in SENSORS:
        csv_path = orient_dir / f"{sensor}.csv"
        if csv_path.exists():
            out[sensor] = read_csv(csv_path)
    return out


def plot_orientation_overlay(orient_dir: Path) -> Path | None:
    """Plot selected orientation angles with both sensors overlaid."""
    stats = load_json(orient_dir / "orientation_stats.json") or {}
    selected = stats.get("selected_method", "?")
    sensor_dfs = _load_selected_sensor_csvs(orient_dir)
    if not sensor_dfs:
        log.warning("No selected orientation CSVs found in %s", orient_dir)
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    any_line = False
    for sensor, df in sensor_dfs.items():
        if "timestamp" not in df.columns:
            continue
        ts_s = timestamps_to_relative_seconds(df["timestamp"])
        for idx, angle_col in enumerate(_ANGLES):
            if angle_col not in df.columns:
                continue
            y = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
            x_plot, y_plot = _prepare_wrapped_angle_plot_xy(ts_s, y)
            if x_plot.size == 0:
                continue
            axes[idx].plot(
                x_plot,
                y_plot,
                lw=0.9,
                alpha=0.95,
                color=SENSOR_COLORS.get(sensor),
                label=sensor,
            )
            any_line = True

    for idx, angle_col in enumerate(_ANGLES):
        axes[idx].set_ylabel(angle_col.replace("_deg", " (deg)"))
        handles, labels = axes[idx].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            axes[idx].legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
        axes[idx].grid(alpha=0.2, lw=0.4)
    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(f"{orient_dir.parent.name} — selected orientation ({selected})")
    fig.tight_layout()

    out_path = orient_dir / "orientation_overlay.png"
    if any_line:
        save_figure(fig, out_path)
        return out_path
    plt.close(fig)
    return None


def plot_orientation_methods_comparison(orient_dir: Path) -> Path | None:
    """Compare orientation methods via per-sensor quality metrics."""
    data = load_json(orient_dir / "all_methods.json") or {}
    methods = data.get("methods", {})
    if not methods:
        log.warning("No all_methods.json found in %s", orient_dir)
        return None

    method_names = list(methods.keys())
    selected = data.get("selected_method")
    x = np.arange(len(method_names), dtype=float)
    width = 0.18

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    metrics = [
        ("gravity_alignment", "Gravity alignment", False),
        ("pitch_std_deg", "Pitch std (deg)", True),
        ("roll_std_deg", "Roll std (deg)", True),
    ]

    for row, (metric, title, lower_is_better) in enumerate(metrics):
        ax = axes[row]
        for sensor_idx, sensor in enumerate(SENSORS):
            vals = []
            for method in method_names:
                vals.append(float(methods.get(method, {}).get(sensor, {}).get(metric, np.nan)))
            offset = (sensor_idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=SENSOR_COLORS.get(sensor),
                alpha=0.85,
                label=sensor,
            )
            if selected in method_names:
                sel_idx = method_names.index(selected)
                bars[sel_idx].set_edgecolor("black")
                bars[sel_idx].set_linewidth(1.2)

        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, lw=0.4)
        if lower_is_better:
            ax.text(
                0.99,
                0.93,
                "lower is better",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
                color="#444",
            )

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xticks(x, method_names)
    axes[-1].set_xlabel("Orientation method")
    fig.suptitle(f"{orient_dir.parent.name} — orientation method comparison")
    fig.tight_layout()

    out_path = orient_dir / "orientation_methods_comparison.png"
    return save_figure(fig, out_path)


def plot_orientation_stage(target: str | Path) -> list[Path]:
    """Generate all orientation plots for a section."""
    orient_dir = _resolve_orientation_dir(target)
    out_paths: list[Path] = []
    for plot_fn in (plot_orientation_overlay, plot_orientation_methods_comparison):
        try:
            p = plot_fn(orient_dir)
        except Exception as exc:
            log.warning("Orientation plot failed for %s: %s", orient_dir.name, exc)
            continue
        if p is not None:
            out_paths.append(p)
    return out_paths


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_orientation")
    parser.add_argument("target", help="Section directory reference or orientation dir")
    args = parser.parse_args(argv)

    try:
        paths = plot_orientation_stage(args.target)
    except Exception as exc:
        log.error("Failed to plot orientation: %s", exc)
        return

    if not paths:
        print("No orientation plots generated.")
    for p in paths:
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
