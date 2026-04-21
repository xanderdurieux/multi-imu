"""Orientation stage plots for one section.

Generates:
- ``{sensor}_orientation.png``  per-sensor Euler angle comparison across all methods run
- ``orientation_overlay.png``   selected method, both sensors overlaid
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from common.paths import read_csv, resolve_data_dir, section_stage_dir
from visualization._utils import (
    SENSOR_COLORS,
    SENSORS,
    filter_valid_plot_xy,
    load_json,
    save_figure,
    timestamps_to_relative_seconds,
)

log = logging.getLogger(__name__)

_ANGLES = ("roll_deg", "pitch_deg", "yaw_deg")
_ANGLE_LABELS = {"roll_deg": "Roll (°)", "pitch_deg": "Pitch (°)", "yaw_deg": "Yaw (°)"}
_METHOD_PALETTE = [
    "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _resolve_orientation_dir(target: str | Path) -> Path:
    base = resolve_data_dir(target)
    if base.name == "orientation":
        return base
    orient = section_stage_dir(base.name, "orientation")
    if orient.is_dir():
        return orient
    raise FileNotFoundError(f"No orientation directory found for: {target}")


def _prepare_wrapped_angle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter invalid rows and insert NaN breaks at ±180° wraparounds."""
    x_plot, y_plot = filter_valid_plot_xy(x, y)
    if x_plot.size < 2:
        return x_plot, y_plot

    jump_idx = np.flatnonzero(np.abs(np.diff(y_plot)) > 180.0)
    if jump_idx.size == 0:
        return x_plot, y_plot

    parts_x, parts_y = [], []
    start = 0
    for idx in jump_idx:
        stop = idx + 1
        parts_x += [x_plot[start:stop], np.array([np.nan])]
        parts_y += [y_plot[start:stop], np.array([np.nan])]
        start = stop
    parts_x.append(x_plot[start:])
    parts_y.append(y_plot[start:])
    return np.concatenate(parts_x), np.concatenate(parts_y)


def plot_orientation_methods_comparison(orient_dir: Path, sensor: str) -> Path | None:
    """Plot Euler angles for all methods run on one sensor.

    Reads ``{sensor}_all_methods.csv`` (columns: timestamp, {method}_roll_deg,
    {method}_pitch_deg, {method}_yaw_deg) and ``orientation_stats.json``.
    """
    csv_path = orient_dir / f"{sensor}_all_methods.csv"
    if not csv_path.exists():
        log.debug("No all-methods CSV for %s in %s", sensor, orient_dir)
        return None

    df = read_csv(csv_path)
    if df.empty or "timestamp" not in df.columns:
        return None

    stats = load_json(orient_dir / "orientation_stats.json") or {}
    selected = stats.get("selected_method", "")
    all_methods_stats = stats.get("all_methods", {}).get(sensor, {})

    # Discover methods from column names.
    methods = sorted({
        col.rsplit("_", 2)[0]
        for col in df.columns
        if col.endswith("_roll_deg")
    })
    if not methods:
        return None

    t_s = timestamps_to_relative_seconds(df["timestamp"])

    # Static window shading from calibration JSON (optional).
    static_windows: list[tuple[float, float, str, str]] = []
    cal_json = section_stage_dir(orient_dir.parent.name, "calibrated") / "calibration.json"
    if cal_json.exists():
        try:
            import json
            cal = json.loads(cal_json.read_text(encoding="utf-8"))
            opening_seq = cal.get("opening_sequence", {}).get(sensor, {})
            closing_seq = cal.get("closing_sequence", {}).get(sensor, {})
            t0_ms = df["timestamp"].iloc[0]
            for seq, color, label_prefix in (
                (opening_seq, "green", "opening"),
                (closing_seq, "#ff7f0e", "closing"),
            ):
                for prefix, suffix in (("pre_static_start_ms", "pre_static_end_ms"),
                                     ("post_static_start_ms", "post_static_end_ms")):
                    s, e = float(seq.get(prefix, 0)), float(seq.get(suffix, 0))
                    if e > s:
                        static_windows.append(
                            ((s - t0_ms) / 1000.0, (e - t0_ms) / 1000.0, color, f"{label_prefix} static")
                        )
        except Exception:
            pass

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for m_idx, method in enumerate(methods):
        color = _METHOD_PALETTE[m_idx % len(_METHOD_PALETTE)]
        lw = 1.8 if method == selected else 1.0
        ls = "-" if method == selected else "--"
        residual = all_methods_stats.get(method, {}).get("gravity_residual_ms2", float("nan"))
        label = f"{method} (res={residual:.3f} m/s²)" if np.isfinite(residual) else method

        for row, angle in enumerate(_ANGLES):
            col = f"{method}_{angle}"
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = _prepare_wrapped_angle(t_s, y)
            if xp.size == 0:
                continue
            axes[row].plot(xp, yp, color=color, lw=lw, ls=ls,
                           label=label if row == 0 else "_")

    for ax, angle in zip(axes, _ANGLES):
        for t0_s, t1_s, color, _label in static_windows:
            ax.axvspan(t0_s, t1_s, alpha=0.12, color=color)
        ax.set_ylabel(_ANGLE_LABELS[angle])
        ax.grid(alpha=0.3, lw=0.4)

    if static_windows:
        patches = [
            mpatches.Patch(color="green", alpha=0.3, label="opening static window"),
            mpatches.Patch(color="#ff7f0e", alpha=0.3, label="closing static window"),
        ]
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles + patches, labels + [p.get_label() for p in patches], fontsize=7, ncol=2)
    else:
        axes[0].legend(fontsize=7, ncol=2)

    axes[0].set_title(
        f"{orient_dir.parent.name} — {sensor} orientation methods (selected: {selected})"
    )
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    out_path = orient_dir / f"{sensor}_orientation.png"
    return save_figure(fig, out_path)


def plot_orientation_overlay(orient_dir: Path) -> Path | None:
    """Plot selected orientation angles with both sensors overlaid."""
    stats = load_json(orient_dir / "orientation_stats.json") or {}
    selected = stats.get("selected_method", "?")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    any_line = False

    for sensor in SENSORS:
        csv_path = orient_dir / f"{sensor}.csv"
        if not csv_path.exists():
            continue
        df = read_csv(csv_path)
        if df.empty or "timestamp" not in df.columns:
            continue
        t_s = timestamps_to_relative_seconds(df["timestamp"])
        color = SENSOR_COLORS.get(sensor)

        for row, angle in enumerate(_ANGLES):
            if angle not in df.columns:
                continue
            y = pd.to_numeric(df[angle], errors="coerce").to_numpy(dtype=float)
            xp, yp = _prepare_wrapped_angle(t_s, y)
            if xp.size == 0:
                continue
            axes[row].plot(xp, yp, color=color, lw=0.9, alpha=0.95,
                           label=sensor if row == 0 else "_")
            any_line = True

    for ax, angle in zip(axes, _ANGLES):
        ax.set_ylabel(_ANGLE_LABELS[angle])
        ax.grid(alpha=0.2, lw=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8, loc="upper right")
    axes[0].set_title(f"{orient_dir.parent.name} — selected orientation ({selected})")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    out_path = orient_dir / "orientation_overlay.png"
    if any_line:
        return save_figure(fig, out_path)
    plt.close(fig)
    return None


def plot_orientation_stage(target: str | Path) -> list[Path]:
    """Generate all orientation plots for a section."""
    orient_dir = _resolve_orientation_dir(target)
    out_paths: list[Path] = []

    for sensor in SENSORS:
        try:
            p = plot_orientation_methods_comparison(orient_dir, sensor)
            if p is not None:
                out_paths.append(p)
        except Exception as exc:
            log.warning("Orientation comparison plot failed for %s/%s: %s", orient_dir.name, sensor, exc)

    try:
        p = plot_orientation_overlay(orient_dir)
        if p is not None:
            out_paths.append(p)
    except Exception as exc:
        log.warning("Orientation overlay plot failed for %s: %s", orient_dir.name, exc)

    return out_paths


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_orientation")
    parser.add_argument("target", help="Section directory or orientation dir")
    args = parser.parse_args(argv)
    try:
        paths = plot_orientation_stage(args.target)
    except Exception as exc:
        log.error("Failed to plot orientation: %s", exc)
        return
    if not paths:
        print("No orientation plots generated.")
    for p in paths:
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
