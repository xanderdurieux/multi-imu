"""Plot derived signals for one section.

Generates:
- ``derived/derived_overview.png``     all derived signals, both sensors
- ``derived/linear_acceleration.png``  orientation-aware linear acc, per axis
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
from common.signals import smooth_moving_average
from visualization._utils import (
    SENSOR_COLORS,
    SENSORS,
    filter_valid_plot_xy,
    load_json,
    relative_seconds,
    save_figure,
    shared_t0_ms,
)

log = logging.getLogger(__name__)

_AXIS_COLORS = {"x": "#e41a1c", "y": "#4daf4a", "z": "#377eb8"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_derived_csvs(section_dir: Path) -> dict[str, pd.DataFrame]:
    derived_dir = section_dir / "derived"
    out: dict[str, pd.DataFrame] = {}
    for name in ("sporsa_signals", "arduino_signals", "cross_sensor_signals"):
        p = derived_dir / f"{name}.csv"
        if p.exists():
            try:
                out[name] = read_csv(p)
            except Exception as exc:
                log.warning("Could not read %s: %s", p, exc)
    return out


def _yclip(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """Return y-limits clipped to the lo/hi percentile of finite values."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return -1.0, 1.0
    ylo, yhi = float(np.percentile(finite, lo)), float(np.percentile(finite, hi))
    pad = max((yhi - ylo) * 0.1, 0.05)
    return ylo - pad, yhi + pad


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def plot_derived_overview(section_dir: Path) -> Path | None:
    """Multi-panel overview of all derived signals for both sensors."""
    dfs = _load_derived_csvs(section_dir)
    if not dfs:
        log.warning("No derived CSVs found in %s", section_dir / "derived")
        return None

    t0 = shared_t0_ms(*dfs.values())

    has_linear = any(
        "acc_linear_norm" in dfs.get(f"{s}_signals", pd.DataFrame()).columns
        for s in SENSORS
    )
    has_cross = "cross_sensor_signals" in dfs

    panels: list[tuple[str, str, float | None]] = [
        ("acc_norm",        "acc norm (m/s²)",     9.81),
        ("acc_vertical",    "acc vertical (m/s²)",  9.81),
        ("acc_hf",          "acc HF (m/s²)",        0.0),
        ("gyro_norm",       "gyro norm (°/s)",      None),
        ("jerk_norm",       "jerk norm (m/s³)",     None),
        ("energy_acc",      "energy acc (m/s²)",    None),
    ]
    if has_linear:
        panels.insert(3, ("acc_linear_norm", "linear acc norm (m/s²)", 0.0))

    n_rows = len(panels) + (1 if has_cross else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax_idx, (col, ylabel, hline) in enumerate(panels):
        ax = axes[ax_idx]
        any_data = False
        for sensor in SENSORS:
            df = dfs.get(f"{sensor}_signals")
            if df is None or col not in df.columns:
                continue
            ts = relative_seconds(
                pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float), t0)
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(ts, y)
            if xp.size == 0:
                continue
            ax.plot(xp, yp, lw=0.6, alpha=0.55,
                    color=SENSOR_COLORS[sensor], label=sensor)
            # Bold smoothed line for readability.
            ys = smooth_moving_average(yp, 50)
            xps, yps = filter_valid_plot_xy(xp, ys)
            if xps.size:
                ax.plot(xps, yps, lw=1.4, alpha=0.9,
                        color=SENSOR_COLORS[sensor])
            any_data = True
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.15, lw=0.4)
        if any_data:
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper right",
                      handlelength=1.2)

    if has_cross:
        ax = axes[len(panels)]
        cross_df = dfs["cross_sensor_signals"]
        ts = relative_seconds(
            pd.to_numeric(cross_df["timestamp"], errors="coerce").to_numpy(dtype=float), t0)
        for col, color, label in [
            ("disagree_score",  "#9467bd", "disagree score"),
            ("acc_correlation", "#8c564b", "acc correlation"),
        ]:
            if col not in cross_df.columns:
                continue
            y = pd.to_numeric(cross_df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(ts, y)
            if xp.size:
                ax.plot(xp, yp, lw=0.7, alpha=0.75, color=color, label=label)
        ax.set_ylabel("cross-sensor", fontsize=8)
        ax.grid(alpha=0.15, lw=0.4)
        ax.legend(fontsize=7, loc="upper right", handlelength=1.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{section_dir.name} — derived signals overview", fontsize=10)
    fig.tight_layout(h_pad=0.3)

    out_path = section_dir / "derived" / "derived_overview.png"
    return save_figure(fig, out_path, dpi=130)


# ---------------------------------------------------------------------------
# Linear acceleration detail
# ---------------------------------------------------------------------------

def plot_linear_acceleration(section_dir: Path) -> Path | None:
    """Per-axis linear acceleration, both sensors overlaid in each panel."""
    dfs = _load_derived_csvs(section_dir)

    lin_cols = ("acc_linear_x", "acc_linear_y", "acc_linear_z", "acc_linear_norm")
    available = {
        s: dfs[f"{s}_signals"]
        for s in SENSORS
        if f"{s}_signals" in dfs
        and all(c in dfs[f"{s}_signals"].columns for c in lin_cols)
    }
    if not available:
        log.warning("No linear acceleration columns in derived CSVs for %s", section_dir.name)
        return None

    t0 = shared_t0_ms(*dfs.values())
    method = _get_method_label(section_dir)

    axis_panels = [
        ("acc_linear_x",    "X  lateral (m/s²)",   _AXIS_COLORS["x"]),
        ("acc_linear_y",    "Y  forward (m/s²)",    _AXIS_COLORS["y"]),
        ("acc_linear_z",    "Z  vertical (m/s²)",   _AXIS_COLORS["z"]),
        ("acc_linear_norm", "norm (m/s²)",           "black"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for row, (col, ylabel, axis_color) in enumerate(axis_panels):
        ax = axes[row]
        all_y: list[np.ndarray] = []

        for sensor in SENSORS:
            df = available.get(sensor)
            if df is None:
                continue
            ts = relative_seconds(
                pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float), t0)
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(ts, y)
            if xp.size == 0:
                continue
            all_y.append(yp)
            # Raw signal, thin + transparent.
            ax.plot(xp, yp, lw=0.5, alpha=0.3, color=SENSOR_COLORS[sensor])
            # Smoothed overlay, bold.
            ys = smooth_moving_average(yp, 30)
            xps, yps = filter_valid_plot_xy(xp, ys)
            if xps.size:
                ax.plot(xps, yps, lw=1.5, alpha=0.9,
                        color=SENSOR_COLORS[sensor], label=sensor)

        ax.axhline(0.0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9, color=axis_color)
        ax.grid(alpha=0.15, lw=0.4)

        # Clip y-axis to signal range, ignoring outlier spikes.
        if all_y:
            combined = np.concatenate(all_y)
            ylo, yhi = _yclip(combined, lo=0.5, hi=99.5)
            if col != "acc_linear_norm":
                bound = max(abs(ylo), abs(yhi), 0.25)
                ylo, yhi = -bound, bound
            ax.set_ylim(ylo, yhi)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right",
                      handlelength=1.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{section_dir.name} — orientation-aware linear acceleration  [{method}]",
        fontsize=10,
    )
    fig.tight_layout(h_pad=0.4)

    out_path = section_dir / "derived" / "linear_acceleration.png"
    return save_figure(fig, out_path, dpi=130)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_method_label(section_dir: Path) -> str:
    data = load_json(section_dir / "orientation" / "orientation_stats.json")
    if data:
        return data.get("selected_method", "orientation")
    return "orientation"


def plot_derived_stage(target: str | Path) -> list[Path]:
    """Generate all derived signal plots for a section."""
    section_dir = resolve_data_dir(target)
    while section_dir != section_dir.parent:
        if (section_dir / "derived").is_dir() or (section_dir / "calibrated").is_dir():
            break
        section_dir = section_dir.parent

    out_paths: list[Path] = []
    for plot_fn in (plot_derived_overview, plot_linear_acceleration):
        try:
            p = plot_fn(section_dir)
        except Exception as exc:
            log.warning("%s failed for %s: %s", plot_fn.__name__, section_dir.name, exc)
            continue
        if p is not None:
            out_paths.append(p)
    return out_paths


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_derived")
    parser.add_argument("target", help="Section directory or name")
    args = parser.parse_args(argv)
    try:
        paths = plot_derived_stage(args.target)
    except Exception as exc:
        log.error("Failed to plot derived signals: %s", exc)
        return
    if not paths:
        print("No derived plots generated.")
    for p in paths:
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
