"""Plot derived helpers for plot pipeline diagnostics and dataset summaries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import read_csv, resolve_data_dir
from common.signals import smooth_moving_average
from labels.parser import LabelRow, load_labels
from visualization._utils import (
    QUALITATIVE_PALETTE,
    SENSOR_COLORS,
    SENSORS,
    UNKNOWN_LABEL_COLOR,
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
    """Load derived csvs."""
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
    """Return get method label."""
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
    for plot_fn in (
        plot_derived_overview,
        plot_linear_acceleration,
        plot_derived_acceleration_family,
        plot_derived_rotation_family,
        plot_derived_energy_family,
        plot_derived_cross_family,
        plot_derived_summary,
    ):
        try:
            p = plot_fn(section_dir)
        except Exception as exc:
            log.warning("%s failed for %s: %s", plot_fn.__name__, section_dir.name, exc)
            continue
        if p is None:
            continue
        if isinstance(p, list):
            out_paths.extend(p)
        else:
            out_paths.append(p)
    return out_paths


# ---------------------------------------------------------------------------
# Per-family derived plots
# ---------------------------------------------------------------------------

_DERIVED_LABEL_ALPHA = 0.18


_ACCELERATION_PANELS: tuple[tuple[str, str, float | None], ...] = (
    ("acc_norm",      "|acc| (m/s²)",           9.81),
    ("acc_vertical",  "acc vertical (m/s²)",     9.81),
    ("acc_horizontal","acc horizontal (m/s²)",   0.0),
    ("acc_lf",        "acc LF (m/s²)",           9.81),
    ("acc_hf",        "acc HF (m/s²)",           0.0),
    ("acc_deviation", "acc deviation (m/s²)",    0.0),
    ("jerk_norm",     "|jerk| (m/s³)",           None),
)

_ROTATION_PANELS: tuple[tuple[str, str, float | None], ...] = (
    ("gyro_norm",   "|gyro| (°/s)",       None),
    ("gyro_lf",     "gyro LF (°/s)",       0.0),
    ("gyro_hf",     "gyro HF (°/s)",       0.0),
    ("alpha_norm",  "|angular accel|",     None),
)

_ENERGY_PANELS: tuple[tuple[str, str, float | None], ...] = (
    ("energy_acc",   "energy acc (m/s²)",  None),
    ("energy_gyro",  "energy gyro (°/s)",  None),
    ("alpha_energy", "alpha energy",       None),
)

_CROSS_PANELS: tuple[tuple[str, str, str], ...] = (
    ("acc_diff_norm",    "|acc bike - acc rider| (m/s²)", "#7f3fbf"),
    ("gyro_diff_norm",   "|gyro bike - gyro rider| (°/s)", "#9b59b6"),
    ("vertical_diff",    "vertical acc diff (m/s²)",       "#5b3a89"),
    ("acc_correlation",  "acc correlation",                "#3498db"),
    ("acc_dominance",    "acc dominance",                  "#16a085"),
    ("gyro_dominance",   "gyro dominance",                 "#1abc9c"),
    ("disagree_score",   "disagree score",                 "#e67e22"),
    ("xcorr_acc_max",    "xcorr acc (max)",                "#e74c3c"),
    ("xcorr_acc_lag_s",  "xcorr acc lag (s)",              "#c0392b"),
    ("xcorr_gyro_max",   "xcorr gyro (max)",               "#d35400"),
    ("xcorr_gyro_lag_s", "xcorr gyro lag (s)",             "#a04000"),
)


def _section_labels(section_dir: Path) -> list[LabelRow]:
    """Return section labels for span overlay (empty if absent)."""
    labels_path = section_dir / "labels" / "labels.csv"
    if not labels_path.exists():
        return []
    return load_labels(labels_path)


def _label_color_map(labels: list[LabelRow]) -> dict[str, str]:
    """Stable color per label name."""
    names = sorted({lr.label for lr in labels if lr.label})
    return {n: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, n in enumerate(names)}


def _draw_label_spans(
    axes: list[plt.Axes],
    labels: list[LabelRow],
    t0_ms: float,
    colors: dict[str, str],
) -> None:
    """Overlay label intervals as colored bands on each axis."""
    for lr in labels:
        if not lr.label:
            continue
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, UNKNOWN_LABEL_COLOR)
        alpha = _DERIVED_LABEL_ALPHA * max(0.3, getattr(lr, "confidence", 1.0) or 1.0)
        for ax in axes:
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _label_legend_handles(colors: dict[str, str]) -> list[mpatches.Patch]:
    """Return label legend handles."""
    return [mpatches.Patch(color=c, alpha=0.6, label=n) for n, c in colors.items()]


def _plot_per_sensor_panels(
    section_dir: Path,
    panels: tuple[tuple[str, str, float | None], ...],
    out_path: Path,
    title: str,
    *,
    smooth_window: int = 30,
) -> Path | None:
    """Plot a stack of derived-signal panels with bike+rider overlaid."""
    dfs = _load_derived_csvs(section_dir)
    if not dfs:
        return None

    # Filter to panels that have data in at least one sensor.
    available = [
        (col, ylabel, hline)
        for col, ylabel, hline in panels
        if any(col in dfs.get(f"{s}_signals", pd.DataFrame()).columns for s in SENSORS)
    ]
    if not available:
        return None

    t0 = shared_t0_ms(*dfs.values())
    labels = _section_labels(section_dir)
    label_colors = _label_color_map(labels)

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3, 1.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, ylabel, hline) in zip(axes, available):
        all_y: list[np.ndarray] = []
        for sensor in SENSORS:
            df = dfs.get(f"{sensor}_signals")
            if df is None or col not in df.columns:
                continue
            ts = relative_seconds(
                pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float), t0,
            )
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(ts, y)
            if xp.size == 0:
                continue
            ax.plot(xp, yp, lw=0.5, alpha=0.4, color=SENSOR_COLORS[sensor])
            ys = smooth_moving_average(yp, smooth_window)
            xps, yps = filter_valid_plot_xy(xp, ys)
            if xps.size:
                ax.plot(xps, yps, lw=1.4, alpha=0.9, color=SENSOR_COLORS[sensor], label=sensor)
            all_y.append(yp)

        _draw_label_spans([ax], labels, t0, label_colors)

        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.6, ls="--", alpha=0.5)
        if all_y:
            combined = np.concatenate(all_y)
            ylo, yhi = _yclip(combined, lo=0.5, hi=99.5)
            ax.set_ylim(ylo, yhi)

        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.18, lw=0.4)

    sensor_handles = [mpatches.Patch(color=SENSOR_COLORS[s], label=s) for s in SENSORS]
    handles = sensor_handles + _label_legend_handles(label_colors)
    if handles:
        axes[0].legend(
            handles=handles, loc="upper right",
            ncol=2, fontsize=7, framealpha=0.85,
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{section_dir.name} — {title}", fontsize=10)
    fig.tight_layout(h_pad=0.3)
    return save_figure(fig, out_path, dpi=130)


def plot_derived_acceleration_family(section_dir: Path) -> Path | None:
    """Plot the acceleration-family derived signals."""
    return _plot_per_sensor_panels(
        section_dir,
        _ACCELERATION_PANELS,
        section_dir / "derived" / "family_acceleration.png",
        "acceleration family",
    )


def plot_derived_rotation_family(section_dir: Path) -> Path | None:
    """Plot the rotation-family derived signals (gyro / alpha)."""
    return _plot_per_sensor_panels(
        section_dir,
        _ROTATION_PANELS,
        section_dir / "derived" / "family_rotation.png",
        "rotation family",
    )


def plot_derived_energy_family(section_dir: Path) -> Path | None:
    """Plot the energy-family derived signals."""
    return _plot_per_sensor_panels(
        section_dir,
        _ENERGY_PANELS,
        section_dir / "derived" / "family_energy.png",
        "energy family",
    )


def plot_derived_cross_family(section_dir: Path) -> Path | None:
    """Plot all cross-sensor derived signals stacked vertically."""
    dfs = _load_derived_csvs(section_dir)
    cross_df = dfs.get("cross_sensor_signals")
    if cross_df is None or cross_df.empty:
        log.debug("No cross_sensor_signals.csv for %s", section_dir.name)
        return None

    available = [
        (col, ylabel, color)
        for col, ylabel, color in _CROSS_PANELS
        if col in cross_df.columns
        and pd.to_numeric(cross_df[col], errors="coerce").notna().any()
    ]
    if not available:
        log.debug("No plottable cross-sensor columns for %s", section_dir.name)
        return None

    t0 = shared_t0_ms(*dfs.values())
    labels = _section_labels(section_dir)
    label_colors = _label_color_map(labels)

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3, 1.5 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    ts = relative_seconds(
        pd.to_numeric(cross_df["timestamp"], errors="coerce").to_numpy(dtype=float), t0,
    )

    for ax, (col, ylabel, color) in zip(axes, available):
        y = pd.to_numeric(cross_df[col], errors="coerce").to_numpy(dtype=float)
        xp, yp = filter_valid_plot_xy(ts, y)
        if xp.size:
            ax.plot(xp, yp, lw=0.6, alpha=0.45, color=color)
            ys = smooth_moving_average(yp, 30)
            xps, yps = filter_valid_plot_xy(xp, ys)
            if xps.size:
                ax.plot(xps, yps, lw=1.4, alpha=0.9, color=color)
            ylo, yhi = _yclip(yp, lo=0.5, hi=99.5)
            ax.set_ylim(ylo, yhi)
        _draw_label_spans([ax], labels, t0, label_colors)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.18, lw=0.4)

    if label_colors:
        axes[0].legend(
            handles=_label_legend_handles(label_colors),
            loc="upper right", fontsize=7, framealpha=0.85, ncol=3,
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{section_dir.name} — cross-sensor family", fontsize=10)
    fig.tight_layout(h_pad=0.3)

    out_path = section_dir / "derived" / "family_cross.png"
    return save_figure(fig, out_path, dpi=130)


# ---------------------------------------------------------------------------
# Derived signals summary
# ---------------------------------------------------------------------------

def _column_stats(series: pd.Series) -> dict[str, float]:
    """Return basic stats for a numeric series (NaN-safe)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p5": float("nan"), "p95": float("nan"),
                "valid_frac": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "std":  float(np.std(finite)),
        "p5":   float(np.percentile(finite, 5)),
        "p95":  float(np.percentile(finite, 95)),
        "valid_frac": float(finite.size / arr.size),
    }


def plot_derived_summary(section_dir: Path) -> Path | None:
    """Summary figure: signal counts, valid coverage, and per-sensor distributions."""
    dfs = _load_derived_csvs(section_dir)
    if not dfs:
        return None

    sporsa_df = dfs.get("sporsa_signals", pd.DataFrame())
    arduino_df = dfs.get("arduino_signals", pd.DataFrame())
    cross_df = dfs.get("cross_sensor_signals", pd.DataFrame())

    # Per-sensor signal coverage (valid fraction per column).
    def _coverage(df: pd.DataFrame) -> dict[str, float]:
        """Coverage."""
        out: dict[str, float] = {}
        if df.empty:
            return out
        for c in df.columns:
            if c == "timestamp":
                continue
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if arr.size == 0:
                continue
            out[c] = float(np.isfinite(arr).mean())
        return out

    cov_sporsa = _coverage(sporsa_df)
    cov_arduino = _coverage(arduino_df)
    cov_cross = _coverage(cross_df)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.45)

    # 1. Number of derived signals per source.
    ax = fig.add_subplot(gs[0, 0])
    counts = {
        "sporsa": len(cov_sporsa),
        "arduino": len(cov_arduino),
        "cross": len(cov_cross),
    }
    bar_colors = [SENSOR_COLORS.get("sporsa", "#1f77b4"),
                  SENSOR_COLORS.get("arduino", "#ff7f0e"),
                  "#7f3fbf"]
    ax.bar(list(counts.keys()), list(counts.values()), color=bar_colors, edgecolor="white")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)
    ax.set_title("Derived signals per source", fontsize=10)
    ax.set_ylabel("# columns")
    ax.grid(axis="y", alpha=0.2)

    # 2. Valid-fraction per signal (sporsa).
    ax = fig.add_subplot(gs[0, 1])
    if cov_sporsa:
        items = sorted(cov_sporsa.items(), key=lambda x: x[1])
        names, vals = zip(*items)
        ax.barh(list(names), list(vals), color=SENSOR_COLORS["sporsa"], edgecolor="white")
        ax.set_xlim(0, 1.02)
    ax.set_title("Sporsa signal valid fraction", fontsize=10)
    ax.set_xlabel("valid fraction")
    ax.grid(axis="x", alpha=0.2)

    # 3. Valid-fraction per signal (arduino).
    ax = fig.add_subplot(gs[0, 2])
    if cov_arduino:
        items = sorted(cov_arduino.items(), key=lambda x: x[1])
        names, vals = zip(*items)
        ax.barh(list(names), list(vals), color=SENSOR_COLORS["arduino"], edgecolor="white")
        ax.set_xlim(0, 1.02)
    ax.set_title("Arduino signal valid fraction", fontsize=10)
    ax.set_xlabel("valid fraction")
    ax.grid(axis="x", alpha=0.2)

    # 4. Per-sensor mean/p5/p95 for the canonical kinematic signals.
    canonical = ("acc_norm", "gyro_norm", "jerk_norm", "acc_hf", "alpha_norm", "energy_acc")
    ax = fig.add_subplot(gs[1, 0:2])
    rows: list[dict] = []
    for sensor, df in (("sporsa", sporsa_df), ("arduino", arduino_df)):
        for col in canonical:
            if col in df.columns:
                stats = _column_stats(df[col])
                rows.append({"sensor": sensor, "signal": col, **stats})
    if rows:
        labels = [f"{r['signal']}\n({r['sensor']})" for r in rows]
        means = [r["mean"] for r in rows]
        p5s = [r["mean"] - r["p5"] for r in rows]
        p95s = [r["p95"] - r["mean"] for r in rows]
        bar_colors = [SENSOR_COLORS[r["sensor"]] for r in rows]
        x = np.arange(len(rows))
        ax.bar(
            x, means, yerr=[p5s, p95s], color=bar_colors, edgecolor="white",
            capsize=2, ecolor="#555", alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("Canonical signals — mean with [p5, p95] whiskers", fontsize=10)
    ax.set_ylabel("value")
    ax.grid(axis="y", alpha=0.2)

    # 5. Cross signal valid-fraction.
    ax = fig.add_subplot(gs[1, 2])
    if cov_cross:
        items = sorted(cov_cross.items(), key=lambda x: x[1])
        names, vals = zip(*items)
        ax.barh(list(names), list(vals), color="#7f3fbf", edgecolor="white")
        ax.set_xlim(0, 1.02)
    ax.set_title("Cross signal valid fraction", fontsize=10)
    ax.set_xlabel("valid fraction")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle(f"{section_dir.name} — derived signals summary", fontsize=12, y=0.995)

    out_path = section_dir / "derived" / "derived_summary.png"
    return save_figure(fig, out_path, dpi=130)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
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
