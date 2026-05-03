"""Signal examples helpers for build report tables, figures, and thesis bundles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv, sections_root
from visualization._utils import (
    AXIS_COLORS as _AXIS_COLORS,
    NORM_COLOR as _NORM_COLOR,
    SENSOR_COLORS,
    label_color as _label_color,
)

log = logging.getLogger(__name__)

_DPI = 200
_BIKE_COLOR = SENSOR_COLORS["sporsa"]
_RIDER_COLOR = SENSOR_COLORS["arduino"]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def find_representative_windows(
    df: pd.DataFrame,
    scenario: str,
    n: int = 1,
) -> pd.DataFrame:
    """Return find representative windows."""
    mask = df["scenario_label"] == scenario
    sub = df[mask].copy()
    if sub.empty:
        return sub

    sort_cols = []
    if "overall_quality_score" in sub.columns:
        sort_cols.append("overall_quality_score")
    if "window_n_samples_sporsa" in sub.columns:
        sort_cols.append("window_n_samples_sporsa")

    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=False)

    return sub.head(n)


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def load_section_signals(section_id: str) -> dict[str, pd.DataFrame]:
    """Load section signals."""
    sec_dir = sections_root() / section_id
    out: dict[str, pd.DataFrame] = {}

    def _load(key: str, path: Path) -> None:
        """Load."""
        out[key] = read_csv(path) if path.exists() else pd.DataFrame()

    _load("bike_cal",         sec_dir / "calibrated" / "sporsa.csv")
    _load("rider_cal",        sec_dir / "calibrated" / "arduino.csv")
    _load("bike_derived",     sec_dir / "derived"    / "sporsa_signals.csv")
    _load("rider_derived",    sec_dir / "derived"    / "arduino_signals.csv")
    _load("bike_orientation", sec_dir / "orientation" / "sporsa.csv")
    _load("labels",           sec_dir / "labels"     / "labels.csv")

    return out


def _crop(df: pd.DataFrame, t_lo_ms: float, t_hi_ms: float) -> pd.DataFrame:
    """Return rows where timestamp (ms) is within [t_lo_ms, t_hi_ms]."""
    if df.empty or "timestamp" not in df.columns:
        return df
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    mask = (ts >= t_lo_ms) & (ts <= t_hi_ms)
    return df[mask].copy()


def _to_s(ts_ms: pd.Series | np.ndarray, t0_ms: float) -> np.ndarray:
    """Convert millisecond timestamps to seconds relative to t0_ms."""
    return (pd.to_numeric(ts_ms, errors="coerce").to_numpy(dtype=float) - t0_ms) / 1000.0


def _norm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Row-wise vector norm, NaN for rows with any NaN."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.full(len(df), np.nan)
    arr = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols]
    )
    out = np.full(arr.shape[0], np.nan)
    valid = np.isfinite(arr).all(axis=1)
    out[valid] = np.linalg.norm(arr[valid], axis=1)
    return out


def _yclim(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Return yclim."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return -1.0, 1.0
    ylo, yhi = float(np.percentile(finite, lo)), float(np.percentile(finite, hi))
    pad = max((yhi - ylo) * 0.12, 0.1)
    return ylo - pad, yhi + pad


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_signal_example(
    section_id: str,
    window_start_ms: float,
    window_end_ms: float,
    output_path: Path,
    *,
    context_s: float = 5.0,
    scenario_label: str = "",
) -> Optional[Path]:
    """Plot signal example."""
    signals = load_section_signals(section_id)

    context_ms = context_s * 1000.0
    view_lo = window_start_ms - context_ms
    view_hi = window_end_ms + context_ms

    # Crop all signals to the view window
    bike_cal   = _crop(signals["bike_cal"],         view_lo, view_hi)
    rider_cal  = _crop(signals["rider_cal"],        view_lo, view_hi)
    bike_der   = _crop(signals["bike_derived"],     view_lo, view_hi)
    rider_der  = _crop(signals["rider_derived"],    view_lo, view_hi)
    bike_ori   = _crop(signals["bike_orientation"], view_lo, view_hi)
    labels_df  = signals["labels"]

    # Need at least bike or rider calibrated data to plot
    if bike_cal.empty and rider_cal.empty:
        log.warning("No calibrated signal data for section %s; skipping", section_id)
        return None

    t0_ms = view_lo  # reference for time axis (start of view)

    # Build plot panels
    n_panels = 6
    height_ratios = [2, 2, 1.5, 1.5, 1.5, 0.6]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )

    def _shade_window(ax: plt.Axes) -> None:
        """Return shade window."""
        ax.axvspan(
            (window_start_ms - t0_ms) / 1000.0,
            (window_end_ms   - t0_ms) / 1000.0,
            alpha=0.12, color="#FFC107", zorder=0,
        )

    # ---- Panel 0: Bike accelerometer ----
    ax = axes[0]
    if not bike_cal.empty and "timestamp" in bike_cal.columns:
        t = _to_s(bike_cal["timestamp"], t0_ms)
        for axis, col in (("x", "ax"), ("y", "ay"), ("z", "az")):
            if col in bike_cal.columns:
                ax.plot(t, pd.to_numeric(bike_cal[col], errors="coerce"),
                        color=_AXIS_COLORS[axis], lw=0.8, alpha=0.7, label=axis)
        norm_vals = _norm(bike_cal, ["ax", "ay", "az"])
        ax.plot(t, norm_vals, color=_NORM_COLOR, lw=1.2, alpha=0.9, label="norm")
        ax.set_ylim(*_yclim(norm_vals))
    ax.set_ylabel("Bike acc\n(m/s²)", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.7, ncol=4)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _shade_window(ax)

    # ---- Panel 1: Rider accelerometer ----
    ax = axes[1]
    if not rider_cal.empty and "timestamp" in rider_cal.columns:
        t = _to_s(rider_cal["timestamp"], t0_ms)
        for axis, col in (("x", "ax"), ("y", "ay"), ("z", "az")):
            if col in rider_cal.columns:
                ax.plot(t, pd.to_numeric(rider_cal[col], errors="coerce"),
                        color=_AXIS_COLORS[axis], lw=0.8, alpha=0.7, label=axis)
        norm_vals = _norm(rider_cal, ["ax", "ay", "az"])
        ax.plot(t, norm_vals, color=_NORM_COLOR, lw=1.2, alpha=0.9, label="norm")
        ax.set_ylim(*_yclim(norm_vals))
    ax.set_ylabel("Rider acc\n(m/s²)", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.7, ncol=4)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _shade_window(ax)

    # ---- Panel 2: Vertical acceleration ----
    ax = axes[2]
    plotted_any = False
    if not bike_der.empty and "acc_vertical" in bike_der.columns:
        t = _to_s(bike_der["timestamp"], t0_ms)
        vals = pd.to_numeric(bike_der["acc_vertical"], errors="coerce")
        ax.plot(t, vals, color=_BIKE_COLOR, lw=1.0, alpha=0.9, label="bike")
        plotted_any = True
    if not rider_der.empty and "acc_vertical" in rider_der.columns:
        t = _to_s(rider_der["timestamp"], t0_ms)
        vals = pd.to_numeric(rider_der["acc_vertical"], errors="coerce")
        ax.plot(t, vals, color=_RIDER_COLOR, lw=1.0, alpha=0.9, label="rider")
        plotted_any = True
    ax.set_ylabel("Vertical acc\n(m/s²)", fontsize=8)
    if plotted_any:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _shade_window(ax)

    # ---- Panel 3: Jerk norm ----
    ax = axes[3]
    plotted_any = False
    if not bike_der.empty and "jerk_norm" in bike_der.columns:
        t = _to_s(bike_der["timestamp"], t0_ms)
        vals = pd.to_numeric(bike_der["jerk_norm"], errors="coerce")
        ax.plot(t, vals, color=_BIKE_COLOR, lw=1.0, alpha=0.9, label="bike")
        plotted_any = True
    if not rider_der.empty and "jerk_norm" in rider_der.columns:
        t = _to_s(rider_der["timestamp"], t0_ms)
        vals = pd.to_numeric(rider_der["jerk_norm"], errors="coerce")
        ax.plot(t, vals, color=_RIDER_COLOR, lw=1.0, alpha=0.9, label="rider")
        plotted_any = True
    ax.set_ylabel("Jerk norm\n(m/s³)", fontsize=8)
    if plotted_any:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _shade_window(ax)

    # ---- Panel 4: Bike pitch and roll ----
    ax = axes[4]
    if not bike_ori.empty and "timestamp" in bike_ori.columns:
        t = _to_s(bike_ori["timestamp"], t0_ms)
        if "pitch_deg" in bike_ori.columns:
            ax.plot(t, pd.to_numeric(bike_ori["pitch_deg"], errors="coerce"),
                    color="#e41a1c", lw=1.0, alpha=0.9, label="pitch")
        if "roll_deg" in bike_ori.columns:
            ax.plot(t, pd.to_numeric(bike_ori["roll_deg"], errors="coerce"),
                    color="#4daf4a", lw=1.0, alpha=0.9, label="roll")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.set_ylabel("Bike orient.\n(deg)", fontsize=8)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _shade_window(ax)

    # ---- Panel 5: Label strip ----
    ax = axes[5]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    label_legend_handles: list[mpatches.Patch] = []
    seen_labels: set[str] = set()

    if not labels_df.empty and {"start_ms", "end_ms", "scenario_label"}.issubset(labels_df.columns):
        all_labels = sorted(labels_df["scenario_label"].dropna().unique().tolist())
        for _, lrow in labels_df.iterrows():
            lbl = str(lrow["scenario_label"]) if pd.notna(lrow["scenario_label"]) else "unlabeled"
            ls_ms = float(pd.to_numeric(lrow["start_ms"], errors="coerce"))
            le_ms = float(pd.to_numeric(lrow["end_ms"],   errors="coerce"))
            if np.isnan(ls_ms) or np.isnan(le_ms):
                continue
            # Only draw if overlaps view
            if le_ms < view_lo or ls_ms > view_hi:
                continue
            # Clip to view
            ls_s = (max(ls_ms, view_lo) - t0_ms) / 1000.0
            le_s = (min(le_ms, view_hi) - t0_ms) / 1000.0
            color = _label_color(lbl, all_labels)
            ax.barh(0.5, le_s - ls_s, left=ls_s, height=0.8,
                    color=color, alpha=0.85, linewidth=0)
            if lbl not in seen_labels:
                label_legend_handles.append(mpatches.Patch(color=color, label=lbl))
                seen_labels.add(lbl)
    else:
        # Grey bar for unlabeled
        view_s = (view_hi - view_lo) / 1000.0
        ax.barh(0.5, view_s, left=0, height=0.8, color="#BDBDBD", alpha=0.7)

    ax.set_ylabel("Labels", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)

    if label_legend_handles:
        ax.legend(handles=label_legend_handles, loc="upper right",
                  fontsize=6, framealpha=0.7)

    _shade_window(ax)

    # ---- Window boundary lines on all axes ----
    win_lo_s = (window_start_ms - t0_ms) / 1000.0
    win_hi_s = (window_end_ms   - t0_ms) / 1000.0
    for ax in axes:
        ax.axvline(win_lo_s, color="#FFC107", lw=1.0, linestyle="--", alpha=0.7, zorder=5)
        ax.axvline(win_hi_s, color="#FFC107", lw=1.0, linestyle="--", alpha=0.7, zorder=5)

    # ---- X-axis limits ----
    view_s = (view_hi - view_lo) / 1000.0
    axes[-1].set_xlim(0, view_s)

    # ---- Title ----
    title = f"Signal example: {scenario_label}" if scenario_label else f"Signal example — {section_id}"
    fig.suptitle(title, fontsize=11, y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote signal example → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Cross-sensor comparison
# ---------------------------------------------------------------------------

def plot_cross_sensor_comparison(
    section_id: str,
    window_start_ms: float,
    window_end_ms: float,
    output_path: Path,
    *,
    context_s: float = 5.0,
    scenario_label: str = "",
) -> Optional[Path]:
    """Two-panel bike vs rider accelerometer norm comparison."""
    signals = load_section_signals(section_id)

    context_ms = context_s * 1000.0
    view_lo = window_start_ms - context_ms
    view_hi = window_end_ms + context_ms
    t0_ms = view_lo

    bike_cal  = _crop(signals["bike_cal"],  view_lo, view_hi)
    rider_cal = _crop(signals["rider_cal"], view_lo, view_hi)

    if bike_cal.empty and rider_cal.empty:
        log.warning("No data for cross-sensor comparison in %s; skipping", section_id)
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"hspace": 0.08})

    win_lo_s = (window_start_ms - t0_ms) / 1000.0
    win_hi_s = (window_end_ms   - t0_ms) / 1000.0

    def _panel(ax: plt.Axes, df: pd.DataFrame, color: str, ylabel: str) -> None:
        """Return panel."""
        if not df.empty and "timestamp" in df.columns:
            t = _to_s(df["timestamp"], t0_ms)
            norm_vals = _norm(df, ["ax", "ay", "az"])
            ax.plot(t, norm_vals, color=color, lw=1.2, alpha=0.9)
            ax.set_ylim(*_yclim(norm_vals))
        ax.axvspan(win_lo_s, win_hi_s, alpha=0.12, color="#FFC107", zorder=0)
        ax.axvline(win_lo_s, color="#FFC107", lw=1.0, linestyle="--", alpha=0.7)
        ax.axvline(win_hi_s, color="#FFC107", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _panel(axes[0], bike_cal,  _BIKE_COLOR,  "Bike acc norm (m/s²)")
    _panel(axes[1], rider_cal, _RIDER_COLOR, "Rider acc norm (m/s²)")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    view_s = (view_hi - view_lo) / 1000.0
    axes[-1].set_xlim(0, view_s)

    title = f"Cross-sensor comparison: {scenario_label}" if scenario_label else "Cross-sensor comparison"
    fig.suptitle(title, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote cross-sensor comparison → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

_CROSS_SENSOR_SCENARIOS = {"braking", "hard_braking", "riding", "cornering", "fall"}


def generate_all_signal_examples(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    context_s: float = 5.0,
    scenarios: list[str] | None = None,
) -> list[Path]:
    """Generate all signal examples."""
    required = {"scenario_label", "section_id", "window_start_ms", "window_end_ms"}
    if not required.issubset(df.columns):
        log.warning("Missing required columns for signal examples; skipping")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    all_scenarios = sorted(df["scenario_label"].dropna().unique().tolist())
    if scenarios is not None:
        all_scenarios = [s for s in all_scenarios if s in scenarios]

    for scenario in all_scenarios:
        windows = find_representative_windows(df, scenario, n=1)
        if windows.empty:
            log.debug("No windows found for scenario '%s'; skipping", scenario)
            continue

        row = windows.iloc[0]
        section_id = str(row["section_id"])
        win_start = float(row["window_start_ms"])
        win_end   = float(row["window_end_ms"])

        out_path = output_dir / f"signal_example_{scenario}.png"
        result = plot_signal_example(
            section_id, win_start, win_end, out_path,
            context_s=context_s, scenario_label=scenario,
        )
        if result is not None:
            generated.append(result)

        # Cross-sensor comparison for selected scenarios
        if scenario in _CROSS_SENSOR_SCENARIOS:
            cross_path = output_dir / f"cross_sensor_{scenario}.png"
            result = plot_cross_sensor_comparison(
                section_id, win_start, win_end, cross_path,
                context_s=context_s, scenario_label=scenario,
            )
            if result is not None:
                generated.append(result)

    log.info(
        "Signal examples: %d written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated
