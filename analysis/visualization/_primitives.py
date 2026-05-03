"""primitives helpers for plot pipeline diagnostics and dataset summaries."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.signals import vector_norm
from visualization._utils import (
    ACC_COLS,
    GYRO_COLS,
    MAG_COLS,
    SENSOR_COLORS,
    filter_valid_plot_xy,
)

# ---------------------------------------------------------------------------
# Column-group labels (mirrors _utils constants, exposed for callers)
# ---------------------------------------------------------------------------

ACC_YLABEL = "Acc (m/s²)"
GYRO_YLABEL = "Gyro (°/s)"
MAG_YLABEL = "Mag (µT)"

# Maps a frozenset of xyz column names to the pre-computed norm column name.
_PRECOMPUTED_NORM: dict[frozenset, str] = {
    frozenset(ACC_COLS): "acc_norm",
    frozenset(GYRO_COLS): "gyro_norm",
    frozenset(MAG_COLS): "mag_norm",
}

_COL_COLORS: dict[str, str] = {
    "ax": "#1f77b4", "ay": "#ff7f0e", "az": "#2ca02c",
    "gx": "#d62728", "gy": "#9467bd", "gz": "#8c564b",
    "mx": "#e377c2", "my": "#7f7f7f", "mz": "#bcbd22",
}


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def imu_figure(
    n_rows: int,
    *,
    row_height: float = 3.0,
    width: float = 12.0,
    sharex: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Return imu figure."""
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(width, row_height * n_rows),
        sharex=sharex,
    )
    return fig, list(np.atleast_1d(axes))


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def draw_signal(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    label: str | None = None,
    color: str | None = None,
    lw: float = 0.8,
    alpha: float = 1.0,
    ylabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw a single time-series line on `ax`."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    xp, yp = filter_valid_plot_xy(t_arr, y_arr)
    kw: dict = {"lw": lw, "alpha": alpha}
    if label is not None:
        kw["label"] = label
    if color is not None:
        kw["color"] = color
    ax.plot(xp, yp, **kw)
    if label is not None:
        ax.legend(loc="upper right", fontsize=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def draw_signal_components(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    ylabel: str | None = None,
    title: str | None = None,
    lw: float = 0.7,
    alpha: float = 1.0,
) -> plt.Axes:
    """Draw signal components."""
    t_arr = np.asarray(t, dtype=float)
    drawn: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        xp, yp = filter_valid_plot_xy(t_arr, y)
        ax.plot(xp, yp, lw=lw, alpha=alpha, label=col, color=_COL_COLORS.get(col))
        drawn.append(col)
    if drawn:
        ax.legend(loc="upper right", fontsize=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def draw_signal_norm(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    label: str | None = None,
    color: str | None = None,
    lw: float = 0.8,
    alpha: float = 1.0,
    ylabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw signal norm."""
    precomputed = _PRECOMPUTED_NORM.get(frozenset(cols))
    if precomputed and precomputed in df.columns:
        norm = pd.to_numeric(df[precomputed], errors="coerce").to_numpy(dtype=float)
    else:
        present = [c for c in cols if c in df.columns]
        if not present:
            return ax
        norm = vector_norm(df, present)
    return draw_signal(
        ax,
        t,
        norm,
        label=label,
        color=color,
        lw=lw,
        alpha=alpha,
        ylabel=ylabel,
        title=title,
    )


def draw_two_streams(
    ax: plt.Axes,
    t_a: np.ndarray | pd.Series,
    y_a: np.ndarray | pd.Series,
    t_b: np.ndarray | pd.Series,
    y_b: np.ndarray | pd.Series,
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str | None = None,
    color_b: str | None = None,
    lw: float = 0.8,
    alpha: float = 0.8,
    ylabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw two streams."""
    color_a = color_a or SENSOR_COLORS.get(label_a.lower(), "#1f77b4")
    color_b = color_b or SENSOR_COLORS.get(label_b.lower(), "#ff7f0e")
    draw_signal(ax, t_a, y_a, label=label_a, color=color_a, lw=lw, alpha=alpha)
    draw_signal(ax, t_b, y_b, label=label_b, color=color_b, lw=lw, alpha=alpha)
    ax.legend(loc="upper right", fontsize=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def draw_vlines(
    ax: plt.Axes,
    times: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    color: str | Sequence[str] = "#555555",
    lw: float = 0.9,
    linestyle: str = "--",
    label_offset: float = 0.02,
) -> plt.Axes:
    """Add labelled vertical marker lines at the given `times`."""
    ymin, ymax = ax.get_ylim()
    for i, t in enumerate(times):
        c = color[i] if isinstance(color, (list, tuple)) and i < len(color) else (
            color if isinstance(color, str) else "#555555"
        )
        ax.axvline(t, color=c, lw=lw, linestyle=linestyle, zorder=3)
        if labels is not None and i < len(labels):
            ax.text(
                t,
                ymin + (ymax - ymin) * (1.0 - label_offset),
                labels[i],
                fontsize=6, ha="left", va="top", color=c,
            )
    return ax


def draw_hlines(
    ax: plt.Axes,
    ys: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    color: str | Sequence[str] = "#555555",
    lw: float = 0.9,
    linestyle: str = "--",
    alpha: float = 1.0,
    label_offset: float = 0.02,
) -> plt.Axes:
    """Add labelled horizontal reference lines at the given `ys`."""
    xmin, xmax = ax.get_xlim()
    for i, y in enumerate(ys):
        c = color[i] if isinstance(color, (list, tuple)) and i < len(color) else (
            color if isinstance(color, str) else "#555555"
        )
        ax.axhline(y, color=c, lw=lw, linestyle=linestyle, alpha=alpha, zorder=3)
        if labels is not None and i < len(labels):
            ax.text(
                xmin + (xmax - xmin) * (1.0 - label_offset),
                y,
                labels[i],
                fontsize=6, ha="right", va="bottom", color=c,
            )
    return ax


def draw_span(
    ax: plt.Axes,
    t_start: float,
    t_end: float,
    *,
    label: str | None = None,
    color: str = "#ffdd57",
    alpha: float = 0.25,
    zorder: int = 1,
) -> plt.Axes:
    """Shade the time region [`t_start`, `t_end`] on `ax`."""
    ax.axvspan(t_start, t_end, color=color, alpha=alpha, zorder=zorder, label=label)
    if label is not None:
        ax.legend(loc="upper right", fontsize=7)
    return ax


# ---------------------------------------------------------------------------
# Bar chart primitives
# ---------------------------------------------------------------------------

def draw_bars(
    ax: plt.Axes,
    positions: Sequence[float],
    heights: Sequence[float],
    *,
    tick_labels: Sequence[str] | None = None,
    color: str | Sequence[str] = "#1f77b4",
    width: float = 0.7,
    alpha: float = 0.9,
    ylabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw vertical bars at `positions` with the given `heights`."""
    for i, (x, h) in enumerate(zip(positions, heights)):
        if not np.isfinite(h):
            continue
        c = color[i] if isinstance(color, (list, tuple)) and i < len(color) else (
            color if isinstance(color, str) else "#1f77b4"
        )
        ax.bar(x, h, width=width, color=c, alpha=alpha)
    if tick_labels is not None:
        ax.set_xticks(list(positions))
        ax.set_xticklabels(list(tick_labels))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def draw_barh(
    ax: plt.Axes,
    positions: Sequence[float],
    widths: Sequence[float],
    *,
    tick_labels: Sequence[str] | None = None,
    color: str | Sequence[str] = "#1f77b4",
    height: float = 0.6,
    alpha: float = 0.9,
    xlabel: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw horizontal bars at `positions` with the given `widths`."""
    for i, (y, w) in enumerate(zip(positions, widths)):
        if not np.isfinite(w):
            continue
        c = color[i] if isinstance(color, (list, tuple)) and i < len(color) else (
            color if isinstance(color, str) else "#1f77b4"
        )
        ax.barh(y, w, height=height, color=c, alpha=alpha)
    if tick_labels is not None:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(list(tick_labels))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Sensor-specific convenience drawers
# ---------------------------------------------------------------------------

def draw_acc(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    *,
    norm: bool = False,
    label: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw accelerometer data (components or norm) on `ax`."""
    if norm:
        return draw_signal_norm(
            ax, t, df, ACC_COLS,
            label=label or "|acc|", color=color,
            ylabel=ACC_YLABEL, title=title,
        )
    return draw_signal_components(ax, t, df, ACC_COLS, ylabel=ACC_YLABEL, title=title)


def draw_gyro(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    *,
    norm: bool = False,
    label: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw gyroscope data (components or norm) on `ax`."""
    if norm:
        return draw_signal_norm(
            ax, t, df, GYRO_COLS,
            label=label or "|gyro|", color=color,
            ylabel=GYRO_YLABEL, title=title,
        )
    return draw_signal_components(ax, t, df, GYRO_COLS, ylabel=GYRO_YLABEL, title=title)


def draw_mag(
    ax: plt.Axes,
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    *,
    norm: bool = False,
    label: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw magnetometer data (components or norm) on `ax`."""
    if norm:
        return draw_signal_norm(
            ax, t, df, MAG_COLS,
            label=label or "|mag|", color=color,
            ylabel=MAG_YLABEL, title=title,
        )
    return draw_signal_components(ax, t, df, MAG_COLS, ylabel=MAG_YLABEL, title=title)


# ---------------------------------------------------------------------------
# Multi-row IMU panel
# ---------------------------------------------------------------------------

def draw_imu_rows(
    axes: list[plt.Axes],
    t: np.ndarray | pd.Series,
    df: pd.DataFrame,
    *,
    show_acc: bool = True,
    show_gyro: bool = True,
    show_mag: bool = True,
    norm: bool = False,
    label_prefix: str = "",
    color: str | None = None,
    title: str | None = None,
) -> list[plt.Axes]:
    """Draw imu rows."""
    row = 0
    row_kwargs: dict = {}
    if color is not None:
        row_kwargs["color"] = color

    def _label(base: str) -> str:
        """Return label."""
        return f"{label_prefix}{base}" if label_prefix else base

    if show_acc:
        ax = axes[row]
        if norm:
            draw_signal_norm(ax, t, df, ACC_COLS, label=_label("|acc|"), ylabel=ACC_YLABEL, **row_kwargs)
        else:
            draw_signal_components(ax, t, df, ACC_COLS, ylabel=ACC_YLABEL)
        if title and row == 0:
            ax.set_title(title)
        row += 1

    if show_gyro:
        ax = axes[row]
        if norm:
            draw_signal_norm(ax, t, df, GYRO_COLS, label=_label("|gyro|"), ylabel=GYRO_YLABEL, **row_kwargs)
        else:
            draw_signal_components(ax, t, df, GYRO_COLS, ylabel=GYRO_YLABEL)
        row += 1

    if show_mag:
        ax = axes[row]
        if norm:
            draw_signal_norm(ax, t, df, MAG_COLS, label=_label("|mag|"), ylabel=MAG_YLABEL, **row_kwargs)
        else:
            draw_signal_components(ax, t, df, MAG_COLS, ylabel=MAG_YLABEL)
        row += 1

    return axes
