"""Plot feature-stage outputs from section-level ``features/features.csv``."""

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
from labels.parser import LabelRow, load_labels
from visualization._utils import (
    QUALITATIVE_PALETTE,
    SENSOR_COLORS,
    UNKNOWN_LABEL_COLOR,
    filter_valid_plot_xy,
    save_figure,
)

log = logging.getLogger(__name__)

# bike features come from the sporsa sensor; rider features from arduino.
_FEATURE_PREFIX_COLORS: dict[str, str] = {
    "bike":  SENSOR_COLORS["sporsa"],
    "rider": SENSOR_COLORS["arduino"],
}

_META_COLS = frozenset({
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "window_n_samples_sporsa",
    "window_n_samples_arduino",
    "window_valid_ratio_sporsa",
    "window_valid_ratio_arduino",
    "window_type",
    "label",
    "scenario_label",
    "scenario_labels",
    "overall_quality_score",
    "overall_quality_label",
    "quality_tier",
    "quality_bike",
    "quality_rider",
    "quality_alignment",
    "quality_calibration",
    "quality_cross",
    "calibration_quality",
    "sync_confidence",
})
_DEFAULT_PREFIXES = ("bike_", "rider_", "cross_")
_LABEL_ALPHA = 0.18


# Feature families used for the per-family time-series plots.
# Each family defines (signals, stats) — both must combine into a column name
# of the form ``{sensor}_{signal}_{stat}`` for bike/rider, or appear as raw
# ``{sensor}_{signal}_{stat}`` columns. The plotting code skips combinations
# that are absent from the dataframe.
_FEATURE_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "kinematic": {
        "signals": ("acc_norm", "gyro_norm", "jerk_norm", "acc_vertical"),
        "stats": ("mean", "std", "max", "energy"),
    },
    "high_freq": {
        "signals": ("acc_hf", "gyro_hf", "acc_norm", "gyro_norm"),
        "stats": ("dominant_freq", "spectral_centroid", "spectral_energy_high", "spectral_energy_mid"),
    },
    "events": {
        "signals": ("acc_deviation", "jerk_norm", "alpha_norm", "energy_acc"),
        "stats": ("peak_count", "peak_max", "peak_prominence_max", "peak_width_mean"),
    },
    "shape": {
        "signals": ("acc_norm", "jerk_norm", "energy_acc", "alpha_norm"),
        "stats": ("skew", "kurtosis", "iqr", "zero_crossings"),
    },
}

_ORIENTATION_FAMILY = {
    "signals": ("pitch", "roll", "yaw_rate"),
    "stats": ("mean", "std", "range", "rate_std"),
}

_FAMILY_TITLES = {
    "kinematic":  "Kinematic features (mean / std / max / energy)",
    "high_freq":  "High-frequency / spectral features",
    "events":     "Event-detection features (peaks)",
    "shape":      "Distribution-shape features",
    "orientation":"Orientation-derived features",
    "cross":      "Cross-sensor features",
}


# ---------------------------------------------------------------------------
# I/O and prep helpers
# ---------------------------------------------------------------------------

def _resolve_features_csv(target: str | Path) -> Path:
    """Resolve features csv from a section dir, features dir, or csv path."""
    base = resolve_data_dir(target)
    if base.name == "features" and (base / "features.csv").exists():
        return base / "features.csv"
    maybe = base / "features" / "features.csv"
    if maybe.exists():
        return maybe
    direct = Path(str(target)).expanduser()
    if direct.is_file() and direct.suffix.lower() == ".csv":
        return direct.resolve()
    raise FileNotFoundError(f"Could not resolve features CSV from: {target}")


def _resolve_section_dir(target: str | Path) -> Path:
    """Resolve the section directory enclosing the target reference."""
    base = resolve_data_dir(target)
    if base.name == "features":
        return base.parent
    return base


def _prepare_feature_time_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Sort feature rows by start time and return centered relative-second x."""
    df = df.copy()
    df["window_start_ms"] = pd.to_numeric(df["window_start_ms"], errors="coerce")
    df["window_end_ms"] = pd.to_numeric(
        df.get("window_end_ms", df["window_start_ms"]), errors="coerce",
    )
    df = (
        df.dropna(subset=["window_start_ms"])
        .sort_values("window_start_ms")
        .reset_index(drop=True)
    )
    end_ms = df["window_end_ms"].fillna(df["window_start_ms"])
    center_ms = ((df["window_start_ms"] + end_ms) / 2.0).to_numpy(dtype=float)
    t0_ms = float(df["window_start_ms"].iloc[0]) if not df.empty else 0.0
    return df, (center_ms - t0_ms) / 1000.0, t0_ms


def _section_labels(section_dir: Path) -> list[LabelRow]:
    """Load section-level labels for span overlays (returns [] if absent)."""
    labels_path = section_dir / "labels" / "labels.csv"
    if not labels_path.exists():
        return []
    return load_labels(labels_path)


def _label_color_map(labels: list[LabelRow]) -> dict[str, str]:
    """Stable colors for each label name, drawn from QUALITATIVE_PALETTE."""
    names = sorted({lr.label for lr in labels if lr.label})
    return {n: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, n in enumerate(names)}


def _draw_label_spans(
    axes: list[plt.Axes],
    labels: list[LabelRow],
    t0_ms: float,
    colors: dict[str, str],
) -> None:
    """Draw colored axvspan regions on each ax for label intervals."""
    for lr in labels:
        if not lr.label:
            continue
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, UNKNOWN_LABEL_COLOR)
        alpha = _LABEL_ALPHA * max(0.3, getattr(lr, "confidence", 1.0) or 1.0)
        for ax in axes:
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _label_legend_handles(colors: dict[str, str]) -> list[mpatches.Patch]:
    """Return label legend handles."""
    return [mpatches.Patch(color=c, alpha=0.6, label=name) for name, c in colors.items()]


# ---------------------------------------------------------------------------
# Existing trend overview (kept; minor cleanup so it sorts time)
# ---------------------------------------------------------------------------

def _select_top_variance_columns(df: pd.DataFrame, top_n: int) -> list[str]:
    """Pick the most variable feature columns, preferring sensor-prefixed ones."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = [c for c in numeric_cols if c not in _META_COLS]
    if not candidates:
        return []
    prefixed = [c for c in candidates if c.startswith(_DEFAULT_PREFIXES)]
    if prefixed:
        candidates = prefixed
    variances = (
        df[candidates]
        .apply(pd.to_numeric, errors="coerce")
        .var(axis=0, skipna=True)
        .sort_values(ascending=False)
    )
    return [c for c in variances.index.tolist() if np.isfinite(variances[c])][:top_n]


def plot_features_stage(
    target: str | Path,
    *,
    top_n: int = 8,
) -> Path | None:
    """Compact multi-panel trend plot of the most variable features over time."""
    features_csv = _resolve_features_csv(target)
    df = read_csv(features_csv)
    if df.empty:
        log.warning("Features CSV is empty: %s", features_csv)
        return None

    if "window_start_ms" not in df.columns:
        log.warning("features.csv missing 'window_start_ms' column: %s", features_csv)
        return None

    df, t_s, _ = _prepare_feature_time_axis(df)
    feat_cols = _select_top_variance_columns(df, top_n=max(1, int(top_n)))
    if not feat_cols:
        log.warning("No numeric feature columns found in %s", features_csv)
        return None

    rows = len(feat_cols)
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3, 2.1 * rows)), sharex=True)
    if rows == 1:
        axes = [axes]

    for idx, col in enumerate(feat_cols):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x_plot, y_plot = filter_valid_plot_xy(t_s, y)
        axes[idx].plot(x_plot, y_plot, lw=0.8, color="#1f77b4")
        axes[idx].set_ylabel(col, fontsize=8)
        axes[idx].grid(alpha=0.2, lw=0.4)

    axes[-1].set_xlabel("Window center (s)")
    section_name = (
        str(df["section_id"].iloc[0])
        if "section_id" in df.columns and len(df["section_id"]) > 0
        else features_csv.parent.parent.name
    )
    fig.suptitle(f"{section_name} — features overview (top {len(feat_cols)} by variance)")
    fig.tight_layout()

    out_path = features_csv.parent / "features_overview.png"
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Per-family time-series figures
# ---------------------------------------------------------------------------

def _column(df: pd.DataFrame, *parts: str) -> str | None:
    """Return the first column matching the joined parts, or None."""
    name = "_".join(parts)
    return name if name in df.columns else None


def _plot_family_grid(
    df: pd.DataFrame,
    t_s: np.ndarray,
    family_def: dict[str, tuple[str, ...]],
    title: str,
    out_path: Path,
    labels: list[LabelRow],
    label_colors: dict[str, str],
    t0_ms: float,
    *,
    sensor_pair: tuple[str, str] = ("bike", "rider"),
) -> Path | None:
    """Plot a (signal x stat) grid where each panel overlays the two sensors."""
    signals = family_def["signals"]
    stats = family_def["stats"]

    # Filter to (signal, stat) pairs that exist for at least one sensor.
    valid_signals: list[str] = []
    for sig in signals:
        if any(
            _column(df, sensor, sig, stat) is not None
            for sensor in sensor_pair
            for stat in stats
        ):
            valid_signals.append(sig)
    if not valid_signals:
        log.debug("No columns for family at %s", out_path.name)
        return None

    n_rows = len(valid_signals)
    n_cols = len(stats)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols + 1.0, 1.9 * n_rows + 0.8),
        sharex=True,
        squeeze=False,
    )

    for r, sig in enumerate(valid_signals):
        for c, stat in enumerate(stats):
            ax = axes[r][c]
            plotted = False
            for sensor in sensor_pair:
                col = _column(df, sensor, sig, stat)
                if col is None:
                    continue
                y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                xp, yp = filter_valid_plot_xy(t_s, y)
                if xp.size == 0:
                    continue
                color = _FEATURE_PREFIX_COLORS.get(sensor, "gray")
                ax.plot(xp, yp, lw=0.8, color=color, alpha=0.9, label=sensor)
                plotted = True
            _draw_label_spans([ax], labels, t0_ms, label_colors)
            if not plotted:
                ax.set_facecolor("#fafafa")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                        color="#888", fontsize=8)
            if r == 0:
                ax.set_title(stat, fontsize=9)
            if c == 0:
                ax.set_ylabel(sig, fontsize=9)
            ax.grid(alpha=0.18, lw=0.4)
            ax.tick_params(labelsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=8)

    sensor_handles = [
        mpatches.Patch(color=_FEATURE_PREFIX_COLORS.get(s, "gray"), label=s)
        for s in sensor_pair
    ]
    handles = sensor_handles + _label_legend_handles(label_colors)
    if handles:
        fig.legend(
            handles=handles, loc="lower center",
            ncol=min(8, len(handles)), fontsize=7, framealpha=0.85,
            bbox_to_anchor=(0.5, -0.005),
        )

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    return save_figure(fig, out_path)


def _plot_orientation_family(
    df: pd.DataFrame,
    t_s: np.ndarray,
    out_path: Path,
    labels: list[LabelRow],
    label_colors: dict[str, str],
    t0_ms: float,
) -> Path | None:
    """Orientation features (pitch/roll/yaw_rate × mean/std/range)."""
    signals = _ORIENTATION_FAMILY["signals"]
    stats = _ORIENTATION_FAMILY["stats"]
    sensor_pair = ("bike", "rider")

    rows: list[tuple[str, str]] = []
    for sig in signals:
        for stat in stats:
            if any(_column(df, sensor, sig, stat) is not None for sensor in sensor_pair):
                rows.append((sig, stat))
    if not rows:
        log.debug("No orientation feature columns")
        return None

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, max(3, 1.3 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (sig, stat) in zip(axes, rows):
        for sensor in sensor_pair:
            col = _column(df, sensor, sig, stat)
            if col is None:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(t_s, y)
            if xp.size == 0:
                continue
            ax.plot(xp, yp, lw=0.8, color=_FEATURE_PREFIX_COLORS.get(sensor, "gray"), label=sensor)
        _draw_label_spans([ax], labels, t0_ms, label_colors)
        ax.set_ylabel(f"{sig}_{stat}", fontsize=8)
        ax.grid(alpha=0.18, lw=0.4)

    axes[-1].set_xlabel("time (s)")

    sensor_handles = [
        mpatches.Patch(color=_FEATURE_PREFIX_COLORS.get(s, "gray"), label=s)
        for s in sensor_pair
    ]
    handles = sensor_handles + _label_legend_handles(label_colors)
    if handles:
        axes[0].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85, ncol=2)

    fig.suptitle(_FAMILY_TITLES["orientation"], fontsize=11)
    fig.tight_layout()
    return save_figure(fig, out_path)


def _plot_cross_family(
    df: pd.DataFrame,
    t_s: np.ndarray,
    out_path: Path,
    labels: list[LabelRow],
    label_colors: dict[str, str],
    t0_ms: float,
) -> Path | None:
    """Plot all cross-sensor features stacked vertically with label spans."""
    cross_cols = sorted(c for c in df.columns if c.startswith("cross_"))
    cross_cols = [c for c in cross_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not cross_cols:
        log.debug("No cross-sensor feature columns")
        return None

    n = len(cross_cols)
    fig, axes = plt.subplots(n, 1, figsize=(13, max(3, 1.1 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cross_cols):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        xp, yp = filter_valid_plot_xy(t_s, y)
        ax.plot(xp, yp, lw=0.8, color="#7f3fbf", alpha=0.9)
        _draw_label_spans([ax], labels, t0_ms, label_colors)
        ax.set_ylabel(col.replace("cross_", ""), fontsize=7)
        ax.grid(alpha=0.18, lw=0.4)

    axes[-1].set_xlabel("time (s)")
    handles = _label_legend_handles(label_colors)
    if handles:
        axes[0].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85, ncol=3)
    fig.suptitle(_FAMILY_TITLES["cross"], fontsize=11)
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_features_per_family(target: str | Path) -> list[Path]:
    """Generate one figure per feature family for a section."""
    features_csv = _resolve_features_csv(target)
    section_dir = _resolve_section_dir(target)
    df_raw = read_csv(features_csv)
    if df_raw.empty or "window_start_ms" not in df_raw.columns:
        log.warning("Cannot build per-family plots: empty or missing window_start_ms in %s", features_csv)
        return []

    df, t_s, t0_ms = _prepare_feature_time_axis(df_raw)
    labels = _section_labels(section_dir)
    label_colors = _label_color_map(labels)

    out_paths: list[Path] = []
    out_dir = features_csv.parent

    for fam_name, fam_def in _FEATURE_FAMILIES.items():
        out = _plot_family_grid(
            df, t_s, fam_def,
            _FAMILY_TITLES.get(fam_name, fam_name),
            out_dir / f"family_{fam_name}.png",
            labels, label_colors, t0_ms,
        )
        if out is not None:
            out_paths.append(out)

    orient_out = _plot_orientation_family(
        df, t_s, out_dir / "family_orientation.png",
        labels, label_colors, t0_ms,
    )
    if orient_out is not None:
        out_paths.append(orient_out)

    cross_out = _plot_cross_family(
        df, t_s, out_dir / "family_cross.png",
        labels, label_colors, t0_ms,
    )
    if cross_out is not None:
        out_paths.append(cross_out)

    return out_paths


# ---------------------------------------------------------------------------
# Window diagnostic plot
# ---------------------------------------------------------------------------

def plot_window_diagnostic(target: str | Path) -> Path | None:
    """Plot per-window diagnostic info (n_samples, valid_ratio, quality, type)."""
    features_csv = _resolve_features_csv(target)
    section_dir = _resolve_section_dir(target)
    df_raw = read_csv(features_csv)
    if df_raw.empty or "window_start_ms" not in df_raw.columns:
        return None

    df, t_s, t0_ms = _prepare_feature_time_axis(df_raw)
    labels = _section_labels(section_dir)
    label_colors = _label_color_map(labels)

    panels: list[tuple[str, list[tuple[str, str, str]]]] = []
    if "window_n_samples_sporsa" in df.columns or "window_n_samples_arduino" in df.columns:
        panels.append((
            "samples per window",
            [
                ("window_n_samples_sporsa", "sporsa", SENSOR_COLORS["sporsa"]),
                ("window_n_samples_arduino", "arduino", SENSOR_COLORS["arduino"]),
            ],
        ))
    if "window_valid_ratio_sporsa" in df.columns or "window_valid_ratio_arduino" in df.columns:
        panels.append((
            "valid ratio",
            [
                ("window_valid_ratio_sporsa", "sporsa", SENSOR_COLORS["sporsa"]),
                ("window_valid_ratio_arduino", "arduino", SENSOR_COLORS["arduino"]),
            ],
        ))
    if "overall_quality_score" in df.columns:
        panels.append(("overall quality score", [("overall_quality_score", "quality", "#2ca02c")]))

    if not panels:
        log.debug("No window-diagnostic columns available")
        return None

    n_rows = len(panels) + 1  # +1 for window-type strip
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 1.5 * n_rows + 1.5), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (ylabel, series) in zip(axes[:-1], panels):
        for col, label, color in series:
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            xp, yp = filter_valid_plot_xy(t_s, y)
            if xp.size:
                ax.plot(xp, yp, lw=0.8, color=color, label=label, alpha=0.9)
        _draw_label_spans([ax], labels, t0_ms, label_colors)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.18, lw=0.4)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, loc="upper right", fontsize=7, framealpha=0.85, ncol=2)

    # Window-type marker strip.
    type_ax = axes[-1]
    if "window_type" in df.columns:
        types = df["window_type"].astype(str).fillna("sliding")
        type_to_y = {"sliding": 0.0, "event_aligned": 1.0}
        type_color = {"sliding": "#1f77b4", "event_aligned": "#d62728"}
        for tname, yv in type_to_y.items():
            mask = types.values == tname
            if mask.any():
                type_ax.scatter(
                    t_s[mask], np.full(mask.sum(), yv),
                    s=10, color=type_color[tname], label=tname, alpha=0.8,
                )
        type_ax.set_yticks([0.0, 1.0])
        type_ax.set_yticklabels(["sliding", "event_aligned"], fontsize=8)
        type_ax.set_ylim(-0.4, 1.4)
    _draw_label_spans([type_ax], labels, t0_ms, label_colors)
    type_ax.set_ylabel("window type", fontsize=8)
    type_ax.set_xlabel("time (s)")
    type_ax.grid(alpha=0.18, lw=0.4)
    h, l = type_ax.get_legend_handles_labels()
    if h:
        type_ax.legend(h, l, loc="upper right", fontsize=7, framealpha=0.85)

    section_name = (
        str(df["section_id"].iloc[0])
        if "section_id" in df.columns and len(df["section_id"]) > 0
        else section_dir.name
    )
    fig.suptitle(f"{section_name} — window diagnostics", fontsize=11)
    fig.tight_layout()

    out_path = features_csv.parent / "window_diagnostics.png"
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Feature summary plot
# ---------------------------------------------------------------------------

def _classify_feature_column(col: str) -> tuple[str, str, str] | None:
    """Return (sensor, signal, stat) inferred from a feature column name.

    Returns None for columns that don't match the ``sensor_signal_stat`` pattern.
    """
    if col in _META_COLS:
        return None
    for sensor in ("bike", "rider", "cross"):
        prefix = f"{sensor}_"
        if not col.startswith(prefix):
            continue
        rest = col[len(prefix):]
        for sig in (
            "acc_norm", "gyro_norm", "acc_vertical", "acc_horizontal",
            "acc_hf", "gyro_hf", "acc_lf", "gyro_lf",
            "acc_deviation", "acc_linear_norm", "acc_linear",
            "jerk_norm", "alpha_norm", "energy_acc", "energy_gyro", "alpha_energy",
            "yaw_rate", "pitch", "roll", "yaw",
            "disagree_score", "acc_correlation", "acc_diff", "gyro_diff",
            "vertical_diff", "xcorr_acc", "xcorr_gyro",
        ):
            if rest == sig:
                return sensor, sig, ""
            if rest.startswith(sig + "_"):
                return sensor, sig, rest[len(sig) + 1:]
        # Fallback: split on first underscore
        head, _, tail = rest.partition("_")
        return sensor, head, tail
    return None


def plot_feature_summary(target: str | Path) -> Path | None:
    """High-level summary: feature counts by sensor / signal / stat + windows + labels."""
    features_csv = _resolve_features_csv(target)
    section_dir = _resolve_section_dir(target)
    df = read_csv(features_csv)
    if df.empty:
        return None

    # Classify every column.
    by_sensor: dict[str, int] = {}
    by_signal: dict[str, int] = {}
    by_stat: dict[str, int] = {}
    for col in df.columns:
        cls = _classify_feature_column(col)
        if cls is None:
            continue
        sensor, signal, stat = cls
        by_sensor[sensor] = by_sensor.get(sensor, 0) + 1
        by_signal[signal] = by_signal.get(signal, 0) + 1
        if stat:
            by_stat[stat] = by_stat.get(stat, 0) + 1

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    # 1. Feature count by sensor family (bike/rider/cross).
    ax = fig.add_subplot(gs[0, 0])
    if by_sensor:
        items = sorted(by_sensor.items(), key=lambda x: -x[1])
        names, vals = zip(*items)
        bar_colors = [_FEATURE_PREFIX_COLORS.get(n, "#7f3fbf") for n in names]
        ax.bar(names, vals, color=bar_colors, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_title("Features by sensor family", fontsize=10)
    ax.set_ylabel("# features")
    ax.grid(axis="y", alpha=0.2)

    # 2. Feature count by signal name.
    ax = fig.add_subplot(gs[0, 1])
    if by_signal:
        items = sorted(by_signal.items(), key=lambda x: -x[1])[:15]
        names, vals = zip(*items)
        ax.barh(list(reversed(names)), list(reversed(vals)), color="#4c72b0", edgecolor="white")
    ax.set_title("Features by signal (top 15)", fontsize=10)
    ax.set_xlabel("# features")
    ax.grid(axis="x", alpha=0.2)

    # 3. Feature count by stat type.
    ax = fig.add_subplot(gs[0, 2])
    if by_stat:
        items = sorted(by_stat.items(), key=lambda x: -x[1])[:15]
        names, vals = zip(*items)
        ax.barh(list(reversed(names)), list(reversed(vals)), color="#dd8452", edgecolor="white")
    ax.set_title("Features by stat type (top 15)", fontsize=10)
    ax.set_xlabel("# features")
    ax.grid(axis="x", alpha=0.2)

    # 4. Window-type counts.
    ax = fig.add_subplot(gs[1, 0])
    if "window_type" in df.columns:
        wcounts = df["window_type"].astype(str).fillna("sliding").value_counts()
        type_color = {"sliding": "#1f77b4", "event_aligned": "#d62728"}
        colors = [type_color.get(n, "gray") for n in wcounts.index]
        ax.bar(wcounts.index, wcounts.values, color=colors, edgecolor="white")
        for i, v in enumerate(wcounts.values):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_title("Window types", fontsize=10)
    ax.set_ylabel("# windows")
    ax.grid(axis="y", alpha=0.2)

    # 5. Quality tier distribution.
    ax = fig.add_subplot(gs[1, 1])
    if "quality_tier" in df.columns:
        tier_color = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
        order = ["A", "B", "C"]
        counts = df["quality_tier"].value_counts()
        present = [t for t in order if t in counts.index] + [
            t for t in counts.index if t not in order
        ]
        vals = [int(counts[t]) for t in present]
        colors = [tier_color.get(str(t), "#95a5a6") for t in present]
        ax.bar([str(t) for t in present], vals, color=colors, edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_title("Quality tier distribution", fontsize=10)
    ax.set_ylabel("# windows")
    ax.grid(axis="y", alpha=0.2)

    # 6. Label distribution (from scenario_labels token set).
    ax = fig.add_subplot(gs[1, 2])
    label_col = "scenario_labels" if "scenario_labels" in df.columns else (
        "scenario_label" if "scenario_label" in df.columns else None
    )
    if label_col:
        token_counts: dict[str, int] = {}
        for cell in df[label_col].fillna("unlabeled").astype(str):
            for tok in cell.split("|"):
                tok = tok.strip()
                if tok:
                    token_counts[tok] = token_counts.get(tok, 0) + 1
        if token_counts:
            items = sorted(token_counts.items(), key=lambda x: -x[1])[:15]
            names, vals = zip(*items)
            colors = [
                QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
                for i in range(len(names))
            ]
            ax.barh(
                list(reversed(names)), list(reversed(vals)),
                color=list(reversed(colors)), edgecolor="white",
            )
    ax.set_title("Window label tokens (top 15)", fontsize=10)
    ax.set_xlabel("# windows")
    ax.grid(axis="x", alpha=0.2)

    section_name = (
        str(df["section_id"].iloc[0])
        if "section_id" in df.columns and len(df) > 0
        else section_dir.name
    )
    n_features = sum(1 for c in df.columns if _classify_feature_column(c) is not None)
    fig.suptitle(
        f"{section_name} — feature summary  "
        f"({len(df)} windows, {n_features} feature columns)",
        fontsize=12, y=0.995,
    )

    out_path = features_csv.parent / "feature_summary.png"
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def plot_features_full(target: str | Path) -> list[Path]:
    """Run the full feature-stage plot bundle for a section."""
    out_paths: list[Path] = []

    for fn in (
        plot_features_stage,
        plot_feature_summary,
        plot_window_diagnostic,
    ):
        try:
            r = fn(target)
        except Exception as exc:
            log.warning("%s failed: %s", fn.__name__, exc)
            continue
        if r is not None:
            out_paths.append(r)

    try:
        out_paths.extend(plot_features_per_family(target))
    except Exception as exc:
        log.warning("plot_features_per_family failed: %s", exc)

    return out_paths


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_features")
    parser.add_argument("target", help="Section directory, features dir, or features CSV")
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of feature series in the variance-overview plot (default: 8).",
    )
    parser.add_argument(
        "--mode",
        choices=("overview", "summary", "windows", "families", "all"),
        default="all",
        help="Which plot bundle to generate (default: all).",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    try:
        if args.mode == "overview":
            r = plot_features_stage(args.target, top_n=args.top_n)
            if r is not None:
                paths.append(r)
        elif args.mode == "summary":
            r = plot_feature_summary(args.target)
            if r is not None:
                paths.append(r)
        elif args.mode == "windows":
            r = plot_window_diagnostic(args.target)
            if r is not None:
                paths.append(r)
        elif args.mode == "families":
            paths.extend(plot_features_per_family(args.target))
        else:
            paths = plot_features_full(args.target)
    except Exception as exc:
        log.error("Failed to plot features stage: %s", exc)
        return

    if not paths:
        print("No features plot generated.")
        return
    for p in paths:
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
