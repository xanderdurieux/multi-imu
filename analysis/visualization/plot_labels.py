"""Plot labels helpers for plot pipeline diagnostics and dataset summaries."""

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

from common.paths import (
    list_csv_files,
    read_csv,
    recording_labels_csv,
    recordings_root,
    resolve_data_dir,
    section_labels_csv,
    sections_root,
)
from labels.parser import LabelRow, load_labels
from common.signals import vector_norm
from visualization._utils import (
    ACC_COLS,
    GYRO_COLS,
    SENSOR_COLORS,
    filter_valid_plot_xy,
    save_figure,
)

log = logging.getLogger(__name__)

_LABEL_ALPHA = 0.22
_FEATURES_TOP_N = 5
_EXCLUDE_FEATURE_COLS = frozenset({
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "section_id",
    "label",
    "scenario_label",
    "scenario_labels",
    "overall_quality_score",
    "overall_quality_label",
})
_FEATURE_PREFIXES = ("bike_", "rider_", "cross_")


# ---------------------------------------------------------------------------
# Color and span helpers
# ---------------------------------------------------------------------------

def _label_colors(labels: list[LabelRow]) -> dict[str, tuple]:
    """Assign a consistent tab10 color to each unique label name."""
    unique = sorted({lr.label for lr in labels if lr.label})
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(unique)}


def _draw_spans(
    axes: list[plt.Axes],
    labels: list[LabelRow],
    t0_ms: float,
    colors: dict[str, tuple],
) -> None:
    """Draw colored axvspan regions for each label interval on all axes."""
    for lr in labels:
        if not lr.label:
            continue
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, "gray")
        alpha = _LABEL_ALPHA * max(0.3, lr.confidence)
        for ax in axes:
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _label_patches(colors: dict[str, tuple]) -> list[mpatches.Patch]:
    """Return label patches."""
    return [
        mpatches.Patch(color=color, alpha=0.6, label=name)
        for name, color in colors.items()
    ]


# ---------------------------------------------------------------------------
# Sensor stage
# ---------------------------------------------------------------------------

def _load_sensor_dfs(stage_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all IMU sensor CSVs from a directory (acc columns required)."""
    result: dict[str, pd.DataFrame] = {}
    for csv_path in list_csv_files(stage_dir):
        try:
            df = read_csv(csv_path)
        except Exception:
            continue
        if "timestamp" not in df.columns:
            continue
        if not any(c in df.columns for c in ACC_COLS):
            continue
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if not df.empty:
            result[csv_path.stem] = df
    return result


def plot_labels_sensor(
    stage_dir: Path,
    labels: list[LabelRow],
    *,
    output_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Plot labels sensor."""
    sensor_dfs = _load_sensor_dfs(stage_dir)
    if not sensor_dfs:
        log.warning("No IMU sensor CSVs found in %s", stage_dir)
        return None

    # Common time origin: earliest timestamp across all sensors
    t0_ms = min(
        float(df["timestamp"].iloc[0])
        for df in sensor_dfs.values()
        if not df.empty
    )

    colors = _label_colors(labels)
    sensor_cmap = plt.get_cmap("Set1")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for si, (sensor_name, df) in enumerate(sorted(sensor_dfs.items())):
        ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        ts_s = (ts_ms - t0_ms) / 1000.0
        sc = sensor_cmap(si % 9)

        acc_cols = [c for c in ACC_COLS if c in df.columns]
        gyro_cols = [c for c in GYRO_COLS if c in df.columns]

        if acc_cols:
            acc_norm = vector_norm(df, acc_cols)
            x, y = filter_valid_plot_xy(ts_s, acc_norm)
            axes[0].plot(x, y, lw=0.7, color=sc, label=sensor_name)

        if gyro_cols:
            gyro_norm = vector_norm(df, gyro_cols)
            x, y = filter_valid_plot_xy(ts_s, gyro_norm)
            axes[1].plot(x, y, lw=0.7, color=sc, label=sensor_name)

    _draw_spans(list(axes), labels, t0_ms, colors)

    axes[0].set_ylabel("|acc| (m/s²)")
    axes[1].set_ylabel("|gyro| (rad/s or °/s)")
    axes[-1].set_xlabel("Time (s)")

    for ax in axes:
        sensor_handles, sensor_labels_text = ax.get_legend_handles_labels()
        all_handles = sensor_handles + _label_patches(colors)
        all_labels = sensor_labels_text + list(colors.keys())
        if all_handles:
            ax.legend(all_handles, all_labels, loc="upper right", fontsize=7, framealpha=0.8)

    fig.suptitle(title or stage_dir.name, fontsize=10)
    fig.tight_layout()

    if output_path is None:
        output_path = stage_dir / "labels_overlay.png"
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Features stage
# ---------------------------------------------------------------------------

def _resolve_features_csv(target_dir: Path) -> Path | None:
    """Resolve features csv."""
    if (target_dir / "features.csv").exists():
        return target_dir / "features.csv"
    nested = target_dir / "features" / "features.csv"
    if nested.exists():
        return nested
    return None


def _select_feature_cols(df: pd.DataFrame, top_n: int) -> list[str]:
    """Select feature cols."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = [c for c in numeric if c not in _EXCLUDE_FEATURE_COLS]
    prefixed = [c for c in candidates if c.startswith(_FEATURE_PREFIXES)]
    pool = prefixed if prefixed else candidates
    if not pool:
        return []
    variances = df[pool].apply(pd.to_numeric, errors="coerce").var(skipna=True)
    return variances.sort_values(ascending=False).dropna().index[:top_n].tolist()


def plot_labels_features(
    section_dir: Path,
    labels: list[LabelRow],
    *,
    top_n: int = _FEATURES_TOP_N,
    output_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Plot labels features."""
    feat_csv = _resolve_features_csv(section_dir)
    if feat_csv is None:
        log.warning("No features.csv found under %s", section_dir)
        return None

    df = read_csv(feat_csv)
    if df.empty:
        log.warning("Features CSV is empty: %s", feat_csv)
        return None

    if "window_start_ms" not in df.columns:
        log.warning("features.csv missing 'window_start_ms' column: %s", feat_csv)
        return None

    t_ms = pd.to_numeric(df["window_start_ms"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(t_ms)
    if not finite_mask.any():
        log.warning("No valid window_start_ms values in %s", feat_csv)
        return None

    t0_ms = t_ms[finite_mask][0]
    t_s = (t_ms - t0_ms) / 1000.0

    feat_cols = _select_feature_cols(df, top_n=top_n)
    if not feat_cols:
        log.warning("No plottable feature columns in %s", feat_csv)
        return None

    colors = _label_colors(labels)

    rows = len(feat_cols)
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3, 2.0 * rows)), sharex=True)
    if rows == 1:
        axes = [axes]

    for i, col in enumerate(feat_cols):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x_plot, y_plot = filter_valid_plot_xy(t_s, y)
        axes[i].plot(x_plot, y_plot, lw=0.9, color="#1f77b4")
        axes[i].set_ylabel(col, fontsize=7)
        axes[i].grid(alpha=0.2, lw=0.4)

    _draw_spans(list(axes), labels, t0_ms, colors)

    axes[-1].set_xlabel("Time (s)")

    if colors:
        axes[0].legend(
            handles=_label_patches(colors),
            loc="upper right",
            fontsize=7,
            framealpha=0.8,
        )

    section_name = section_dir.name
    fig.suptitle(title or f"{section_name} — features with label overlay", fontsize=10)
    fig.tight_layout()

    if output_path is None:
        output_path = feat_csv.parent / "labels_overlay.png"
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Calibrated stage
# ---------------------------------------------------------------------------

def plot_labels_calibrated(
    section_dir: Path,
    labels: list[LabelRow],
    *,
    output_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Plot labels calibrated."""
    cal_dir = section_dir / "calibrated"
    if not cal_dir.is_dir():
        log.warning("No calibrated directory found under %s", section_dir)
        return None

    if output_path is None:
        output_path = cal_dir / "labels_overlay.png"

    return plot_labels_sensor(
        cal_dir,
        labels,
        output_path=output_path,
        title=title or f"{section_dir.name} — calibrated with label overlay",
    )


# ---------------------------------------------------------------------------
# Orientation stage
# ---------------------------------------------------------------------------

_ORIENT_ANGLES = ("yaw_deg", "pitch_deg", "roll_deg")


def plot_labels_orientation(
    section_dir: Path,
    labels: list[LabelRow],
    *,
    output_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Plot labels orientation."""
    orient_dir = section_dir / "orientation"
    if not orient_dir.is_dir():
        log.warning("No orientation directory found under %s", section_dir)
        return None

    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in ("sporsa", "arduino"):
        csv_path = orient_dir / f"{sensor}.csv"
        if csv_path.exists():
            try:
                df = read_csv(csv_path)
                if "timestamp" in df.columns and any(c in df.columns for c in _ORIENT_ANGLES):
                    sensor_dfs[sensor] = df
            except Exception:
                pass

    if not sensor_dfs:
        log.warning("No orientation CSVs with angle columns found in %s", orient_dir)
        return None

    t0_ms = min(
        float(pd.to_numeric(df["timestamp"], errors="coerce").dropna().iloc[0])
        for df in sensor_dfs.values()
    )

    colors = _label_colors(labels)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    for sensor, df in sensor_dfs.items():
        ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        ts_s = (ts_ms - t0_ms) / 1000.0
        sc = SENSOR_COLORS.get(sensor, "gray")
        for idx, angle_col in enumerate(_ORIENT_ANGLES):
            if angle_col not in df.columns:
                continue
            y = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
            x_plot, y_plot = filter_valid_plot_xy(ts_s, y)
            if x_plot.size:
                axes[idx].plot(x_plot, y_plot, lw=0.8, alpha=0.9, color=sc, label=sensor)

    _draw_spans(list(axes), labels, t0_ms, colors)

    for idx, angle_col in enumerate(_ORIENT_ANGLES):
        axes[idx].set_ylabel(angle_col.replace("_deg", " (deg)"), fontsize=8)
        axes[idx].grid(alpha=0.2, lw=0.4)
        sensor_handles, sensor_labels_text = axes[idx].get_legend_handles_labels()
        all_handles = sensor_handles + _label_patches(colors)
        all_labels = sensor_labels_text + list(colors.keys())
        if all_handles:
            axes[idx].legend(all_handles, all_labels, loc="upper right", fontsize=7, framealpha=0.8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or f"{section_dir.name} — orientation with label overlay", fontsize=10)
    fig.tight_layout()

    if output_path is None:
        output_path = orient_dir / "labels_overlay.png"
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Derived stage
# ---------------------------------------------------------------------------

_DERIVED_PANELS = [
    ("acc_norm",     "acc norm (m/s²)"),
    ("gyro_norm",    "gyro norm (°/s)"),
    ("jerk_norm",    "jerk norm (m/s³)"),
    ("acc_hf",       "acc HF (m/s²)"),
]


def plot_labels_derived(
    section_dir: Path,
    labels: list[LabelRow],
    *,
    output_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Plot labels derived."""
    derived_dir = section_dir / "derived"
    if not derived_dir.is_dir():
        log.warning("No derived directory found under %s", section_dir)
        return None

    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in ("sporsa", "arduino"):
        csv_path = derived_dir / f"{sensor}_signals.csv"
        if csv_path.exists():
            try:
                df = read_csv(csv_path)
                if "timestamp" in df.columns:
                    sensor_dfs[sensor] = df
            except Exception:
                pass

    if not sensor_dfs:
        log.warning("No derived signal CSVs found in %s", derived_dir)
        return None

    # Determine which panels have any data
    available_panels = [
        (col, ylabel)
        for col, ylabel in _DERIVED_PANELS
        if any(col in df.columns for df in sensor_dfs.values())
    ]
    if not available_panels:
        log.warning("No plottable derived columns found in %s", derived_dir)
        return None

    t0_ms = min(
        float(pd.to_numeric(df["timestamp"], errors="coerce").dropna().iloc[0])
        for df in sensor_dfs.values()
    )

    colors = _label_colors(labels)
    n_rows = len(available_panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(3, 2.0 * n_rows)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax_idx, (col, ylabel) in enumerate(available_panels):
        ax = axes[ax_idx]
        for sensor, df in sensor_dfs.items():
            if col not in df.columns:
                continue
            ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
            ts_s = (ts_ms - t0_ms) / 1000.0
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            x_plot, y_plot = filter_valid_plot_xy(ts_s, y)
            if x_plot.size:
                sc = SENSOR_COLORS.get(sensor, "gray")
                ax.plot(x_plot, y_plot, lw=0.7, alpha=0.8, color=sc, label=sensor)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
        sensor_handles, sensor_labels_text = ax.get_legend_handles_labels()
        all_handles = sensor_handles + _label_patches(colors)
        all_labels = sensor_labels_text + list(colors.keys())
        if all_handles:
            ax.legend(all_handles, all_labels, loc="upper right", fontsize=7, framealpha=0.8)

    _draw_spans(list(axes), labels, t0_ms, colors)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or f"{section_dir.name} — derived signals with label overlay", fontsize=10)
    fig.tight_layout()

    if output_path is None:
        output_path = derived_dir / "labels_overlay.png"
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

def _infer_labels_path(target_dir: Path) -> Path:
    """Infer the canonical label CSV for a recording stage or section directory."""
    recs_root = recordings_root()
    secs_root = sections_root()

    try:
        rel = target_dir.relative_to(recs_root)
        recording_name = rel.parts[0]
        return recording_labels_csv(recording_name)
    except ValueError:
        pass

    try:
        rel = target_dir.relative_to(secs_root)
        section_name = rel.parts[0]
        return section_labels_csv(secs_root / section_name)
    except ValueError:
        pass

    # Fallback: look adjacent to target
    return target_dir / "labels" / "labels.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_section_root(target_dir: Path) -> Path:
    """Return the section root for *target_dir*, resolving subdirectory references."""
    secs_root = sections_root()
    try:
        rel = target_dir.relative_to(secs_root)
        return secs_root / rel.parts[0]
    except ValueError:
        return target_dir


def plot_labels(
    target: str | Path,
    *,
    stage: str = "auto",
    top_n: int = _FEATURES_TOP_N,
    output_path: Path | None = None,
) -> Path | None:
    """Plot labels."""
    target_dir = resolve_data_dir(target)
    labels_path = _infer_labels_path(target_dir)
    labels = load_labels(labels_path)
    if not labels:
        log.info("No labels at %s — spans will be omitted", labels_path)

    section_root = _resolve_section_root(target_dir)

    if stage == "calibrated":
        return plot_labels_calibrated(section_root, labels, output_path=output_path)

    if stage == "orientation":
        return plot_labels_orientation(section_root, labels, output_path=output_path)

    if stage == "derived":
        return plot_labels_derived(section_root, labels, output_path=output_path)

    if stage == "features":
        return plot_labels_features(
            section_root, labels, top_n=top_n, output_path=output_path
        )

    if stage == "sensor":
        return plot_labels_sensor(target_dir, labels, output_path=output_path)

    # auto: try calibrated → sensor → features
    if (section_root / "calibrated").is_dir():
        return plot_labels_calibrated(section_root, labels, output_path=output_path)
    if _load_sensor_dfs(target_dir):
        return plot_labels_sensor(target_dir, labels, output_path=output_path)
    if _resolve_features_csv(section_root) is not None:
        return plot_labels_features(
            section_root, labels, top_n=top_n, output_path=output_path
        )
    log.warning("Could not auto-detect a plottable stage under %s", target_dir)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    import sys

    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_labels",
        description="Overlay labels on sensor data or feature windows.",
    )
    parser.add_argument(
        "target",
        help=(
            "Data directory reference: recording stage (e.g. 2026-02-26_r1/synced), "
            "section name (e.g. 2026-02-26_r1s1), or absolute path."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["auto", "calibrated", "orientation", "derived", "sensor", "features"],
        default="auto",
        help="Stage to visualize (default: auto-detect from directory contents).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=_FEATURES_TOP_N,
        help=f"Top feature columns to show in features mode (default: {_FEATURES_TOP_N}).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PNG path (auto-derived from stage directory if omitted).",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output) if args.output else None
    try:
        saved = plot_labels(
            args.target,
            stage=args.stage,
            top_n=args.top_n,
            output_path=out_path,
        )
    except Exception as exc:
        log.error("Failed to plot labels: %s", exc)
        return

    if saved is None:
        print("No plot generated.")
    else:
        print(f"Saved → {saved}")


if __name__ == "__main__":
    main()
