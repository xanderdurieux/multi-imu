"""Stage summaries helpers for build report tables, figures, and thesis bundles."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv
from visualization._utils import SENSOR_COLORS as _SENSOR_COLORS

log = logging.getLogger(__name__)

_DPI = 200

_SENSOR_LABELS = {"sporsa": "Bike (SPORSA)", "arduino": "Rider (Arduino)"}

_QUALITY_COLORS = {
    "good":     "#66BB6A",
    "marginal": "#FFA726",
    "poor":     "#ef5350",
}

_SYNC_METHOD_COLORS = {
    "multi_anchor":          "#2ca02c",
    "one_anchor_adaptive":   "#8c564b",
    "one_anchor_prior":      "#9467bd",
    "signal_only":           "#1f77b4",
    "unknown":               "#95a5a6",
}
_SYNC_METHOD_LABELS = {
    "multi_anchor":          "Multi-anchor",
    "one_anchor_adaptive":   "Adaptive",
    "one_anchor_prior":      "Prior",
    "signal_only":           "Signal-only",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violin_or_box(
    ax: plt.Axes,
    data_groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    title: str,
    *,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
    show_points: bool = True,
) -> None:
    """Draw a violin plot for each group; fall back to box plot if too few points."""
    n_groups = len(data_groups)
    positions = np.arange(1, n_groups + 1)

    enough = [d[np.isfinite(d)] for d in data_groups]
    use_violin = all(len(d) >= 4 for d in enough)

    if use_violin:
        parts = ax.violinplot(
            [d for d in enough],
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.6,
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.0)
    else:
        bp = ax.boxplot(
            [d for d in enough],
            positions=positions,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "lw": 1.5},
            whiskerprops={"lw": 1.0},
            capprops={"lw": 1.0},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    if show_points:
        for i, (d, pos) in enumerate(zip(enough, positions)):
            if len(d) == 0:
                continue
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(d))
            ax.scatter(pos + jitter, d, color=colors[i], s=18, alpha=0.5, zorder=3)

    if hline is not None:
        ax.axhline(hline, color="gray", lw=1.0, ls="--",
                   label=hline_label or str(hline))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _session_from_recording(recording_name: str) -> str:
    """Return session from recording."""
    parts = str(recording_name).rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("r") and parts[1][1:].isdigit():
        return parts[0]
    return str(recording_name)


def _safe_path_part(value: str) -> str:
    """Return NaN-safe path part."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe or "unknown_session"


# ===========================================================================
# CALIBRATION SUMMARIES
# ===========================================================================

def plot_calibration_bias_distributions(cal_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Violin plot of gyro biases (x, y, z) per sensor across all sections."""
    sensors = [s for s in ("sporsa", "arduino") if f"{s}_gyro_bias_x" in cal_df.columns]
    if not sensors:
        log.warning("No gyro bias columns in calibration params; skipping bias distributions")
        return None

    axes_letters = ("x", "y", "z")
    n_sensors = len(sensors)
    fig, grid = plt.subplots(1, n_sensors, figsize=(5 * n_sensors, 4.5), squeeze=False)

    for col_idx, sensor in enumerate(sensors):
        ax = grid[0, col_idx]
        groups = []
        labels_list = []
        colors = []
        for axis in axes_letters:
            col = f"{sensor}_gyro_bias_{axis}"
            if col in cal_df.columns:
                vals = pd.to_numeric(cal_df[col], errors="coerce").dropna().to_numpy(float)
                groups.append(vals)
                labels_list.append(axis)
                colors.append({"x": "#e41a1c", "y": "#4daf4a", "z": "#377eb8"}[axis])

        _violin_or_box(
            ax, groups, labels_list, colors,
            ylabel="Bias (deg/s)",
            title=f"{_SENSOR_LABELS.get(sensor, sensor)} — gyro bias",
            hline=0.0,
        )

    fig.suptitle("Gyro bias distribution across sections", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote calibration bias distributions → %s", project_relative_path(output_path))
    return output_path


def plot_calibration_gravity_residuals(cal_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Violin plot of gravity residuals per sensor across all sections."""
    sensors = [s for s in ("sporsa", "arduino") if f"{s}_gravity_residual_ms2" in cal_df.columns]
    if not sensors:
        log.warning("No gravity residual columns; skipping")
        return None

    groups = []
    labels_list = []
    colors = []
    for sensor in sensors:
        col = f"{sensor}_gravity_residual_ms2"
        vals = pd.to_numeric(cal_df[col], errors="coerce").dropna().to_numpy(float)
        groups.append(vals)
        labels_list.append(_SENSOR_LABELS.get(sensor, sensor))
        colors.append(_SENSOR_COLORS[sensor])

    fig, ax = plt.subplots(figsize=(5, 4))
    _violin_or_box(
        ax, groups, labels_list, colors,
        ylabel="Gravity residual (m/s²)",
        title="Gravity residual distribution",
        hline=0.0,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote calibration gravity residuals → %s", project_relative_path(output_path))
    return output_path


def plot_calibration_quality_overview(cal_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Stacked bar of calibration quality per section, plus protocol detection rate."""
    if "calibration_quality" not in cal_df.columns:
        log.warning("No 'calibration_quality' column; skipping quality overview")
        return None

    qual_counts = (
        cal_df["calibration_quality"]
        .value_counts()
        .reindex(["good", "marginal", "poor"], fill_value=0)
    )
    protocol_rate = (
        cal_df["protocol_detected"].sum() / len(cal_df)
        if "protocol_detected" in cal_df.columns
        else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: quality pie
    ax = axes[0]
    colors = [_QUALITY_COLORS.get(q, "#ccc") for q in qual_counts.index]
    non_zero = qual_counts[qual_counts > 0]
    ax.pie(
        non_zero.values,
        labels=non_zero.index,
        colors=[_QUALITY_COLORS.get(q, "#ccc") for q in non_zero.index],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Calibration quality distribution")

    # Right: protocol detection bar
    ax2 = axes[1]
    if protocol_rate is not None:
        detected = int(cal_df["protocol_detected"].sum())
        total = len(cal_df)
        not_detected = total - detected
        ax2.bar(
            ["Protocol\ndetected", "Protocol\nnot detected"],
            [detected, not_detected],
            color=["#66BB6A", "#ef5350"],
            edgecolor="white",
            linewidth=0.5,
        )
        for x, v in enumerate([detected, not_detected]):
            ax2.text(x, v + 0.1, str(v), ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Number of sections")
        ax2.set_title(f"Protocol detection ({detected}/{total} sections)")
        ax2.set_ylim(0, total * 1.2)
        ax2.grid(axis="y", alpha=0.3, lw=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle("Calibration summary across all sections", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote calibration quality overview → %s", project_relative_path(output_path))
    return output_path


def plot_calibration_acc_bias(cal_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Violin of accelerometer biases (x, y, z) for the rider sensor (arduino only)."""
    sensor = "arduino"
    bias_cols = {
        ax: f"{sensor}_acc_bias_{ax}"
        for ax in ("x", "y", "z")
        if f"{sensor}_acc_bias_{ax}" in cal_df.columns
    }
    if not bias_cols:
        log.info("No acc bias columns for arduino; skipping acc bias plot")
        return None

    groups = []
    labels_list = []
    colors_list = []
    axis_colors = {"x": "#e41a1c", "y": "#4daf4a", "z": "#377eb8"}
    for axis, col in bias_cols.items():
        vals = pd.to_numeric(cal_df[col], errors="coerce").dropna().to_numpy(float)
        groups.append(vals)
        labels_list.append(axis)
        colors_list.append(axis_colors[axis])

    fig, ax = plt.subplots(figsize=(5, 4))
    _violin_or_box(
        ax, groups, labels_list, colors_list,
        ylabel="Acc bias (m/s²)",
        title=f"{_SENSOR_LABELS.get(sensor, sensor)} — accelerometer bias",
        hline=0.0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote calibration acc bias → %s", project_relative_path(output_path))
    return output_path


def generate_calibration_summary(
    cal_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all calibration summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for fn, name in (
        (plot_calibration_bias_distributions,  "calibration_gyro_bias.png"),
        (plot_calibration_gravity_residuals,    "calibration_gravity_residuals.png"),
        (plot_calibration_quality_overview,     "calibration_quality_overview.png"),
        (plot_calibration_acc_bias,             "calibration_acc_bias.png"),
    ):
        try:
            p = fn(cal_df, output_dir / name)
            if p is not None:
                generated.append(p)
        except Exception as exc:
            log.warning("Calibration summary plot '%s' failed: %s", name, exc)

    log.info(
        "Calibration summary: %d figures written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated


# ===========================================================================
# SYNC SUMMARIES
# ===========================================================================

def plot_sync_method_selection(sync_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Horizontal bar chart: which sync method was chosen per recording."""
    if "selected_method" not in sync_df.columns:
        log.warning("No 'selected_method' column in sync params; skipping")
        return None

    counts = sync_df["selected_method"].value_counts()
    methods = counts.index.tolist()
    values = counts.values.tolist()
    colors = [_SYNC_METHOD_COLORS.get(m, "#95a5a6") for m in methods]
    labels_display = [_SYNC_METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(methods) + 1.5)))
    bars = ax.barh(labels_display, values, color=colors, edgecolor="white", linewidth=0.4)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of recordings")
    ax.set_title("Sync method selected per recording")
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote sync method selection → %s", project_relative_path(output_path))
    return output_path


def plot_sync_correlation_comparison(sync_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Grouped bar: cross-correlation score per available method per recording."""
    methods = ["multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"]
    corr_cols = {m: f"{m}_corr_offset_and_drift" for m in methods}
    available_methods = [m for m in methods if corr_cols[m] in sync_df.columns]

    if not available_methods:
        log.warning("No correlation columns in sync params; skipping")
        return None

    recordings = sync_df["recording_name"].tolist() if "recording_name" in sync_df.columns else \
        sync_df.index.astype(str).tolist()
    n_rec = len(recordings)
    n_methods = len(available_methods)
    x = np.arange(n_rec)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_rec), 4))

    for i, method in enumerate(available_methods):
        col = corr_cols[method]
        vals = pd.to_numeric(sync_df[col], errors="coerce").fillna(0).tolist()
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset, vals, width * 0.9,
            label=_SYNC_METHOD_LABELS.get(method, method),
            color=_SYNC_METHOD_COLORS.get(method, "#95a5a6"),
            alpha=0.85, edgecolor="white",
        )

    # Highlight selected method per recording
    if "selected_method" in sync_df.columns:
        for xi, (_, row) in zip(x, sync_df.iterrows()):
            sel = row.get("selected_method", "")
            if sel in available_methods:
                ax.annotate(
                    "★",
                    xy=(xi + (available_methods.index(sel) - (n_methods - 1) / 2) * width, 0),
                    xytext=(xi + (available_methods.index(sel) - (n_methods - 1) / 2) * width,
                            -0.08),
                    fontsize=10, ha="center", color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(recordings, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Cross-correlation score")
    ax.set_title("Sync cross-correlation by method (★ = selected)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote sync correlation comparison → %s", project_relative_path(output_path))
    return output_path


def plot_sync_drift(sync_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """Scatter plot of final drift (ppm) per recording, coloured by selected method."""
    if "drift_ppm" not in sync_df.columns:
        log.warning("No 'drift_ppm' column in sync params; skipping drift plot")
        return None

    drift_vals = pd.to_numeric(sync_df["drift_ppm"], errors="coerce")
    recordings = sync_df["recording_name"].tolist() if "recording_name" in sync_df.columns else \
        [str(i) for i in range(len(sync_df))]
    selected = sync_df["selected_method"].tolist() if "selected_method" in sync_df.columns else \
        ["unknown"] * len(sync_df)

    x = np.arange(len(recordings))
    colors = [_SYNC_METHOD_COLORS.get(m, "#95a5a6") for m in selected]

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(recordings)), 4))

    sc = ax.scatter(x, drift_vals, c=colors, s=80, zorder=4, edgecolors="white", linewidths=0.5)
    ax.bar(x, drift_vals, color=colors, alpha=0.3, width=0.5)
    ax.axhline(0, color="black", lw=0.7, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(recordings, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Clock drift (ppm)")
    ax.set_title("Synchronisation drift per recording")

    # Legend
    seen: set[str] = set()
    handles = []
    for m, c in zip(selected, colors):
        if m not in seen:
            handles.append(mpatches.Patch(color=c, label=_SYNC_METHOD_LABELS.get(m, m)))
            seen.add(m)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)

    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote sync drift → %s", project_relative_path(output_path))
    return output_path


def generate_sync_summary(
    sync_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all synchronisation summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for fn, name in (
        (plot_sync_method_selection,       "sync_method_selection.png"),
        (plot_sync_correlation_comparison, "sync_correlation_comparison.png"),
        (plot_sync_drift,                  "sync_drift.png"),
    ):
        try:
            p = fn(sync_df, output_dir / name)
            if p is not None:
                generated.append(p)
        except Exception as exc:
            log.warning("Sync summary plot '%s' failed: %s", name, exc)

    if "recording_name" in sync_df.columns:
        session_df = sync_df.copy()
        if "session" not in session_df.columns:
            session_df["session"] = session_df["recording_name"].map(_session_from_recording)

        sessions_root = output_dir / "sessions"
        for session in sorted(session_df["session"].dropna().astype(str).unique()):
            sub = session_df[session_df["session"].astype(str) == session].copy()
            if sub.empty:
                continue
            session_dir = sessions_root / _safe_path_part(session)
            session_dir.mkdir(parents=True, exist_ok=True)
            for fn, name in (
                (plot_sync_correlation_comparison, "sync_correlation_comparison.png"),
                (plot_sync_drift, "sync_drift.png"),
            ):
                try:
                    p = fn(sub, session_dir / name)
                    if p is not None:
                        generated.append(p)
                except Exception as exc:
                    log.warning(
                        "Sync session summary plot '%s/%s' failed: %s",
                        session,
                        name,
                        exc,
                    )

    log.info(
        "Sync summary: %d figures written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated


# ===========================================================================
# ORIENTATION SUMMARIES
# ===========================================================================

def generate_orientation_summary(
    ori_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate orientation summary plots from ``orientation_stats.csv`` exports."""
    if ori_df.empty:
        log.warning("orientation_stats.csv is empty — skipping orientation summary")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "orientation_gravity_alignment.png",
        "orientation_angle_stability.png",
        "orientation_quality_per_section.png",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    from visualization.plot_exports_orientation import (
        plot_orientation_quality_overview,
        plot_orientation_residuals,
    )

    generated: list[Path] = []
    for fn in (
        plot_orientation_quality_overview,
        plot_orientation_residuals,
    ):
        try:
            p = fn(ori_df, output_dir)
            if p is not None:
                generated.append(p)
        except Exception as exc:
            log.warning("Orientation summary plot '%s' failed: %s", fn.__name__, exc)

    log.info(
        "Orientation summary: %d figures written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated


# ===========================================================================
# PARSED STAGE SUMMARIES
# ===========================================================================

def generate_parsed_summary(
    exports_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate parsed-stage aggregate plots from ``parsed_recording_summary.csv``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_csv = exports_dir / "parsed_recording_summary.csv"
    if not parsed_csv.exists():
        log.warning(
            "parsed_recording_summary.csv not found in %s — skipping parsed summary",
            project_relative_path(exports_dir),
        )
        return []

    summary_df = read_csv(parsed_csv)
    if summary_df.empty:
        log.warning("parsed_recording_summary.csv is empty — skipping parsed summary")
        return []

    from visualization.plot_parsed_stats import (
        plot_interval_jitter,
        plot_packet_loss,
        plot_session_bar_chart,
    )

    generated: list[Path] = []

    for fn, name in (
        (plot_session_bar_chart, "parsed_session_bar_chart.png"),
        (plot_interval_jitter,   "parsed_interval_jitter.png"),
        (plot_packet_loss,       "parsed_packet_loss.png"),
    ):
        try:
            p = fn(summary_df, output_dir / name)
            if p is not None:
                generated.append(p)
        except Exception as exc:
            log.warning("parsed summary plot '%s' failed: %s", name, exc)

    log.info(
        "Parsed summary: %d figures written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated


# ===========================================================================
# TOP-LEVEL ENTRY POINT
# ===========================================================================

def generate_all_stage_summaries(
    exports_dir: Path,
    sections_root_dir: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Generate all stage summaries."""
    results: dict[str, list[Path]] = {
        "parsed": [],
        "calibration": [],
        "sync": [],
        "orientation": [],
    }

    # ---- Parsed ----
    try:
        results["parsed"] = generate_parsed_summary(
            exports_dir, output_dir / "parsed"
        )
    except Exception as exc:
        log.warning("Parsed summary generation failed: %s", exc)

    # ---- Calibration ----
    cal_csv = exports_dir / "calibration_params.csv"
    if cal_csv.exists():
        try:
            cal_df = read_csv(cal_csv)
            results["calibration"] = generate_calibration_summary(
                cal_df, output_dir / "calibration"
            )
        except Exception as exc:
            log.warning("Calibration summary generation failed: %s", exc)
    else:
        log.warning("calibration_params.csv not found in %s — skipping calibration summary",
                    project_relative_path(exports_dir))

    # ---- Sync ----
    sync_csv = exports_dir / "sync_params.csv"
    if sync_csv.exists():
        try:
            sync_df = read_csv(sync_csv)
            results["sync"] = generate_sync_summary(
                sync_df, output_dir / "sync"
            )
        except Exception as exc:
            log.warning("Sync summary generation failed: %s", exc)
    else:
        log.warning("sync_params.csv not found in %s — skipping sync summary",
                    project_relative_path(exports_dir))

    # ---- Orientation ----
    orientation_csv = exports_dir / "orientation_stats.csv"
    if orientation_csv.exists():
        try:
            ori_df = read_csv(orientation_csv)
            results["orientation"] = generate_orientation_summary(
                ori_df, output_dir / "orientation"
            )
        except Exception as exc:
            log.warning("Orientation summary generation failed: %s", exc)
    else:
        log.warning("orientation_stats.csv not found in %s — skipping orientation summary",
                    project_relative_path(exports_dir))

    total = sum(len(v) for v in results.values())
    log.info("Stage summaries complete: %d figures total", total)
    return results
