"""Calibration-stage visualizations.

Functions
---------
plot_calibration_raw_vs_calibrated
    Individual-axis and norm comparison (raw → calibrated) per sensor.
plot_calibration_parameters
    Gyro bias, rotation matrix, gravity vector, and quality summary per sensor.
plot_calibration_world_frame
    World-frame acceleration and gyroscope signals with reference lines.
plot_calibration_methods
    Method-comparison bar charts from ``all_methods.json`` (if present).
plot_calibration_stage
    Master entry point — calls all of the above for a section directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv, resolve_data_dir
from visualization._utils import filter_valid_plot_xy, strict_vector_norm, timestamps_to_relative_seconds

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}
_RAW_ALPHA = 0.55
_CAL_ALPHA = 0.90
_G = 9.81


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_section_dir(target: str | Path) -> Path:
    base = resolve_data_dir(target)
    if (base / "calibrated").is_dir():
        return base
    if base.name == "calibrated" and base.parent.is_dir():
        return base.parent
    raise FileNotFoundError(
        f"Could not resolve section directory with calibrated outputs from: {target}"
    )


def _load_calibration_json(section_dir: Path) -> dict:
    cal_json = section_dir / "calibrated" / "calibration.json"
    if not cal_json.exists():
        return {}
    return json.loads(cal_json.read_text(encoding="utf-8"))


def _load_all_methods_json(section_dir: Path) -> dict | None:
    path = section_dir / "calibrated" / "all_methods.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _static_calibration_reference(section_dir: Path) -> dict | None:
    data = _load_all_methods_json(section_dir)
    if not data:
        return None
    ref = data.get("static_calibration_reference")
    return ref if isinstance(ref, dict) else None


def _ts_s(df: pd.DataFrame) -> np.ndarray:
    return timestamps_to_relative_seconds(df["timestamp"])


# ---------------------------------------------------------------------------
# Plot 1: Raw vs calibrated — individual axes + norms
# ---------------------------------------------------------------------------

def plot_calibration_raw_vs_calibrated(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Per-sensor 4-panel plot: |acc|, |gyro|, and individual axes before vs after calibration.

    Outputs
    -------
    ``calibrated/<sensor>_calibration_comparison.png``
        Existing norm comparison (acc + gyro norms).
    ``calibrated/<sensor>_axes_comparison.png``
        Individual axes (ax/ay/az) before vs after calibration.
    """
    selected = sensors or list(_SENSORS)
    out_paths: list[Path] = []
    cal_dir = section_dir / "calibrated"

    for sensor in selected:
        raw_csv = section_dir / f"{sensor}.csv"
        cal_csv = cal_dir / f"{sensor}.csv"
        if not raw_csv.exists() or not cal_csv.exists():
            log.warning("Missing raw or calibrated CSV for %s in %s", sensor, section_dir.name)
            continue

        df_raw = read_csv(raw_csv)
        df_cal = read_csv(cal_csv)
        if "timestamp" not in df_raw.columns or "timestamp" not in df_cal.columns:
            continue

        ts_raw = _ts_s(df_raw)
        ts_cal = _ts_s(df_cal)
        color = _SENSOR_COLORS.get(sensor, "steelblue")

        # --- Plot A: norm comparison ---
        acc_raw = strict_vector_norm(df_raw, [c for c in ("ax", "ay", "az") if c in df_raw.columns])
        gyro_raw = strict_vector_norm(df_raw, [c for c in ("gx", "gy", "gz") if c in df_raw.columns])
        acc_cal = strict_vector_norm(df_cal, [c for c in ("ax", "ay", "az") if c in df_cal.columns])
        gyro_cal = strict_vector_norm(df_cal, [c for c in ("gx", "gy", "gz") if c in df_cal.columns])

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        for ax_obj, ts, vals, label, alpha in (
            (axes[0], ts_raw, acc_raw, "raw |acc|", _RAW_ALPHA),
            (axes[0], ts_cal, acc_cal, "calibrated |acc|", _CAL_ALPHA),
            (axes[1], ts_raw, gyro_raw, "raw |gyro|", _RAW_ALPHA),
            (axes[1], ts_cal, gyro_cal, "calibrated |gyro|", _CAL_ALPHA),
        ):
            x, y = filter_valid_plot_xy(ts, vals)
            ax_obj.plot(x, y, lw=0.8, alpha=alpha, label=label)
        axes[0].axhline(_G, color="gray", lw=0.8, ls="--", label=f"g = {_G} m/s²")
        axes[0].set_ylabel("|acc| (m/s²)")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].grid(alpha=0.2, lw=0.4)
        axes[1].set_ylabel("|gyro| (deg/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(alpha=0.2, lw=0.4)
        fig.suptitle(f"{section_dir.name} / {sensor} — calibration effect (norms)")
        fig.tight_layout()
        out_a = cal_dir / f"{sensor}_calibration_comparison.png"
        fig.savefig(out_a, dpi=120)
        plt.close(fig)
        out_paths.append(out_a)
        log.info("Plot written: %s", project_relative_path(out_a))

        # --- Plot B: individual axes ---
        acc_axes = [c for c in ("ax", "ay", "az") if c in df_raw.columns and c in df_cal.columns]
        gyro_axes = [c for c in ("gx", "gy", "gz") if c in df_raw.columns and c in df_cal.columns]
        n_rows = max(len(acc_axes), len(gyro_axes))
        if n_rows == 0:
            continue

        fig, axes_grid = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), sharex=False, squeeze=False)
        units = {"ax": "m/s²", "ay": "m/s²", "az": "m/s²", "gx": "deg/s", "gy": "deg/s", "gz": "deg/s"}
        axis_colors = {"ax": "#d62728", "ay": "#2ca02c", "az": "#1f77b4",
                       "gx": "#9467bd", "gy": "#8c564b", "gz": "#e377c2"}

        for col_idx, cols in enumerate((acc_axes, gyro_axes)):
            for row_idx, col in enumerate(cols):
                ax_obj = axes_grid[row_idx][col_idx]
                c = axis_colors.get(col, "gray")
                raw_vals = pd.to_numeric(df_raw[col], errors="coerce").to_numpy(dtype=float)
                cal_vals = pd.to_numeric(df_cal[col], errors="coerce").to_numpy(dtype=float)
                xr, yr = filter_valid_plot_xy(ts_raw, raw_vals)
                xc, yc = filter_valid_plot_xy(ts_cal, cal_vals)
                ax_obj.plot(xr, yr, lw=0.7, alpha=_RAW_ALPHA, color=c, label="raw")
                ax_obj.plot(xc, yc, lw=0.9, alpha=_CAL_ALPHA, color=c, ls="--", label="calibrated")
                ax_obj.axhline(0, color="gray", lw=0.5, ls=":")
                ax_obj.set_ylabel(f"{col} ({units.get(col, '')})")
                ax_obj.legend(fontsize=7, loc="upper right")
                ax_obj.grid(alpha=0.2, lw=0.4)
            for row_idx in range(len(cols), n_rows):
                axes_grid[row_idx][col_idx].set_visible(False)

        for ax_obj in axes_grid[-1]:
            if ax_obj.get_visible():
                ax_obj.set_xlabel("Time (s)")
        axes_grid[0][0].set_title("Accelerometer axes")
        if len(gyro_axes) > 0:
            axes_grid[0][1].set_title("Gyroscope axes")
        fig.suptitle(f"{section_dir.name} / {sensor} — per-axis calibration effect", y=1.01)
        fig.tight_layout()
        out_b = cal_dir / f"{sensor}_axes_comparison.png"
        fig.savefig(out_b, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_b)
        log.info("Plot written: %s", project_relative_path(out_b))

    return out_paths


# ---------------------------------------------------------------------------
# Plot 2: Calibration parameters — biases and rotation matrix
# ---------------------------------------------------------------------------

def plot_calibration_parameters(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> Path | None:
    """Gyro bias bars, rotation matri, and gravity bars per sensor.

    For Arduino, if ``all_methods.json`` includes a static calibration reference, the gyro row
    shows dynamic vs static gyro bias (grouped bars). Accelerometer static reference is not shown.

    Output
    ------
    ``calibrated/calibration_parameters.png``
    """
    cal_data = _load_calibration_json(section_dir)
    if not cal_data:
        log.warning("No calibration.json found in %s", section_dir.name)
        return None

    selected = sensors or list(_SENSORS)
    sensors_present = [s for s in selected if s in cal_data]
    if not sensors_present:
        return None

    static_ref = _static_calibration_reference(section_dir)

    n = len(sensors_present)
    # Plain grid: rows = gyro, rotation matrix, gravity; columns = sensors
    fig, axes_grid = plt.subplots(3, n, figsize=(4 * n + 1, 8), squeeze=False)
    fig.suptitle(f"{section_dir.name} — calibration parameters", fontsize=12)

    axes_labels = ["x", "y", "z"]
    bar_x = np.arange(3)
    bar_w = 0.35
    _STATIC_REF_COLOR = "#d62728"

    for s_idx, sensor in enumerate(sensors_present):
        params = cal_data[sensor]
        gyro_bias = params.get("gyro_bias", [0, 0, 0])
        rot_mat = params.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        grav_vec = params.get("gravity_vector_body", [0, 0, _G])
        color = _SENSOR_COLORS.get(sensor, "steelblue")

        ax_gyro = axes_grid[0, s_idx]
        if sensor == "arduino" and static_ref:
            static_gyro = static_ref.get("gyroscope_bias_deg_s", {})
            static_gyro_vals = [static_gyro.get(a, float("nan")) for a in axes_labels]
            ax_gyro.bar(
                bar_x - bar_w / 2,
                gyro_bias,
                bar_w,
                label="dynamic",
                color=color,
                alpha=0.8,
            )
            ax_gyro.bar(
                bar_x + bar_w / 2,
                static_gyro_vals,
                bar_w,
                label="static ref",
                color=_STATIC_REF_COLOR,
                alpha=0.8,
                hatch="//",
            )
            ax_gyro.legend(fontsize=8, loc="upper right", framealpha=0.92)
            static_warnings = static_ref.get("warnings", [])
            if static_warnings:
                ax_gyro.text(
                    0.02,
                    0.98,
                    "⚠ " + "; ".join(static_warnings),
                    transform=ax_gyro.transAxes,
                    fontsize=7,
                    va="top",
                    ha="left",
                    color="#8c564b",
                )
        else:
            ax_gyro.bar(bar_x, gyro_bias, color=color, alpha=0.8)
        ax_gyro.axhline(0, color="gray", lw=0.7, ls="--")
        ax_gyro.set_xticks(bar_x)
        ax_gyro.set_xticklabels(axes_labels)
        ax_gyro.set_ylabel("Bias (deg/s)")
        ax_gyro.set_title(f"{sensor} — gyro bias")
        ax_gyro.grid(axis="y", alpha=0.3)

        ax_rot = axes_grid[1, s_idx]
        R = np.array(rot_mat, dtype=float)
        im = ax_rot.imshow(R, cmap="RdBu", vmin=-1.5, vmax=1.5, aspect="equal")
        ax_rot.set_xticks([0, 1, 2])
        ax_rot.set_yticks([0, 1, 2])
        ax_rot.set_xticklabels([])
        ax_rot.set_yticklabels([])
        for i in range(3):
            for j in range(3):
                ax_rot.text(
                    j,
                    i,
                    f"{R[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(R[i, j]) < 0.7 else "white",
                )
        fig.colorbar(im, ax=ax_rot)
        ax_rot.set_title(f"{sensor} — rotation matrix")

        ax_grav = axes_grid[2, s_idx]
        g = np.array(grav_vec, dtype=float)
        ax_grav.bar(bar_x, g, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.8)
        ax_grav.axhline(_G, color="gray", lw=0.8, ls="--")
        ax_grav.axhline(-_G, color="gray", lw=0.8, ls="--")
        ax_grav.set_xticks(bar_x)
        ax_grav.set_xticklabels(axes_labels)
        ax_grav.set_ylabel("m/s²")
        ax_grav.set_title(f"{sensor} — gravity (body frame)")
        ax_grav.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = section_dir / "calibrated" / "calibration_parameters.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Plot 3: World-frame signals
# ---------------------------------------------------------------------------

def plot_calibration_world_frame(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Plot world-frame acc and gyro signals from the calibrated CSV.

    ``az_world`` should remain near +9.81 m/s² when the sensor is on a bike
    moving roughly level.  This plot makes gravity alignment quality immediately
    visible.

    Output
    ------
    ``calibrated/<sensor>_world_frame.png``
    """
    selected = sensors or list(_SENSORS)
    out_paths: list[Path] = []
    cal_dir = section_dir / "calibrated"

    for sensor in selected:
        cal_csv = cal_dir / f"{sensor}.csv"
        if not cal_csv.exists():
            continue
        df = read_csv(cal_csv)
        if "timestamp" not in df.columns:
            continue

        world_acc = [c for c in ("ax_world", "ay_world", "az_world") if c in df.columns]
        world_gyro = [c for c in ("gx_world", "gy_world", "gz_world") if c in df.columns]
        if not world_acc and not world_gyro:
            log.info("No world-frame columns in %s/%s — skipping world-frame plot", section_dir.name, sensor)
            continue

        ts = _ts_s(df)
        axis_colors = {
            "ax_world": "#d62728", "ay_world": "#2ca02c", "az_world": "#1f77b4",
            "gx_world": "#9467bd", "gy_world": "#8c564b", "gz_world": "#e377c2",
        }
        n_rows = int(bool(world_acc)) + int(bool(world_gyro))
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=True, squeeze=False)
        axes = axes.flatten()

        row = 0
        if world_acc:
            ax = axes[row]
            for col in world_acc:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                x, y = filter_valid_plot_xy(ts, vals)
                ax.plot(x, y, lw=0.8, alpha=0.85, color=axis_colors[col], label=col)
            ax.axhline(_G, color="gray", lw=1.0, ls="--", label=f"+g ({_G} m/s²)")
            ax.axhline(-_G, color="gray", lw=0.7, ls=":", label=f"−g")
            ax.axhline(0, color="black", lw=0.4, ls=":")
            ax.set_ylabel("World-frame acc (m/s²)")
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.grid(alpha=0.2, lw=0.4)
            row += 1

        if world_gyro:
            ax = axes[row]
            for col in world_gyro:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                x, y = filter_valid_plot_xy(ts, vals)
                ax.plot(x, y, lw=0.8, alpha=0.85, color=axis_colors[col], label=col)
            ax.axhline(0, color="black", lw=0.4, ls=":")
            ax.set_ylabel("World-frame gyro (deg/s)")
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.grid(alpha=0.2, lw=0.4)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{section_dir.name} / {sensor} — world-frame signals after calibration")
        fig.tight_layout()
        out_path = cal_dir / f"{sensor}_world_frame.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        out_paths.append(out_path)
        log.info("Plot written: %s", project_relative_path(out_path))

    return out_paths


# ---------------------------------------------------------------------------
# Plot 4: Method comparison (all_methods.json)
# ---------------------------------------------------------------------------

def plot_calibration_methods(
    section_dir: Path,
) -> Path | None:
    """Bar chart comparison of all calibration methods from ``all_methods.json``.

    Shows gravity residuals and forward confidence per method/sensor.

    Output
    ------
    ``calibrated/methods_comparison.png``
    """
    data = _load_all_methods_json(section_dir)
    if data is None:
        return None

    methods_data: dict = data.get("methods", {})
    selected_method: str = data.get("selected_method", "")
    method_names = list(methods_data.keys())
    if not method_names:
        return None

    n_methods = len(method_names)
    sensors = list(_SENSORS)
    bar_x = np.arange(n_methods)
    bar_w = 0.35

    n_rows = 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    fig.suptitle(f"{section_dir.name} — calibration method comparison", fontsize=12)

    method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    _METHOD_DISPLAY = {
        "gravity_only": "gravity only",
        "gravity_plus_forward": "gravity + forward",
        "gravity_plus_mag": "gravity + mag",
    }

    # Row 0: Gravity residuals per sensor
    for s_idx, sensor in enumerate(sensors):
        ax = axes[0][s_idx]
        residuals = [
            methods_data[m].get(sensor, {}).get("gravity_residual_ms2", float("nan"))
            for m in method_names
        ]
        colors = [
            method_colors[i % len(method_colors)] for i in range(n_methods)
        ]
        bars = ax.bar(bar_x, residuals, color=colors, alpha=0.8)
        ax.axhline(1.0, color="orange", lw=1.0, ls="--", label="marginal threshold (1.0)")
        ax.axhline(2.0, color="red", lw=1.0, ls="--", label="poor threshold (2.0)")
        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            [_METHOD_DISPLAY.get(m, m) for m in method_names], rotation=15, ha="right", fontsize=9
        )
        ax.set_ylabel("Gravity residual (m/s²)")
        ax.set_title(f"{sensor} — gravity residual (lower = better)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        for bar, v, m in zip(bars, residuals, method_names):
            if np.isfinite(v):
                label = f"{v:.3f}"
                if m == selected_method:
                    label += " ★"
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, label,
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold" if m == selected_method else "normal")

    # Row 1: Alignment confidence (forward / mag) + mag field norm secondary axis
    ax_fwd = axes[1][0]
    for i, method in enumerate(method_names):
        fwd_confs = [
            methods_data[method].get(s, {}).get("forward_confidence", 0.0)
            for s in sensors
        ]
        x_pos = bar_x[i] + np.array([-bar_w / 2, bar_w / 2])
        for j, (sensor, fc) in enumerate(zip(sensors, fwd_confs)):
            ax_fwd.bar(x_pos[j], fc, width=bar_w * 0.9,
                       color=_SENSOR_COLORS.get(sensor, "gray"), alpha=0.8,
                       label=sensor if i == 0 else "_nolegend_")
        # Overlay mag field norm for gravity_plus_mag method.
        if method == "gravity_plus_mag":
            for sensor in sensors:
                mag_norm = methods_data[method].get(sensor, {}).get("mag_field_norm")
                if mag_norm is not None:
                    ax_fwd.annotate(
                        f"|B|={mag_norm:.1f}",
                        xy=(bar_x[i], 0.02), fontsize=6, ha="center", va="bottom",
                        color="#7f7f7f",
                    )
                    break
    ax_fwd.axhline(0.3, color="orange", lw=1.0, ls="--", label="min threshold (0.3)")
    ax_fwd.set_xticks(bar_x)
    ax_fwd.set_xticklabels(
        [_METHOD_DISPLAY.get(m, m) for m in method_names], rotation=15, ha="right", fontsize=9
    )
    ax_fwd.set_ylabel("Alignment confidence [0–1]")
    ax_fwd.set_title("Alignment confidence (forward / mag)")
    ax_fwd.legend(fontsize=8)
    ax_fwd.set_ylim(0, 1.05)
    ax_fwd.grid(axis="y", alpha=0.3)

    # Quality labels
    ax_q = axes[1][1]
    ax_q.axis("off")
    col_headers = ["Method", "Selected"] + sensors
    cell_data = []
    for method in method_names:
        row_data = [_METHOD_DISPLAY.get(method, method), "★" if method == selected_method else ""]
        for sensor in sensors:
            q = methods_data[method].get(sensor, {}).get("quality", "?")
            fb = methods_data[method].get(sensor, {}).get("fallback_used", False)
            mag_tags = methods_data[method].get(sensor, {}).get("quality_tags", [])
            extra = ""
            if fb:
                extra = " ⚠"
            elif "mag_yaw_used" in mag_tags:
                extra = " [mag]"
            elif "mag_fallback_to_pca" in mag_tags:
                extra = " (pca)"
            row_data.append(f"{q}{extra}")
        cell_data.append(row_data)

    table = ax_q.table(
        cellText=cell_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    _quality_bg = {"good": "#d4edda", "marginal": "#fff3cd", "poor": "#f8d7da"}
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
        elif col >= 2:
            text = cell.get_text().get_text()
            for q, bg in _quality_bg.items():
                if q in text:
                    cell.set_facecolor(bg)
                    break
    ax_q.set_title("Quality summary")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = section_dir / "calibrated" / "methods_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def plot_calibration_stage(
    target: str | Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Run all calibration plots for a section directory.

    Produces:
    - ``<sensor>_calibration_comparison.png``  — norm before/after
    - ``<sensor>_axes_comparison.png``          — per-axis before/after
    - ``calibration_parameters.png``            — biases + rotation matrix
    - ``<sensor>_world_frame.png``              — world-frame signals
    - ``methods_comparison.png``                — method comparison (if all_methods.json present)
    """
    section_dir = _resolve_section_dir(target)
    out_paths: list[Path] = []

    try:
        out_paths += plot_calibration_raw_vs_calibrated(section_dir, sensors=sensors)
    except Exception as exc:
        log.warning("Raw-vs-calibrated plot failed for %s: %s", section_dir.name, exc)

    try:
        p = plot_calibration_parameters(section_dir, sensors=sensors)
        if p:
            out_paths.append(p)
    except Exception as exc:
        log.warning("Calibration parameters plot failed for %s: %s", section_dir.name, exc)

    try:
        out_paths += plot_calibration_world_frame(section_dir, sensors=sensors)
    except Exception as exc:
        log.warning("World-frame plot failed for %s: %s", section_dir.name, exc)

    try:
        p = plot_calibration_methods(section_dir)
        if p:
            out_paths.append(p)
    except Exception as exc:
        log.warning("Methods comparison plot failed for %s: %s", section_dir.name, exc)

    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_calibration",
        description="Generate calibration plots for a section directory.",
    )
    parser.add_argument("target", help="Section directory reference or calibrated dir")
    parser.add_argument(
        "--sensor",
        action="append",
        choices=list(_SENSORS),
        dest="sensors",
        help="Limit to one or more sensors (repeat option).",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "raw_vs_cal", "parameters", "world_frame", "methods"],
        default="all",
        help="Which plot(s) to generate (default: all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        section_dir = _resolve_section_dir(args.target)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return

    dispatch = {
        "raw_vs_cal": lambda: plot_calibration_raw_vs_calibrated(section_dir, sensors=args.sensors),
        "parameters": lambda: [plot_calibration_parameters(section_dir, sensors=args.sensors)],
        "world_frame": lambda: plot_calibration_world_frame(section_dir, sensors=args.sensors),
        "methods": lambda: [plot_calibration_methods(section_dir)],
        "all": lambda: plot_calibration_stage(section_dir, sensors=args.sensors),
    }

    paths = dispatch[args.plot]()
    paths = [p for p in (paths or []) if p is not None]
    if not paths:
        print("No plots generated.")
    for p in paths:
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
