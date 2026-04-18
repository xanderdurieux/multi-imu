"""Sync-stage visualizations — diagnostic plots for per-recording synchronization.

CLI (from the ``analysis`` directory)::

    python -m visualization.plot_sync <recording>
    python -m visualization.plot_sync all <recording>
    python -m visualization.plot_sync before-after <recording>
    python -m visualization.plot_sync offset-drift <recording>
    python -m visualization.plot_sync offset-drift-zoomed <recording> [--zoom-start S] [--zoom-end S]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from common.paths import recording_stage_dir
from visualization._primitives import draw_acc, draw_signal, draw_two_streams, imu_figure
from visualization._utils import (
    SENSOR_COLORS,
    SENSORS,
    acc_norm,
    load_sensor_df,
    relative_seconds,
    save_figure,
    shared_t0_ms,
    timestamps_to_relative_seconds,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_METHODS: tuple[str, ...] = (
    "multi_anchor",
    "one_anchor_adaptive",
    "one_anchor_prior",
    "signal_only",
)

_METHOD_COLORS: dict[str, str] = {
    "multi_anchor":        "#2ca02c",
    "one_anchor_adaptive": "#8c564b",
    "one_anchor_prior":    "#9467bd",
    "signal_only":         "#1f77b4",
}

_METHOD_LABELS: dict[str, str] = {
    "multi_anchor":        "Multi-anchor protocol",
    "one_anchor_adaptive": "One-anchor adaptive",
    "one_anchor_prior":    "One-anchor prior",
    "signal_only":         "Signal-only",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_sync_infos(recording_name: str) -> dict[str, dict | None]:
    """Load sync model info for every method.

    Your pipeline produces a consistent layout:
    - ``synced/sync_info.json`` contains the selected model plus per-method summaries.
    """
    synced_root = recording_stage_dir(recording_name, "synced")
    sync_info_path = synced_root / "sync_info.json"
    if not sync_info_path.exists():
        raise FileNotFoundError(f"Missing sync_info.json under {synced_root}")

    payload = json.loads(sync_info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sync_info.json: expected object at {sync_info_path}")

    target_time_origin_seconds = payload.get("target_time_origin_seconds")
    calibration = payload.get("calibration")
    methods = payload.get("methods")
    if target_time_origin_seconds is None or not isinstance(calibration, dict):
        raise ValueError(f"Invalid sync_info.json: missing origin/calibration under {sync_info_path}")

    result: dict[str, dict | None] = {}

    # Build the flattened shape expected by the plotting code.
    for method in _ALL_METHODS:
        method_summary = methods.get(method) if isinstance(methods, dict) else None
        if not isinstance(method_summary, dict):
            result[method] = None
            continue

        offset_s = method_summary.get("offset_seconds")
        drift_s = method_summary.get("drift_seconds_per_second")
        if offset_s is None or drift_s is None:
            result[method] = None
            continue

        result[method] = {
            "target_time_origin_seconds": target_time_origin_seconds,
            "offset_seconds": float(offset_s),
            "drift_seconds_per_second": float(drift_s),
            "calibration": calibration,  # includes `anchors`
        }

    return result


def _load_sync_info_payload(recording_name: str) -> dict:
    """Load the raw ``sync_info.json`` payload for *recording_name*."""
    synced_root = recording_stage_dir(recording_name, "synced")
    sync_info_path = synced_root / "sync_info.json"
    if not sync_info_path.exists():
        raise FileNotFoundError(f"Missing sync_info.json under {synced_root}")

    payload = json.loads(sync_info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sync_info.json: expected object at {sync_info_path}")
    return payload


def _method_metrics_rows(payload: dict) -> list[dict[str, object]]:
    """Build per-method metric rows from a ``sync_info.json`` payload."""
    methods = payload.get("methods")
    selected_method = str(payload.get("selected_method") or "")
    rows: list[dict[str, object]] = []

    for method in _ALL_METHODS:
        method_block = methods.get(method) if isinstance(methods, dict) else None
        if not isinstance(method_block, dict):
            rows.append(
                {
                    "method": method,
                    "label": _METHOD_LABELS[method],
                    "available": False,
                    "selected": method == selected_method,
                    "offset_seconds": np.nan,
                    "drift_seconds_per_second": np.nan,
                    "correlation": np.nan,
                }
            )
            continue

        rows.append(
            {
                "method": method,
                "label": _METHOD_LABELS[method],
                "available": bool(method_block.get("available", False)),
                "selected": method == selected_method,
                "offset_seconds": (
                    float(method_block["offset_seconds"])
                    if method_block.get("offset_seconds") is not None
                    else np.nan
                ),
                "drift_seconds_per_second": (
                    float(method_block["drift_seconds_per_second"])
                    if method_block.get("drift_seconds_per_second") is not None
                    else np.nan
                ),
                "correlation": (
                    float(method_block["corr_offset_and_drift"])
                    if method_block.get("corr_offset_and_drift") is not None
                    else np.nan
                ),
            }
        )

    return rows


def _format_metric(value: float, fmt: str) -> str:
    """Format a numeric value for a table cell, preserving missing entries."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def _collect_anchors(sync_infos: dict[str, dict | None]) -> list[dict]:
    """Return calibration anchor list from the first method that has one."""
    for method in _ALL_METHODS:
        info = sync_infos.get(method)
        if not isinstance(info, dict):
            continue
        cal = info.get("calibration")
        if isinstance(cal, dict):
            anchors = cal.get("anchors")
            if anchors:
                return list(anchors)
    return []


def _tgt_origin_s(sync_infos: dict[str, dict | None]) -> float | None:
    """Return ``target_time_origin_seconds`` from the first available method."""
    for method in _ALL_METHODS:
        info = sync_infos.get(method)
        if isinstance(info, dict):
            v = info.get("target_time_origin_seconds")
            if v is not None:
                return float(v)
    return None


def _recording_duration_s(recording_name: str) -> float:
    """Estimate recording duration (s) from the arduino parsed CSV."""
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    df = load_sensor_df(parsed_dir, "arduino") if parsed_dir.is_dir() else None
    if df is not None:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        ts = ts[np.isfinite(ts)]
        if ts.size > 1:
            return float(ts[-1] - ts[0]) / 1000.0
    return 3600.0


def _build_zoom_windows(
    overlap_start_s: float,
    overlap_end_s: float,
    *,
    window_s: float = 45.0,
) -> list[tuple[str, float, float]]:
    """Return first/middle/end windows within the shared synced overlap."""
    overlap_duration_s = max(0.0, overlap_end_s - overlap_start_s)
    if overlap_duration_s <= 0.0:
        return []

    actual_window_s = min(window_s, overlap_duration_s)
    midpoint_s = overlap_start_s + overlap_duration_s / 2.0

    windows = [
        ("Start", overlap_start_s, overlap_start_s + actual_window_s),
        (
            "Middle",
            max(overlap_start_s, midpoint_s - actual_window_s / 2.0),
            min(overlap_end_s, midpoint_s + actual_window_s / 2.0),
        ),
        ("End", overlap_end_s - actual_window_s, overlap_end_s),
    ]
    return windows

# ---------------------------------------------------------------------------
# Ax-level drawing primitives (sync-specific)
# ---------------------------------------------------------------------------

def draw_sync_model_comparison(
    ax: plt.Axes,
    sync_infos: dict[str, dict | None],
    *,
    duration_s: float,
    origin_s: float,
    anchors: list[dict] | None = None,
    show_anchor_scorebar: bool = False,
) -> plt.Axes:
    """Draw sync method offset lines + calibration anchor scatter on *ax*.

    **Coordinate system** (origin = ``target_time_origin_seconds``):

    - x-axis: τ_tgt = t_tgt − origin_s  (target clock, s from recording start)
    - y-axis: Δ = t_ref − t_tgt  (local offset between clocks, s)

    Each method is drawn as the offset model
    ``Δ(τ_tgt) = offset_seconds + drift_seconds_per_second * τ_tgt``.
    Calibration anchors appear as score-colored dots at their measured
    ``(τ_tgt, offset_s)`` positions.
    """
    if anchors is None:
        anchors = _collect_anchors(sync_infos)

    # Compute baseline to normalise y-axis (removes the ~1.77e9 Unix-timestamp component)
    raw_offsets = [
        float(info["offset_seconds"])
        for info in sync_infos.values()
        if isinstance(info, dict) and info.get("offset_seconds") is not None
    ]
    y_baseline = float(np.median(raw_offsets)) if raw_offsets else 0.0

    t_range = np.linspace(0.0, duration_s, 500)
    y_values: list[float] = []
    primary_line: tuple[np.ndarray, np.ndarray, str] | None = None
    intercept_specs: list[dict[str, float | str]] = []
    drift_label_specs: list[dict[str, float | str]] = []

    # Per-method model lines
    for method in _ALL_METHODS:
        info = sync_infos.get(method)
        if not isinstance(info, dict):
            continue
        offset_s = info.get("offset_seconds")
        if offset_s is None:
            continue
        offset_s = float(offset_s) - y_baseline
        drift    = float(info.get("drift_seconds_per_second") or 0.0)
        color    = _METHOD_COLORS[method]

        y_line = drift * t_range + offset_s
        ax.plot(t_range, y_line, color=color, lw=1.8,
                label=_METHOD_LABELS[method], zorder=3)
        y_values.extend([float(np.nanmin(y_line)), float(np.nanmax(y_line))])
        if primary_line is None:
            primary_line = (t_range.copy(), y_line.copy(), color)
        label_x = 0.72 * duration_s
        label_y = float(np.interp(label_x, t_range, y_line))
        drift_label_specs.append(
            {
                "x": label_x,
                "y": label_y,
                "color": color,
                "label": f"{drift * 1e6:+.1f} ppm",
            }
        )

        # Y-intercept marker + label
        ax.scatter([0.0], [offset_s], color=color, s=55, marker="o", zorder=5)
        intercept_specs.append(
            {
                "y": offset_s,
                "color": color,
                "label": f"{offset_s:+.3f} s",
            }
        )

    # Calibration anchors
    if anchors:
        τ_tgt_arr = np.array([float(a["tgt_ms"]) / 1000.0 - origin_s for a in anchors])
        offset_arr = np.array(
            [float(a["offset_s"]) - y_baseline for a in anchors],
            dtype=float,
        )
        anchor_scores = np.array(
            [float(a.get("score")) if a.get("score") is not None else np.nan for a in anchors],
            dtype=float,
        )
        finite_scores = anchor_scores[np.isfinite(anchor_scores)]
        score_norm = None
        if finite_scores.size:
            score_norm = colors.Normalize(
                vmin=min(0.0, float(finite_scores.min())),
                vmax=max(1.0, float(finite_scores.max())),
            )

        anchor_scatter = ax.scatter(
            τ_tgt_arr, offset_arr,
            c=anchor_scores if finite_scores.size else "#444444",
            cmap="RdYlGn",
            norm=score_norm,
            s=65,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            alpha=0.95,
            label="Calibration anchors",
        )
        y_values.extend(offset_arr[np.isfinite(offset_arr)].tolist())

        if primary_line is not None:
            line_x, line_y, line_color = primary_line
            residual_specs: list[dict[str, float]] = []
            for xt, ya in zip(τ_tgt_arr, offset_arr):
                if not (np.isfinite(xt) and np.isfinite(ya)):
                    continue
                y_model = float(np.interp(xt, line_x, line_y))
                residual_ms = (ya - y_model) * 1000.0
                ax.plot(
                    [xt, xt],
                    [y_model, ya],
                    color=line_color,
                    lw=1.0,
                    alpha=0.8,
                    zorder=4,
                )
                residual_specs.append(
                    {
                        "x": float(xt),
                        "y_anchor": float(ya),
                        "y_model": y_model,
                        "residual_ms": residual_ms,
                    }
                )
                y_values.append(y_model)

            residual_specs.sort(key=lambda item: (item["x"], item["y_anchor"]))
            if residual_specs:
                # Span from anchor range only — not global y_values which can be
                # dominated by signal_only's offset.
                anchor_ys = [s["y_anchor"] for s in residual_specs]
                anchor_span = max(max(anchor_ys) - min(anchor_ys), 0.02)
                min_gap = max(0.04 * anchor_span, 0.005)
                placed_y: list[float] = []
                for spec in residual_specs:
                    direction = 1.0 if spec["residual_ms"] >= 0.0 else -1.0
                    base_y = spec["y_anchor"] + direction * (0.015 * anchor_span)
                    text_y = base_y
                    while any(abs(text_y - prev_y) < min_gap for prev_y in placed_y):
                        text_y += direction * min_gap
                    placed_y.append(text_y)
                    va = "bottom" if direction > 0 else "top"
                    ax.text(
                        spec["x"] + 3.0,
                        text_y,
                        f"{spec['residual_ms']:+.1f} ms",
                        fontsize=7,
                        color=line_color,
                        va=va,
                        ha="left",
                        zorder=7,
                        clip_on=True,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
                    )
        if show_anchor_scorebar and finite_scores.size:
            cbar = ax.figure.colorbar(anchor_scatter, ax=ax, pad=0.015)
            cbar.set_label("Anchor score", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    if intercept_specs:
        intercept_specs.sort(key=lambda item: float(item["y"]))
        # Use a tight y-span (without signal_only outliers) for min_gap so labels
        # don't get pushed far outside the axes.
        core_ys = [float(item["y"]) for item in intercept_specs]
        if len(core_ys) >= 3:
            q25, q75 = float(np.percentile(core_ys, 25)), float(np.percentile(core_ys, 75))
            iqr = max(q75 - q25, 1e-6)
            core_ys_filtered = [v for v in core_ys if abs(v - np.median(core_ys)) <= 3 * iqr]
        else:
            core_ys_filtered = core_ys
        core_span = max(
            max(core_ys_filtered) - min(core_ys_filtered) if core_ys_filtered else 0.02,
            0.02,
        )
        min_gap = 0.08 * core_span
        placed_y: list[float] = []
        for spec in intercept_specs:
            text_y = float(spec["y"])
            while any(abs(text_y - prev_y) < min_gap for prev_y in placed_y):
                text_y += min_gap
            placed_y.append(text_y)
            ax.annotate(
                str(spec["label"]),
                xy=(0.0, float(spec["y"])),
                xytext=(duration_s * 0.015, text_y),
                textcoords="data",
                fontsize=7,
                color=str(spec["color"]),
                va="center",
                annotation_clip=True,
                arrowprops={"arrowstyle": "-", "color": str(spec["color"]), "lw": 0.5},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
            )

    if drift_label_specs:
        drift_label_specs.sort(key=lambda item: float(item["y"]))
        # Use only the drift-label y positions themselves for min_gap, ignoring
        # outlier methods (signal_only) so close methods don't get pushed off-screen.
        drift_ys = [float(s["y"]) for s in drift_label_specs]
        if len(drift_ys) >= 2:
            dq25 = float(np.percentile(drift_ys, 25))
            dq75 = float(np.percentile(drift_ys, 75))
            d_iqr = max(dq75 - dq25, 1e-6)
            drift_ys_core = [v for v in drift_ys if abs(v - np.median(drift_ys)) <= 3 * d_iqr]
        else:
            drift_ys_core = drift_ys
        drift_span = max(max(drift_ys_core) - min(drift_ys_core) if drift_ys_core else 0.02, 0.02)
        min_gap = max(0.06 * drift_span, 0.005)
        placed_y: list[float] = []
        for spec in drift_label_specs:
            text_y = float(spec["y"])
            while any(abs(text_y - prev_y) < min_gap for prev_y in placed_y):
                text_y += min_gap
            placed_y.append(text_y)
            ax.text(
                float(spec["x"]),
                text_y,
                str(spec["label"]),
                fontsize=7,
                color=str(spec["color"]),
                va="center",
                ha="left",
                zorder=7,
                clip_on=True,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
            )

    ax.set_xlabel("Target time from origin (s)")
    ax.set_ylabel(f"Δ offset (s)  [ref ≈ {y_baseline:.2f} s]")
    if y_values:
        # Tukey IQR fence: clip extreme outliers (e.g. signal_only diverging by
        # tens of seconds) so calibration methods remain visible.
        arr = np.array(y_values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            iqr = max(q75 - q25, 0.05)   # floor at 50 ms so axis never collapses
            lo = q25 - 2.0 * iqr
            hi = q75 + 2.0 * iqr
            span = max(hi - lo, 0.1)
            margin = max(0.02, 0.15 * span)
            ax.set_ylim(lo - margin, hi + margin)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8, loc="upper left")
    return ax


# ---------------------------------------------------------------------------
# Figure-level public functions
# ---------------------------------------------------------------------------


def plot_before_after_imu(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Accelerometer |acc|: parsed sensors separately, then merged synced pair.

    Three-panel layout:

    - **Before (top two)**: SPORSA and Arduino each at their own t = 0
    - **After (bottom)**: both sensors on a shared timeline post-sync

    Output: ``synced/before_after_comparison.png``.
    """
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    synced_dir = recording_stage_dir(recording_name, "synced")

    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed directory not found: {parsed_dir}")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    parsed = {s: df for s in SENSORS if (df := load_sensor_df(parsed_dir, s)) is not None}
    synced = {s: df for s in SENSORS if (df := load_sensor_df(synced_dir, s)) is not None}

    if len(parsed) < 2:
        raise FileNotFoundError(f"Both sensor CSVs required under {parsed_dir}")
    if len(synced) < 2:
        raise FileNotFoundError(f"Both sensor CSVs required under {synced_dir}")

    if output_path is None:
        output_path = synced_dir / "before_after_comparison.png"

    fig = plt.figure(figsize=(12, 7.0))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    gs_before = gs_outer[0].subgridspec(2, 1, hspace=0.0)

    ax_sp     = fig.add_subplot(gs_before[0, 0])
    ax_ar     = fig.add_subplot(gs_before[1, 0], sharex=ax_sp)
    ax_merged = fig.add_subplot(gs_outer[1, 0])

    # --- Before panels: each sensor at its own t = 0 ---
    for sensor, ax in (("sporsa", ax_sp), ("arduino", ax_ar)):
        df  = parsed[sensor]
        t_s = timestamps_to_relative_seconds(df["timestamp"])
        draw_acc(ax, t_s, df, norm=True, label=sensor, color=SENSOR_COLORS[sensor])
        ax.grid(alpha=0.2, lw=0.4)

    ax_sp.set_title("Before sync", fontsize=10, pad=4)
    ax_sp.tick_params(axis="x", which="both", labelbottom=False, bottom=False)

    # --- After panel: shared t0 across both synced streams ---
    t0_ms = shared_t0_ms(*synced.values())
    for sensor, df in synced.items():
        t_s  = relative_seconds(df["timestamp"].to_numpy(dtype=float), t0_ms)
        norm = acc_norm(df)
        if norm is not None:
            draw_signal(
                ax_merged, t_s, norm,
                label=sensor, color=SENSOR_COLORS[sensor],
                lw=0.85, alpha=0.88,
            )

    ax_merged.set_title("After sync", fontsize=10, pad=4)
    ax_merged.set_xlabel("Time from recording start (s)")
    ax_merged.set_ylabel("|acc| (m/s²)")
    ax_merged.legend(fontsize=8, loc="upper right")
    ax_merged.grid(alpha=0.2, lw=0.4)

    fig.suptitle(
        f"{recording_name} — |acc|: before (separate) vs after (merged)",
        fontsize=13, y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.91, bottom=0.055)
    return save_figure(fig, output_path, dpi=150)


def plot_sync_offset_drift_model_comparison(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Offset vs target time: all sync methods as model lines + calibration anchors.

    The coordinate origin is ``target_time_origin_seconds`` (the Arduino epoch),
    so each method line starts at ``offset_seconds`` and changes with slope
    ``drift_seconds_per_second``. This keeps the y-axis in offset space, which
    makes anchor-vs-model differences visible at millisecond scale.

    Output: ``synced/offset_drift_comparison.png``.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    sync_infos = _load_sync_infos(recording_name)
    if not any(v is not None for v in sync_infos.values()):
        raise FileNotFoundError(
            f"No sync_info.json found for any method under {synced_dir}"
        )

    origin_s = _tgt_origin_s(sync_infos)
    if origin_s is None:
        raise ValueError("No target_time_origin_seconds found in any sync_info.")

    duration_s = _recording_duration_s(recording_name)

    if output_path is None:
        output_path = synced_dir / "offset_drift_comparison.png"

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_sync_model_comparison(
        ax,
        sync_infos,
        duration_s=duration_s,
        origin_s=origin_s,
        show_anchor_scorebar=True,
    )
    ax.set_title(
        f"{recording_name} — sync models: offset vs target time",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=150)


def plot_sync_zoomed_comparison(
    recording_name: str,
    *,
    zoom_start_s: float | None = None,
    zoom_end_s: float | None = None,
    output_path: Path | None = None,
) -> Path:
    """Three-window synced comparison between SPORSA and Arduino.

    Creates three stacked comparison panels from the synced streams:
    the first, middle, and final 45 s of the shared recording overlap.
    ``zoom_start_s`` / ``zoom_end_s`` remain accepted for CLI compatibility,
    but are ignored by this view.

    Output: ``synced/zoomed_comparison.png``.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    sporsa_df = load_sensor_df(synced_dir, "sporsa")
    arduino_df = load_sensor_df(synced_dir, "arduino")
    if sporsa_df is None or arduino_df is None:
        raise FileNotFoundError(
            f"Both synced sensor CSVs are required under {synced_dir}"
        )

    t0_ms = shared_t0_ms(sporsa_df, arduino_df)
    sporsa_t_s = relative_seconds(
        sporsa_df["timestamp"].to_numpy(dtype=float),
        t0_ms,
    )
    arduino_t_s = relative_seconds(
        arduino_df["timestamp"].to_numpy(dtype=float),
        t0_ms,
    )

    sporsa_acc = acc_norm(sporsa_df)
    arduino_acc = acc_norm(arduino_df)
    if sporsa_acc is None or arduino_acc is None:
        raise ValueError("Both synced streams need accelerometer data for comparison.")

    sporsa_valid = sporsa_t_s[np.isfinite(sporsa_t_s)]
    arduino_valid = arduino_t_s[np.isfinite(arduino_t_s)]
    if sporsa_valid.size == 0 or arduino_valid.size == 0:
        raise ValueError("Synced timestamps are empty after filtering.")

    overlap_start_s = max(float(sporsa_valid[0]), float(arduino_valid[0]))
    overlap_end_s = min(float(sporsa_valid[-1]), float(arduino_valid[-1]))
    windows = _build_zoom_windows(overlap_start_s, overlap_end_s)
    if not windows:
        raise ValueError("No shared synced overlap available for zoomed comparison.")

    if output_path is None:
        output_path = synced_dir / "zoomed_comparison.png"

    fig, axes = imu_figure(3, row_height=2.8, width=12.0, sharex=False)

    for ax, (label, window_start_s, window_end_s) in zip(axes, windows):
        draw_two_streams(
            ax,
            sporsa_t_s,
            sporsa_acc,
            arduino_t_s,
            arduino_acc,
            label_a="sporsa",
            label_b="arduino",
            ylabel="|acc| (m/s²)",
            title=f"{label} [{window_start_s:.0f}–{window_end_s:.0f} s]",
            lw=0.9,
            alpha=0.85,
        )
        ax.set_xlim(window_start_s, window_end_s)
        ax.grid(alpha=0.2, lw=0.4)

    axes[-1].set_xlabel("Time from shared synced start (s)")
    fig.suptitle(
        f"{recording_name} — synced SPORSA vs Arduino comparison",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return save_figure(fig, output_path, dpi=150)


def plot_sync_method_metrics(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Plot method correlation bars with selected-method highlight and metrics table.

    Output: ``synced/method_metrics_comparison.png``.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    payload = _load_sync_info_payload(recording_name)
    rows = _method_metrics_rows(payload)
    selected_method = str(payload.get("selected_method") or "")

    if output_path is None:
        output_path = synced_dir / "method_metrics_comparison.png"

    labels = [str(row["label"]) for row in rows]
    correlations = np.array(
        [
            float(row["correlation"]) if np.isfinite(float(row["correlation"])) else np.nan
            for row in rows
        ],
        dtype=float,
    )
    x = np.arange(len(rows))
    base_colors = [_METHOD_COLORS[str(row["method"])] for row in rows]
    bar_colors = []
    bar_edges = []
    bar_widths = []
    for row, color in zip(rows, base_colors, strict=False):
        is_selected = bool(row["selected"])
        available = bool(row["available"])
        bar_colors.append(color if available else "#d9d9d9")
        bar_edges.append("black" if is_selected else "#666666")
        bar_widths.append(2.0 if is_selected else 0.8)

    fig = plt.figure(figsize=(11.5, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.45], hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])

    finite_corr = correlations[np.isfinite(correlations)]
    if finite_corr.size:
        ymin = min(0.0, float(finite_corr.min()) - 0.05)
        ymax = max(0.05, float(finite_corr.max()) + 0.08)
    else:
        ymin, ymax = 0.0, 1.0

    bars = ax.bar(
        x,
        np.nan_to_num(correlations, nan=0.0),
        color=bar_colors,
        edgecolor=bar_edges,
        linewidth=bar_widths,
        zorder=3,
    )

    for bar, row in zip(bars, rows, strict=False):
        corr = float(row["correlation"])
        label = f"{corr:.3f}" if np.isfinite(corr) else "n/a"
        y = corr if np.isfinite(corr) else 0.0
        va = "bottom" if y >= 0 else "top"
        offset = 0.015 if y >= 0 else -0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            label,
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold" if bool(row["selected"]) else "normal",
        )

    for idx, row in enumerate(rows):
        if bool(row["selected"]):
            ax.scatter(
                [idx],
                [max(float(row["correlation"]) if np.isfinite(float(row["correlation"])) else 0.0, ymin) + 0.04],
                marker="*",
                s=180,
                color="#f1c40f",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                clip_on=False,
            )

    ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.8, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Correlation score")
    ax.set_title(f"{recording_name} — sync method correlation comparison", fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.4, zorder=1)

    for tick, row in zip(ax.get_xticklabels(), rows, strict=False):
        if bool(row["selected"]):
            tick.set_fontweight("bold")
            tick.set_color(_METHOD_COLORS[str(row["method"])])

    table_rows = [
        [
            str(row["label"]),
            _format_metric(float(row["offset_seconds"]), ".3f"),
            _format_metric(float(row["drift_seconds_per_second"]), ".6f"),
            _format_metric(float(row["correlation"]), ".3f"),
        ]
        for row in rows
    ]

    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Method", "Offset (s)", "Drift (s/s)", "Correlation"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.0, 0.96, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row_idx == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
            continue
        data_row = rows[row_idx - 1]
        if bool(data_row["selected"]):
            cell.set_facecolor("#fff6cc")
        elif not bool(data_row["available"]):
            cell.set_facecolor("#f7f7f7")
            cell.set_text_props(color="#777777")
        if col_idx == 0:
            cell.set_text_props(ha="left")

    selection_suffix = f"selected: {_METHOD_LABELS.get(selected_method, selected_method)}" if selected_method else "no selected method"
    fig.suptitle(f"{recording_name} — sync method summary ({selection_suffix})", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return save_figure(fig, output_path, dpi=150)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def plot_anchor_fits(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Per-anchor acc_norm alignment: ref vs tgt under coarse and refined offset.

    One panel per calibration anchor.  Each panel overlays:

    - **ref** (blue): acc_norm over the full segment window.
    - **tgt coarse** (orange dashed): tgt acc_norm shifted by the shake-center offset.
    - **tgt refined** (orange solid): tgt acc_norm shifted by the xcorr-refined offset.
    - Detected peaks for both sensors as vertical dashed lines.
    - Shake zone shaded in light grey.

    Output: ``synced/anchor_fits.png``.
    """
    from parser.calibration_segments import load_calibration_segments_from_json
    from sync.stream_io import load_stream, resample_stream

    _RATE = 100.0
    _REF_SENSOR = "sporsa"
    _TGT_SENSOR = "arduino"
    _COLOR_REF = SENSOR_COLORS.get(_REF_SENSOR, "#1f77b4")
    _COLOR_TGT = SENSOR_COLORS.get(_TGT_SENSOR, "#ff7f0e")

    ref_segs = load_calibration_segments_from_json(recording_name, _REF_SENSOR)
    tgt_segs = load_calibration_segments_from_json(recording_name, _TGT_SENSOR)

    if len(ref_segs) != len(tgt_segs):
        raise ValueError(
            f"Segment count mismatch for {recording_name!r}: "
            f"{_REF_SENSOR}={len(ref_segs)}, {_TGT_SENSOR}={len(tgt_segs)}"
        )

    payload = _load_sync_info_payload(recording_name)
    anchors_data = (
        payload.get("calibration", {}).get("anchors", [])
    )

    parsed_dir = recording_stage_dir(recording_name, "parsed")
    ref_df = load_stream(parsed_dir / f"{_REF_SENSOR}.csv")
    tgt_df = load_stream(parsed_dir / f"{_TGT_SENSOR}.csv")

    def _shake_center(seg) -> float:
        return (seg.start_ms + seg.static_pre_ms + seg.end_ms - seg.static_post_ms) / 2.0

    n = len(ref_segs)
    fig, axes = imu_figure(n, row_height=2.6, width=14.0, sharex=False)

    for i, (ref_seg, tgt_seg) in enumerate(zip(ref_segs, tgt_segs)):
        ax = axes[i]

        ref_center = _shake_center(ref_seg)
        tgt_center = _shake_center(tgt_seg)
        coarse_offset_s = (ref_center - tgt_center) / 1000.0

        if i < len(anchors_data):
            refined_offset_s = float(anchors_data[i]["offset_s"])
            score_raw = anchors_data[i].get("score")
            score_str = f"{score_raw:.3f}" if score_raw is not None else "n/a"
        else:
            refined_offset_s = coarse_offset_s
            score_str = "n/a"

        ref_start = ref_seg.start_ms
        ref_end = ref_seg.end_ms

        # Reference signal on its natural timeline
        ref_win = resample_stream(
            ref_df, _RATE, start_ms=ref_start, end_ms=ref_end, columns=["acc_norm"]
        )
        ref_t_s = (ref_win["timestamp"].to_numpy(dtype=float) - ref_start) / 1000.0
        ref_y = ref_win["acc_norm"].to_numpy(dtype=float) if "acc_norm" in ref_win.columns else np.full(len(ref_t_s), np.nan)

        # Target signal fetched in a common window, then shifted to ref timeline
        # Use the refined-offset window (slightly wider via coarse margin if large delta)
        delta_s = abs(refined_offset_s - coarse_offset_s)
        margin_ms = max(200.0, delta_s * 1500.0)
        tgt_fetch_start = ref_start - refined_offset_s * 1000.0 - margin_ms
        tgt_fetch_end = ref_end - refined_offset_s * 1000.0 + margin_ms

        tgt_win = resample_stream(
            tgt_df, _RATE, start_ms=tgt_fetch_start, end_ms=tgt_fetch_end, columns=["acc_norm"]
        )
        tgt_raw_t_ms = tgt_win["timestamp"].to_numpy(dtype=float)
        tgt_y = tgt_win["acc_norm"].to_numpy(dtype=float) if "acc_norm" in tgt_win.columns else np.full(len(tgt_raw_t_ms), np.nan)

        # Shift tgt timestamps to ref timeline for each alignment
        tgt_coarse_t_s = (tgt_raw_t_ms + coarse_offset_s * 1000.0 - ref_start) / 1000.0
        tgt_refined_t_s = (tgt_raw_t_ms + refined_offset_s * 1000.0 - ref_start) / 1000.0

        duration_s = (ref_end - ref_start) / 1000.0

        # Shade shake zone
        shake_start_s = ref_seg.static_pre_ms / 1000.0
        shake_end_s = duration_s - ref_seg.static_post_ms / 1000.0
        ax.axvspan(shake_start_s, shake_end_s, color="#e8e8e8", alpha=0.6, zorder=0, label="shake zone")

        # Draw signals
        draw_signal(ax, ref_t_s, ref_y, label=f"{_REF_SENSOR} (ref)", color=_COLOR_REF, lw=1.0, alpha=0.9)
        draw_signal(ax, tgt_coarse_t_s, tgt_y, label=f"{_TGT_SENSOR} coarse ({coarse_offset_s:+.3f}s)", color=_COLOR_TGT, lw=0.8, alpha=0.45)
        draw_signal(ax, tgt_refined_t_s, tgt_y, label=f"{_TGT_SENSOR} refined ({refined_offset_s:+.3f}s)", color=_COLOR_TGT, lw=1.0, alpha=0.9)

        # Ref peaks
        for peak_ms in ref_seg.peak_ms:
            peak_rel_s = (peak_ms - ref_start) / 1000.0
            ax.axvline(peak_rel_s, color=_COLOR_REF, lw=0.7, linestyle=":", alpha=0.8, zorder=4)

        # Tgt peaks shifted to ref timeline (refined)
        for peak_ms in tgt_seg.peak_ms:
            peak_rel_s = (peak_ms + refined_offset_s * 1000.0 - ref_start) / 1000.0
            ax.axvline(peak_rel_s, color=_COLOR_TGT, lw=0.7, linestyle=":", alpha=0.8, zorder=4)

        ax.set_xlim(-0.5, duration_s + 0.5)
        ax.set_ylabel("|acc| (m/s²)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.grid(alpha=0.2, lw=0.35)

        delta_str = f"Δ={refined_offset_s - coarse_offset_s:+.3f}s"
        ax.set_title(
            f"Anchor {i}  |  coarse={coarse_offset_s:+.4f}s  refined={refined_offset_s:+.4f}s  {delta_str}  score={score_str}",
            fontsize=9,
        )

    axes[-1].set_xlabel("Time from ref segment start (s)")
    fig.suptitle(f"{recording_name} — calibration anchor fits", fontsize=12, y=0.999)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))

    if output_path is None:
        output_path = recording_stage_dir(recording_name, "synced") / "anchor_fits.png"

    return save_figure(fig, output_path, dpi=150)


def plot_sync_stage(recording_name: str, output_path: Path | None = None):
    """Plot the sync stage for a recording."""
    plot_before_after_imu(recording_name, output_path=output_path)
    plot_sync_offset_drift_model_comparison(recording_name, output_path=output_path)
    plot_sync_zoomed_comparison(recording_name, output_path=output_path)
    plot_sync_method_metrics(recording_name, output_path=output_path)
    plot_anchor_fits(recording_name, output_path=output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_sync",
        description=(
            "Sync-stage diagnostic plots.\n\n"
            "  <recording>                      — generate all sync plots\n"
            "  --plot <plot>                    — plot only this plot: before-after, offset-drift-model, zoomed, method-metrics, or all."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("recording_name", nargs="?", help="Recording folder name")
    parser.add_argument(
        "--plot",
        choices=["before-after", "offset-drift-model", "zoomed", "method-metrics", "anchor-fits", "all"],
        default="all",
        help="Plot only this plot: before-after, offset-drift-model, zoomed, method-metrics, anchor-fits, or all.",
    )
    parser.add_argument(
        "--output",
        help="Output filename for the plots."
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    recording_name = args.recording_name
    output_name = args.output

    if args.plot == "all":
        plot_sync_stage(recording_name, output_path=output_name)
    elif args.plot == "before-after":
        plot_before_after_imu(recording_name, output_path=output_name)
    elif args.plot == "offset-drift-model":
        plot_sync_offset_drift_model_comparison(recording_name, output_path=output_name)
    elif args.plot == "zoomed":
        plot_sync_zoomed_comparison(recording_name, output_path=output_name)
    elif args.plot == "method-metrics":
        plot_sync_method_metrics(recording_name, output_path=output_name)
    elif args.plot == "anchor-fits":
        plot_anchor_fits(recording_name, output_path=output_name)


if __name__ == "__main__":
    main()
