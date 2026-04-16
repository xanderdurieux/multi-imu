"""Sync-stage visualizations — diagnostic plots for per-recording synchronization.

CLI (from the ``analysis`` directory)::

    python -m visualization.plot_sync before-after  <recording>
    python -m visualization.plot_sync offset-drift  <recording>
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

from common.paths import recording_stage_dir
from sync.sync_info_format import flatten_sync_info_dict
from visualization._primitives import draw_acc, draw_signal
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

_METHOD_STAGES: dict[str, str] = {
    "multi_anchor":        "synced/multi_anchor",
    "one_anchor_adaptive": "synced/one_anchor_adaptive",
    "one_anchor_prior":    "synced/one_anchor_prior",
    "signal_only":         "synced/signal_only",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_sync_infos(recording_name: str) -> dict[str, dict | None]:
    """Load and flatten ``sync_info.json`` for every method stage."""
    result: dict[str, dict | None] = {}
    for method, stage in _METHOD_STAGES.items():
        path = recording_stage_dir(recording_name, stage) / "sync_info.json"
        if not path.exists():
            result[method] = None
            continue
        try:
            result[method] = flatten_sync_info_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            log.warning("Could not load sync_info for %s/%s: %s", recording_name, method, exc)
            result[method] = None
    return result


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


def _sync_model_ylim(
    sync_infos: dict[str, dict | None],
    xlim: tuple[float, float],
    anchors: list[dict],
    origin_s: float,
    *,
    margin: float = 1.0,
) -> tuple[float, float]:
    """Tight y-range for the sync model view over *xlim*, including anchors."""
    x1, x2 = xlim
    ys: list[float] = [x1, x2]  # the y = x reference touches these corners

    for method in _ALL_METHODS:
        info = sync_infos.get(method)
        if not isinstance(info, dict):
            continue
        offset_s = info.get("offset_seconds")
        if offset_s is None:
            continue
        drift = float(info.get("drift_seconds_per_second") or 0.0)
        for x in (x1, x2):
            ys.append((1.0 + drift) * x + float(offset_s))

    for a in anchors:
        τ_tgt = float(a["t_tgt_s"]) - origin_s
        if x1 <= τ_tgt <= x2:
            ys.append(float(a["t_ref_s"]) - origin_s)

    return min(ys) - margin, max(ys) + margin


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
) -> plt.Axes:
    """Draw sync method lines + calibration anchor scatter on *ax*.

    **Coordinate system** (origin = ``target_time_origin_seconds``):

    - x-axis: τ_tgt = t_tgt − origin_s  (target clock, s from recording start)
    - y-axis: τ_ref = t_ref − origin_s  (reference clock, s from recording start)

    Each method is drawn as a line that passes through
    ``(τ_tgt=0, offset_seconds)`` with slope ``(1 + drift_seconds_per_second)``.
    Calibration anchors appear as black triangles at their measured
    ``(τ_tgt, τ_ref)`` positions.
    """
    if anchors is None:
        anchors = _collect_anchors(sync_infos)

    t_range = np.linspace(0.0, duration_s, 500)

    # Perfect-sync reference diagonal
    ax.plot(
        t_range, t_range,
        color="gray", lw=1.0, ls="--", alpha=0.4,
        label="Perfect sync (y = x)", zorder=1,
    )

    # Per-method model lines
    for method in _ALL_METHODS:
        info = sync_infos.get(method)
        if not isinstance(info, dict):
            continue
        offset_s = info.get("offset_seconds")
        if offset_s is None:
            continue
        offset_s = float(offset_s)
        drift    = float(info.get("drift_seconds_per_second") or 0.0)
        color    = _METHOD_COLORS[method]

        y_line = (1.0 + drift) * t_range + offset_s
        ax.plot(t_range, y_line, color=color, lw=1.8,
                label=_METHOD_LABELS[method], zorder=3)

        # Y-intercept marker + label
        ax.scatter([0.0], [offset_s], color=color, s=55, marker="o", zorder=5)
        ax.annotate(
            f"{offset_s:+.2f} s",
            xy=(0.0, offset_s), xytext=(6, 0),
            textcoords="offset points",
            fontsize=7, color=color, va="center",
        )

    # Calibration anchors
    if anchors:
        τ_tgt_arr = np.array([float(a["t_tgt_s"]) - origin_s for a in anchors])
        τ_ref_arr = np.array([float(a["t_ref_s"]) - origin_s for a in anchors])
        ax.scatter(
            τ_tgt_arr, τ_ref_arr,
            c="black", s=70, marker="^", zorder=6, alpha=0.85,
            label="Calibration anchors",
        )
        for i, (xt, yr) in enumerate(zip(τ_tgt_arr, τ_ref_arr)):
            ax.annotate(
                str(i), xy=(xt, yr), xytext=(4, 4),
                textcoords="offset points",
                fontsize=7, color="black", zorder=7,
            )

    ax.set_xlabel("Target time from origin (s)")
    ax.set_ylabel("Reference time from origin (s)")
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

    Output: ``synced/sync_before_after_acc.png``.
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
        output_path = synced_dir / "sync_before_after_acc.png"

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


def plot_sync_offset_drift_comparison(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Reference vs target time: all sync methods as model lines + calibration anchors.

    The coordinate origin is ``target_time_origin_seconds`` (the Arduino epoch),
    so each method line passes through ``(τ_tgt=0, offset_seconds)`` with slope
    ``(1 + drift_seconds_per_second)``.  This lets you directly read off the
    initial offset from the y-intercept and compare drift angles across methods.
    Calibration anchors (black triangles) are the ground-truth observations the
    models were fitted to.

    Output: ``synced/sync_offset_drift_comparison.png``.
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
        output_path = synced_dir / "sync_offset_drift_comparison.png"

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_sync_model_comparison(ax, sync_infos, duration_s=duration_s, origin_s=origin_s)
    ax.set_title(
        f"{recording_name} — sync models: reference vs target time",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=150)


def plot_sync_offset_drift_zoomed(
    recording_name: str,
    *,
    zoom_start_s: float | None = None,
    zoom_end_s: float | None = None,
    output_path: Path | None = None,
) -> Path:
    """Zoomed reference vs target time plot around the calibration anchor region.

    *zoom_start_s* and *zoom_end_s* are in seconds relative to
    ``target_time_origin_seconds``.  When omitted they are auto-derived from the
    anchor timestamps ± 60 s buffer.  Falls back to ``[0, 300]`` if no anchors
    are available.

    Output: ``synced/sync_offset_drift_zoomed.png``.
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

    anchors  = _collect_anchors(sync_infos)
    buffer_s = 60.0

    # Auto-derive zoom window from anchor positions when not explicitly given
    if zoom_start_s is None or zoom_end_s is None:
        if anchors:
            τ_vals = [float(a["t_tgt_s"]) - origin_s for a in anchors]
            auto_start = max(0.0, min(τ_vals) - buffer_s)
            auto_end   = max(τ_vals) + buffer_s
        else:
            auto_start, auto_end = 0.0, 300.0
        zoom_start_s = zoom_start_s if zoom_start_s is not None else auto_start
        zoom_end_s   = zoom_end_s   if zoom_end_s   is not None else auto_end

    if output_path is None:
        output_path = synced_dir / "sync_offset_drift_zoomed.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_sync_model_comparison(
        ax, sync_infos,
        duration_s=zoom_end_s + buffer_s,
        origin_s=origin_s,
        anchors=anchors,
    )
    ax.set_xlim(zoom_start_s, zoom_end_s)
    y_lo, y_hi = _sync_model_ylim(
        sync_infos, (zoom_start_s, zoom_end_s), anchors, origin_s
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        f"{recording_name} — sync models zoomed"
        f" [{zoom_start_s:.0f}–{zoom_end_s:.0f} s]",
        fontsize=11,
    )
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=150)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_sync",
        description=(
            "Sync-stage diagnostic plots.\n\n"
            "  before-after <recording>          — |acc| before/after sync\n"
            "  offset-drift <recording>          — full offset/drift model comparison\n"
            "  offset-drift-zoomed <recording>   — zoomed view around calibration anchors"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        choices=["before-after", "offset-drift", "offset-drift-zoomed"],
    )
    parser.add_argument("recording_name", help="Recording folder name")
    parser.add_argument("--zoom-start", type=float, default=None,
                        help="Zoom window start (s from target origin)")
    parser.add_argument("--zoom-end", type=float, default=None,
                        help="Zoom window end (s from target origin)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.subcommand == "before-after":
        out = plot_before_after_imu(args.recording_name)
    elif args.subcommand == "offset-drift":
        out = plot_sync_offset_drift_comparison(args.recording_name)
    else:
        out = plot_sync_offset_drift_zoomed(
            args.recording_name,
            zoom_start_s=args.zoom_start,
            zoom_end_s=args.zoom_end,
        )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
