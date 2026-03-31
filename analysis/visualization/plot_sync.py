"""Sync-stage visualizations.

Functions
---------
plot_sync_overlay
    Before-vs-after comparison: parsed (unsynced) and synced sensor signals.
plot_sync_methods_comparison
    Bar charts comparing all sync method metrics from ``all_methods.json``.
plot_sync_detail
    Offset/drift summary and calibration anchor visualization from ``sync_info.json``.
plot_sync_stage
    Master entry point — calls all of the above for a recording's synced dir.

CLI usage::

    python -m visualization.plot_sync <recording_name>
    python -m visualization.plot_sync <recording_name> --plot methods
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv, recording_stage_dir
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}
_METHOD_COLORS = {
    "calibration": "#2ca02c",
    "lida":        "#1f77b4",
    "sda":         "#ff7f0e",
    "online":      "#9467bd",
}
_METHOD_LABELS = {
    "calibration": "Calibration",
    "lida":        "SDA + LIDA",
    "sda":         "SDA only",
    "online":      "Online",
}
_ALL_METHODS = ["calibration", "lida", "sda", "online"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def _load_sensor_df(stage_dir: Path, sensor: str) -> pd.DataFrame | None:
    csv = stage_dir / f"{sensor}.csv"
    if not csv.exists():
        return None
    df = read_csv(csv)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _shared_t0(dfs: list[pd.DataFrame]) -> float:
    """Return the earliest finite timestamp across all DataFrames (ms)."""
    starts = []
    for df in dfs:
        ts = df["timestamp"].to_numpy(dtype=float)
        finite = ts[np.isfinite(ts)]
        if finite.size > 0:
            starts.append(finite[0])
    return min(starts) if starts else 0.0


def _ts_s(df: pd.DataFrame, t0: float) -> np.ndarray:
    ts = df["timestamp"].to_numpy(dtype=float)
    return (ts - t0) / 1000.0


def _plot_sensor_pair(axes, sensor_dfs: dict[str, pd.DataFrame], t0: float,
                      title: str, lw: float = 0.8) -> None:
    """Plot |acc| and |gyro| norms for each sensor onto two axes (shared t0)."""
    for sensor, df in sensor_dfs.items():
        ts = _ts_s(df, t0)
        color = _SENSOR_COLORS.get(sensor)
        acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
        gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
        if acc_cols:
            norm = strict_vector_norm(df, acc_cols)
            x, y = filter_valid_plot_xy(ts, norm)
            axes[0].plot(x, y, lw=lw, color=color, label=f"{sensor}", alpha=0.85)
        if gyro_cols:
            norm = strict_vector_norm(df, gyro_cols)
            x, y = filter_valid_plot_xy(ts, norm)
            axes[1].plot(x, y, lw=lw, color=color, label=f"{sensor}", alpha=0.85)
    axes[0].set_ylabel("|acc| (m/s²)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.2, lw=0.4)
    axes[0].set_title(title)
    axes[1].set_ylabel("|gyro| (deg/s)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.2, lw=0.4)


# ---------------------------------------------------------------------------
# Plot 1: Before vs after sync overlay
# ---------------------------------------------------------------------------

def plot_sync_overlay(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Four-panel plot: acc + gyro norms before sync (parsed) and after sync.

    Before-sync signals each start at t=0 independently, making temporal
    misalignment visible.  After-sync signals share a common t0 so aligned
    features overlap.

    Output
    ------
    ``synced/overlay_before_after.png``
    """
    parsed_dir = synced_dir.parent / "parsed"
    if output_path is None:
        output_path = synced_dir / "overlay_before_after.png"

    # Load synced data (shared timeline).
    synced_dfs = {s: df for s in _SENSORS
                  if (df := _load_sensor_df(synced_dir, s)) is not None}
    # Load parsed data (independent timelines).
    parsed_dfs = {s: df for s in _SENSORS
                  if parsed_dir.exists() and (df := _load_sensor_df(parsed_dir, s)) is not None}

    has_parsed = bool(parsed_dfs)
    n_rows = 4 if has_parsed else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=False)

    rec_name = synced_dir.parent.name

    if has_parsed:
        # Each sensor relative to its own t0 so both start at 0 — misalignment is visible.
        parsed_t0s = {s: _shared_t0([df]) for s, df in parsed_dfs.items()}
        for sensor, df in parsed_dfs.items():
            ts = _ts_s(df, parsed_t0s[sensor])
            color = _SENSOR_COLORS.get(sensor)
            acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
            gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
            if acc_cols:
                norm = strict_vector_norm(df, acc_cols)
                x, y = filter_valid_plot_xy(ts, norm)
                axes[0].plot(x, y, lw=0.7, color=color, label=sensor, alpha=0.8)
            if gyro_cols:
                norm = strict_vector_norm(df, gyro_cols)
                x, y = filter_valid_plot_xy(ts, norm)
                axes[1].plot(x, y, lw=0.7, color=color, label=sensor, alpha=0.8)
        axes[0].set_ylabel("|acc| (m/s²)")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].grid(alpha=0.2, lw=0.4)
        axes[0].set_title(f"{rec_name} — BEFORE sync (per-sensor relative time)")
        axes[1].set_ylabel("|gyro| (deg/s)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(alpha=0.2, lw=0.4)
        axes[1].set_xlabel("Time (s)")
        sync_axes = axes[2:4]
    else:
        sync_axes = axes[0:2]

    # Synced: single shared t0.
    if synced_dfs:
        t0 = _shared_t0(list(synced_dfs.values()))
        _plot_sensor_pair(
            sync_axes, synced_dfs, t0,
            title=f"{rec_name} — AFTER sync (shared timeline)",
        )
        sync_axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Plot 2: Method comparison
# ---------------------------------------------------------------------------

def plot_sync_methods_comparison(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """Bar charts comparing all sync methods from ``all_methods.json``.

    Shows correlation, drift, and calibration anchor scores.  The selected
    method is highlighted with a star annotation.

    Output
    ------
    ``synced/methods_comparison.png``
    """
    data = _load_json(synced_dir / "all_methods.json")
    if data is None:
        log.info("No all_methods.json in %s — skipping method comparison plot", synced_dir)
        return None

    if output_path is None:
        output_path = synced_dir / "methods_comparison.png"

    selected = data.get("selected_method", "")
    methods = [m for m in _ALL_METHODS if data.get(m) is not None]
    if not methods:
        return None

    n_methods = len(methods)
    bar_x = np.arange(n_methods)
    bar_w = 0.6
    method_labels = [_METHOD_LABELS.get(m, m) for m in methods]
    colors = [_METHOD_COLORS.get(m, "gray") for m in methods]
    edge_colors = ["black" if m == selected else "none" for m in methods]
    lw_edges = [2.0 if m == selected else 0.0 for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"{synced_dir.parent.name} — sync method comparison  (★ = selected: {_METHOD_LABELS.get(selected, selected)})",
        fontsize=12,
    )

    def _annotate_selected(ax, x_vals, y_vals):
        for xi, yi, m in zip(x_vals, y_vals, methods):
            if m == selected and np.isfinite(yi):
                ax.text(xi, yi, "★", ha="center", va="bottom", fontsize=14,
                        color="black", fontweight="bold")

    # --- Correlation (offset + drift) ---
    corr_vals = [data[m].get("corr_offset_and_drift") or float("nan") for m in methods]
    ax = axes[0][0]
    bars = ax.bar(bar_x, corr_vals, color=colors, edgecolor=edge_colors,
                  linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axhline(0.2, color="orange", lw=1.0, ls=":", label="min threshold (0.2)")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("Correlation")
    ax.set_title("Cross-correlation (offset + drift) — higher is better")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _annotate_selected(ax, bar_x, corr_vals)
    for xi, v in zip(bar_x, corr_vals):
        if np.isfinite(v):
            ax.text(xi, max(v, 0) + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # --- Drift (ppm) ---
    drift_vals = [abs(data[m].get("drift_ppm") or 0.0) for m in methods]
    ax = axes[0][1]
    ax.bar(bar_x, drift_vals, color=colors, edgecolor=edge_colors,
           linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(400, color="orange", lw=1.0, ls=":", label="typical Arduino drift (~400 ppm)")
    ax.axhline(5000, color="red", lw=1.0, ls=":", label="poor threshold (5000 ppm)")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("|drift| (ppm)")
    ax.set_title("Absolute drift — lower is better")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _annotate_selected(ax, bar_x, drift_vals)
    for xi, v in zip(bar_x, drift_vals):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # --- Calibration scores ---
    ax = axes[1][0]
    cal = data.get("calibration") or {}
    if cal.get("available"):
        open_score = cal.get("calibration_open_score") or float("nan")
        close_score = cal.get("calibration_close_score") or float("nan")
        span_s = cal.get("calibration_span_s") or float("nan")
        bx = np.array([0, 1])
        scores = [open_score, close_score]
        score_colors = [
            "#2ca02c" if (s or 0) >= 0.5 else "#d62728" for s in scores
        ]
        ax.bar(bx, scores, color=score_colors, width=0.4, alpha=0.85)
        ax.axhline(0.5, color="orange", lw=1.0, ls="--", label="min threshold (0.5)")
        ax.set_xticks(bx)
        ax.set_xticklabels(["Opening anchor", "Closing anchor"])
        ax.set_ylabel("Calibration anchor score")
        ax.set_title(f"Calibration method — anchor scores  (span: {span_s:.1f} s)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(bx, scores):
            if np.isfinite(v):
                ax.text(xi, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Calibration method\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")

    # --- Summary table ---
    ax = axes[1][1]
    ax.axis("off")
    col_headers = ["Method", "Corr", "Drift (ppm)", "Avail.", "Selected"]
    rows = []
    for m in methods:
        md = data[m]
        corr = md.get("corr_offset_and_drift")
        drift = md.get("drift_ppm")
        avail = "✓" if md.get("available") else "✗"
        sel = "★" if m == selected else ""
        rows.append([
            _METHOD_LABELS.get(m, m),
            f"{corr:.4f}" if corr is not None else "N/A",
            f"{drift:.1f}" if drift is not None else "N/A",
            avail,
            sel,
        ])
    table = ax.table(cellText=rows, colLabels=col_headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
        elif rows[row - 1][4] == "★":
            cell.set_facecolor("#d4edda")
    ax.set_title("Method summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Plot 3: Sync detail (selected method)
# ---------------------------------------------------------------------------

def plot_sync_detail(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """Summary panel for the selected sync method from ``sync_info.json``.

    Shows offset, drift, correlation scores, and calibration anchor details
    (when the calibration method was selected).

    Output
    ------
    ``synced/sync_detail.png``
    """
    info = _load_json(synced_dir / "sync_info.json")
    if info is None:
        log.info("No sync_info.json in %s — skipping detail plot", synced_dir)
        return None

    if output_path is None:
        output_path = synced_dir / "sync_detail.png"

    method = info.get("sync_method", "?")
    offset_s = info.get("offset_seconds")
    drift_sps = info.get("drift_seconds_per_second")
    drift_ppm = (drift_sps * 1e6) if drift_sps is not None else None
    corr = info.get("correlation") or {}
    cal_block = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    rec_name = synced_dir.parent.name

    has_cal = cal_block is not None
    fig_h = 8 if has_cal else 4
    fig, axes = plt.subplots(2 if has_cal else 1, 2,
                             figsize=(14, fig_h), squeeze=False)
    fig.suptitle(f"{rec_name} — sync detail  ({method})", fontsize=12)

    # --- Row 0, left: key metrics as horizontal bar (gauge-style) ---
    ax = axes[0][0]
    ax.axis("off")
    lines = [
        f"Method:             {method}",
        f"Offset:             {offset_s:.6f} s" if offset_s is not None else "Offset:             N/A",
        f"Drift:              {drift_ppm:.2f} ppm" if drift_ppm is not None else "Drift:              N/A",
        f"Drift source:       {info.get('drift_source', 'N/A')}",
        f"Corr (offset only): {corr.get('offset_only', float('nan')):.4f}",
        f"Corr (offset+drift):{corr.get('offset_and_drift', float('nan')):.4f}",
        f"Signal:             {corr.get('signal', 'N/A')}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Sync parameters")

    # --- Row 0, right: correlation scores as gauge bars ---
    ax = axes[0][1]
    corr_metrics = {
        "offset only": corr.get("offset_only"),
        "offset + drift": corr.get("offset_and_drift"),
    }
    valid = {k: v for k, v in corr_metrics.items() if v is not None}
    bx = np.arange(len(valid))
    bar_colors = ["#2ca02c" if v >= 0.2 else "#d62728" for v in valid.values()]
    bars = ax.barh(bx, list(valid.values()), color=bar_colors, alpha=0.85)
    ax.axvline(0.2, color="orange", lw=1.0, ls="--", label="min threshold (0.2)")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_yticks(bx)
    ax.set_yticklabels(list(valid.keys()))
    ax.set_xlabel("Pearson r")
    ax.set_title("Cross-correlation quality")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, valid.values()):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    # --- Row 1: calibration anchors (only if calibration method) ---
    if has_cal:
        opening = cal_block.get("opening") or {}
        closing = cal_block.get("closing") or {}
        span_s = cal_block.get("calibration_span_s")

        # Left: anchor scores
        ax = axes[1][0]
        anchor_names = ["Opening", "Closing"]
        anchor_scores = [opening.get("score"), closing.get("score")]
        anchor_colors = [
            "#2ca02c" if (s or 0) >= 0.5 else "#d62728"
            for s in anchor_scores
        ]
        bx2 = np.arange(2)
        bars2 = ax.bar(bx2, [s or 0 for s in anchor_scores],
                       color=anchor_colors, width=0.4, alpha=0.85)
        ax.axhline(0.5, color="orange", lw=1.0, ls="--", label="min threshold (0.5)")
        ax.set_xticks(bx2)
        ax.set_xticklabels(anchor_names)
        ax.set_ylabel("Anchor score")
        ax.set_title(f"Calibration anchors  (span: {span_s:.1f} s)" if span_s else "Calibration anchors")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars2, anchor_scores):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        # Right: anchor timeline
        ax = axes[1][1]
        ax.axis("off")
        o_t = opening.get("t_tgt_s")
        c_t = closing.get("t_tgt_s")
        o_w = opening.get("window_duration_s", 0)
        c_w = closing.get("window_duration_s", 0)
        o_off = opening.get("offset_s")
        c_off = closing.get("offset_s")
        anchor_lines = [
            "Anchor details:",
            "",
            f"  Opening:",
            f"    t_target = {o_t:.3f} s" if o_t is not None else "    t_target = N/A",
            f"    window   = {o_w:.2f} s",
            f"    offset   = {o_off:.6f} s" if o_off is not None else "    offset   = N/A",
            f"    score    = {opening.get('score'):.4f}" if opening.get("score") is not None else "    score    = N/A",
            "",
            f"  Closing:",
            f"    t_target = {c_t:.3f} s" if c_t is not None else "    t_target = N/A",
            f"    window   = {c_w:.2f} s",
            f"    offset   = {c_off:.6f} s" if c_off is not None else "    offset   = N/A",
            f"    score    = {closing.get('score'):.4f}" if closing.get("score") is not None else "    score    = N/A",
            "",
            f"  Span: {span_s:.1f} s" if span_s is not None else "  Span: N/A",
        ]
        ax.text(0.05, 0.95, "\n".join(anchor_lines), transform=ax.transAxes,
                fontsize=10, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        ax.set_title("Calibration anchor details")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def plot_sync_stage(
    synced_dir: Path,
) -> list[Path]:
    """Generate all sync plots for a recording's synced directory.

    Produces:
    - ``overlay_before_after.png``   — before/after sync signal comparison
    - ``methods_comparison.png``     — method metrics bar charts
    - ``sync_detail.png``            — selected method parameters
    - ``comparison.png``             — sensor overlay (via plot_comparison)
    - ``sporsa.png`` / ``arduino.png`` — individual sensor signals
    """
    out_paths: list[Path] = []

    try:
        p = plot_sync_overlay(synced_dir)
        out_paths.append(p)
    except Exception as exc:
        log.warning("sync overlay plot failed for %s: %s", synced_dir.name, exc)

    try:
        p = plot_sync_methods_comparison(synced_dir)
        if p:
            out_paths.append(p)
    except Exception as exc:
        log.warning("sync methods comparison plot failed for %s: %s", synced_dir.name, exc)

    try:
        p = plot_sync_detail(synced_dir)
        if p:
            out_paths.append(p)
    except Exception as exc:
        log.warning("sync detail plot failed for %s: %s", synced_dir.name, exc)

    # Existing per-sensor + overlay comparison.
    try:
        from visualization.plot_comparison import plot_stage_data
        out_paths += plot_stage_data(synced_dir)
    except Exception as exc:
        log.warning("sync comparison plot failed for %s: %s", synced_dir.name, exc)

    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_sync",
        description="Generate sync plots for a recording.",
    )
    parser.add_argument(
        "recording_name",
        help="Recording name (e.g. 2026-02-26_r3) or path to synced/ directory.",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "overlay", "methods", "detail"],
        default="all",
        help="Which plot(s) to generate (default: all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve synced dir from recording name or direct path.
    p = Path(args.recording_name)
    if p.is_dir() and p.name == "synced":
        synced_dir = p
    elif p.is_dir():
        synced_dir = p / "synced"
    else:
        synced_dir = recording_stage_dir(args.recording_name, "synced")

    if not synced_dir.is_dir():
        log.error("Synced directory not found: %s", synced_dir)
        return

    dispatch = {
        "overlay": lambda: [plot_sync_overlay(synced_dir)],
        "methods": lambda: [plot_sync_methods_comparison(synced_dir)],
        "detail":  lambda: [plot_sync_detail(synced_dir)],
        "all":     lambda: plot_sync_stage(synced_dir),
    }

    paths = dispatch[args.plot]()
    paths = [p for p in (paths or []) if p is not None]
    if not paths:
        print("No plots generated.")
    for p in paths:
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
