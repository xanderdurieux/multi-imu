"""Sync-stage visualizations — diagnostic plots for per-recording synchronization.

Functions
---------
plot_sync_overlay
    Before-vs-after comparison: parsed (unsynced) and synced sensor signals.
plot_sync_methods_comparison
    Bar charts comparing all sync method metrics from ``all_methods.json``.
plot_sync_detail
    Offset/drift summary and calibration anchor visualization from ``sync_info.json``.
plot_sync_model_dashboard
    Grid of key model parameters, correlation gauges, and anchor detail.
plot_sync_zoom_alignment
    Multi-panel zoomed overlay at opening, mid-recording, and closing.
plot_sync_clock_correction
    Clock correction curve (delta_ms) and drift-only term vs Arduino time.
plot_sync_roll_corr_global
    Rolling Pearson *r* between SPORSA and synced Arduino |acc| over time.
plot_sync_scatter_zoom
    |acc| scatter in a mid-recording window for aligned and unaligned signals.
plot_sync_anchor_offsets
    Per-anchor offset vs target time, sized/colored by score, with drift fit.
plot_sync_difference_profile
    Resampled |acc|_ref - |acc|_tgt residual over the full session.
plot_sync_stage
    Master entry point — calls all of the above for a recording's synced dir.

CLI usage::

    python -m visualization.plot_sync <recording_name>
    python -m visualization.plot_sync <recording_name> --plot methods
    python -m visualization.plot_sync <recording_name> --plot dashboard
    python -m visualization.plot_sync <recording_name> --plot all
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
from scipy.signal import correlate

from common.paths import project_relative_path, read_csv, recording_stage_dir
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}
_METHOD_COLORS = {
    "multi_anchor":          "#2ca02c",
    "one_anchor_adaptive":   "#8c564b",
    "one_anchor_prior":      "#9467bd",
    "signal_only":           "#1f77b4",
}
_METHOD_LABELS = {
    "multi_anchor":          "Multi-anchor protocol",
    "one_anchor_adaptive":   "One-anchor adaptive",
    "one_anchor_prior":      "One-anchor prior",
    "signal_only":           "Signal-only",
}
_ALL_METHODS = ["multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"]
_TARGET_HZ = 20.0


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
    starts = []
    for df in dfs:
        ts = df["timestamp"].to_numpy(dtype=float)
        finite = ts[np.isfinite(ts)]
        if finite.size > 0:
            starts.append(finite[0])
    return min(starts) if starts else 0.0


def _ts_s(df: pd.DataFrame, t0: float) -> np.ndarray:
    return (df["timestamp"].to_numpy(dtype=float) - t0) / 1000.0


def _acc_norm(df: pd.DataFrame) -> np.ndarray | None:
    cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not cols:
        return None
    return strict_vector_norm(df, cols)


def _save(fig: plt.Figure, path: Path, dpi: int = 120) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: %s", project_relative_path(path))
    return path


def _resample_acc_norm_pair(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    hz: float = _TARGET_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Resample both sensors' |acc| to a common uniform grid.

    Returns ``(time_s, ref_norm, tgt_norm)`` or *None* if insufficient data.
    """
    ref_norm = _acc_norm(ref_df)
    tgt_norm = _acc_norm(tgt_df)
    if ref_norm is None or tgt_norm is None:
        return None

    ref_ts = ref_df["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_df["timestamp"].to_numpy(dtype=float)

    t_start = max(np.nanmin(ref_ts), np.nanmin(tgt_ts))
    t_end = min(np.nanmax(ref_ts), np.nanmax(tgt_ts))
    if t_end <= t_start:
        return None

    step_ms = 1000.0 / hz
    grid = np.arange(t_start, t_end, step_ms)
    if grid.size < 10:
        return None

    ref_interp = np.interp(grid, ref_ts, ref_norm)
    tgt_interp = np.interp(grid, tgt_ts, tgt_norm)
    time_s = (grid - grid[0]) / 1000.0
    return time_s, ref_interp, tgt_interp


def _rolling_pearson(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson *r* using a centered window. Returns NaN at edges."""
    n = len(a)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        x = a[i - half : i + half]
        y = b[i - half : i + half]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out


def _plot_sensor_pair(axes, sensor_dfs: dict[str, pd.DataFrame], t0: float,
                      title: str, lw: float = 0.8) -> None:
    for sensor, df in sensor_dfs.items():
        ts = _ts_s(df, t0)
        color = _SENSOR_COLORS.get(sensor)
        acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
        gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
        if acc_cols:
            norm = strict_vector_norm(df, acc_cols)
            x, y = filter_valid_plot_xy(ts, norm)
            axes[0].plot(x, y, lw=lw, color=color, label=sensor, alpha=0.85)
        if gyro_cols:
            norm = strict_vector_norm(df, gyro_cols)
            x, y = filter_valid_plot_xy(ts, norm)
            axes[1].plot(x, y, lw=lw, color=color, label=sensor, alpha=0.85)
    axes[0].set_ylabel("|acc| (m/s\u00b2)")
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
    """``synced/overlay_before_after.png``"""
    parsed_dir = synced_dir.parent / "parsed"
    if output_path is None:
        output_path = synced_dir / "overlay_before_after.png"

    synced_dfs = {s: df for s in _SENSORS
                  if (df := _load_sensor_df(synced_dir, s)) is not None}
    parsed_dfs = {s: df for s in _SENSORS
                  if parsed_dir.exists() and (df := _load_sensor_df(parsed_dir, s)) is not None}

    has_parsed = bool(parsed_dfs)
    n_rows = 4 if has_parsed else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=False)
    rec_name = synced_dir.parent.name

    if has_parsed:
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
        axes[0].set_ylabel("|acc| (m/s\u00b2)")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].grid(alpha=0.2, lw=0.4)
        axes[0].set_title(f"{rec_name} \u2014 BEFORE sync (per-sensor relative time)")
        axes[1].set_ylabel("|gyro| (deg/s)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(alpha=0.2, lw=0.4)
        axes[1].set_xlabel("Time (s)")
        sync_axes = axes[2:4]
    else:
        sync_axes = axes[0:2]

    if synced_dfs:
        t0 = _shared_t0(list(synced_dfs.values()))
        _plot_sensor_pair(sync_axes, synced_dfs, t0,
                          title=f"{rec_name} \u2014 AFTER sync (shared timeline)")
        sync_axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 2: Method comparison
# ---------------------------------------------------------------------------

def plot_sync_methods_comparison(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/methods_comparison.png``"""
    data = _load_json(synced_dir / "all_methods.json")
    if data is None:
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
        f"{synced_dir.parent.name} \u2014 sync method comparison  "
        f"(\u2605 = selected: {_METHOD_LABELS.get(selected, selected)})",
        fontsize=12,
    )

    def _annotate(ax, xs, ys, fmt=".4f", star=True):
        ymax = max((abs(float(v)) for v in ys if np.isfinite(v)), default=1.0)
        for xi, yi, m in zip(xs, ys, methods):
            if not np.isfinite(yi):
                continue
            ax.text(xi, yi + ymax * 0.03, format(yi, fmt),
                    ha="center", va="bottom", fontsize=8)
            if star and m == selected:
                ax.text(xi, yi + ymax * 0.10, "\u2605", ha="center",
                        va="bottom", fontsize=14, fontweight="bold")

    # Correlation (offset + drift)
    corr_vals = np.array([data[m].get("corr_offset_and_drift") or np.nan for m in methods])
    ax = axes[0][0]
    ax.bar(bar_x, corr_vals, color=colors, edgecolor=edge_colors,
           linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axhline(0.2, color="orange", lw=1.0, ls=":", label="min threshold (0.2)")
    ax.set_xticks(bar_x); ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("Correlation"); ax.set_title("Cross-corr (offset + drift)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _annotate(ax, bar_x, corr_vals)

    # |drift| (ppm)
    drift_vals = np.array([abs(data[m].get("drift_ppm") or 0.0) for m in methods])
    ax = axes[0][1]
    ax.bar(bar_x, drift_vals, color=colors, edgecolor=edge_colors,
           linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(400, color="orange", lw=1.0, ls=":", label="typical Arduino (~400 ppm)")
    ax.axhline(5000, color="red", lw=1.0, ls=":", label="implausible (5000 ppm)")
    ax.set_xticks(bar_x); ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("|drift| (ppm)"); ax.set_title("Absolute drift")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _annotate(ax, bar_x, drift_vals, fmt=".1f")

    # Heatmap of normalized metrics
    ax = axes[1][0]
    metric_keys = ["corr_offset_and_drift", "drift_ppm", "calibration_fit_r2",
                   "calibration_n_windows"]
    metric_labels = ["corr", "|drift| ppm", "fit R\u00b2", "n anchors"]
    mat = np.full((len(metric_keys), n_methods), np.nan)
    for j, m in enumerate(methods):
        md = data[m]
        mat[0, j] = md.get("corr_offset_and_drift") or np.nan
        mat[1, j] = abs(md.get("drift_ppm") or np.nan)
        mat[2, j] = md.get("calibration_fit_r2") or np.nan
        mat[3, j] = md.get("calibration_n_windows") or np.nan
    # Row-normalize to [0, 1] for display
    mat_norm = np.full_like(mat, np.nan)
    for row_i in range(mat.shape[0]):
        row = mat[row_i]
        finite = row[np.isfinite(row)]
        if finite.size > 0:
            lo, hi = finite.min(), finite.max()
            span = hi - lo if hi > lo else 1.0
            mat_norm[row_i] = (row - lo) / span
    im = ax.imshow(mat_norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(method_labels, rotation=15, ha="right", fontsize=8)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=9)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if np.isfinite(v):
                fmt = ".0f" if ri in (1, 3) else ".3f"
                ax.text(ci, ri, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="white" if (mat_norm[ri, ci] or 0) > 0.5 else "black")
    ax.set_title("Normalized metrics heatmap")

    # Summary table
    ax = axes[1][1]
    ax.axis("off")
    col_headers = ["Method", "Corr", "Drift (ppm)", "Avail.", "Sel."]
    rows = []
    for m in methods:
        md = data[m]
        corr = md.get("corr_offset_and_drift")
        drift = md.get("drift_ppm")
        avail = "\u2713" if md.get("available") else "\u2717"
        sel = "\u2605" if m == selected else ""
        rows.append([
            _METHOD_LABELS.get(m, m),
            f"{corr:.4f}" if corr is not None else "N/A",
            f"{drift:.1f}" if drift is not None else "N/A",
            avail, sel,
        ])
    table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
        elif rows[row - 1][4] == "\u2605":
            cell.set_facecolor("#d4edda")
    ax.set_title("Method summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 3: Sync detail (selected method)
# ---------------------------------------------------------------------------

def plot_sync_detail(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/sync_detail.png``"""
    info = _load_json(synced_dir / "sync_info.json")
    if info is None:
        return None
    if output_path is None:
        output_path = synced_dir / "sync_detail.png"

    method = info.get("sync_method", "?")
    offset_s = info.get("offset_seconds")
    drift_sps = info.get("drift_seconds_per_second")
    drift_ppm = (drift_sps * 1e6) if drift_sps is not None else None
    corr = info.get("correlation") or {}
    cal_block = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    adap_block = info.get("adaptive") if isinstance(info.get("adaptive"), dict) else None
    rec_name = synced_dir.parent.name

    has_cal = cal_block is not None
    has_adap = adap_block is not None
    n_rows = 1 + int(has_cal) + int(has_adap)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)
    fig.suptitle(f"{rec_name} \u2014 sync detail  ({method})", fontsize=12)

    # Row 0 left: key metrics
    ax = axes[0][0]
    ax.axis("off")
    lines = [
        f"Method:             {method}",
        f"Offset:             {offset_s:.6f} s" if offset_s is not None else "Offset:             N/A",
        f"Drift:              {drift_ppm:.2f} ppm" if drift_ppm is not None else "Drift:              N/A",
        f"Drift source:       {info.get('drift_source', 'N/A')}",
        f"Signal mode:        {info.get('signal_mode', 'N/A')}",
        f"Corr (offset only): {corr.get('offset_only', float('nan')):.4f}",
        f"Corr (offset+drift):{corr.get('offset_and_drift', float('nan')):.4f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Sync parameters")

    # Row 0 right: correlation gauges
    ax = axes[0][1]
    valid = {k: v for k, v in {
        "offset only": corr.get("offset_only"),
        "offset + drift": corr.get("offset_and_drift"),
    }.items() if v is not None}
    bx = np.arange(len(valid))
    bar_colors = ["#2ca02c" if v >= 0.2 else "#d62728" for v in valid.values()]
    bars = ax.barh(bx, list(valid.values()), color=bar_colors, alpha=0.85)
    ax.axvline(0.2, color="orange", lw=1.0, ls="--", label="min threshold")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_yticks(bx); ax.set_yticklabels(list(valid.keys()))
    ax.set_xlabel("Pearson r"); ax.set_title("Cross-correlation quality")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, valid.values()):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    # Calibration row
    if has_cal:
        anchors = cal_block.get("anchors")
        anchors = anchors if isinstance(anchors, list) and anchors else []
        opening = cal_block.get("opening") or {}
        closing = cal_block.get("closing") or {}
        span_s = cal_block.get("anchor_span_s") or cal_block.get("calibration_span_s")
        n_windows = cal_block.get("n_anchors") or cal_block.get("n_windows_used")
        fit_r2 = cal_block.get("fit_r2")

        ax = axes[1][0]
        if anchors:
            names = [f"A{i+1}" for i in range(len(anchors))]
            scores = [a.get("score") or 0 for a in anchors]
        else:
            names = ["Opening", "Closing"]
            scores = [opening.get("score") or 0, closing.get("score") or 0]
        cols = ["#2ca02c" if s >= 0.5 else "#d62728" for s in scores]
        ax.bar(range(len(scores)), scores, color=cols, width=0.5, alpha=0.85)
        ax.axhline(0.5, color="orange", lw=1.0, ls="--", label="min (0.5)")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Score")
        title = "Calibration anchors"
        extras = []
        if span_s is not None:
            extras.append(f"span={span_s:.1f}s")
        if n_windows is not None:
            extras.append(f"n={n_windows}")
        if fit_r2 is not None:
            extras.append(f"R\u00b2={fit_r2:.3f}")
        if extras:
            title += f"  ({', '.join(extras)})"
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(scores):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax = axes[1][1]
        ax.axis("off")
        if anchors:
            detail_lines = ["Anchor details:"]
            for idx, a in enumerate(anchors[:8], 1):
                detail_lines.append(
                    f"  A{idx}: t={a.get('t_tgt_s', 0):.1f}s  "
                    f"off={a.get('offset_s', 0):.6f}s  "
                    f"score={a.get('score', 0):.3f}"
                )
            if len(anchors) > 8:
                detail_lines.append(f"  ... {len(anchors)-8} more ...")
        else:
            detail_lines = [
                f"Opening: t={opening.get('t_tgt_s', 0):.1f}s  off={opening.get('offset_s', 0):.6f}s",
                f"Closing: t={closing.get('t_tgt_s', 0):.1f}s  off={closing.get('offset_s', 0):.6f}s",
            ]
        ax.text(0.05, 0.95, "\n".join(detail_lines), transform=ax.transAxes,
                fontsize=10, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        ax.set_title("Anchor details")

    # Adaptive row
    if has_adap:
        adap_row = 1 + int(has_cal)
        n_wins = adap_block.get("n_windows_used")
        fit_r2 = adap_block.get("fit_r2")
        init_meth = adap_block.get("initial_method", "N/A")

        ax = axes[adap_row][0]
        r2_val = fit_r2 if fit_r2 is not None else 0.0
        bar_color = "#2ca02c" if r2_val >= 0.1 else "#d62728"
        ax.bar([0], [r2_val], color=bar_color, width=0.4, alpha=0.85)
        ax.axhline(0.1, color="orange", lw=1.0, ls="--", label="min R\u00b2")
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, max(1.1, r2_val * 1.2))
        ax.set_xticks([0]); ax.set_xticklabels(["Drift fit R\u00b2"]); ax.set_ylabel("R\u00b2")
        ax.set_title(f"Adaptive \u2014 {n_wins or '?'} windows (init: {init_meth})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        if fit_r2 is not None:
            ax.text(0, r2_val + 0.02, f"{r2_val:.4f}", ha="center", va="bottom", fontsize=9)

        ax = axes[adap_row][1]
        ax.axis("off")
        anchor = adap_block.get("opening_anchor") or {}
        adap_lines = [
            f"Init method:  {init_meth}",
            f"Init offset:  {adap_block.get('initial_offset_s', 0):.6f} s",
            f"Opening: t={anchor.get('t_tgt_s', 0):.1f}s  score={anchor.get('score', 0):.3f}",
            f"Windows:      {n_wins}",
            f"Fit R\u00b2:       {fit_r2:.4f}" if fit_r2 is not None else "Fit R\u00b2: N/A",
        ]
        ax.text(0.05, 0.95, "\n".join(adap_lines), transform=ax.transAxes,
                fontsize=10, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        ax.set_title("Adaptive details")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 4: Model dashboard
# ---------------------------------------------------------------------------

def plot_sync_model_dashboard(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """Grid: offset, drift ppm, correlations, anchor summary. ``synced/model_dashboard.png``"""
    info = _load_json(synced_dir / "sync_info.json")
    allm = _load_json(synced_dir / "all_methods.json")
    if info is None:
        return None
    if output_path is None:
        output_path = synced_dir / "model_dashboard.png"

    method = info.get("sync_method", "?")
    offset_s = info.get("offset_seconds")
    drift_sps = info.get("drift_seconds_per_second")
    drift_ppm = (drift_sps * 1e6) if drift_sps is not None else None
    corr = info.get("correlation") or {}
    cal = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    rec_name = synced_dir.parent.name

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"{rec_name} \u2014 sync model dashboard", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # (0,0) Key metrics text
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        f"Method:  {_METHOD_LABELS.get(method, method)}",
        f"Offset:  {offset_s:.6f} s" if offset_s is not None else "Offset:  N/A",
        f"Drift:   {drift_ppm:.2f} ppm" if drift_ppm is not None else "Drift:   N/A",
        f"Source:  {info.get('drift_source', 'N/A')}",
        f"Signal:  {info.get('signal_mode', 'N/A')}",
    ]
    ax.text(0.05, 0.92, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef6ff", alpha=0.9))
    ax.set_title("Selected model", fontsize=10)

    # (0,1) Correlation bar
    ax = fig.add_subplot(gs[0, 1])
    labels_corr = ["offset only", "offset+drift"]
    vals_corr = [corr.get("offset_only") or 0, corr.get("offset_and_drift") or 0]
    cols_corr = ["#2ca02c" if v >= 0.2 else "#d62728" for v in vals_corr]
    bars = ax.barh(range(2), vals_corr, color=cols_corr, alpha=0.85)
    ax.axvline(0.2, color="orange", lw=1.0, ls="--")
    ax.set_yticks(range(2)); ax.set_yticklabels(labels_corr)
    ax.set_xlabel("Pearson r"); ax.set_title("Correlation quality", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, vals_corr):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)

    # (0,2) Drift gauge — all methods
    ax = fig.add_subplot(gs[0, 2])
    if allm:
        avail = [m for m in _ALL_METHODS if allm.get(m)]
        d_vals = [abs(allm[m].get("drift_ppm") or 0) for m in avail]
        cols = [_METHOD_COLORS.get(m, "gray") for m in avail]
        ax.barh(range(len(avail)), d_vals, color=cols, alpha=0.85)
        ax.axvline(400, color="orange", lw=1.0, ls=":", label="~400 ppm")
        ax.set_yticks(range(len(avail)))
        ax.set_yticklabels([_METHOD_LABELS.get(m, m) for m in avail], fontsize=8)
        for i, v in enumerate(d_vals):
            ax.text(v + 10, i, f"{v:.0f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=12, color="gray")
    ax.set_xlabel("|drift| (ppm)"); ax.set_title("Drift comparison", fontsize=10)
    ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=7, loc="lower right")

    # (1,0:2) Anchor bars (scores + offset timeline)
    anchors = []
    if cal:
        anchors = cal.get("anchors") or []
    if anchors:
        ax = fig.add_subplot(gs[1, 0:2])
        n_a = len(anchors)
        scores = [a.get("score", 0) for a in anchors]
        t_vals = [a.get("t_tgt_s", 0) for a in anchors]
        off_vals = [a.get("offset_s", 0) for a in anchors]
        x_a = np.arange(n_a)
        cols_a = ["#2ca02c" if s >= 0.5 else "#d62728" for s in scores]
        ax.bar(x_a, scores, color=cols_a, width=0.55, alpha=0.85)
        ax.axhline(0.5, color="orange", lw=1.0, ls="--")
        for i, s in enumerate(scores):
            ax.text(i, s + 0.015, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(x_a, t_vals, "o-", color="#1f77b4", ms=5, lw=1.2, label="t_target (s)")
        ax2.set_ylabel("t_target (s)", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_xticks(x_a); ax.set_xticklabels([f"A{i+1}" for i in x_a])
        ax.set_ylabel("Score"); ax.set_title("Calibration anchors", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # (1,2) Anchor offset scatter
        ax3 = fig.add_subplot(gs[1, 2])
        sizes = np.array(scores) * 80 + 20
        sc = ax3.scatter(t_vals, off_vals, s=sizes, c=scores, cmap="RdYlGn",
                         vmin=0, vmax=1, edgecolors="black", linewidths=0.5)
        if n_a >= 2:
            z = np.polyfit(t_vals, off_vals, 1)
            fit_line = np.poly1d(z)
            t_range = np.linspace(min(t_vals), max(t_vals), 50)
            ax3.plot(t_range, fit_line(t_range), "--", color="gray", lw=1.0,
                     label=f"drift fit ({z[0]*1e6:.0f} ppm)")
            ax3.legend(fontsize=7)
        plt.colorbar(sc, ax=ax3, label="score", shrink=0.8)
        ax3.set_xlabel("t_target (s)"); ax3.set_ylabel("offset (s)")
        ax3.set_title("Anchor offset vs time", fontsize=10); ax3.grid(alpha=0.3)
    else:
        ax = fig.add_subplot(gs[1, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "No calibration anchors available",
                ha="center", va="center", fontsize=12, color="gray", transform=ax.transAxes)

    # (2,0:3) Per-method correlation comparison
    ax = fig.add_subplot(gs[2, :])
    if allm:
        avail = [m for m in _ALL_METHODS if allm.get(m)]
        n_m = len(avail)
        x_m = np.arange(n_m)
        corr_vals = [allm[m].get("corr_offset_and_drift") or 0 for m in avail]
        cols_m = [_METHOD_COLORS.get(m, "gray") for m in avail]
        edges = ["black" if m == (allm.get("selected_method") or "") else "none" for m in avail]
        lws = [2.0 if m == (allm.get("selected_method") or "") else 0.0 for m in avail]
        ax.bar(x_m, corr_vals, color=cols_m, edgecolor=edges, linewidth=lws,
               width=0.6, alpha=0.85)
        ax.axhline(0.2, color="orange", lw=1.0, ls=":")
        ax.set_xticks(x_m)
        ax.set_xticklabels([_METHOD_LABELS.get(m, m) for m in avail], fontsize=9)
        for i, v in enumerate(corr_vals):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
            if avail[i] == allm.get("selected_method"):
                ax.text(i, v + 0.05, "\u2605", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("Corr (offset+drift)"); ax.set_title("All methods \u2014 correlation", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 5: Zoom alignment (3-panel)
# ---------------------------------------------------------------------------

def plot_sync_zoom_alignment(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/zoom_alignment.png`` — 3 windows: opening, mid, late."""
    parsed_dir = synced_dir.parent / "parsed"
    if output_path is None:
        output_path = synced_dir / "zoom_alignment.png"

    ref_s = _load_sensor_df(synced_dir, "sporsa")
    tgt_s = _load_sensor_df(synced_dir, "arduino")
    ref_p = _load_sensor_df(parsed_dir, "sporsa") if parsed_dir.exists() else None
    tgt_p = _load_sensor_df(parsed_dir, "arduino") if parsed_dir.exists() else None

    if any(x is None for x in (ref_s, tgt_s)):
        return None

    ref_norm_s = _acc_norm(ref_s)
    tgt_norm_s = _acc_norm(tgt_s)
    if ref_norm_s is None or tgt_norm_s is None:
        return None

    t0 = _shared_t0([ref_s, tgt_s])
    ref_ts = _ts_s(ref_s, t0)
    tgt_ts = _ts_s(tgt_s, t0)
    duration_s = max(ref_ts[-1], tgt_ts[-1])
    window_s = 45.0

    windows = [
        ("Opening (~0\u201345 s)", 0.0),
        ("Mid-recording", max(0.0, duration_s / 2 - window_s / 2)),
        ("Late recording", max(0.0, duration_s - window_s - 5)),
    ]

    fig, axes = plt.subplots(len(windows), 1, figsize=(16, 3.5 * len(windows)))
    if len(windows) == 1:
        axes = [axes]
    fig.suptitle(f"{synced_dir.parent.name} \u2014 zoom alignment", fontsize=12)

    for ax, (label, start) in zip(axes, windows):
        end = start + window_s
        # Synced
        m_r = (ref_ts >= start) & (ref_ts <= end)
        m_t = (tgt_ts >= start) & (tgt_ts <= end)
        if m_r.sum() > 5:
            ax.plot(ref_ts[m_r], ref_norm_s[m_r], lw=0.9, color=_SENSOR_COLORS["sporsa"],
                    label="SPORSA (synced)", alpha=0.9)
        if m_t.sum() > 5:
            ax.plot(tgt_ts[m_t], tgt_norm_s[m_t], lw=0.9, color=_SENSOR_COLORS["arduino"],
                    label="Arduino (synced)", alpha=0.9)

        # Parsed (unsynced) as dashed if available
        if ref_p is not None and tgt_p is not None:
            ref_norm_p = _acc_norm(ref_p)
            tgt_norm_p = _acc_norm(tgt_p)
            if ref_norm_p is not None and tgt_norm_p is not None:
                # Each sensor relative to its own first timestamp
                ref_pt = _ts_s(ref_p, ref_p["timestamp"].iloc[0])
                tgt_pt = _ts_s(tgt_p, tgt_p["timestamp"].iloc[0])
                m_rp = (ref_pt >= start) & (ref_pt <= end)
                m_tp = (tgt_pt >= start) & (tgt_pt <= end)
                if m_rp.sum() > 5:
                    ax.plot(ref_pt[m_rp], ref_norm_p[m_rp], lw=0.6,
                            color=_SENSOR_COLORS["sporsa"], ls="--", alpha=0.5,
                            label="SPORSA (parsed)")
                if m_tp.sum() > 5:
                    ax.plot(tgt_pt[m_tp], tgt_norm_p[m_tp], lw=0.6,
                            color=_SENSOR_COLORS["arduino"], ls="--", alpha=0.5,
                            label="Arduino (parsed)")

        ax.set_ylabel("|acc| (m/s\u00b2)")
        ax.set_title(f"{label}  [{start:.0f}\u2013{end:.0f} s]", fontsize=10)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.2, lw=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 6: Clock correction curve
# ---------------------------------------------------------------------------

def plot_sync_clock_correction(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/clock_correction.png`` — delta_ms and drift-only term vs Arduino time."""
    info = _load_json(synced_dir / "sync_info.json")
    parsed_dir = synced_dir.parent / "parsed"
    if info is None:
        return None
    if output_path is None:
        output_path = synced_dir / "clock_correction.png"

    tgt_p = _load_sensor_df(parsed_dir, "arduino") if parsed_dir.exists() else None
    if tgt_p is None:
        return None

    offset_s = info.get("offset_seconds")
    drift_sps = info.get("drift_seconds_per_second")
    t0_s = info.get("target_time_origin_seconds")
    if offset_s is None or drift_sps is None or t0_s is None:
        return None

    ts_raw_ms = tgt_p["timestamp"].to_numpy(dtype=float)
    ts_raw_s = ts_raw_ms / 1000.0
    delta_total_ms = (offset_s + drift_sps * (ts_raw_s - t0_s)) * 1000.0
    drift_only_ms = drift_sps * (ts_raw_s - t0_s) * 1000.0

    t_plot = ts_raw_s - ts_raw_s[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{synced_dir.parent.name} \u2014 clock correction", fontsize=12)

    ax = axes[0]
    ax.plot(t_plot, delta_total_ms, lw=1.0, color="#2ca02c")
    ax.set_ylabel("\u0394 total (ms)")
    ax.set_title("Total correction (offset + drift)")
    ax.grid(alpha=0.2, lw=0.4)

    ax = axes[1]
    ax.plot(t_plot, drift_only_ms, lw=1.0, color="#e67e22")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("\u0394 drift (ms)")
    ax.set_xlabel("Arduino time (s, relative)")
    ax.set_title(f"Drift-only term  ({drift_sps*1e6:.1f} ppm)")
    ax.grid(alpha=0.2, lw=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 7: Rolling global correlation
# ---------------------------------------------------------------------------

def plot_sync_roll_corr_global(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
    window_s: float = 12.0,
) -> Path | None:
    """``synced/rolling_correlation.png`` — time-varying Pearson *r* between sensors."""
    if output_path is None:
        output_path = synced_dir / "rolling_correlation.png"

    ref = _load_sensor_df(synced_dir, "sporsa")
    tgt = _load_sensor_df(synced_dir, "arduino")
    if any(x is None for x in (ref, tgt)):
        return None

    result = _resample_acc_norm_pair(ref, tgt, hz=_TARGET_HZ)
    if result is None:
        return None
    time_s, ref_n, tgt_n = result

    win_samples = max(10, int(window_s * _TARGET_HZ))
    r = _rolling_pearson(ref_n, tgt_n, win_samples)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"{synced_dir.parent.name} \u2014 rolling alignment quality", fontsize=12)

    # Top: overlaid acc norms
    ax = axes[0]
    ax.plot(time_s, ref_n, lw=0.6, color=_SENSOR_COLORS["sporsa"],
            label="SPORSA", alpha=0.8)
    ax.plot(time_s, tgt_n, lw=0.6, color=_SENSOR_COLORS["arduino"],
            label="Arduino", alpha=0.8)
    ax.set_ylabel("|acc| (m/s\u00b2)"); ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)

    # Bottom: rolling r
    ax = axes[1]
    valid = np.isfinite(r)
    ax.fill_between(time_s[valid], 0, r[valid],
                    where=r[valid] >= 0, color="#2ca02c", alpha=0.3)
    ax.fill_between(time_s[valid], 0, r[valid],
                    where=r[valid] < 0, color="#d62728", alpha=0.3)
    ax.plot(time_s[valid], r[valid], lw=0.8, color="black")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axhline(0.2, color="orange", lw=0.8, ls=":", label="r = 0.2")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"Rolling Pearson r ({window_s:.0f}s window)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(alpha=0.2, lw=0.4)
    ax.set_title("Time-varying alignment correlation")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 8: Scatter zoom
# ---------------------------------------------------------------------------

def plot_sync_scatter_zoom(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/scatter_zoom.png`` — |acc| vs |acc| in a mid-recording window."""
    if output_path is None:
        output_path = synced_dir / "scatter_zoom.png"

    ref = _load_sensor_df(synced_dir, "sporsa")
    tgt = _load_sensor_df(synced_dir, "arduino")
    if any(x is None for x in (ref, tgt)):
        return None

    result = _resample_acc_norm_pair(ref, tgt, hz=_TARGET_HZ)
    if result is None:
        return None
    time_s, ref_n, tgt_n = result

    total = time_s[-1]
    win = 60.0
    mid_start = max(0, total / 2 - win / 2)
    mid_end = mid_start + win
    mask = (time_s >= mid_start) & (time_s <= mid_end)
    if mask.sum() < 10:
        return None

    r_seg = ref_n[mask]
    t_seg = tgt_n[mask]
    r_val = np.corrcoef(r_seg, t_seg)[0, 1] if np.std(r_seg) > 1e-12 and np.std(t_seg) > 1e-12 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{synced_dir.parent.name} \u2014 scatter zoom  "
                 f"[{mid_start:.0f}\u2013{mid_end:.0f} s]", fontsize=12)

    ax = axes[0]
    ax.scatter(r_seg, t_seg, s=6, alpha=0.4, c="#2ca02c", edgecolors="none")
    lo = min(r_seg.min(), t_seg.min())
    hi = max(r_seg.max(), t_seg.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, label="y=x")
    ax.set_xlabel("|acc| SPORSA (m/s\u00b2)"); ax.set_ylabel("|acc| Arduino (m/s\u00b2)")
    ax.set_title(f"Scatter (r={r_val:.3f})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Time-series of this window
    ax = axes[1]
    t_win = time_s[mask]
    ax.plot(t_win, r_seg, lw=0.8, color=_SENSOR_COLORS["sporsa"], label="SPORSA")
    ax.plot(t_win, t_seg, lw=0.8, color=_SENSOR_COLORS["arduino"], label="Arduino")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|acc| (m/s\u00b2)")
    ax.set_title("Zoomed overlay"); ax.legend(fontsize=8); ax.grid(alpha=0.2, lw=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 9: Anchor offsets vs time
# ---------------------------------------------------------------------------

def plot_sync_anchor_offsets(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/anchor_offsets.png`` — scatter of per-anchor offset vs target time."""
    info = _load_json(synced_dir / "sync_info.json")
    if info is None:
        return None
    cal = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    if cal is None:
        return None
    anchors = cal.get("anchors") or []
    if len(anchors) < 2:
        return None
    if output_path is None:
        output_path = synced_dir / "anchor_offsets.png"

    t_vals = np.array([a.get("t_tgt_s", 0) for a in anchors])
    off_vals = np.array([a.get("offset_s", 0) for a in anchors])
    scores = np.array([a.get("score", 0.5) for a in anchors])

    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = scores * 100 + 30
    sc = ax.scatter(t_vals, off_vals, s=sizes, c=scores, cmap="RdYlGn",
                    vmin=0, vmax=1, edgecolors="black", linewidths=0.6, zorder=3)

    # Linear fit
    z = np.polyfit(t_vals, off_vals, 1)
    fit_fn = np.poly1d(z)
    t_range = np.linspace(t_vals.min(), t_vals.max(), 100)
    drift_ppm = z[0] * 1e6
    ax.plot(t_range, fit_fn(t_range), "--", color="gray", lw=1.2,
            label=f"Linear fit (drift \u2248 {drift_ppm:.0f} ppm)")

    # Residuals
    residuals = off_vals - fit_fn(t_vals)
    for t, o, r in zip(t_vals, off_vals, residuals):
        ax.annotate(f"{r*1e3:+.1f} ms", (t, o), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="gray")

    plt.colorbar(sc, ax=ax, label="Anchor score", shrink=0.8)
    ax.set_xlabel("Arduino target time (s)")
    ax.set_ylabel("Estimated offset (s)")
    ax.set_title(f"{synced_dir.parent.name} \u2014 anchor offsets vs time")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 10: Difference profile
# ---------------------------------------------------------------------------

def plot_sync_difference_profile(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/difference_profile.png`` — resampled |acc|_ref - |acc|_tgt."""
    if output_path is None:
        output_path = synced_dir / "difference_profile.png"

    ref = _load_sensor_df(synced_dir, "sporsa")
    tgt = _load_sensor_df(synced_dir, "arduino")
    if any(x is None for x in (ref, tgt)):
        return None

    result = _resample_acc_norm_pair(ref, tgt, hz=_TARGET_HZ)
    if result is None:
        return None
    time_s, ref_n, tgt_n = result
    diff = ref_n - tgt_n

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"{synced_dir.parent.name} \u2014 acc norm difference profile", fontsize=12)

    # Top: overlaid norms
    ax = axes[0]
    ax.plot(time_s, ref_n, lw=0.5, color=_SENSOR_COLORS["sporsa"],
            label="SPORSA", alpha=0.8)
    ax.plot(time_s, tgt_n, lw=0.5, color=_SENSOR_COLORS["arduino"],
            label="Arduino", alpha=0.8)
    ax.set_ylabel("|acc|"); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.2, lw=0.4)

    # Bottom: difference
    ax = axes[1]
    ax.fill_between(time_s, 0, diff, where=diff >= 0, color="#2ca02c", alpha=0.3)
    ax.fill_between(time_s, 0, diff, where=diff < 0, color="#d62728", alpha=0.3)
    ax.plot(time_s, diff, lw=0.5, color="black", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    # Running mean
    win = max(1, int(10.0 * _TARGET_HZ))
    if len(diff) > win:
        smooth = np.convolve(diff, np.ones(win) / win, mode="same")
        ax.plot(time_s, smooth, lw=1.5, color="#9b59b6", label=f"10 s mean")
        ax.legend(fontsize=7)
    ax.set_ylabel("|acc|_ref \u2212 |acc|_tgt")
    ax.set_xlabel("Time (s)")
    ax.set_title("Residual difference (SPORSA \u2212 Arduino)")
    ax.grid(alpha=0.2, lw=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def plot_sync_stage(
    synced_dir: Path,
) -> list[Path]:
    """Generate all sync plots for a recording's synced directory."""
    out_paths: list[Path] = []

    for name, fn in [
        ("overlay",       lambda: plot_sync_overlay(synced_dir)),
        ("methods",       lambda: plot_sync_methods_comparison(synced_dir)),
        ("dashboard",     lambda: plot_sync_model_dashboard(synced_dir)),
        ("zoom",          lambda: plot_sync_zoom_alignment(synced_dir)),
        ("clock",         lambda: plot_sync_clock_correction(synced_dir)),
        ("rolling_corr",  lambda: plot_sync_roll_corr_global(synced_dir)),
        ("scatter",       lambda: plot_sync_scatter_zoom(synced_dir)),
        ("anchors",       lambda: plot_sync_anchor_offsets(synced_dir)),
        ("difference",    lambda: plot_sync_difference_profile(synced_dir)),
        ("detail",        lambda: plot_sync_detail(synced_dir)),
    ]:
        try:
            p = fn()
            if p is not None:
                out_paths.append(p)
        except Exception as exc:
            log.warning("plot_sync %s failed for %s: %s", name, synced_dir.name, exc)

    try:
        from visualization.plot_comparison import plot_stage_data
        out_paths += plot_stage_data(synced_dir)
    except Exception as exc:
        log.warning("sync comparison plot failed for %s: %s", synced_dir.name, exc)

    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PLOT_CHOICES = [
    "all", "overlay", "methods", "detail", "dashboard", "zoom",
    "clock", "rolling_corr", "scatter", "anchors", "difference",
]


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_sync",
        description="Generate sync plots for a recording.",
    )
    parser.add_argument("recording_name",
                        help="Recording name or path to synced/ directory.")
    parser.add_argument("--plot", choices=_PLOT_CHOICES, default="all",
                        help="Which plot(s) to generate (default: all).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
        "overlay":      lambda: [plot_sync_overlay(synced_dir)],
        "methods":      lambda: [plot_sync_methods_comparison(synced_dir)],
        "detail":       lambda: [plot_sync_detail(synced_dir)],
        "dashboard":    lambda: [plot_sync_model_dashboard(synced_dir)],
        "zoom":         lambda: [plot_sync_zoom_alignment(synced_dir)],
        "clock":        lambda: [plot_sync_clock_correction(synced_dir)],
        "rolling_corr": lambda: [plot_sync_roll_corr_global(synced_dir)],
        "scatter":      lambda: [plot_sync_scatter_zoom(synced_dir)],
        "anchors":      lambda: [plot_sync_anchor_offsets(synced_dir)],
        "difference":   lambda: [plot_sync_difference_profile(synced_dir)],
        "all":          lambda: plot_sync_stage(synced_dir),
    }

    paths = dispatch[args.plot]()
    paths = [p for p in (paths or []) if p is not None]
    if not paths:
        print("No plots generated.")
    for p in paths:
        print(f"Saved \u2192 {p}")


if __name__ == "__main__":
    main()
