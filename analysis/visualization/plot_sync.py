"""Sync-stage visualizations — diagnostic plots for per-recording synchronization.

Functions
---------
plot_sync_overlay
    Before-vs-after comparison: parsed (unsynced) and synced sensor signals.
plot_sync_methods_comparison
    Bar charts comparing all sync method metrics from ``all_methods.json``.
plot_sync_method_offsets
    \u0394-offset vs selected method (\u00b5s) with human-readable absolute values.
plot_sync_model_dashboard
    Grid of key model parameters, correlation gauges, and anchor detail.
plot_sync_zoom_alignment
    Multi-panel zoomed overlay (synced signals only) at opening, mid, and late.
plot_sync_roll_corr_global
    Rolling Pearson *r* between SPORSA and synced Arduino |acc| over time.
plot_sync_scatter_zoom
    Per-sample |acc| vs |acc| in a mid window (alignment sanity check).
plot_sync_anchor_offsets
    Per-anchor offset vs target time, sized/colored by score, with drift fit.
plot_sync_anchor_timeline
    Full-session |acc| with calibration / opening anchor times marked.
plot_sync_anchor_model_fit
    Per-method linear clock model vs measured anchor offsets (from ``all_methods.json``).
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
from matplotlib.transforms import blended_transform_factory
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
_METHOD_MARKERS = {
    "multi_anchor":          "o",
    "one_anchor_adaptive":   "s",
    "one_anchor_prior":      "D",
    "signal_only":           "^",
}


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


def _anchor_time(a: dict) -> float:
    """Prefer ``t_ref_s`` (reference-axis) when present; fall back to ``t_tgt_s``."""
    return float(a.get("t_ref_s") or a.get("t_tgt_s", 0.0))


def _anchor_markers_from_sync_info(info: dict) -> list[tuple[float, str, float | None]]:
    """Return ``(t_ref_s, label, score)`` for sync calibration / opening markers.

    Uses reference-side anchor time when available so markers align with
    the SPORSA |acc| trace; falls back to ``t_tgt_s`` for older metadata.
    """
    out: list[tuple[float, str, float | None]] = []
    cal = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    if cal:
        anchors = cal.get("anchors") or []
        if anchors:
            for i, a in enumerate(anchors):
                out.append((_anchor_time(a), f"A{i + 1}", a.get("score")))
        else:
            opening = cal.get("opening") or {}
            closing = cal.get("closing") or {}
            if opening:
                out.append((_anchor_time(opening), "Opening", opening.get("score")))
            if closing:
                out.append((_anchor_time(closing), "Closing", closing.get("score")))
    if not out:
        adap = info.get("adaptive") if isinstance(info.get("adaptive"), dict) else None
        if adap:
            oa = adap.get("opening_anchor") or {}
            if oa:
                out.append((_anchor_time(oa), "Opening", oa.get("score")))
    return out


def _arduino_abs_time_range_s(synced_dir: Path) -> tuple[float, float] | None:
    """Absolute target time (s) at first / last Arduino sample."""
    df = _load_sensor_df(synced_dir, "arduino")
    if df is None or df.empty:
        return None
    ts = df["timestamp"].to_numpy(dtype=float)
    if ts.size < 1:
        return None
    return float(ts[0]) / 1000.0, float(ts[-1]) / 1000.0


def _method_row_drift_s_per_s(m: dict) -> float:
    """Drift (s/s) from ``all_methods`` method row."""
    d = m.get("drift_seconds_per_second")
    if d is not None:
        try:
            return float(d)
        except (TypeError, ValueError):
            pass
    ppm = m.get("drift_ppm")
    if ppm is not None:
        try:
            return float(ppm) * 1e-6
        except (TypeError, ValueError):
            pass
    return 0.0


def _measured_anchor_points_from_method_row(
    m: dict,
) -> list[tuple[float, float, str]]:
    """``(t_tgt_s, offset_s, tag)`` from calibration anchors and adaptive opening."""
    pts: list[tuple[float, float, str]] = []
    cal = m.get("calibration_anchors")
    if isinstance(cal, list) and cal:
        for i, a in enumerate(cal):
            if not isinstance(a, dict):
                continue
            t = a.get("t_tgt_s")
            o = a.get("offset_s")
            if t is None or o is None:
                continue
            try:
                pts.append((float(t), float(o), f"A{i + 1}"))
            except (TypeError, ValueError):
                pass
    oa = m.get("adaptive_opening_anchor")
    if isinstance(oa, dict):
        t = oa.get("t_tgt_s")
        o = oa.get("offset_s")
        if t is not None and o is not None:
            try:
                pts.append((float(t), float(o), "Opn"))
            except (TypeError, ValueError):
                pass
    return pts


def _format_offset_interval_s(seconds: float | None) -> str:
    """Format a clock offset in seconds for tables and annotations."""
    if seconds is None:
        return "N/A"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(s):
        return "N/A"
    a = abs(s)
    if a >= 1.0:
        return f"{s:.5f} s"
    if a >= 1e-3:
        return f"{s * 1e3:.4f} ms"
    if a >= 1e-6:
        return f"{s * 1e6:.2f} \u00b5s"
    if a >= 1e-9:
        return f"{s * 1e9:.2f} ns"
    return f"{s:.2e} s"


def _format_delta_us(delta_us: float) -> str:
    """Compact string for an offset difference in microseconds."""
    if not np.isfinite(delta_us):
        return "?"
    if abs(delta_us) >= 1000.0:
        return f"{delta_us / 1000.0:+.4f} ms"
    return f"{delta_us:+.2f} \u00b5s"


def _plot_method_offset_delta_horizontal(
    ax: plt.Axes,
    data: dict,
    *,
    footnote: str | None = None,
) -> bool:
    """Draw horizontal bars: Δ offset vs selected method (\u00b5s). Returns False if no data."""
    methods = [m for m in _ALL_METHODS if isinstance(data.get(m), dict)]
    if not methods:
        return False

    selected = data.get("selected_method", "")
    off_s = np.full(len(methods), np.nan, dtype=float)
    avail = np.zeros(len(methods), dtype=bool)
    for i, m in enumerate(methods):
        md = data[m] or {}
        if not md.get("available"):
            continue
        raw = md.get("offset_seconds")
        if raw is None:
            continue
        try:
            fv = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            off_s[i] = fv
            avail[i] = True
    if not avail.any():
        return False

    ref: float | None = None
    if selected and selected in methods:
        j = methods.index(selected)
        if avail[j]:
            ref = float(off_s[j])
    if ref is None:
        ref = float(np.median(off_s[avail]))

    deltas_us = (off_s - ref) * 1e6
    y = np.arange(len(methods))
    method_labels = [_METHOD_LABELS.get(m, m) for m in methods]
    finite_d = deltas_us[avail]
    max_abs_d = float(np.nanmax(np.abs(finite_d))) if finite_d.size else 0.0
    span = max(max_abs_d, 0.5)

    for i, m in enumerate(methods):
        if not avail[i]:
            ax.scatter([0], [i], marker="x", color="#888888", s=36, zorder=4)
            ax.text(0, i, "  n/a", va="center", ha="left", fontsize=8, color="gray")
            continue
        d = float(deltas_us[i])
        col = _METHOD_COLORS.get(m, "gray")
        ax.barh(
            i, d, color=col, height=0.55, alpha=0.88,
            edgecolor="black" if m == selected else "none",
            linewidth=1.8 if m == selected else 0,
        )
        abs_lbl = _format_offset_interval_s(float(off_s[i]))
        delta_lbl = _format_delta_us(d)
        pad = 0.035 * span
        if abs(d) < 1e-12:
            tx = pad
            ha = "left"
        else:
            tx = d + (pad if d >= 0 else -pad)
            ha = "left" if d >= 0 else "right"
        ax.text(tx, i, f" {delta_lbl}  | abs {abs_lbl}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="#333333", lw=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(method_labels, fontsize=9)
    ref_lbl = (
        _METHOD_LABELS.get(selected, selected)
        if selected and selected in methods and avail[methods.index(selected)]
        else "median"
    )
    ax.set_xlabel(f"\u0394 offset vs {ref_lbl} (\u00b5s)")
    ax.set_title("Inter-method offset spread", fontsize=10)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.grid(axis="x", alpha=0.28)
    if footnote:
        ax.text(0.5, -0.22, footnote, transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    return True


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

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 1.12, 0.88], hspace=0.38, wspace=0.32,
    )
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_drift = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_off = fig.add_subplot(gs[2, :])
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

    def _annotate_offset_under_bars(ax, xs) -> None:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for xi, m in zip(xs, methods):
            md = data[m] or {}
            off = md.get("offset_seconds")
            if off is None or not md.get("available"):
                lbl = ""
            else:
                try:
                    lbl = _format_offset_interval_s(float(off))
                except (TypeError, ValueError):
                    lbl = ""
            if lbl:
                ax.text(xi, -0.08, lbl, transform=trans, ha="center", va="top",
                        fontsize=7, color="#444444")

    # Correlation (offset + drift)
    corr_vals = np.array([data[m].get("corr_offset_and_drift") or np.nan for m in methods])
    ax = ax_corr
    ax.bar(bar_x, corr_vals, color=colors, edgecolor=edge_colors,
           linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axhline(0.2, color="orange", lw=1.0, ls=":", label="min threshold (0.2)")
    ax.set_xticks(bar_x); ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("Correlation"); ax.set_title("Cross-corr (offset + drift)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _annotate(ax, bar_x, corr_vals)
    _annotate_offset_under_bars(ax, bar_x)
    ax.margins(y=0.22)

    # |drift| (ppm)
    drift_vals = np.array([abs(data[m].get("drift_ppm") or 0.0) for m in methods])
    ax = ax_drift
    ax.bar(bar_x, drift_vals, color=colors, edgecolor=edge_colors,
           linewidth=lw_edges, width=bar_w, alpha=0.85)
    ax.axhline(400, color="orange", lw=1.0, ls=":", label="typical Arduino (~400 ppm)")
    ax.axhline(5000, color="red", lw=1.0, ls=":", label="implausible (5000 ppm)")
    ax.set_xticks(bar_x); ax.set_xticklabels(method_labels, rotation=15, ha="right")
    ax.set_ylabel("|drift| (ppm)"); ax.set_title("Absolute drift")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _annotate(ax, bar_x, drift_vals, fmt=".1f")
    _annotate_offset_under_bars(ax, bar_x)
    ax.margins(y=0.22)

    # Heatmap of normalized metrics
    ax = ax_hm
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
    ax = ax_tbl
    ax.axis("off")
    col_headers = ["Method", "Corr", "Drift (ppm)", "Offset", "Avail.", "Sel."]
    rows = []
    for m in methods:
        md = data[m]
        corr = md.get("corr_offset_and_drift")
        drift = md.get("drift_ppm")
        off_raw = md.get("offset_seconds")
        off_cell = (
            _format_offset_interval_s(float(off_raw))
            if md.get("available") and off_raw is not None
            else "N/A"
        )
        avail = "\u2713" if md.get("available") else "\u2717"
        sel = "\u2605" if m == selected else ""
        rows.append([
            _METHOD_LABELS.get(m, m),
            f"{corr:.4f}" if corr is not None else "N/A",
            f"{drift:.1f}" if drift is not None else "N/A",
            off_cell,
            avail, sel,
        ])
    table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.65)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
        elif rows[row - 1][5] == "\u2605":
            cell.set_facecolor("#d4edda")
    ax.set_title("Method summary")

    if not _plot_method_offset_delta_horizontal(
        ax_off,
        data,
        footnote=(
            "\u0394 vs selected (\u2605) method, in \u00b5s. "
            "\u201cabs\u201d = absolute clock offset for that method."
        ),
    ):
        ax_off.axis("off")
        ax_off.text(
            0.5, 0.5,
            "No per-method offset_seconds in all_methods.json (re-run sync to refresh).",
            ha="center", va="center", fontsize=10, color="gray",
            transform=ax_off.transAxes,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 3: Per-method offset comparison
# ---------------------------------------------------------------------------

def plot_sync_method_offsets(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/method_offsets.png`` — inter-method offset spread (\u0394 vs selected)."""
    data = _load_json(synced_dir / "all_methods.json")
    if data is None:
        return None
    if output_path is None:
        output_path = synced_dir / "method_offsets.png"

    selected = data.get("selected_method", "")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.suptitle(
        f"{synced_dir.parent.name} \u2014 clock offset comparison  "
        f"(\u2605 = selected: {_METHOD_LABELS.get(selected, selected)})",
        fontsize=12,
    )
    if not _plot_method_offset_delta_horizontal(
        ax,
        data,
        footnote=(
            "\u0394 vs selected method. Values after \u201cabs\u201d use auto scale (ms / \u00b5s / ns)."
        ),
    ):
        plt.close(fig)
        return None

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
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

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{rec_name} \u2014 sync model dashboard", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(
        4, 3, hspace=0.45, wspace=0.35, height_ratios=[1.0, 1.15, 0.9, 1.0],
    )

    # (0,0) Key metrics text
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        f"Method:  {_METHOD_LABELS.get(method, method)}",
        f"Offset:  {_format_offset_interval_s(offset_s)}" if offset_s is not None else "Offset:  N/A",
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
            md_r = allm[avail[i]] or {}
            off_t = ""
            if md_r.get("available") and md_r.get("offset_seconds") is not None:
                try:
                    off_t = _format_offset_interval_s(float(md_r["offset_seconds"]))
                except (TypeError, ValueError):
                    off_t = ""
            suffix = f"  | off {off_t}" if off_t else ""
            ax.text(v + 10, i, f"{v:.0f}{suffix}", va="center", fontsize=8)
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

    # (2,0:3) Per-method offset spread
    ax_off = fig.add_subplot(gs[2, :])
    if allm:
        if not _plot_method_offset_delta_horizontal(
            ax_off,
            allm,
            footnote="\u0394 vs selected (\u2605) method; \u201cabs\u201d = absolute offset.",
        ):
            ax_off.axis("off")
    else:
        ax_off.axis("off")

    # (3,0:3) Per-method correlation comparison
    ax = fig.add_subplot(gs[3, :])
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
        trans_c = blended_transform_factory(ax.transData, ax.transAxes)
        for i, v in enumerate(corr_vals):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
            if avail[i] == allm.get("selected_method"):
                ax.text(i, v + 0.05, "\u2605", ha="center", fontsize=13, fontweight="bold")
            md = allm[avail[i]] or {}
            off_c = md.get("offset_seconds")
            if md.get("available") and off_c is not None:
                try:
                    ax.text(
                        i, -0.07, _format_offset_interval_s(float(off_c)),
                        transform=trans_c, ha="center", va="top", fontsize=7, color="#444444",
                    )
                except (TypeError, ValueError):
                    pass
        ax.margins(y=0.2)
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
    """``synced/zoom_alignment.png`` — 3 windows: opening, mid, late (synced only)."""
    if output_path is None:
        output_path = synced_dir / "zoom_alignment.png"

    ref_s = _load_sensor_df(synced_dir, "sporsa")
    tgt_s = _load_sensor_df(synced_dir, "arduino")

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
                    label="SPORSA", alpha=0.9)
        if m_t.sum() > 5:
            ax.plot(tgt_ts[m_t], tgt_norm_s[m_t], lw=0.9, color=_SENSOR_COLORS["arduino"],
                    label="Arduino", alpha=0.9)

        ax.set_ylabel("|acc| (m/s\u00b2)")
        ax.set_title(f"{label}  [{start:.0f}\u2013{end:.0f} s]", fontsize=10)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.2, lw=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 6: Rolling global correlation
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
# Plot 7: Scatter zoom
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

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.suptitle(
        f"{synced_dir.parent.name} \u2014 |acc| agreement (mid session "
        f"{mid_start:.0f}\u2013{mid_end:.0f} s)",
        fontsize=12,
    )
    ax.scatter(r_seg, t_seg, s=7, alpha=0.35, c="#2ca02c", edgecolors="none")
    lo = min(r_seg.min(), t_seg.min())
    hi = max(r_seg.max(), t_seg.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.9, label="Ideal (y = x)")
    ax.set_xlabel("|acc| SPORSA (m/s\u00b2)")
    ax.set_ylabel("|acc| Arduino (m/s\u00b2)")
    ax.set_title(
        f"Each point is one time sample (resampled to {_TARGET_HZ:.0f} Hz); "
        f"Pearson r = {r_val:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.text(
        0.5, 0.02,
        "After sync, both axes share the same timeline; a tight cloud on the diagonal means "
        "the two sensors report similar acceleration magnitude at the same instants.",
        ha="center", fontsize=9, color="#444444",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 8: Anchors on reference |acc| trace
# ---------------------------------------------------------------------------

def plot_sync_anchor_timeline(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/anchor_timeline.png`` — full SPORSA |acc| with anchor times marked."""
    info = _load_json(synced_dir / "sync_info.json")
    if info is None:
        return None
    markers = _anchor_markers_from_sync_info(info)
    if not markers:
        return None
    if output_path is None:
        output_path = synced_dir / "anchor_timeline.png"

    ref = _load_sensor_df(synced_dir, "sporsa")
    if ref is None:
        return None
    acc_cols = [c for c in ("ax", "ay", "az") if c in ref.columns]
    if not acc_cols:
        return None

    t0 = _shared_t0([ref])
    ts = _ts_s(ref, t0)
    norm = strict_vector_norm(ref, acc_cols)
    x, y = filter_valid_plot_xy(ts, norm)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, y, lw=0.55, color=_SENSOR_COLORS["sporsa"], alpha=0.85, label="|acc| SPORSA")

    y_hi = float(np.nanmax(y)) if y.size else 1.0
    y_lo = float(np.nanmin(y)) if y.size else 0.0
    y_rng = y_hi - y_lo if y_hi > y_lo else 1.0
    t0_s = t0 / 1000.0
    for t_abs, label, score in markers:
        t_evt = t_abs - t0_s
        if t_evt < x.min() - 1 or t_evt > x.max() + 1:
            continue
        sc = 0.5 if score is None else float(score)
        color = "#2ca02c" if sc >= 0.5 else "#d62728"
        ax.axvline(t_evt, color=color, ls="--", lw=1.0, alpha=0.75)
        ax.text(
            t_evt + 0.02 * (x.max() - x.min() if x.size > 1 else 1),
            y_lo + 0.92 * y_rng,
            f"{label}\n({sc:.2f})" if score is not None else label,
            fontsize=8,
            color=color,
            va="top",
        )

    ax.set_xlabel("Time (s)"); ax.set_ylabel("|acc| (m/s\u00b2)")
    ax.set_title(f"{synced_dir.parent.name} \u2014 calibration / anchor times on reference signal")
    ax.grid(alpha=0.2, lw=0.4)
    fig.tight_layout()
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
    if len(anchors) < 1:
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

    if len(anchors) >= 2:
        z = np.polyfit(t_vals, off_vals, 1)
        fit_fn = np.poly1d(z)
        t_range = np.linspace(t_vals.min(), t_vals.max(), 100)
        drift_ppm = z[0] * 1e6
        ax.plot(t_range, fit_fn(t_range), "--", color="gray", lw=1.2,
                label=f"Linear fit (drift \u2248 {drift_ppm:.0f} ppm)")
        residuals = off_vals - fit_fn(t_vals)
        for t, o, r in zip(t_vals, off_vals, residuals):
            ax.annotate(f"{r*1e3:+.1f} ms", (t, o), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="gray")

    plt.colorbar(sc, ax=ax, label="Anchor score", shrink=0.8)
    ax.set_xlabel("Arduino target time (s)")
    ax.set_ylabel("Estimated offset (s)")
    ax.set_title(f"{synced_dir.parent.name} \u2014 anchor offsets vs time")
    if len(anchors) >= 2:
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 10: Anchor measurements vs fitted linear model (all methods)
# ---------------------------------------------------------------------------

def plot_sync_anchor_model_fit(
    synced_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path | None:
    """``synced/anchor_model_fit.png`` — implied offset(t) vs measured anchors per method."""
    data = _load_json(synced_dir / "all_methods.json")
    if data is None:
        return None
    if output_path is None:
        output_path = synced_dir / "anchor_model_fit.png"

    selected = data.get("selected_method", "")
    methods = [m for m in _ALL_METHODS if isinstance(data.get(m), dict)]
    if not methods:
        return None

    bounds = _arduino_abs_time_range_s(synced_dir)
    t0_fallback = float(bounds[0]) if bounds else None
    for m in methods:
        md = data[m] or {}
        if md.get("available") and md.get("target_time_origin_seconds") is not None:
            try:
                t0_fallback = float(md["target_time_origin_seconds"])
                break
            except (TypeError, ValueError):
                pass
    if t0_fallback is None:
        return None

    t_lo = t0_fallback
    t_hi = t0_fallback + 1.0
    if bounds:
        t_lo, t_hi = bounds[0], bounds[1]
    anchor_ts: list[float] = []
    for m in methods:
        md = data[m] or {}
        if not md.get("available"):
            continue
        for t, _o, _tag in _measured_anchor_points_from_method_row(md):
            anchor_ts.append(t)
    if anchor_ts:
        t_lo = min(t_lo, min(anchor_ts))
        t_hi = max(t_hi, max(anchor_ts))
    margin = max(30.0, 0.02 * (t_hi - t_lo))
    t_lo -= margin
    t_hi += margin
    t_fine = np.linspace(t_lo, t_hi, max(400, int((t_hi - t_lo) * 3)))

    fig, ax = plt.subplots(figsize=(14, 6))
    any_line = False
    y_vals: list[float] = []

    for m in methods:
        md = data[m] or {}
        if not md.get("available"):
            continue
        b = md.get("offset_seconds")
        if b is None:
            continue
        try:
            b = float(b)
        except (TypeError, ValueError):
            continue
        t0 = md.get("target_time_origin_seconds")
        if t0 is None:
            t0 = t0_fallback
        else:
            try:
                t0 = float(t0)
            except (TypeError, ValueError):
                t0 = t0_fallback
        a = _method_row_drift_s_per_s(md)
        y_line = b + a * (t_fine - t0)
        col = _METHOD_COLORS.get(m, "gray")
        mk = _METHOD_MARKERS.get(m, "o")
        lbl = _METHOD_LABELS.get(m, m)
        if m == selected:
            lbl += " (\u2605)"
        ax.plot(t_fine, y_line, color=col, lw=2.0, ls="-", label=lbl, alpha=0.9)
        any_line = True
        y_vals.extend([float(np.nanmin(y_line)), float(np.nanmax(y_line))])

        pts = _measured_anchor_points_from_method_row(md)
        for t_a, o_a, tag in pts:
            ax.scatter(
                [t_a], [o_a], s=72, marker=mk, color=col,
                edgecolors="black", linewidths=0.7, zorder=5,
            )
            ax.annotate(
                tag, (t_a, o_a), textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=col,
            )
            y_vals.extend([o_a, b + a * (t_a - t0)])

    if not any_line and not y_vals:
        plt.close(fig)
        return None

    ax.set_xlabel("Target time (s, Arduino clock)")
    ax.set_ylabel("Offset (s)  \u2014 maps target \u2192 reference")
    ax.set_title(
        f"{synced_dir.parent.name} \u2014 linear model vs measured anchor offsets",
    )
    ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    fig.text(
        0.5, 0.01,
        "Curves: offset(t) = offset_at_t\u2080 + drift\u00b7(t \u2212 t\u2080) from each method\u2019s sync_info. "
        "Markers: per-anchor cross-correlation offsets (tags A1\u2026 / Opn = adaptive opening). "
        "Re-run sync to refresh all_methods.json if models are missing.",
        ha="center", fontsize=8, color="#444444",
    )
    if y_vals:
        pad = max(0.05 * (max(y_vals) - min(y_vals)), 0.02)
        ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
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
        ("overlay",         lambda: plot_sync_overlay(synced_dir)),
        ("methods",         lambda: plot_sync_methods_comparison(synced_dir)),
        ("offsets",         lambda: plot_sync_method_offsets(synced_dir)),
        ("dashboard",       lambda: plot_sync_model_dashboard(synced_dir)),
        ("zoom",            lambda: plot_sync_zoom_alignment(synced_dir)),
        ("rolling_corr",    lambda: plot_sync_roll_corr_global(synced_dir)),
        ("scatter",         lambda: plot_sync_scatter_zoom(synced_dir)),
        ("anchors",         lambda: plot_sync_anchor_offsets(synced_dir)),
        ("anchor_timeline", lambda: plot_sync_anchor_timeline(synced_dir)),
        ("anchor_model",    lambda: plot_sync_anchor_model_fit(synced_dir)),
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
    "all", "overlay", "methods", "offsets", "dashboard", "zoom",
    "rolling_corr", "scatter", "anchors", "anchor_timeline", "anchor_model",
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
        "overlay":         lambda: [plot_sync_overlay(synced_dir)],
        "methods":         lambda: [plot_sync_methods_comparison(synced_dir)],
        "offsets":         lambda: [plot_sync_method_offsets(synced_dir)],
        "dashboard":       lambda: [plot_sync_model_dashboard(synced_dir)],
        "zoom":            lambda: [plot_sync_zoom_alignment(synced_dir)],
        "rolling_corr":    lambda: [plot_sync_roll_corr_global(synced_dir)],
        "scatter":         lambda: [plot_sync_scatter_zoom(synced_dir)],
        "anchors":         lambda: [plot_sync_anchor_offsets(synced_dir)],
        "anchor_timeline": lambda: [plot_sync_anchor_timeline(synced_dir)],
        "anchor_model":    lambda: [plot_sync_anchor_model_fit(synced_dir)],
        "all":             lambda: plot_sync_stage(synced_dir),
    }

    paths = dispatch[args.plot]()
    paths = [p for p in (paths or []) if p is not None]
    if not paths:
        print("No plots generated.")
    for p in paths:
        print(f"Saved \u2192 {p}")


if __name__ == "__main__":
    main()
