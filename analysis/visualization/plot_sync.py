"""Synchronisation quality visualizations.

Produces two figure types per recording, saved in the ``synced/`` stage
directory:

1. **Sync method comparison** (``sync_method_comparison.png``):
   Horizontal bar chart comparing the acc_norm cross-correlation score
   (offset + drift) achieved by each of the four sync methods — SDA,
   LIDA, calibration-window, and online.  The selected (best) method
   is highlighted.  A secondary panel shows the estimated clock drift
   in ppm per method.  This plot motivates the choice of the
   calibration-window method as the primary sync strategy.

2. **Pre-/post-sync alignment** (``sync_alignment.png``):
   Two-panel overlay of the accelerometer vector norm for Sporsa (handlebar)
   and Arduino (helmet).

   - *Left*: pre-sync (``parsed`` stage).  Both sensors are shown on their
     own time axes (Sporsa uses epoch ms; Arduino uses device uptime ms)
     to illustrate that the clocks are unrelated without alignment.
   - *Right*: post-sync (``synced`` stage).  Both streams share a common
     time axis, demonstrating the alignment quality of the selected method.

CLI::

    python -m visualization.plot_sync 2026-02-26_5
    python -m visualization.plot_sync 2026-02-26_5 --no-alignment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir
from ._utils import mask_dropout_packets, acc_norm_series

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_METHOD_ORDER = ("calibration", "lida", "sda", "online")
_METHOD_LABELS = {
    "calibration": "Calibration windows",
    "lida": "LIDA",
    "sda": "SDA",
    "online": "Online",
}
_METHOD_COLORS = {
    "calibration": "#2ca02c",
    "lida": "#9467bd",
    "sda": "#ff7f0e",
    "online": "#1f77b4",
}
_SELECTED_EDGE = "#d62728"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_methods(recording_name: str) -> dict:
    path = recording_stage_dir(recording_name, "synced") / "all_methods.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_sensor(recording_name: str, stage: str, sensor: str) -> pd.DataFrame:
    try:
        csv_path = find_sensor_csv(recording_name, stage, sensor)
        return mask_dropout_packets(load_dataframe(csv_path))
    except FileNotFoundError:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot 1: Sync method comparison
# ---------------------------------------------------------------------------

def plot_sync_method_comparison(recording_name: str) -> Path | None:
    """Horizontal bar chart comparing the four sync methods.

    Returns the path of the saved PNG, or ``None`` if ``all_methods.json``
    was not found.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    data = _load_all_methods(recording_name)
    if not data:
        log.warning("[%s/synced] all_methods.json not found — skipping sync comparison.", recording_name)
        return None

    selected = data.get("selected_method", "")
    methods = [
        m for m in _METHOD_ORDER
        if m in data and isinstance(data[m], dict) and data[m].get("available", False)
    ]

    if not methods:
        log.warning("[%s/synced] no available sync methods in all_methods.json — skipping.", recording_name)
        return None

    def _float_or_nan(val) -> float:
        return float(val) if val is not None else np.nan

    corr_scores = [_float_or_nan(data[m].get("corr_offset_and_drift")) for m in methods]
    drift_ppms = [abs(_float_or_nan(data[m].get("drift_ppm")) or 0.0) for m in methods]
    labels = [_METHOD_LABELS.get(m, m) for m in methods]
    colors = [_METHOD_COLORS.get(m, "gray") for m in methods]
    edge_widths = [2.5 if m == selected else 0.5 for m in methods]
    edge_colors = [_SELECTED_EDGE if m == selected else "#888888" for m in methods]

    fig, (ax_corr, ax_drift) = plt.subplots(
        1, 2,
        figsize=(10, max(3.5, 0.8 * len(methods) + 1.5)),
        constrained_layout=True,
    )

    # Left: correlation scores
    y_pos = np.arange(len(methods))
    bars = ax_corr.barh(
        y_pos, corr_scores,
        color=colors, edgecolor=edge_colors, linewidth=edge_widths,
        height=0.6,
    )
    ax_corr.set_yticks(y_pos)
    ax_corr.set_yticklabels(labels, fontsize=9)
    ax_corr.set_xlabel("Pearson r (acc_norm, offset + drift)", fontsize=9)
    ax_corr.set_title("Alignment quality", fontsize=10)
    finite_corr = [v for v in corr_scores if np.isfinite(v)]
    ax_corr.set_xlim(0, max(max(finite_corr) if finite_corr else 0.05, 0.05) * 1.25)
    ax_corr.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, corr_scores):
        if np.isfinite(val):
            ax_corr.text(
                val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )

    # Right: drift magnitude
    bars_d = ax_drift.barh(
        y_pos, drift_ppms,
        color=colors, edgecolor=edge_colors, linewidth=edge_widths,
        height=0.6,
    )
    ax_drift.set_yticks(y_pos)
    ax_drift.set_yticklabels(labels, fontsize=9)
    ax_drift.set_xlabel("Clock drift magnitude [ppm]", fontsize=9)
    ax_drift.set_title("Estimated drift", fontsize=10)
    ax_drift.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars_d, drift_ppms):
        if np.isfinite(val):
            ax_drift.text(
                val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=8,
            )

    legend_patch = mpatches.Patch(
        facecolor="none", edgecolor=_SELECTED_EDGE, linewidth=2.0,
        label=f"selected: {_METHOD_LABELS.get(selected, selected)}",
    )
    fig.legend(handles=[legend_patch], loc="upper right", fontsize=8, framealpha=0.8)

    fig.suptitle(f"{recording_name} — Sync method comparison", fontsize=11)
    out_path = synced_dir / "sync_method_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/synced] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: Pre / post sync alignment
# ---------------------------------------------------------------------------

def plot_pre_post_sync(recording_name: str, *, downsample: int = 5) -> Path | None:
    """Two-panel acc_norm overlay: pre-sync (parsed) vs post-sync (synced).

    Returns the path of the saved PNG, or ``None`` if the necessary CSVs
    were not found.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.exists():
        log.warning("[%s] synced/ directory not found — skipping alignment plot.", recording_name)
        return None

    parsed_dfs: dict[str, pd.DataFrame] = {}
    synced_dfs: dict[str, pd.DataFrame] = {}

    for sensor in _SENSORS:
        df_p = _load_sensor(recording_name, "parsed", sensor)
        df_s = _load_sensor(recording_name, "synced", sensor)
        if not df_p.empty:
            parsed_dfs[sensor] = df_p
        if not df_s.empty:
            synced_dfs[sensor] = df_s

    if len(parsed_dfs) < 2 or len(synced_dfs) < 2:
        log.warning("[%s] need both sensors in parsed and synced — skipping alignment.", recording_name)
        return None

    fig, (ax_pre, ax_post) = plt.subplots(
        1, 2,
        figsize=(14, 4),
        constrained_layout=True,
    )

    def _plot_acc_norm_overlay(ax: plt.Axes, dfs: dict[str, pd.DataFrame], shared_clock: bool) -> None:
        handles = []
        ts_all: list[float] = []
        for sensor, df in dfs.items():
            ts_all.extend([float(df["timestamp"].iloc[0]), float(df["timestamp"].iloc[-1])])

        t_ref = min(ts_all) if shared_clock else None
        for sensor, df in dfs.items():
            acc = acc_norm_series(df)
            valid = np.isfinite(acc)
            ts = df["timestamp"].astype(float)
            if shared_clock:
                time_s = (ts - t_ref) / 1000.0
            else:
                time_s = (ts - float(ts.iloc[0])) / 1000.0

            stride = downsample
            ax.plot(
                time_s.to_numpy()[valid][::stride],
                acc[valid][::stride],
                color=_SENSOR_COLORS.get(sensor, "gray"),
                linewidth=0.6, alpha=0.75, label=sensor,
            )
            handles.append(
                plt.Line2D([], [], color=_SENSOR_COLORS.get(sensor, "gray"), label=sensor)
            )
        ax.set_ylabel("‖acc‖ [m/s²]", fontsize=9)
        ax.set_xlabel("Time [s]" if shared_clock else "Time from start [s]", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Pre-sync: each sensor on its own t=0
    _plot_acc_norm_overlay(ax_pre, parsed_dfs, shared_clock=False)
    ax_pre.set_title("Before sync  (each sensor starts at t = 0)", fontsize=9)

    # Post-sync: shared epoch clock
    _plot_acc_norm_overlay(ax_post, synced_dfs, shared_clock=True)
    ax_post.set_title("After sync  (shared time axis)", fontsize=9)

    fig.suptitle(f"{recording_name} — Synchronisation alignment  (‖acc‖)", fontsize=11)
    out_path = synced_dir / "sync_alignment.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/synced] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: run all sync plots for one recording
# ---------------------------------------------------------------------------

def plot_sync_stage(recording_name: str, *, alignment: bool = True) -> None:
    """Generate all sync quality plots for *recording_name*.

    Produces:
    - ``synced/sync_method_comparison.png``: method quality bar chart.
    - ``synced/sync_alignment.png``: pre/post-sync acc_norm overlay (if ``alignment=True``).
    """
    plot_sync_method_comparison(recording_name)
    if alignment:
        plot_pre_post_sync(recording_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_sync",
        description="Generate synchronisation quality plots for one or more recordings.",
    )
    parser.add_argument(
        "recording_names",
        nargs="+",
        help="One or more recording names (e.g. 2026-02-26_5).",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Skip the pre/post-sync alignment plot.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for recording_name in args.recording_names:
        plot_sync_stage(recording_name, alignment=not args.no_alignment)


if __name__ == "__main__":
    main()
