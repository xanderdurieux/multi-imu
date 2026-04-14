"""Thesis-specific figures (high-level, publication-oriented).

CLI (from the ``analysis`` directory)::

    python -m visualization.thesis_plots 2026-02-26_r3
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

from common.paths import project_relative_path, read_csv, recording_stage_dir
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}
_ACC_COLS = ("ax", "ay", "az")


def _load_sensor_df(stage_dir: Path, sensor: str) -> pd.DataFrame | None:
    csv = stage_dir / f"{sensor}.csv"
    if not csv.exists():
        return None
    df = read_csv(csv)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["timestamp"], errors="coerce")


def _timestamps_to_relative_seconds(df: pd.DataFrame) -> np.ndarray:
    return (_timestamp_series(df).to_numpy(dtype=float) - _timestamp_series(df).min()) / 1000.0


def _recording_start_ms(sensor_dfs: dict[str, pd.DataFrame]) -> float:
    """Earliest finite ``timestamp`` (ms) among streams — one origin for a shared *x* axis."""
    mins: list[float] = []
    for df in sensor_dfs.values():
        ts = _timestamp_series(df).to_numpy(dtype=float)
        finite = ts[np.isfinite(ts)]
        if finite.size:
            mins.append(float(finite.min()))
    if not mins:
        raise ValueError("No timestamps in sensor dataframes.")
    return min(mins)


def _relative_seconds_from_t0(df: pd.DataFrame, t0_ms: float) -> np.ndarray:
    return (_timestamp_series(df).to_numpy(dtype=float) - t0_ms) / 1000.0


def _plot_acc_single(ax: plt.Axes, df: pd.DataFrame, *, sensor: str) -> None:
    cols = [c for c in _ACC_COLS if c in df.columns]
    if not cols:
        return
    color = _SENSOR_COLORS[sensor]
    t_s = _timestamps_to_relative_seconds(df)
    norm = strict_vector_norm(df, cols)
    x, y = filter_valid_plot_xy(t_s, norm)
    ax.plot(x, y, lw=0.85, color=color, label=sensor, alpha=0.88)
    ax.set_ylabel("|acc| (m/s²)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)


def _plot_acc_merged(
    ax: plt.Axes,
    sensor_dfs: dict[str, pd.DataFrame]
) -> None:
    """Plot |acc| on one axis; synced streams share ``t0`` = min timestamp across both."""

    if len(sensor_dfs) < 2:
        raise ValueError("Need at least two sensors to plot merged acc.")

    t0_ms = _recording_start_ms(sensor_dfs)
    for sensor, df in sensor_dfs.items():
        cols = [c for c in _ACC_COLS if c in df.columns]
        if not cols:
            continue
        color = _SENSOR_COLORS.get(sensor)
        t_s = _relative_seconds_from_t0(df, t0_ms)
        norm = strict_vector_norm(df, cols)
        x, y = filter_valid_plot_xy(t_s, norm)
        ax.plot(x, y, lw=0.85, color=color, label=sensor, alpha=0.88)
    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel("|acc| (m/s²)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)


def plot_parsed_vs_synced_imu(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Accelerometer |acc|: parsed Sporsa, parsed Arduino, then merged synced pair.

    *x* is seconds from each stream’s first ``timestamp`` in the before panels; the merged panel uses one
    origin (earliest ``timestamp`` across both synced CSVs) so both traces share the same relative time.
    """
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed directory not found: {parsed_dir}")

    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    parsed = {s: df for s in _SENSORS if (df := _load_sensor_df(parsed_dir, s)) is not None}
    synced = {s: df for s in _SENSORS if (df := _load_sensor_df(synced_dir, s)) is not None}

    if len(parsed) < 2:
        raise FileNotFoundError(f"Need sporsa and arduino under parsed: {parsed_dir}")
    if len(synced) < 2:
        raise FileNotFoundError(f"Need sporsa and arduino under synced: {synced_dir}")

    if output_path is None:
        output_path = synced_dir / "thesis_parsed_vs_synced_acc.png"

    fig = plt.figure(figsize=(12, 7.0))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    gs_before = gs_outer[0].subgridspec(2, 1, hspace=0.0)

    ax_sp = fig.add_subplot(gs_before[0, 0])
    ax_ar = fig.add_subplot(gs_before[1, 0], sharex=ax_sp)
    ax_merged = fig.add_subplot(gs_outer[1, 0])

    ax_sp.set_title("Before sync", fontsize=10, pad=4)
    ax_sp.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
    ax_merged.set_title("After sync", fontsize=10, pad=4)

    _plot_acc_single(ax_sp, parsed["sporsa"], sensor="sporsa")
    _plot_acc_single(ax_ar, parsed["arduino"], sensor="arduino")
    _plot_acc_merged(ax_merged, synced)

    fig.suptitle(
        f"{recording_name} — accelerometer: before (separate) vs after (merged)",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.992, top=0.91, bottom=0.055)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("Thesis plot written: %s", project_relative_path(output_path))
    return output_path


def _build_activity_signal(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build standardized Δ||acc|| activity signal per §4.3.3 of the thesis.

    Returns ``(timestamps_s, activity)`` where ``timestamps_s`` is absolute time
    in seconds derived from the ``timestamp`` column (milliseconds).
    """
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols:
        return ts / 1000.0, np.zeros(len(ts))
    arr = df[acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(arr ** 2, axis=1))
    diff = np.diff(norm, prepend=norm[0])
    mu = float(np.nanmean(diff))
    sigma = float(np.nanstd(diff))
    if sigma < 1e-9:
        return ts / 1000.0, np.zeros_like(diff)
    return ts / 1000.0, (diff - mu) / sigma


def plot_sync_method_hierarchy(
    *,
    output_path: Path | None = None,
) -> Path:
    """Structured comparison table of the four synchronization methods (§4.3.4).

    Produces the "synchronization comparison framework" table suggested in the
    thesis: one row per method ordered from strongest to weakest, with columns
    for offset source, drift source, required anchors, key assumption, and role
    in the analysis.

    Output
    ------
    ``thesis_sync_method_hierarchy.png`` in the current directory when no
    *output_path* is given, or the explicit *output_path*.
    """
    if output_path is None:
        output_path = Path("thesis_sync_method_hierarchy.png")

    # ------------------------------------------------------------------
    # Table data — ordered strongest → weakest following the thesis hierarchy
    # ------------------------------------------------------------------
    col_headers = ["Method", "Offset source", "Drift source", "Anchors needed",
                   "Key assumption", "Role in analysis"]

    rows = [
        [
            "Multi-anchor\nprotocol",
            "Opening calibration\nsequence",
            "Calibration window\nspan (open → close)",
            "≥ 2\n(open + close)",
            "Devices co-located\nduring protocol",
            "Reference — closest\napproximation to truth",
        ],
        [
            "One-anchor\nadaptive",
            "Opening calibration\nsequence",
            "Activity-signal\nwindowed fit\n(post-anchor)",
            "1\n(opening only)",
            "Riding-signal motion\ncorrespondence holds\nafter anchor",
            "Drift responds to\ncurrent recording;\navoids future-anchor data",
        ],
        [
            "One-anchor\nprior",
            "Opening calibration\nsequence",
            "Pre-characterised\nclock-drift prior\n(~300 ppm)",
            "1\n(opening only)",
            "Prior drift\ngeneralises to\ncurrent session",
            "Avoids fitting drift\nfrom riding signals;\nstable but less adaptive",
        ],
        [
            "Signal-only",
            "Activity-signal\ncross-correlation\n(full recording)",
            "Activity-signal\nwindowed drift fit\n(full recording)",
            "None",
            "Riding signals are\nsimultaneous across\nboth sensors",
            "Weakest; used to\nassess limits of\npost-hoc sync",
        ],
        [
            "Signal-only\n(no drift)",
            "Activity-signal\ncross-correlation\n(full recording)",
            "None\n(zero drift assumed)",
            "None",
            "No accumulating\nclock drift over\nsession",
            "Baseline degenerate\ncase; offset only",
        ],
    ]

    # ------------------------------------------------------------------
    # Row colours — highlight the hierarchy level
    # ------------------------------------------------------------------
    row_colors = [
        ["#d4edda"] * len(col_headers),   # green — reference
        ["#cce5ff"] * len(col_headers),   # blue — one-anchor adaptive
        ["#e2d9f3"] * len(col_headers),   # purple — one-anchor prior
        ["#fff3cd"] * len(col_headers),   # yellow — signal-only
        ["#f8d7da"] * len(col_headers),   # red — weakest / degenerate
    ]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="upper center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 3.5)

    # Bold header row
    for col in range(len(col_headers)):
        tbl[(0, col)].set_facecolor("#343a40")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    # Column width hints — must sum to <= 1.0 to avoid overflow
    col_widths = [0.12, 0.16, 0.17, 0.10, 0.22, 0.22]
    for col, w in enumerate(col_widths):
        for row in range(len(rows) + 1):
            tbl[(row, col)].set_width(w)

    ax.set_title(
        "Synchronization method hierarchy — offset source, drift source, and role",
        fontsize=11,
        pad=8,
    )

    # Legend chips — placed below table with extra bottom margin
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc="#d4edda", ec="gray", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, fc="#cce5ff", ec="gray", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, fc="#e2d9f3", ec="gray", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, fc="#fff3cd", ec="gray", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, fc="#f8d7da", ec="gray", lw=0.8),
    ]
    legend_labels = [
        "Multi-anchor protocol (reference)",
        "One-anchor adaptive",
        "One-anchor prior",
        "Signal-only",
        "Signal-only, no drift (degenerate)",
    ]
    ax.legend(legend_items, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8,
              framealpha=0.8, handlelength=1.2, handleheight=1.0)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Thesis plot written: %s", output_path)
    return output_path


def plot_sync_example_result(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Four-panel example synchronization result (§4.3.4 / §4.3.5).

    Illustrates the synchronization process for one representative recording:

    - **Panel 1 – Before sync (opening region)**: first ~120 s, each sensor at
      own t=0.  Opening anchor marked on each sensor's clock.
    - **Panel 2 – After offset only (opening region)**: same 120 s window, Arduino
      shifted by offset so both share SPORSA's absolute timeline.  Anchor peaks
      now align.  Displays recording-relative offset (small number).
    - **Panel 3 – After full sync (last 100 s)**: zoom to tail of recording using
      synced CSVs (offset + drift applied), showing residual drift was removed.
    - **Panel 4 – After full sync (full recording)**: full duration, both sensors
      on shared timeline using synced CSVs.

    Output
    ------
    ``synced/thesis_sync_example_result.png``
    """
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    synced_dir = recording_stage_dir(recording_name, "synced")
    sync_json = synced_dir / "sync_info.json"

    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed directory not found: {parsed_dir}")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")
    if not sync_json.exists():
        raise FileNotFoundError(f"sync_info.json not found: {sync_json}")

    sporsa_parsed = _load_sensor_df(parsed_dir, "sporsa")
    arduino_parsed = _load_sensor_df(parsed_dir, "arduino")
    sporsa_synced = _load_sensor_df(synced_dir, "sporsa")
    arduino_synced = _load_sensor_df(synced_dir, "arduino")

    for name, df in [("parsed/sporsa", sporsa_parsed), ("parsed/arduino", arduino_parsed),
                     ("synced/sporsa", sporsa_synced), ("synced/arduino", arduino_synced)]:
        if df is None:
            raise FileNotFoundError(f"Missing sensor CSV: {name}")

    info = json.loads(sync_json.read_text(encoding="utf-8"))
    offset_s = float(info.get("offset_seconds", 0.0))
    drift_sps = float(info.get("drift_seconds_per_second", 0.0))
    cal_block = info.get("calibration") if isinstance(info.get("calibration"), dict) else None

    if output_path is None:
        output_path = synced_dir / "thesis_sync_example_result.png"

    # ------------------------------------------------------------------
    # Build activity signals
    # ------------------------------------------------------------------
    sp_ts, sp_act = _build_activity_signal(sporsa_parsed)
    ar_ts, ar_act = _build_activity_signal(arduino_parsed)
    sp_ts_syn, sp_act_syn = _build_activity_signal(sporsa_synced)
    ar_ts_syn, ar_act_syn = _build_activity_signal(arduino_synced)

    # Panel 1: each sensor at own t=0
    sp_rel = sp_ts - sp_ts[0]
    ar_rel = ar_ts - ar_ts[0]

    # Panel 2: Arduino shifted by offset, shared absolute timeline
    ar_ts_offset_only = ar_ts + offset_s
    t0_abs = min(sp_ts[0], ar_ts_offset_only[0])
    sp_offset_rel = sp_ts - t0_abs
    ar_offset_rel = ar_ts_offset_only - t0_abs

    # Recording-relative offset: how many seconds earlier/later Arduino started
    # relative to SPORSA (positive = Arduino started later than SPORSA).
    rec_rel_offset_s = (ar_ts[0] + offset_s) - sp_ts[0]

    # Panels 3 & 4: synced CSVs on shared timeline
    t0_syn = min(sp_ts_syn[0], ar_ts_syn[0])
    sp_syn_rel = sp_ts_syn - t0_syn
    ar_syn_rel = ar_ts_syn - t0_syn

    # ------------------------------------------------------------------
    # Calibration anchor positions
    # ------------------------------------------------------------------
    opening_ar_rel: float | None = None
    opening_sp_rel: float | None = None
    opening_sp_on_abs: float | None = None   # for panel 2 shared axis
    opening_ar_on_abs: float | None = None
    if cal_block:
        opening = cal_block.get("opening") or {}
        t_tgt = opening.get("t_tgt_s")
        anchor_offset = opening.get("offset_s")
        if t_tgt is not None and anchor_offset is not None:
            t_tgt = float(t_tgt)
            anchor_offset = float(anchor_offset)
            # Panel 1: each sensor's own relative time
            opening_ar_rel = t_tgt - ar_ts[0]
            opening_sp_rel = (t_tgt + anchor_offset) - sp_ts[0]
            # Panel 2: on the shared absolute timeline (t0_abs subtracted)
            opening_sp_on_abs = (t_tgt + anchor_offset) - t0_abs
            opening_ar_on_abs = (t_tgt + offset_s) - t0_abs

    # ------------------------------------------------------------------
    # Figure — 4 panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    lw = 0.75
    lw_full = 0.5      # lighter weight for the full-recording panel
    alpha = 0.82
    sp_color = _SENSOR_COLORS["sporsa"]
    ar_color = _SENSOR_COLORS["arduino"]
    clip = 5.0          # clip extreme activity values for readability
    zoom_s = 120.0      # first N seconds for panels 1 & 2
    tail_s = 100.0      # last N seconds for panel 3

    def _anchor_vline(ax: plt.Axes, x: float, color: str, label: str, text_x_offset: float = 1.5) -> None:
        """Draw a dashed anchor line and label it at a fixed y-fraction of clipped range."""
        ax.axvline(x, color=color, ls="--", lw=1.3, alpha=0.75, zorder=3)
        # Use hardcoded y in data coordinates (fraction of clip range), not ax.get_ylim()
        y_top = clip * 0.72     # upper third of the clipped range
        y_bot = -clip * 0.72    # lower third
        y_pos = y_top if color == sp_color else y_bot
        va = "top" if color == sp_color else "bottom"
        ax.text(x + text_x_offset, y_pos, label, fontsize=7, color=color,
                va=va, ha="left", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # --- Panel 1: Before sync — opening region (first 120 s) ---
    ax = axes[0]
    xsp, ysp = filter_valid_plot_xy(sp_rel, np.clip(sp_act, -clip, clip))
    xar, yar = filter_valid_plot_xy(ar_rel, np.clip(ar_act, -clip, clip))
    ax.plot(xsp, ysp, lw=lw, color=sp_color, alpha=alpha, label="SPORSA (own $t=0$)")
    ax.plot(xar, yar, lw=lw, color=ar_color, alpha=alpha, label="Arduino (own $t=0$)")
    if opening_sp_rel is not None:
        _anchor_vline(ax, opening_sp_rel, sp_color, "opening anchor\n(SPORSA)")
    if opening_ar_rel is not None:
        _anchor_vline(ax, opening_ar_rel, ar_color, "opening anchor\n(Arduino)")
    ax.set_xlim(0, zoom_s)
    ax.set_ylabel("Activity $s_{\\mathrm{acc}}$")
    ax.set_title("Before sync — opening region (each sensor at own $t=0$)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(-clip - 0.5, clip + 0.5)

    # --- Panel 2: After offset only — opening region ---
    ax = axes[1]
    xsp2, ysp2 = filter_valid_plot_xy(sp_offset_rel, np.clip(sp_act, -clip, clip))
    xar2, yar2 = filter_valid_plot_xy(ar_offset_rel, np.clip(ar_act, -clip, clip))
    ax.plot(xsp2, ysp2, lw=lw, color=sp_color, alpha=alpha, label="SPORSA")
    ax.plot(xar2, yar2, lw=lw, color=ar_color, alpha=alpha,
            label=f"Arduino (offset = {rec_rel_offset_s:+.1f} s, no drift)")
    if opening_sp_on_abs is not None:
        _anchor_vline(ax, opening_sp_on_abs, sp_color, "opening anchor\n(SPORSA)")
    if opening_ar_on_abs is not None:
        # Arduino anchor (after offset) should land at nearly the same position as SPORSA anchor;
        # offset the text slightly right so both labels are readable side-by-side.
        _anchor_vline(ax, opening_ar_on_abs, ar_color, "opening anchor\n(Arduino)", text_x_offset=3.5)
    # Set x-limits to the same zoom window (relative to t0_abs)
    sp_start_on_abs = sp_ts[0] - t0_abs
    ax.set_xlim(sp_start_on_abs, sp_start_on_abs + zoom_s)
    ax.set_ylabel("Activity $s_{\\mathrm{acc}}$")
    ax.set_title("After offset alignment — opening region (anchor peaks align)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(-clip - 0.5, clip + 0.5)

    # --- Panel 3: After full sync — last 100 s ---
    ax = axes[2]
    max_end = max(sp_syn_rel[-1] if len(sp_syn_rel) else 0,
                  ar_syn_rel[-1] if len(ar_syn_rel) else 0)
    tail_start = max_end - tail_s
    xsp3, ysp3 = filter_valid_plot_xy(sp_syn_rel, np.clip(sp_act_syn, -clip, clip))
    xar3, yar3 = filter_valid_plot_xy(ar_syn_rel, np.clip(ar_act_syn, -clip, clip))
    ax.plot(xsp3, ysp3, lw=lw, color=sp_color, alpha=alpha, label="SPORSA")
    ax.plot(xar3, yar3, lw=lw, color=ar_color, alpha=alpha,
            label=f"Arduino (drift = {drift_sps * 1e6:+.1f} ppm applied)")
    ax.set_xlim(tail_start, max_end)
    ax.set_ylabel("Activity $s_{\\mathrm{acc}}$")
    sync_method = info.get("sync_method", "?")
    ax.set_title(
        f"After full sync ({sync_method}) — last {tail_s:.0f} s (drift correction visible)", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(-clip - 0.5, clip + 0.5)

    # --- Panel 4: After full sync — full recording ---
    ax = axes[3]
    ax.plot(xsp3, ysp3, lw=lw_full, color=sp_color, alpha=alpha, label="SPORSA")
    ax.plot(xar3, yar3, lw=lw_full, color=ar_color, alpha=alpha, label="Arduino (synced)")
    ax.set_ylabel("Activity $s_{\\mathrm{acc}}$")
    ax.set_xlabel("Time from recording start (s)")
    ax.set_title("After full sync — complete recording", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(-clip - 0.5, clip + 0.5)

    fig.suptitle(f"{recording_name} — Example synchronization result", fontsize=12, y=1.005)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Thesis plot written: %s", project_relative_path(output_path))
    return output_path


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.thesis_plots",
        description=(
            "Thesis figures for the synchronization section.\n\n"
            "Usage:\n"
            "  hierarchy                          — method comparison table\n"
            "  sync-example <recording_name>      — 3-panel sync result\n"
            "  acc <recording_name>               — before/after acc figure\n"
            "  <recording_name>                   — shorthand for 'acc'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("figure_or_recording",
                        help="Figure type ('hierarchy', 'sync-example', 'acc') or recording name.")
    parser.add_argument("recording_name", nargs="?", default=None,
                        help="Recording folder name (required for sync-example and acc).")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    figure_or_rec = args.figure_or_recording.strip()
    extra_rec = args.recording_name.strip() if args.recording_name else None

    if figure_or_rec == "hierarchy":
        out = plot_sync_method_hierarchy()
        print(f"Saved → {out}")
        return

    if figure_or_rec == "sync-example":
        rec = extra_rec
        if not rec:
            parser.error("sync-example requires a recording_name argument.")
        out = plot_sync_example_result(rec)
        print(f"Saved → {out}")
        return

    if figure_or_rec == "acc":
        rec = extra_rec
        if not rec:
            parser.error("acc requires a recording_name argument.")
    else:
        # Treat positional as recording name → legacy acc behaviour
        rec = figure_or_rec

    out = plot_parsed_vs_synced_imu(rec)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
