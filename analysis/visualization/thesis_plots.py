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

from common.paths import iter_sections_for_recording, read_csv, recording_stage_dir
from sync.signals import build_activity_signal
from sync.stream_io import VECTOR_AXES
from sync.sync_info_format import flatten_sync_info_dict
from visualization._utils import (
    SENSOR_COLORS,
    filter_valid_plot_xy,
    load_sensor_df,
    save_figure,
)
from visualization.plot_sync import plot_before_after_imu

log = logging.getLogger(__name__)


def plot_parsed_vs_synced_imu(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Accelerometer |acc|: parsed sensors separately, then merged synced pair.

    Thin wrapper around :func:`visualization.plot_sync.plot_before_after_imu`
    kept for backwards compatibility.  New code should call that function directly.
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if output_path is None:
        output_path = synced_dir / "thesis_parsed_vs_synced_acc.png"
    return plot_before_after_imu(recording_name, output_path=output_path)


def _build_activity_signal(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build standardized Δ||acc|| activity signal per §4.3.3 of the thesis.

    Delegates to :func:`sync.signals.build_activity_signal` so the
    z-scored diff-of-norms logic is defined in one place.

    Returns ``(timestamps_s, activity)`` where ``timestamps_s`` is absolute time
    in seconds derived from the ``timestamp`` column (milliseconds).
    """
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    activity, _mode = build_activity_signal(
        df,
        vector_axes=VECTOR_AXES,
        signal_mode="acc_norm_diff",
    )
    return ts / 1000.0, activity


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
    return save_figure(fig, output_path, dpi=150)


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

    sporsa_parsed = load_sensor_df(parsed_dir, "sporsa")
    arduino_parsed = load_sensor_df(parsed_dir, "arduino")
    sporsa_synced = load_sensor_df(synced_dir, "sporsa")
    arduino_synced = load_sensor_df(synced_dir, "arduino")

    for name, df in [("parsed/sporsa", sporsa_parsed), ("parsed/arduino", arduino_parsed),
                     ("synced/sporsa", sporsa_synced), ("synced/arduino", arduino_synced)]:
        if df is None:
            raise FileNotFoundError(f"Missing sensor CSV: {name}")

    info = flatten_sync_info_dict(
        json.loads(sync_json.read_text(encoding="utf-8"))
    ) or {}
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
        anchors = cal_block.get("anchors")
        opening = anchors[0] if isinstance(anchors, list) and anchors else {}
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
    sp_color = SENSOR_COLORS["sporsa"]
    ar_color = SENSOR_COLORS["arduino"]
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
    return save_figure(fig, output_path, dpi=150)


def plot_sync_landmarks_and_sections_timeline(
    recording_name: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Single-recording timeline with calibration landmarks and section boundaries.

    The figure overlays:
    - synced activity traces (SPORSA + Arduino) on a shared timeline,
    - detected calibration anchors from ``synced/all_methods.json`` (or
      ``synced/sync_info.json`` fallback),
    - vertical start/end boundaries of split sections for this recording.

    Output
    ------
    ``synced/thesis_sync_landmarks_sections_timeline.png``
    """
    synced_dir = recording_stage_dir(recording_name, "synced")
    if not synced_dir.is_dir():
        raise FileNotFoundError(f"Synced directory not found: {synced_dir}")

    sporsa_synced = load_sensor_df(synced_dir, "sporsa")
    arduino_synced = load_sensor_df(synced_dir, "arduino")
    if sporsa_synced is None or arduino_synced is None:
        raise FileNotFoundError(f"Missing synced sensor CSVs under {synced_dir}")

    if output_path is None:
        output_path = synced_dir / "thesis_sync_landmarks_sections_timeline.png"

    # ------------------------------------------------------------------
    # Activity signals on shared synced timeline
    # ------------------------------------------------------------------
    sp_ts_syn, sp_act_syn = _build_activity_signal(sporsa_synced)
    ar_ts_syn, ar_act_syn = _build_activity_signal(arduino_synced)
    t0_syn = min(sp_ts_syn[0], ar_ts_syn[0])
    sp_rel = sp_ts_syn - t0_syn
    ar_rel = ar_ts_syn - t0_syn

    # ------------------------------------------------------------------
    # Calibration anchors from all_methods.json (fallback: sync_info.json)
    # Convert to "seconds from synced start" using t_ref_s.
    # ------------------------------------------------------------------
    anchor_times_rel: list[float] = []
    anchor_scores: list[float | None] = []
    all_methods_path = synced_dir / "all_methods.json"
    sync_info_path = synced_dir / "sync_info.json"
    cal_block: dict | None = None

    if all_methods_path.exists():
        payload = json.loads(all_methods_path.read_text(encoding="utf-8"))
        shared = payload.get("shared") if isinstance(payload, dict) else None
        if isinstance(shared, dict):
            candidate = shared.get("calibration")
            if isinstance(candidate, dict):
                cal_block = candidate
    if cal_block is None and sync_info_path.exists():
        payload = flatten_sync_info_dict(json.loads(sync_info_path.read_text(encoding="utf-8"))) or {}
        candidate = payload.get("calibration")
        if isinstance(candidate, dict):
            cal_block = candidate

    if cal_block:
        anchors = cal_block.get("anchors")
        if isinstance(anchors, list):
            for a in anchors:
                if not isinstance(a, dict):
                    continue
                t_ref = a.get("t_ref_s")
                if t_ref is None:
                    continue
                try:
                    anchor_times_rel.append(float(t_ref) - t0_syn)
                    score_v = a.get("score")
                    anchor_scores.append(float(score_v) if score_v is not None else None)
                except (TypeError, ValueError):
                    continue

    # ------------------------------------------------------------------
    # Section boundaries from section CSV extents.
    # ------------------------------------------------------------------
    section_bounds: list[tuple[str, float, float]] = []
    for sec_dir in iter_sections_for_recording(recording_name):
        sec_sporsa = sec_dir / "sporsa.csv"
        if not sec_sporsa.exists():
            continue
        try:
            sec_df = read_csv(sec_sporsa)
            if sec_df.empty or "timestamp" not in sec_df.columns:
                continue
            ts = pd.to_numeric(sec_df["timestamp"], errors="coerce").to_numpy(dtype=float)
            ts = ts[np.isfinite(ts)]
            if len(ts) < 2:
                continue
            section_bounds.append((sec_dir.name, float(ts[0]) / 1000.0 - t0_syn, float(ts[-1]) / 1000.0 - t0_syn))
        except Exception as exc:
            log.warning("Failed to read section timestamps for %s: %s", sec_dir.name, exc)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5.5))
    clip = 5.0
    lw = 0.75
    alpha = 0.85
    sp_color = SENSOR_COLORS["sporsa"]
    ar_color = SENSOR_COLORS["arduino"]

    xsp, ysp = filter_valid_plot_xy(sp_rel, np.clip(sp_act_syn, -clip, clip))
    xar, yar = filter_valid_plot_xy(ar_rel, np.clip(ar_act_syn, -clip, clip))
    ax.plot(xsp, ysp, lw=lw, color=sp_color, alpha=alpha, label="SPORSA (synced)")
    ax.plot(xar, yar, lw=lw, color=ar_color, alpha=alpha, label="Arduino (synced)")

    # Section boundaries.
    for i, (sec_name, start_s, end_s) in enumerate(sorted(section_bounds, key=lambda x: x[1])):
        ax.axvline(start_s, color="#7f8c8d", lw=0.9, alpha=0.45, zorder=2)
        ax.axvline(end_s, color="#7f8c8d", lw=0.9, alpha=0.45, zorder=2)
        ax.axvspan(start_s, end_s, color="#95a5a6", alpha=0.06, zorder=1)
        mid_s = 0.5 * (start_s + end_s)
        ax.text(mid_s, clip + 0.2, sec_name, fontsize=7, color="#555555", ha="center", va="bottom")

    # Calibration landmarks.
    for idx, t_anchor in enumerate(anchor_times_rel):
        score = anchor_scores[idx] if idx < len(anchor_scores) else None
        lbl = f"cal {idx + 1}"
        if score is not None and np.isfinite(score):
            lbl += f" (score {score:.2f})"
        ax.axvline(t_anchor, color="#d35400", ls="--", lw=1.4, alpha=0.8, zorder=3)
        ax.text(
            t_anchor + 0.6,
            -clip - 0.15,
            lbl,
            fontsize=7,
            color="#d35400",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
            zorder=4,
        )

    ax.set_ylim(-clip - 0.5, clip + 0.7)
    ax.set_xlabel("Time from recording start (s)")
    ax.set_ylabel("Activity $s_{\\mathrm{acc}}$")
    ax.set_title(
        "Synchronized recording with calibration landmarks and section boundaries",
        fontsize=11,
    )
    ax.grid(alpha=0.22, lw=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, output_path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m visualization.thesis_plots",
        description=(
            "Thesis figures for the synchronization section.\n\n"
            "Usage:\n"
            "  --figure hierarchy\n"
            "  --figure sync-example --recording <recording_name>\n"
            "  --figure sync-sections --recording <recording_name>\n"
            "  --figure acc --recording <recording_name>\n"
            "  --recording <recording_name>   (defaults to --figure acc)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--figure",
        choices=("hierarchy", "sync-example", "sync-sections", "acc"),
        default="acc",
        help="Figure type to generate (default: acc).",
    )
    parser.add_argument(
        "--recording",
        dest="recording_name",
        default=None,
        help="Recording folder name (required for sync-example, sync-sections, and acc).",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    figure = args.figure.strip()
    rec = args.recording_name.strip() if args.recording_name else None

    if figure == "hierarchy":
        out = plot_sync_method_hierarchy()
        print(f"Saved → {out}")
        return

    if figure == "sync-example":
        if not rec:
            parser.error("sync-example requires --recording.")
        out = plot_sync_example_result(rec)
        print(f"Saved → {out}")
        return

    if figure == "sync-sections":
        if not rec:
            parser.error("sync-sections requires --recording.")
        out = plot_sync_landmarks_and_sections_timeline(rec)
        print(f"Saved → {out}")
        return

    if not rec:
        parser.error("acc requires --recording.")
    out = plot_parsed_vs_synced_imu(rec)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
