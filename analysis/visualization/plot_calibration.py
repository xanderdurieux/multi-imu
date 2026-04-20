"""Calibration-stage visualizations (schema v2 — protocol-aware).

Only norm-based and parameter-space figures are exported.  Per-axis body- or
world-frame time series are intentionally omitted because the calibration
stage applies a single static body-to-world rotation and does not track
orientation over time; axis-resolved values after the alignment window are
not physically interpretable without the orientation stage.  See the
docstring of :mod:`calibration.core` for details.

Functions
---------
plot_calibration_raw_vs_calibrated
    Norm comparison (|acc|, |gyro|) raw → calibrated per sensor.
plot_calibration_parameters
    Gyro bias bars, alignment rotation matrix, gravity vector per sensor.
plot_calibration_protocol
    Protocol-detection overview: opening sequence, mount transition, alignment window.
plot_calibration_stage
    Master entry point — calls all of the above for a section directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import read_csv, resolve_data_dir
from common.signals import vector_norm
from visualization._utils import (
    ACC_COLS,
    SENSOR_COLORS,
    SENSORS,
    filter_valid_plot_xy,
    load_json,
    save_figure,
    timestamps_to_relative_seconds,
)

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Plot 1: Raw vs calibrated — individual axes + norms
# ---------------------------------------------------------------------------

def plot_calibration_raw_vs_calibrated(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Per-sensor norm comparison (|acc|, |gyro|) before vs after calibration.

    Per-axis body-frame signals are intentionally omitted — see the module
    docstring.

    Output
    ------
    ``calibrated/<sensor>_calibration_comparison.png``
    """
    selected = sensors or list(SENSORS)
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

        ts_raw = timestamps_to_relative_seconds(df_raw["timestamp"])
        ts_cal = timestamps_to_relative_seconds(df_cal["timestamp"])

        acc_raw = vector_norm(df_raw, [c for c in ("ax", "ay", "az") if c in df_raw.columns])
        gyro_raw = vector_norm(df_raw, [c for c in ("gx", "gy", "gz") if c in df_raw.columns])
        acc_cal = vector_norm(df_cal, [c for c in ("ax", "ay", "az") if c in df_cal.columns])
        gyro_cal = vector_norm(df_cal, [c for c in ("gx", "gy", "gz") if c in df_cal.columns])

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
        out_paths.append(save_figure(fig, out_a))

    return out_paths


# ---------------------------------------------------------------------------
# Plot 2: Calibration parameters — gyro bias and alignment rotation matrix
# ---------------------------------------------------------------------------

def plot_calibration_parameters(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> Path | None:
    """Gyro bias bars, alignment rotation matrix and gravity estimate per sensor.

    Output
    ------
    ``calibrated/calibration_parameters.png``
    """
    cal_data = load_json(section_dir / "calibrated" / "calibration.json") or {}
    if not cal_data:
        log.warning("No calibration.json found in %s", section_dir.name)
        return None

    selected = sensors or list(SENSORS)
    intrinsics_block = cal_data.get("intrinsics", {})
    alignment_block = cal_data.get("alignment", {})
    sensors_present = [s for s in selected if s in intrinsics_block or s in alignment_block]
    if not sensors_present:
        return None

    n = len(sensors_present)
    fig, axes_grid = plt.subplots(3, n, figsize=(4 * n + 1, 8), squeeze=False)
    fig.suptitle(
        f"{section_dir.name} — calibration parameters"
        f"  (protocol_detected={cal_data.get('protocol_detected', '?')})",
        fontsize=11,
    )

    axes_labels = ["x", "y", "z"]
    bar_x = np.arange(3)

    for s_idx, sensor in enumerate(sensors_present):
        intr = intrinsics_block.get(sensor, {})
        aln = alignment_block.get(sensor, {})
        gyro_bias = intr.get("gyro_bias", [0, 0, 0])
        rot_mat = aln.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        grav_est = aln.get("gravity_estimate", [0, 0, _G])
        yaw_src = aln.get("yaw_source", "?")
        yaw_conf = aln.get("yaw_confidence", 0.0)
        color = SENSOR_COLORS.get(sensor, "steelblue")

        # Row 0: gyro bias
        ax_gyro = axes_grid[0, s_idx]
        ax_gyro.bar(bar_x, gyro_bias, color=color, alpha=0.8)
        ax_gyro.axhline(0, color="gray", lw=0.7, ls="--")
        ax_gyro.set_xticks(bar_x)
        ax_gyro.set_xticklabels(axes_labels)
        ax_gyro.set_ylabel("Bias (deg/s)")
        intr_q = intr.get("quality", "?")
        ax_gyro.set_title(f"{sensor} — gyro bias  [intrinsics quality={intr_q}]")
        ax_gyro.grid(axis="y", alpha=0.3)

        # Row 1: alignment rotation matrix
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
                    j, i, f"{R[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(R[i, j]) < 0.7 else "white",
                )
        fig.colorbar(im, ax=ax_rot)
        ax_rot.set_title(
            f"{sensor} — alignment rotation  "
            f"[yaw={yaw_src} conf={yaw_conf:.2f}]"
        )

        # Row 2: gravity estimate (body frame)
        ax_grav = axes_grid[2, s_idx]
        g = np.array(grav_est, dtype=float)
        ax_grav.bar(bar_x, g, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.8)
        ax_grav.axhline(_G, color="gray", lw=0.8, ls="--")
        ax_grav.axhline(-_G, color="gray", lw=0.8, ls="--")
        ax_grav.set_xticks(bar_x)
        ax_grav.set_xticklabels(axes_labels)
        ax_grav.set_ylabel("m/s²")
        residual = aln.get("gravity_residual_ms2", float("nan"))
        ax_grav.set_title(f"{sensor} — gravity (body frame)  [residual={residual:.3f} m/s²]")
        ax_grav.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = section_dir / "calibrated" / "calibration_parameters.png"
    return save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3: Protocol detection overview
# ---------------------------------------------------------------------------

def plot_calibration_protocol(
    section_dir: Path,
    *,
    sensors: list[str] | None = None,
) -> list[Path]:
    """Protocol-detection overview per sensor.

    For each sensor shows the acc norm with annotated regions:
    - Opening pre-tap static (bias estimation window)
    - Tap cluster
    - Opening post-tap static
    - Mount transition gap (Arduino only)
    - Alignment window (post-mount for Arduino, opening for Sporsa)

    Output
    ------
    ``calibrated/<sensor>_protocol.png``
    """
    cal_data = load_json(section_dir / "calibrated" / "calibration.json") or {}
    if not cal_data:
        log.warning("No calibration.json found in %s — skipping protocol plot", section_dir.name)
        return []

    selected = sensors or list(SENSORS)
    out_paths: list[Path] = []
    cal_dir = section_dir / "calibrated"

    opening_block = cal_data.get("opening_sequence") or {}
    alignment_block = cal_data.get("alignment", {})
    protocol_detected = cal_data.get("protocol_detected", False)

    for sensor in selected:
        opening_seq = opening_block.get(sensor) or {}
        raw_csv = section_dir / f"{sensor}.csv"
        if not raw_csv.exists():
            continue
        df = read_csv(raw_csv)
        if "timestamp" not in df.columns:
            continue

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        ts = df["timestamp"].to_numpy(dtype=float)
        t0 = ts[0]
        ts_s = (ts - t0) / 1000.0

        acc_cols = [c for c in ACC_COLS if c in df.columns]
        if not acc_cols:
            continue
        acc_norm = vector_norm(df, acc_cols)

        fig, ax = plt.subplots(figsize=(15, 4))
        color = SENSOR_COLORS.get(sensor, "steelblue")
        ax.plot(ts_s, acc_norm, lw=0.7, color=color, alpha=0.8, label="|acc|")
        ax.axhline(_G, color="gray", lw=0.8, ls="--", label=f"g = {_G} m/s²")

        def _shade(start_ms: float, end_ms: float, facecolor: str, label: str, alpha: float = 0.18) -> None:
            x0 = (start_ms - t0) / 1000.0
            x1 = (end_ms - t0) / 1000.0
            if x0 < x1:
                ax.axvspan(x0, x1, facecolor=facecolor, alpha=alpha, label=label)

        if protocol_detected and opening_seq:
            _shade(opening_seq.get("pre_static_start_ms", 0),
                   opening_seq.get("pre_static_end_ms", 0),
                   "green", "pre-tap static (bias)")
            _shade(opening_seq.get("post_static_start_ms", 0),
                   opening_seq.get("post_static_end_ms", 0),
                   "lime", "post-tap static")

            # Mark tap peaks
            tap_times_ms = opening_seq.get("tap_times_ms") or []
            if tap_times_ms:
                tap_ts_s: list[float] = []
                tap_vals: list[float] = []
                for tap_ms in tap_times_ms:
                    idx = int(np.abs(ts - float(tap_ms)).argmin())
                    tap_ts_s.append((ts[idx] - t0) / 1000.0)
                    tap_vals.append(float(acc_norm[idx]))
                ax.scatter(tap_ts_s, tap_vals, color="red", s=25, zorder=5, label="taps")
            else:
                tap_indices = opening_seq.get("tap_cluster", [])
                if tap_indices:
                    tap_ts_s = [(ts[i] - t0) / 1000.0 for i in tap_indices if 0 <= i < len(ts)]
                    tap_vals = [acc_norm[i] for i in tap_indices if 0 <= i < len(acc_norm)]
                    ax.scatter(tap_ts_s, tap_vals, color="red", s=25, zorder=5, label="taps")

        # Alignment window
        aln = alignment_block.get(sensor, {})
        win_start = aln.get("alignment_window_start_ms")
        win_end = aln.get("alignment_window_end_ms")
        if win_start is not None and win_end is not None and win_end > win_start:
            _shade(win_start, win_end, "blue", "alignment window", alpha=0.15)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|acc| (m/s²)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2, lw=0.4)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

        yaw_src = aln.get("yaw_source", "?")
        yaw_conf = aln.get("yaw_confidence", 0.0)
        residual = aln.get("gravity_residual_ms2", float("nan"))
        detected_str = "detected" if protocol_detected else "NOT detected"
        ax.set_title(
            f"{section_dir.name} / {sensor} — protocol {detected_str}  "
            f"| yaw={yaw_src} ({yaw_conf:.2f})  residual={residual:.3f} m/s²"
        )

        fig.tight_layout()
        out_path = cal_dir / f"{sensor}_protocol.png"
        out_paths.append(save_figure(fig, out_path))

    return out_paths


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
    - ``calibration_parameters.png``            — biases + rotation matrix
    - ``<sensor>_protocol.png``                 — protocol detection overview
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
        out_paths += plot_calibration_protocol(section_dir, sensors=sensors)
    except Exception as exc:
        log.warning("Protocol plot failed for %s: %s", section_dir.name, exc)

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
        choices=list(SENSORS),
        dest="sensors",
        help="Limit to one or more sensors (repeat option).",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "raw_vs_cal", "parameters", "protocol"],
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
        "protocol": lambda: plot_calibration_protocol(section_dir, sensors=args.sensors),
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
