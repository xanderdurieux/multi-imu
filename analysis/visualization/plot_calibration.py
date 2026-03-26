"""Plots for the calibrated sensor stage.

Produces two figure types per recording:

1. **Per-sensor world-frame plot** (``<sensor>_world.png``):
   Accelerometer, gyroscope and magnetometer signals after rotation to the
   ENU world frame.  Axis labels are annotated as East / North / Up.
   A dashed reference line at +9.81 m/s² is drawn on the vertical (Up / Z)
   accelerometer panel to visualise gravity alignment quality.
   A text-box summary of the calibration quality metrics is included.

2. **Two-sensor comparison plot** (``sporsa_vs_arduino_world.png`` /
   ``_norm.png``):
   Overlays the calibrated Sporsa (handlebar) and Arduino (helmet) streams.
   Because both are now expressed in the same ENU world frame the overlay
   directly shows how similarly the two sensors perceive the same world-frame
   signals during shared motion periods.

CLI::

    python -m visualization.plot_calibration 2026-02-26_r5
    python -m visualization.plot_calibration 2026-02-26_r5 --no-comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir
from ._utils import mask_valid_plot_x

log = logging.getLogger(__name__)

_STAGE = "calibrated"
_SENSORS = ("sporsa", "arduino")

# ENU world-frame axis labels
_WORLD_LABELS = {
    "acc":  {"cols": ("ax", "ay", "az"),  "axis_names": ("East (X)", "North (Y)", "Up (Z)"),  "unit": "m/s²"},
    "gyro": {"cols": ("gx", "gy", "gz"),  "axis_names": ("ω_X",      "ω_Y",       "ω_Z"),      "unit": "deg/s"},
    "mag":  {"cols": ("mx", "my", "mz"),  "axis_names": ("M_X",      "M_Y",       "M_Z"),      "unit": "µT"},
}

_GRAVITY_MS2 = 9.81    # expected az in ENU after calibration (specific force points up = +Z)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_calibration_json(recording_name: str) -> dict:
    cal_path = recording_stage_dir(recording_name, _STAGE) / "calibration.json"
    if not cal_path.exists():
        return {}
    try:
        return json.loads(cal_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _quality_text(cal: dict, sensor: str) -> str:
    """Build a one-block annotation string from calibration quality fields."""
    s = cal.get(sensor, {})
    if not s:
        return ""
    q = s.get("quality", {})
    lines = [
        f"n_static = {q.get('n_static_samples', '?')}",
        f"gravity |g| = {s.get('gravity_magnitude_m_per_s2', float('nan')):.3f} m/s²",
        f"residual = {q.get('gravity_residual_m_per_s2', float('nan')):.4f} m/s²",
        f"yaw_cal = {q.get('yaw_calibrated', '?')}",
    ]
    if "n_mag_samples" in q:
        lines.insert(1, f"n_mag = {q['n_mag_samples']}")
    return "\n".join(lines)


def _time_axis(df: pd.DataFrame) -> pd.Series:
    ts = df["timestamp"].astype(float)
    return (ts - ts.iloc[0]) / 1000.0


# ---------------------------------------------------------------------------
# Per-sensor world-frame plot
# ---------------------------------------------------------------------------

def plot_sensor_world(
    recording_name: str,
    sensor_name: str,
    *,
    cal: dict | None = None,
) -> Path | None:
    """Plot one calibrated sensor in ENU world frame.

    Returns the path of the saved PNG, or ``None`` if the CSV was not found.
    """
    try:
        csv_path = find_sensor_csv(recording_name, _STAGE, sensor_name)
    except FileNotFoundError:
        log.warning("[%s/%s] no CSV found for %s — skipping.", recording_name, _STAGE, sensor_name)
        return None

    df = load_dataframe(csv_path)
    if df.empty:
        return None

    if cal is None:
        cal = _load_calibration_json(recording_name)

    time_s = _time_axis(df)
    tx = time_s.to_numpy(dtype=float)
    xm = mask_valid_plot_x(tx)
    stage_dir = recording_stage_dir(recording_name, _STAGE)

    sensor_types = ["acc", "gyro", "mag"]
    n_rows = len(sensor_types)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.5 * n_rows),
        sharex=True,
        constrained_layout=True,
    )

    colors = ["#e05c44", "#4c9be8", "#44c17a"]  # X=red-ish, Y=blue, Z=green

    for row, stype in enumerate(sensor_types):
        meta = _WORLD_LABELS[stype]
        cols = meta["cols"]
        axis_names = meta["axis_names"]
        unit = meta["unit"]

        has_data = all(c in df.columns for c in cols)

        for col_idx, (col, label) in enumerate(zip(cols, axis_names)):
            ax = axes[row][col_idx]

            if has_data:
                series = df[col].astype(float)
                valid = series.notna().to_numpy(dtype=bool) & xm
                ax.plot(
                    tx[valid], series[valid].to_numpy(dtype=float, copy=False),
                    color=colors[col_idx], linewidth=0.6, alpha=0.85,
                )

                # Gravity reference line on the vertical (Up/Z) acc panel
                if stype == "acc" and col_idx == 2:
                    ax.axhline(
                        _GRAVITY_MS2, color="k", linestyle="--",
                        linewidth=1.0, alpha=0.5, label=f"g = {_GRAVITY_MS2} m/s²",
                    )
                    ax.legend(fontsize=7, loc="upper right")

                # Zero reference on gyro
                if stype == "gyro":
                    ax.axhline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.4)

            ax.set_title(f"{stype.upper()} — {label}", fontsize=9)
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(True, alpha=0.25)
            if row == n_rows - 1:
                ax.set_xlabel("Time [s]", fontsize=8)

    # Calibration quality text box on acc_z panel
    quality_txt = _quality_text(cal, sensor_name)
    if quality_txt:
        axes[0][2].text(
            0.02, 0.97, quality_txt,
            transform=axes[0][2].transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    fig.suptitle(
        f"{recording_name} / {_STAGE} — {sensor_name}  (ENU world frame)",
        fontsize=11,
    )

    out_path = stage_dir / f"{sensor_name}_world.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{_STAGE}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Two-sensor comparison in world frame
# ---------------------------------------------------------------------------

def plot_comparison_world(
    recording_name: str,
    *,
    norm: bool = False,
) -> Path | None:
    """Overlay both calibrated sensors in the ENU world frame.

    Parameters
    ----------
    norm:
        When ``True``, plot vector norms instead of individual axes.

    Returns the path of the saved PNG.
    """
    stage_dir = recording_stage_dir(recording_name, _STAGE)

    dfs: dict[str, pd.DataFrame] = {}
    for sensor in _SENSORS:
        try:
            csv_path = find_sensor_csv(recording_name, _STAGE, sensor)
            df = load_dataframe(csv_path)
            if not df.empty:
                dfs[sensor] = df
        except FileNotFoundError:
            pass

    if len(dfs) < 2:
        log.warning(
            "[%s/%s] need both sensors for comparison; found %s.",
            recording_name, _STAGE, list(dfs),
        )
        return None

    df_a = dfs[_SENSORS[0]]
    df_b = dfs[_SENSORS[1]]

    # Normalise time: use shared reference if timestamps overlap, else each to own start
    ts_a = df_a["timestamp"].astype(float)
    ts_b = df_b["timestamp"].astype(float)
    overlap = min(ts_a.max(), ts_b.max()) - max(ts_a.min(), ts_b.min())
    if overlap > 0:
        t0 = min(ts_a.min(), ts_b.min())
        time_a = (ts_a - t0) / 1000.0
        time_b = (ts_b - t0) / 1000.0
    else:
        time_a = (ts_a - ts_a.min()) / 1000.0
        time_b = (ts_b - ts_b.min()) / 1000.0

    ta = time_a.to_numpy(dtype=float)
    tb = time_b.to_numpy(dtype=float)
    mx_a = mask_valid_plot_x(ta)
    mx_b = mask_valid_plot_x(tb)

    sensor_types = ["acc", "gyro"]
    n_cols = 1 if norm else 3

    fig, axes = plt.subplots(
        len(sensor_types), n_cols,
        figsize=(5 * n_cols, 3.5 * len(sensor_types)),
        sharex=False,
        constrained_layout=True,
    )
    if len(sensor_types) == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    sensor_colors = {_SENSORS[0]: "#e05c44", _SENSORS[1]: "#4c9be8"}

    for row, stype in enumerate(sensor_types):
        meta = _WORLD_LABELS[stype]
        cols = meta["cols"]
        axis_names = meta["axis_names"]
        unit = meta["unit"]

        if norm:
            panels = [(None, f"{stype.upper()} norm", unit)]
        else:
            panels = [(c, f"{stype.upper()} — {lbl}", unit) for c, lbl in zip(cols, axis_names)]

        for col_idx, (col, title, ylabel) in enumerate(panels):
            ax = axes[row][col_idx]

            for sensor, df, time_arr, mx in (
                (_SENSORS[0], df_a, ta, mx_a),
                (_SENSORS[1], df_b, tb, mx_b),
            ):
                if not all(c in df.columns for c in cols):
                    continue
                if norm:
                    data_raw = df[list(cols)].astype(float)
                    valid = data_raw.notna().all(axis=1).to_numpy(dtype=bool) & mx
                    series = np.sqrt((data_raw[valid] ** 2).sum(axis=1))
                    ax.plot(time_arr[valid], series, color=sensor_colors[sensor],
                            linewidth=0.7, alpha=0.8, label=sensor)
                else:
                    series = df[col].astype(float)
                    valid = series.notna().to_numpy(dtype=bool) & mx
                    ax.plot(time_arr[valid], series[valid].to_numpy(dtype=float, copy=False),
                            color=sensor_colors[sensor],
                            linewidth=0.7, alpha=0.8, label=sensor)

                # Gravity reference on acc Up (Z) panel
                if stype == "acc" and (norm or col_idx == 2):
                    ref = 9.81 if norm else _GRAVITY_MS2
                    ax.axhline(ref, color="k", linestyle="--", linewidth=0.9,
                               alpha=0.4, label=f"ref = {ref:.2f}")

            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("Time [s]", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25)

    suffix = "_norm" if norm else ""
    fig.suptitle(
        f"{recording_name} / {_STAGE} — {_SENSORS[0]} vs {_SENSORS[1]}"
        f"  (ENU world frame{', norm' if norm else ''})",
        fontsize=11,
    )

    out_path = stage_dir / f"sporsa_vs_arduino_world{suffix}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{_STAGE}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: run all calibration plots for one recording
# ---------------------------------------------------------------------------

def plot_calibration_stage(recording_name: str) -> None:
    """Generate all calibration plots for *recording_name*.

    Produces:
    - ``<sensor>_world.png`` for each sensor found in ``calibrated/``.
    - ``sporsa_vs_arduino_world.png`` (axes comparison).
    - ``sporsa_vs_arduino_world_norm.png`` (norm comparison).
    """
    cal = _load_calibration_json(recording_name)
    for sensor in _SENSORS:
        plot_sensor_world(recording_name, sensor, cal=cal)
    plot_comparison_world(recording_name, norm=False)
    plot_comparison_world(recording_name, norm=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_calibration",
        description=(
            "Plot calibrated sensor outputs in the ENU world frame for one or "
            "more recordings."
        ),
    )
    parser.add_argument(
        "recording_names",
        nargs="+",
        help="One or more recording names (e.g. 2026-02-26_r5).",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the two-sensor comparison plots.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for recording_name in args.recording_names:
        cal = _load_calibration_json(recording_name)
        for sensor in _SENSORS:
            plot_sensor_world(recording_name, sensor, cal=cal)
        if not args.no_comparison:
            plot_comparison_world(recording_name, norm=False)
            plot_comparison_world(recording_name, norm=True)


if __name__ == "__main__":
    main()
