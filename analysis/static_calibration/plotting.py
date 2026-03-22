"""Plotting helpers for static calibration recordings (time series + analysis interval)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .imu_static import AXES, FACE_ORDER, GRAVITY


AXIS_COLORS = {
    "x": "#dc2626",
    "y": "#0284c7",
    "z": "#16a34a",
}

FACE_COLORS = {
    "x+": "#b91c1c",
    "x-": "#ef4444",
    "y+": "#0369a1",
    "y-": "#38bdf8",
    "z+": "#166534",
    "z-": "#4ade80",
}


def _relative_time_seconds(df: pd.DataFrame) -> np.ndarray:
    timestamps = df["timestamp"].to_numpy(dtype=float)
    return (timestamps - timestamps[0]) / 1000.0


def _selected_interval_seconds(record: dict, df: pd.DataFrame) -> tuple[float, float]:
    origin = float(df["timestamp"].iloc[0])
    start = (float(record["t_start_ms"]) - origin) / 1000.0
    end = (float(record["t_end_ms"]) - origin) / 1000.0
    return start, end


def plot_recordings_overview(
    per_recording: list[dict],
    parsed_by_stem: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Plot all calibration recordings in one overview figure."""

    records = sorted(per_recording, key=lambda r: r["stem"])
    figure, axes = plt.subplots(
        len(records),
        2,
        figsize=(15, max(10, 2.8 * len(records))),
        constrained_layout=True,
        sharex=False,
    )
    if len(records) == 1:
        axes = np.array([axes])

    for row, record in enumerate(records):
        stem = record["stem"]
        df = parsed_by_stem[stem]
        color = FACE_COLORS[record["dominant_face"]]

        acc = df.dropna(subset=["ax", "ay", "az"]).copy()
        if not acc.empty:
            time_s = _relative_time_seconds(acc)
            start_s, end_s = _selected_interval_seconds(record, acc)
            for axis in AXES:
                axes[row, 0].plot(time_s, acc[f"a{axis}"], color=AXIS_COLORS[axis], linewidth=0.9, label=axis.upper())
            axes[row, 0].axvspan(start_s, end_s, color=color, alpha=0.12)
            axes[row, 0].axhline(GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
            axes[row, 0].axhline(-GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
            axes[row, 0].axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)

        gyro = df.dropna(subset=["gx", "gy", "gz"]).copy()
        if not gyro.empty:
            time_s = _relative_time_seconds(gyro)
            start_s, end_s = _selected_interval_seconds(record, gyro)
            for axis in AXES:
                axes[row, 1].plot(time_s, gyro[f"g{axis}"], color=AXIS_COLORS[axis], linewidth=0.9)
            axes[row, 1].axvspan(start_s, end_s, color=color, alpha=0.12)
            axes[row, 1].axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)

        axes[row, 0].set_ylabel(f"{stem}\nacc [m/s²]")
        axes[row, 1].set_ylabel("gyro [deg/s]")
        axes[row, 0].set_title(
            f"{stem} · {record['dominant_face']} · interval {record['selected_duration_seconds']:.1f} s",
            loc="left",
            fontsize=10,
        )
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 1].grid(True, alpha=0.25)

    axes[0, 0].legend(loc="upper right", ncol=3, frameon=False)
    axes[0, 0].set_title("Accelerometer axes with analysis interval", loc="left", fontsize=11)
    axes[0, 1].set_title("Gyroscope axes with analysis interval", loc="left", fontsize=11)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    figure.suptitle("Static calibration recordings overview", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_recording_details(
    per_recording: list[dict],
    parsed_by_stem: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """Write one detailed plot per calibration recording."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for record in sorted(per_recording, key=lambda r: r["stem"]):
        stem = record["stem"]
        df = parsed_by_stem[stem]
        acc = df.dropna(subset=["ax", "ay", "az"]).copy()
        gyro = df.dropna(subset=["gx", "gy", "gz"]).copy()
        color = FACE_COLORS[record["dominant_face"]]

        figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=False)

        if not acc.empty:
            time_s = _relative_time_seconds(acc)
            start_s, end_s = _selected_interval_seconds(record, acc)
            for axis in AXES:
                axes[0].plot(time_s, acc[f"a{axis}"], color=AXIS_COLORS[axis], linewidth=1.0, label=axis.upper())
            axes[0].axvspan(start_s, end_s, color=color, alpha=0.12)
            axes[0].axhline(GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
            axes[0].axhline(-GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
            axes[0].axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)
            axes[0].legend(loc="upper right", ncol=3, frameon=False)

        if not gyro.empty:
            time_s = _relative_time_seconds(gyro)
            start_s, end_s = _selected_interval_seconds(record, gyro)
            for axis in AXES:
                axes[1].plot(time_s, gyro[f"g{axis}"], color=AXIS_COLORS[axis], linewidth=1.0)
            axes[1].axvspan(start_s, end_s, color=color, alpha=0.12)
            axes[1].axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)

        axes[0].set_title(f"{stem} · {record['dominant_face']} · accelerometer", loc="left")
        axes[1].set_title("Gyroscope", loc="left")
        axes[0].set_ylabel("Acceleration [m/s²]")
        axes[1].set_ylabel("Angular velocity [deg/s]")
        axes[1].set_xlabel("Time [s]")
        axes[0].grid(True, alpha=0.25)
        axes[1].grid(True, alpha=0.25)

        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(path)

    return output_paths


def _analysis_window_column(df: pd.DataFrame, record: dict[str, Any], column: str) -> np.ndarray:
    t0 = float(record["t_start_ms"])
    t1 = float(record["t_end_ms"])
    sub = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
    return pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy(dtype=float)


def plot_calibration_parameters(
    per_recording: list[dict[str, Any]],
    calibration: dict[str, Any],
    output_path: Path,
    *,
    parsed_by_stem: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Bars for fitted parameters and per-recording means; boxplots for spread in each analysis window."""

    records = sorted(per_recording, key=lambda r: r["stem"])
    fig = plt.figure(figsize=(14, 17), constrained_layout=True)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.0, 1.0, 1.1, 1.1, 1.25, 1.25])

    x3 = np.arange(len(AXES))
    acc_bias = [calibration["accelerometer"]["bias"][a] for a in AXES]
    acc_scale = [calibration["accelerometer"]["scale"][a] for a in AXES]
    gyro_bias = [calibration["gyroscope"]["bias_deg_s"][a] for a in AXES]

    # accelerometer bias
    ax_acc_bias = fig.add_subplot(gs[0, 0])
    ax_acc_bias.bar(x3, acc_bias, color=[AXIS_COLORS[a] for a in AXES])
    ax_acc_bias.set_xticks(x3)
    ax_acc_bias.set_xticklabels([a.upper() for a in AXES])
    ax_acc_bias.axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_acc_bias.set_title("Estimated accelerometer bias")
    ax_acc_bias.set_ylabel("m/s²")
    ax_acc_bias.grid(True, axis="y", alpha=0.25)

    # accelerometer scale
    ax_acc_scale = fig.add_subplot(gs[0, 1])
    ax_acc_scale.bar(x3, acc_scale, color=[AXIS_COLORS[a] for a in AXES])
    ax_acc_scale.set_xticks(x3)
    ax_acc_scale.set_xticklabels([a.upper() for a in AXES])
    ax_acc_scale.axhline(1.0, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_acc_scale.set_title("Estimated accelerometer scale")
    ax_acc_scale.set_ylabel("—")
    ax_acc_scale.grid(True, axis="y", alpha=0.25)

    # gyroscope bias
    ax_gyro_bias = fig.add_subplot(gs[1, 0])
    ax_gyro_bias.bar(x3, gyro_bias, color=[AXIS_COLORS[a] for a in AXES])
    ax_gyro_bias.set_xticks(x3)
    ax_gyro_bias.set_xticklabels([a.upper() for a in AXES])
    ax_gyro_bias.axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_gyro_bias.set_title("Estimated gyroscope bias")
    ax_gyro_bias.set_ylabel("deg/s")
    ax_gyro_bias.grid(True, axis="y", alpha=0.25)

    # face counts
    ax_faces = fig.add_subplot(gs[1, 1])
    fc = calibration.get("face_counts", {})
    xf = np.arange(len(FACE_ORDER))
    counts = [fc.get(face, 0) for face in FACE_ORDER]
    ax_faces.bar(xf, counts, color=[FACE_COLORS[f] for f in FACE_ORDER])
    ax_faces.set_xticks(xf)
    ax_faces.set_xticklabels(list(FACE_ORDER), rotation=30, ha="right")
    ax_faces.set_title("Recordings per classified face")
    ax_faces.set_ylabel("count")
    ax_faces.grid(True, axis="y", alpha=0.25)

    # accelerometer and gyroscope means
    ax_acc_means = fig.add_subplot(gs[2, :])
    ax_gyro_means = fig.add_subplot(gs[3, :])

    n = len(records)
    if n > 0:
        width = min(0.22, 0.8 / max(3, n))
        positions = np.arange(n)
        for i, axis in enumerate(AXES):
            vals = [r["mean_acc"][axis] for r in records]
            offset = (i - 1) * width
            ax_acc_means.bar(
                positions + offset,
                vals,
                width,
                label=axis.upper(),
                color=AXIS_COLORS[axis],
            )
        ax_acc_means.set_xticks(positions)
        ax_acc_means.set_xticklabels([r["stem"] for r in records], rotation=35, ha="right")
        ax_acc_means.axhline(GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
        ax_acc_means.axhline(-GRAVITY, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35)
        ax_acc_means.axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)
        ax_acc_means.set_title("Per-recording mean accelerometer (trimmed window)")
        ax_acc_means.set_ylabel("m/s²")
        ax_acc_means.legend(loc="upper right", ncol=3, frameon=False)
        ax_acc_means.grid(True, axis="y", alpha=0.25)

        for i, axis in enumerate(AXES):
            vals = [r["mean_gyro"][axis] for r in records]
            offset = (i - 1) * width
            ax_gyro_means.bar(
                positions + offset,
                vals,
                width,
                label=axis.upper(),
                color=AXIS_COLORS[axis],
            )
        ax_gyro_means.set_xticks(positions)
        ax_gyro_means.set_xticklabels([r["stem"] for r in records], rotation=35, ha="right")
        ax_gyro_means.axhline(0.0, color="#111827", linestyle=":", linewidth=0.8, alpha=0.35)
        ax_gyro_means.set_title("Per-recording mean gyroscope (trimmed window)")
        ax_gyro_means.set_ylabel("deg/s")
        ax_gyro_means.legend(loc="upper right", ncol=3, frameon=False)
        ax_gyro_means.grid(True, axis="y", alpha=0.25)

    else:
        ax_acc_means.set_title("Per-recording mean accelerometer (no recordings)")
        ax_gyro_means.set_title("Per-recording mean gyroscope (no recordings)")

    fig.suptitle("Calibration parameters: fit, means", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
