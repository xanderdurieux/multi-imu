"""Parsed-stage plots for thesis reporting.

Aggregate plots (read from parsed_recording_summary.csv)
---------------------------------------------------------
plot_session_bar_chart(summary_df, output_path)
    Grouped bar chart of recording durations and sample counts.

plot_interval_jitter(summary_df, output_path)
    Per-recording per-sensor median Δt with IQR error bars.

plot_packet_loss(summary_df, output_path)
    Estimated BLE packet loss rate per recording (Arduino only).

Per-recording diagnostic plots
-------------------------------
plot_timestamp_continuity(dfs, output_path)
    Timestamp progression over sample index per sensor; flags large gaps.

plot_interval_distribution(dfs, output_path)
    Histogram of inter-sample intervals per sensor with median marker.

plot_packet_loss_received(arduino_df, output_path)
    Device Δt vs host-received Δt; highlights BLE dropout events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._utils import SENSOR_COLORS, save_figure


_DPI = 200

_SENSOR_LABELS = {"sporsa": "Bike (SPORSA)", "arduino": "Rider (Arduino)"}

_QUALITY_COLORS = {
    "good":    "#66BB6A",
    "usable":  "#FFA726",
    "limited": "#ef5350",
}


def _quality_badge(category: str) -> str:
    return {"good": "G", "usable": "U", "limited": "L"}.get(
        str(category).strip().lower(), "?"
    )


def _quality_badge_color(category: str) -> str:
    return _QUALITY_COLORS.get(str(category).strip().lower(), "#90A4AE")


# ===========================================================================
# Aggregate plots (reporting stage)
# ===========================================================================

def plot_session_bar_chart(summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Grouped bar chart of recording durations and sample counts."""
    if summary_df.empty:
        raise ValueError("summary_df must not be empty")

    df = summary_df.copy().reset_index(drop=True)
    labels = df["recording_name"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    duration_ax = axes[0]
    duration_sporsa = pd.to_numeric(df["sporsa_duration_s"], errors="coerce").to_numpy(dtype=float)
    duration_arduino = pd.to_numeric(df["arduino_duration_s"], errors="coerce").to_numpy(dtype=float)
    duration_ax.bar(
        x - width / 2, np.nan_to_num(duration_sporsa, nan=0.0), width=width,
        color=SENSOR_COLORS["sporsa"], alpha=0.9, label="SPORSA",
    )
    duration_ax.bar(
        x + width / 2, np.nan_to_num(duration_arduino, nan=0.0), width=width,
        color=SENSOR_COLORS["arduino"], alpha=0.9, label="Arduino",
    )
    duration_ax.set_ylabel("Duration (s)")
    duration_ax.set_title("Recording durations")
    duration_ax.grid(axis="y", alpha=0.25, lw=0.5)
    duration_ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    duration_ax.spines["top"].set_visible(False)
    duration_ax.spines["right"].set_visible(False)

    count_ax = axes[1]
    sporsa_samples = pd.to_numeric(df["sporsa_num_samples"], errors="coerce").to_numpy(dtype=float)
    arduino_samples = pd.to_numeric(df["arduino_num_samples"], errors="coerce").to_numpy(dtype=float)
    count_ax.bar(
        x - width / 2, np.nan_to_num(sporsa_samples, nan=0.0), width=width,
        color=SENSOR_COLORS["sporsa"], alpha=0.9, label="SPORSA",
    )
    count_ax.bar(
        x + width / 2, np.nan_to_num(arduino_samples, nan=0.0), width=width,
        color=SENSOR_COLORS["arduino"], alpha=0.9, label="Arduino",
    )
    count_ax.set_ylabel("Samples")
    count_ax.set_title("Sample counts")
    count_ax.grid(axis="y", alpha=0.25, lw=0.5)
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)

    for idx, category in enumerate(df["quality_category"].tolist()):
        badge = _quality_badge(str(category))
        color = _quality_badge_color(str(category))
        count_ax.text(
            x[idx], 0.98, badge,
            transform=count_ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": color,
                "edgecolor": "#666666",
                "alpha": 0.75,
            },
        )

    count_ax.set_xticks(x)
    count_ax.set_xticklabels(labels, rotation=20, ha="right")
    count_ax.set_xlabel("Recording")

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)


def plot_interval_jitter(summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Per-recording per-sensor median inter-sample interval with IQR error bars."""
    df = summary_df.copy().reset_index(drop=True)
    labels = df["recording_name"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5))

    for sensor, offset in (("sporsa", -width / 2), ("arduino", width / 2)):
        median_col = f"{sensor}_median_dt_ms"
        iqr_col = f"{sensor}_iqr_dt_ms"
        if median_col not in df.columns:
            continue
        medians = pd.to_numeric(df[median_col], errors="coerce").to_numpy(dtype=float)
        iqrs = (
            pd.to_numeric(df[iqr_col], errors="coerce").to_numpy(dtype=float)
            if iqr_col in df.columns
            else np.zeros(len(df))
        )
        iqrs = np.nan_to_num(iqrs, nan=0.0)
        valid = np.isfinite(medians)
        color = SENSOR_COLORS[sensor]
        label = "SPORSA" if sensor == "sporsa" else "Arduino"
        ax.bar(x[valid] + offset, medians[valid], width=width, color=color, alpha=0.8, label=label)
        ax.errorbar(
            x[valid] + offset, medians[valid],
            yerr=iqrs[valid] / 2.0,
            fmt="none", color="black", capsize=3, lw=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median Δt (ms)")
    ax.set_title("Inter-sample interval — median with IQR error bars")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)


def plot_packet_loss(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    """Estimated BLE packet loss rate per recording (Arduino only)."""
    col = "arduino_estimated_missing_rate_pct"
    if col not in summary_df.columns:
        return None

    df = summary_df.copy().reset_index(drop=True)
    labels = df["recording_name"].tolist()
    x = np.arange(len(labels), dtype=float)
    loss_pct = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    colors = [
        "#ef5350" if v > 10.0 else "#FFA726" if v > 2.0 else "#66BB6A"
        for v in np.nan_to_num(loss_pct, nan=0.0)
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, np.nan_to_num(loss_pct, nan=0.0), color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(10.0, color="#ef5350", lw=1.0, ls="--", label="10 % threshold (quality cutoff)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Estimated missing samples (%)")
    ax.set_title("Packet loss proxy — Arduino BLE stream")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)


# ===========================================================================
# Per-recording diagnostic plots (parser stage)
# ===========================================================================

def plot_timestamp_continuity(
    dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path | None:
    """Timestamp progression over sample index; large gaps highlighted in red."""
    sensors = [s for s in ("sporsa", "arduino") if s in dfs]
    if not sensors:
        return None

    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 3.5 * len(sensors)), squeeze=False)

    for ax, sensor in zip(axes[:, 0], sensors):
        df = dfs[sensor]
        if "timestamp" not in df.columns:
            ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — no timestamp column")
            continue

        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().reset_index(drop=True)
        if ts.empty:
            ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — no valid timestamps")
            continue

        t_s = (ts - ts.iloc[0]) / 1000.0
        color = SENSOR_COLORS[sensor]

        ax.plot(t_s.index, t_s.values, lw=0.7, color=color, alpha=0.8)

        # Mark samples where the gap to the next sample is > 3× median
        dt = ts.diff().iloc[1:]
        finite_dt = dt[np.isfinite(dt)]
        if not finite_dt.empty:
            median_dt = float(np.median(finite_dt[finite_dt > 0])) if (finite_dt > 0).any() else 0.0
            if median_dt > 0:
                gap_mask = dt > 3.0 * median_dt
                gap_idx = gap_mask[gap_mask].index
                if len(gap_idx):
                    ax.scatter(
                        gap_idx, t_s.loc[gap_idx],
                        color="#ef5350", s=25, zorder=5, label=f"{len(gap_idx)} gap(s)",
                    )
                    ax.legend(fontsize=8, framealpha=0.85)

        ax.set_ylabel("Time from start (s)")
        ax.set_xlabel("Sample index")
        ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — timestamp continuity")
        ax.grid(alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)


def plot_interval_distribution(
    dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path | None:
    """Histogram of inter-sample intervals per sensor with median marker."""
    sensors = [s for s in ("sporsa", "arduino") if s in dfs]
    if not sensors:
        return None

    fig, axes = plt.subplots(1, len(sensors), figsize=(6 * len(sensors), 4), squeeze=False)

    for ax, sensor in zip(axes[0], sensors):
        df = dfs[sensor]
        if "timestamp" not in df.columns:
            ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — no timestamp")
            continue

        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().sort_values().reset_index(drop=True)
        dt = ts.diff().iloc[1:].dropna().to_numpy(dtype=float)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.size == 0:
            ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — no valid intervals")
            continue

        median_dt = float(np.median(dt))
        # Clip extreme outliers for display clarity (> 10× median)
        dt_clipped = dt[dt <= 10.0 * median_dt]

        ax.hist(dt_clipped, bins=60, color=SENSOR_COLORS[sensor], alpha=0.8, edgecolor="white", lw=0.3)
        ax.axvline(median_dt, color="black", lw=1.5, ls="--", label=f"median {median_dt:.1f} ms")
        ax.set_xlabel("Inter-sample interval (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{_SENSOR_LABELS.get(sensor, sensor)} — Δt distribution")
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)


def plot_packet_loss_received(
    arduino_df: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    """Device Δt vs host-received Δt; highlights BLE dropout events."""
    if "timestamp" not in arduino_df.columns or "timestamp_received" not in arduino_df.columns:
        return None

    ts_device = pd.to_numeric(arduino_df["timestamp"], errors="coerce")
    ts_recv = pd.to_numeric(arduino_df["timestamp_received"], errors="coerce")

    valid = ts_device.notna() & ts_recv.notna()
    ts_device = ts_device[valid].reset_index(drop=True)
    ts_recv = ts_recv[valid].reset_index(drop=True)

    if len(ts_device) < 2:
        return None

    dt_device = ts_device.diff().iloc[1:].to_numpy(dtype=float)
    dt_recv = ts_recv.diff().iloc[1:].to_numpy(dtype=float)
    idx = np.arange(1, len(ts_device))

    finite_device = dt_device[(dt_device > 0) & np.isfinite(dt_device)]
    median_device = float(np.median(finite_device)) if finite_device.size > 0 else 1.0

    # BLE dropout: received gap much larger than device interval
    dropout_mask = (dt_recv > 2.0 * median_device) & np.isfinite(dt_recv)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax_dev = axes[0]
    ax_dev.plot(idx, dt_device, lw=0.6, color=SENSOR_COLORS["arduino"], alpha=0.7, label="Device Δt")
    ax_dev.axhline(median_device, color="black", lw=1.0, ls="--", label=f"median {median_device:.1f} ms")
    ax_dev.set_ylabel("Device Δt (ms)")
    ax_dev.set_title("Arduino — device clock intervals")
    ax_dev.legend(fontsize=8, framealpha=0.85)
    ax_dev.grid(alpha=0.2, lw=0.5)
    ax_dev.spines["top"].set_visible(False)
    ax_dev.spines["right"].set_visible(False)

    ax_recv = axes[1]
    ax_recv.plot(idx, dt_recv, lw=0.6, color="#5C6BC0", alpha=0.7, label="Received Δt")
    if dropout_mask.any():
        ax_recv.scatter(
            idx[dropout_mask], dt_recv[dropout_mask],
            color="#ef5350", s=20, zorder=5, label=f"{dropout_mask.sum()} dropout(s)",
        )
    ax_recv.axhline(median_device, color="black", lw=1.0, ls="--", label=f"device median {median_device:.1f} ms")
    ax_recv.set_ylabel("Received Δt (ms)")
    ax_recv.set_xlabel("Sample index")
    ax_recv.set_title("Arduino — host-received BLE intervals (dropouts in red)")
    ax_recv.legend(fontsize=8, framealpha=0.85)
    ax_recv.grid(alpha=0.2, lw=0.5)
    ax_recv.spines["top"].set_visible(False)
    ax_recv.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_path, dpi=_DPI)
