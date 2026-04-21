"""Shared constants and helpers for the plot_exports_* family."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

DPI = 150
LABEL_CMAP = "tab10"

QUALITY_COLORS = {"good": "#2ecc71", "marginal": "#f39c12", "poor": "#e74c3c", "": "#95a5a6"}
ALL_SYNC_METHODS = ["multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"]
METHOD_COLORS = {
    "multi_anchor": "#2ecc71",
    "one_anchor_adaptive": "#e74c3c",
    "one_anchor_prior": "#e67e22",
    "signal_only": "#3498db",
}
ALL_ORIENTATION_METHODS = ["mahony"]
ORIENTATION_METHOD_COLORS = {
    "mahony": "#9467bd",
    "unknown": "#95a5a6",
}

META_COLS = frozenset({
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "scenario_label",
    "overall_quality_label",
    "overall_quality_score",
    "quality_tier",
    "calibration_quality",
    "sync_confidence",
    "window_n_samples_sporsa",
    "window_n_samples_arduino",
    "window_valid_ratio_sporsa",
})


def label_colors(labels: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap(LABEL_CMAP)
    return {lbl: cmap(i % 10) for i, lbl in enumerate(sorted(labels))}


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def short_section(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else name


def short_recording(name: str) -> str:
    parts = name.split("_")
    return parts[-1] if parts else name


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario_label" not in df.columns:
        return pd.DataFrame()
    return df[df["scenario_label"].notna() & (df["scenario_label"] != "unlabeled")].copy()


def session_from_row(name: str) -> str:
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1][:1] == "r" and parts[1][1:].isdigit():
        return parts[0]
    return name
