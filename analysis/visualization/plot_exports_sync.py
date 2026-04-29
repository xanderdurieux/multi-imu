"""Per-recording and per-session sync audit figures."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._exports_common import (
    ALL_SYNC_METHODS,
    DPI,
    METHOD_COLORS,
    session_from_row,
    short_recording,
)
from visualization._utils import save_figure

log = logging.getLogger(__name__)


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe or "unknown_session"


def plot_sync_method_selection(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_method_selection.png"
    if df.empty or "recording_name" not in df.columns or "selected_method" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None
    methods = df["selected_method"].fillna("unknown").tolist()
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.7 + 2), 4))
    ax.bar(range(n), [1] * n, color=colors, edgecolor="white", linewidth=0.5)
    for i, method in enumerate(methods):
        ax.text(i, 0.5, method, ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Selected sync method per recording")
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_method_availability(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_method_availability.png"
    if df.empty or "recording_name" not in df.columns:
        return None
    available_cols = [f"{m}_available" for m in ALL_SYNC_METHODS if f"{m}_available" in df.columns]
    if not available_cols:
        return None
    recordings = df["recording_name"].tolist()
    n_rec = len(recordings)
    method_names = [c.replace("_available", "") for c in available_cols]
    n_methods = len(method_names)
    matrix = np.zeros((n_methods, n_rec), dtype=float)
    for j, col in enumerate(available_cols):
        avail = df[col].fillna(False).tolist()
        for i, val in enumerate(avail):
            matrix[j, i] = 1.0 if bool(val) else 0.0
    fig, ax = plt.subplots(figsize=(max(7, n_rec * 0.6 + 2), max(3, n_methods * 0.6 + 1.5)))
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_rec))
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(method_names, fontsize=9)
    ax.set_title("Sync method availability per recording")
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_correlation_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_correlation_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None
    available_methods = [m for m in ALL_SYNC_METHODS if f"{m}_corr_offset_and_drift" in df.columns]
    if not available_methods and "corr_offset_and_drift" not in df.columns:
        return None
    x = np.arange(n)
    width = 0.8 / max(len(available_methods), 1)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7 + 2), 4))
    if available_methods:
        legend_handles: list[plt.Rectangle] = []
        for i, method in enumerate(available_methods):
            col = f"{method}_corr_offset_and_drift"
            vals = pd.to_numeric(df[col], errors="coerce").tolist()
            offset = (i - len(available_methods) / 2 + 0.5) * width
            color = METHOD_COLORS.get(method, "gray")
            ax.bar(
                x + offset,
                [v if v is not None else 0 for v in vals],
                width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=method))
    else:
        vals = pd.to_numeric(df["corr_offset_and_drift"], errors="coerce").tolist()
        ax.bar(x, [v if v is not None else 0 for v in vals], 0.6, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correlation (offset + drift)")
    ax.set_title("Sync correlation scores per recording and method")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    if available_methods:
        ax.legend(
            handles=legend_handles,
            title="Method",
            loc="upper right",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.8,
        )
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_drift_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_drift_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None

    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None

    available_methods = [
        m for m in ALL_SYNC_METHODS
        if f"{m}_drift_ppm" in df.columns
    ]

    if not available_methods:
        drift_col = "drift_ppm"
        if drift_col not in df.columns:
            if "drift_seconds_per_second" in df.columns:
                df = df.copy()
                df[drift_col] = pd.to_numeric(df["drift_seconds_per_second"], errors="coerce") * 1e6
            else:
                return None
        drift_vals = pd.to_numeric(df[drift_col], errors="coerce").tolist()
        fig, ax = plt.subplots(figsize=(max(7, n * 0.7 + 2), 4))
        colors = ["#e74c3c" if (v is not None and abs(v) > 1000) else "#2980b9" for v in drift_vals]
        ax.bar(range(n), [v if pd.notna(v) else 0.0 for v in drift_vals], color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Drift (ppm)")
        ax.set_title("Selected sync drift per recording")
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.tight_layout()
        return save_figure(fig, out_path, dpi=DPI)

    x = np.arange(n)
    width = 0.8 / len(available_methods)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7 + 2), 4))
    legend_handles: list[plt.Rectangle] = []

    for i, method in enumerate(available_methods):
        col = f"{method}_drift_ppm"
        vals = pd.to_numeric(df[col], errors="coerce").tolist()
        offset = (i - len(available_methods) / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "gray")
        ax.bar(
            x + offset,
            [v if pd.notna(v) else 0.0 for v in vals],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=method))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(400, color="gray", linestyle="--", linewidth=0.7)
    ax.axhline(-400, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Drift (ppm)")
    ax.set_title("Sync drift per recording and method")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(handles=legend_handles, title="Method", loc="upper right", fontsize=8, title_fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_calibration_anchor_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_calibration_anchors_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None

    n_win_col = next(
        (c for c in ("multi_anchor_cal_n_windows", "calibration_cal_n_windows") if c in df.columns),
        None,
    )
    r2_col = next(
        (c for c in ("multi_anchor_cal_fit_r2", "calibration_cal_fit_r2") if c in df.columns),
        None,
    )
    if n_win_col is None and r2_col is None:
        return None

    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None

    n_windows = pd.to_numeric(df[n_win_col], errors="coerce") if n_win_col else pd.Series([np.nan] * n)
    fit_r2 = pd.to_numeric(df[r2_col], errors="coerce") if r2_col else pd.Series([np.nan] * n)
    if n_windows.isna().all() and fit_r2.isna().all():
        return None

    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7 + 2), 6), sharex=True)

    ax = axes[0]
    vals = n_windows.fillna(0.0).tolist()
    ax.bar(x, vals, color="#2ca02c", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title("Calibration anchors used per recording")
    ax.grid(axis="y", alpha=0.3, lw=0.5)

    ax = axes[1]
    vals = fit_r2.fillna(0.0).tolist()
    colors = ["#2ca02c" if v >= 0.2 else "#d62728" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(0.2, color="orange", linestyle="--", linewidth=0.9)
    ax.set_ylabel("R\u00b2")
    ax.set_title("Calibration all-anchor fit quality (R\u00b2)")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_offset_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_offset_overview.png"
    if df.empty or "recording_name" not in df.columns or "offset_seconds" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    offset_vals = pd.to_numeric(df["offset_seconds"], errors="coerce").tolist()
    valid_offsets = [float(v) for v in offset_vals if pd.notna(v)]
    if not valid_offsets:
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(recordings) * 0.7 + 2), 4))
    ax.bar(range(len(recordings)), [v or 0 for v in offset_vals], color="#9b59b6", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(recordings)))
    ax.set_xticklabels([short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Offset (s)")
    ax.set_title("Selected sync time offset per recording")
    min_v = min(valid_offsets)
    max_v = max(valid_offsets)
    span = max_v - min_v
    pad = max(1e-3, span * 0.12)
    ax.set_ylim(min_v - pad, max_v + pad)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_session_corr(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Bar chart of mean selected corr_offset_and_drift per session."""
    out_path = output_dir / "sync_by_session_corr.png"
    if df.empty or "corr_offset_and_drift" not in df.columns:
        return None

    df = df.copy()
    if "session" not in df.columns:
        df["session"] = df["recording_name"].map(session_from_row)

    sessions = sorted(df["session"].unique())
    if len(sessions) < 1:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(sessions) * 1.5 + 2), 4))
    for i, sess in enumerate(sessions):
        sub = df[df["session"] == sess]
        vals = pd.to_numeric(sub["corr_offset_and_drift"], errors="coerce").dropna()
        if vals.empty:
            continue
        mean_v = float(vals.mean())
        ax.bar(i, mean_v, color="#3498db", edgecolor="white", linewidth=0.5, alpha=0.85)
        for _, row in sub.iterrows():
            v = row.get("corr_offset_and_drift")
            if pd.notna(v):
                ax.plot(i, v, "o", color=METHOD_COLORS.get(row.get("selected_method", ""), "#95a5a6"),
                        ms=5, zorder=3)
        ax.text(i, mean_v + 0.01, f"{mean_v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Corr (offset+drift)")
    ax.set_title("Sync correlation by session (bar = mean, dots = individual)")
    ax.axhline(0.2, color="orange", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_method_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Heatmap: rows = recordings, columns = methods, value = corr_offset_and_drift."""
    out_path = output_dir / "sync_method_heatmap.png"
    if df.empty or "recording_name" not in df.columns:
        return None

    corr_cols = [f"{m}_corr_offset_and_drift" for m in ALL_SYNC_METHODS
                 if f"{m}_corr_offset_and_drift" in df.columns]
    if not corr_cols:
        return None

    recordings = df["recording_name"].tolist()
    method_labels = [c.replace("_corr_offset_and_drift", "") for c in corr_cols]
    mat = df[corr_cols].apply(pd.to_numeric, errors="coerce").values

    fig, ax = plt.subplots(figsize=(max(6, len(corr_cols) * 1.8 + 2),
                                    max(4, len(recordings) * 0.45 + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(recordings)))
    ax.set_yticklabels([short_recording(r) for r in recordings], fontsize=8)

    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if np.isfinite(v):
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")

    if "selected_method" in df.columns:
        for ri, sel in enumerate(df["selected_method"]):
            for ci, ml in enumerate(method_labels):
                if sel == ml:
                    ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                               fill=False, edgecolor="black", lw=2))

    plt.colorbar(im, ax=ax, label="Corr (offset+drift)", shrink=0.8)
    ax.set_title("Sync method correlation heatmap (box = selected)")
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_session_strip(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Strip plot: one panel per session, x = recording suffix, y = correlation, color = method."""
    out_path = output_dir / "sync_session_strip.png"
    if df.empty or "corr_offset_and_drift" not in df.columns:
        return None

    df = df.copy()
    if "session" not in df.columns:
        df["session"] = df["recording_name"].map(session_from_row)
    if "recording_suffix" not in df.columns:
        df["recording_suffix"] = df["recording_name"].apply(
            lambda n: n.rsplit("_", 1)[-1] if "_" in n else n
        )

    sessions = sorted(df["session"].unique())
    n_sess = len(sessions)
    if n_sess < 1:
        return None

    fig, axes = plt.subplots(1, n_sess, figsize=(max(5, n_sess * 5), 4),
                             sharey=True, squeeze=False)

    for idx, sess in enumerate(sessions):
        ax = axes[0][idx]
        sub = df[df["session"] == sess].sort_values("recording_suffix")
        x_labels = sub["recording_suffix"].tolist()
        corr_vals = pd.to_numeric(sub["corr_offset_and_drift"], errors="coerce")
        methods = sub["selected_method"].fillna("unknown").tolist() if "selected_method" in sub.columns else ["unknown"] * len(sub)
        colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]
        ax.scatter(range(len(x_labels)), corr_vals, c=colors, s=60, edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.2, color="orange", lw=0.8, ls=":")
        ax.set_title(sess, fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        if idx == 0:
            ax.set_ylabel("Corr (offset+drift)")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, ms=8, label=m)
               for m, c in METHOD_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Sync quality by session (color = selected method)", fontsize=11, y=1.04)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sync_session_drift(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Per-session drift overview: dots per recording colored by method."""
    out_path = output_dir / "sync_by_session_drift.png"
    if df.empty or "drift_ppm" not in df.columns:
        return None

    df = df.copy()
    if "session" not in df.columns:
        df["session"] = df["recording_name"].map(session_from_row)

    sessions = sorted(df["session"].unique())
    fig, ax = plt.subplots(figsize=(max(6, len(sessions) * 1.5 + 2), 4))
    for i, sess in enumerate(sessions):
        sub = df[df["session"] == sess]
        for _, row in sub.iterrows():
            v = row.get("drift_ppm")
            if pd.notna(v):
                method = row.get("selected_method", "unknown")
                ax.plot(i, v, "o", color=METHOD_COLORS.get(method, "#95a5a6"),
                        ms=7, zorder=3, markeredgecolor="black", markeredgewidth=0.4)

    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(400, color="gray", ls="--", lw=0.7, label="\u00b1400 ppm")
    ax.axhline(-400, color="gray", ls="--", lw=0.7)
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Selected drift (ppm)")
    ax.set_title("Sync drift by session")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def run_sync_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if df.empty:
        log.warning("Sync params DataFrame is empty; skipping sync EDA")
        return []
    figures_dir = Path(output_dir) / "figures" / "sync"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    def _add(result):
        if result is not None:
            generated.append(result)

    _add(plot_sync_method_selection(df, figures_dir))
    _add(plot_sync_method_availability(df, figures_dir))
    _add(plot_sync_correlation_overview(df, figures_dir))
    _add(plot_sync_drift_overview(df, figures_dir))
    _add(plot_sync_calibration_anchor_overview(df, figures_dir))
    _add(plot_sync_offset_overview(df, figures_dir))
    _add(plot_sync_session_corr(df, figures_dir))
    _add(plot_sync_method_heatmap(df, figures_dir))
    _add(plot_sync_session_strip(df, figures_dir))
    _add(plot_sync_session_drift(df, figures_dir))

    session_df = df.copy()
    if "session" not in session_df.columns and "recording_name" in session_df.columns:
        session_df["session"] = session_df["recording_name"].map(session_from_row)
    if "session" in session_df.columns:
        sessions_root = figures_dir / "sessions"
        for session in sorted(session_df["session"].dropna().astype(str).unique()):
            sub = session_df[session_df["session"].astype(str) == session].copy()
            if sub.empty:
                continue
            session_dir = sessions_root / _safe_path_part(session)
            session_dir.mkdir(parents=True, exist_ok=True)
            _add(plot_sync_correlation_overview(sub, session_dir))
            _add(plot_sync_method_heatmap(sub, session_dir))
            _add(plot_sync_drift_overview(sub, session_dir))
            _add(plot_sync_offset_overview(sub, session_dir))

    log.info("Sync EDA complete: %d figures", len(generated))
    return generated
