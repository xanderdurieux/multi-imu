"""Static, thesis-ready figures for feature-level and aggregate analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style
from ._utils import mask_valid_plot_x

_META_COLUMNS = frozenset(
    {
        "section",
        "window_start_s",
        "window_end_s",
        "window_center_s",
        "scenario_label",
        "recording",
    }
)


def _numeric_feature_columns(df: pd.DataFrame, max_cols: int = 24) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        return []
    # Prefer higher-variance columns for crowded heatmaps / PCA
    var_series = df[cols].var(numeric_only=True, skipna=True).sort_values(ascending=False)
    ordered = [c for c in var_series.index if np.isfinite(var_series[c]) and var_series[c] > 0]
    return ordered[:max_cols] if len(ordered) > max_cols else ordered


def _pca_2d(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (scores Nx2, loadings Kx2, evr1, evr2). X is N x K, already finite."""
    n, k = X.shape
    if n < 3 or k < 2:
        z = np.zeros((n, 2))
        return z, np.zeros((k, 2)), 0.0, 0.0
    Xc = X - np.mean(X, axis=0)
    std = np.std(Xc, axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    Xz = Xc / std
    u, s, vt = np.linalg.svd(Xz, full_matrices=False)
    scores = u[:, :2] * s[:2]
    loadings = vt[:2, :].T
    total_var = np.sum(s**2)
    evr = (s[:2] ** 2) / total_var if total_var > 0 else np.array([0.0, 0.0])
    return scores, loadings, float(evr[0]), float(evr[1]) if len(evr) > 1 else 0.0


def plot_feature_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Pearson correlation matrix for top-variance numeric features."""
    apply_matplotlib_thesis_style()
    feat_cols = _numeric_feature_columns(df, max_cols=20)
    if len(feat_cols) < 2:
        return
    sub = df[feat_cols].dropna(how="all")
    if len(sub) < 5:
        return
    c = sub.corr(numeric_only=True, min_periods=3)
    if c.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    im = ax.imshow(c.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(feat_cols)))
    ax.set_yticks(np.arange(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_yticklabels(feat_cols)
    ax.set_title("Feature correlation (Pearson)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Correlation")
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_distributions_grid(df: pd.DataFrame, out_path: Path, n_features: int = 6) -> None:
    """Histograms of the highest-variance features (z-scored for display)."""
    apply_matplotlib_thesis_style()
    feat_cols = _numeric_feature_columns(df, max_cols=max(n_features, 12))
    if not feat_cols:
        return
    take = feat_cols[:n_features]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, col in zip(axes_flat, take):
        v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(v) < 2:
            ax.set_visible(False)
            continue
        v = (v - np.nanmean(v)) / max(np.nanstd(v), 1e-9)
        ax.hist(v, bins=min(30, max(10, len(v) // 5)), color=THESIS_COLORS[0], edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("z-score")
        ax.set_ylabel("Count")
    for j in range(len(take), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Distributions of dominant features (standardized)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca_scatter(
    df: pd.DataFrame,
    out_path: Path,
    *,
    color_key: str | None = "recording",
) -> None:
    """2D PCA of numeric features; points colored by metadata column if present."""
    apply_matplotlib_thesis_style()
    feat_cols = _numeric_feature_columns(df, max_cols=32)
    if len(feat_cols) < 3:
        return
    work = df[feat_cols].copy()
    mask = work.notna().all(axis=1)
    if mask.sum() < 5:
        return
    X = work.loc[mask].to_numpy(dtype=float)
    scores, _loadings, ev1, ev2 = _pca_2d(X)

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    if color_key and color_key in df.columns:
        labels = df.loc[mask, color_key].astype(str)
        cats = pd.unique(labels)
        for i, cat in enumerate(cats):
            m = (labels == cat).to_numpy()
            color = THESIS_COLORS[i % len(THESIS_COLORS)]
            ax.scatter(scores[m, 0], scores[m, 1], s=22, alpha=0.75, label=str(cat), c=color, edgecolors="none")
        ax.legend(loc="best", fontsize=8, title=color_key.replace("_", " "), markerscale=1.2)
    else:
        ax.scatter(scores[:, 0], scores[:, 1], s=22, alpha=0.75, c=THESIS_COLORS[0], edgecolors="none")

    ax.set_xlabel(f"PC1 ({ev1 * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev2 * 100:.1f}% var.)")
    ax.set_title("Principal components of motion features")
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_sensor_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    """Time series of cross-sensor agreement metrics when columns exist."""
    need = {"window_center_s", "acc_norm_corr", "pitch_corr"}
    if not need.issubset(df.columns):
        return
    apply_matplotlib_thesis_style()
    t = pd.to_numeric(df["window_center_s"], errors="coerce")
    c1 = pd.to_numeric(df["acc_norm_corr"], errors="coerce")
    c2 = pd.to_numeric(df["pitch_corr"], errors="coerce")
    tx = t.to_numpy(dtype=float)
    xm = mask_valid_plot_x(tx)
    c1a = c1.to_numpy(dtype=float)
    c2a = c2.to_numpy(dtype=float)
    mask1 = xm & np.isfinite(c1a)
    mask2 = xm & np.isfinite(c2a)
    if not (mask1.any() or mask2.any()):
        return

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(tx[mask1], c1a[mask1], color=THESIS_COLORS[0], linewidth=1.0, label="acc_norm_corr")
    axes[0].set_ylabel(r"$\rho$ (acc norm)")
    axes[0].axhline(0, color="#999999", linewidth=0.6)
    axes[0].set_title("Cross-sensor correlation over time")
    axes[1].plot(tx[mask2], c2a[mask2], color=THESIS_COLORS[1], linewidth=1.0, label="pitch_corr")
    axes[1].set_ylabel(r"$\rho$ (pitch)")
    axes[1].set_xlabel("Window center [s]")
    axes[1].axhline(0, color="#999999", linewidth=0.6)
    fig.savefig(out_path)
    plt.close(fig)


def recording_window_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: recording, n_windows; sorted descending by count."""
    if "recording" not in df.columns:
        return pd.DataFrame(columns=["recording", "n_windows"])
    g = df.groupby("recording", dropna=False).size().reset_index(name="n_windows")
    return g.sort_values("n_windows", ascending=False).reset_index(drop=True)


def plot_recording_coverage(profile: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of window counts per recording (data quantity overview)."""
    if "recording" not in profile.columns or "n_windows" not in profile.columns:
        return
    apply_matplotlib_thesis_style()
    rec = profile["recording"].astype(str)
    n = profile["n_windows"].to_numpy(dtype=int)
    order = np.argsort(n)[::-1]
    rec = rec.iloc[order]
    n = n[order]

    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(rec))), constrained_layout=True)
    y = np.arange(len(rec))
    ax.barh(y, n, color=THESIS_COLORS[0], alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(rec)
    ax.invert_yaxis()
    ax.set_xlabel("Number of feature windows")
    ax.set_title("Recording coverage (feature windows)")
    fig.savefig(out_path)
    plt.close(fig)


def write_static_insight_bundle(df: pd.DataFrame, static_dir: Path) -> None:
    """Write all static figures to ``static_dir``."""
    static_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    plot_feature_correlation_heatmap(df, static_dir / "feature_correlation.png")
    plot_feature_distributions_grid(df, static_dir / "feature_distributions.png")
    plot_pca_scatter(df, static_dir / "feature_pca.png", color_key="recording" if "recording" in df.columns else None)
    plot_cross_sensor_timeseries(df, static_dir / "cross_sensor_timeseries.png")
    if "recording" in df.columns:
        prof = recording_window_counts(df)
        plot_recording_coverage(prof, static_dir / "recording_coverage.png")
