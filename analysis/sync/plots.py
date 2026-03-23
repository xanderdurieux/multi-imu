"""Figures written to ``synced/`` after synchronisation (diagnostic quality)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_stage_dir, recordings_root
from visualization._utils import mask_dropout_packets
from visualization.labels import SENSOR_COMPONENTS

from .core import add_vector_norms, remove_dropouts, resample_stream

# Keep in sync with selection.METHOD_STAGES / METHOD_LABELS / ALL_METHODS order
# Same order as selection.ALL_METHODS (preference / tie-break display)
_METHOD_ORDER = ("calibration", "lida", "sda", "online")
_METHOD_STAGES: dict[str, str] = {
    "sda": "synced/sda",
    "lida": "synced/lida",
    "calibration": "synced/cal",
    "online": "synced/online",
}
_METHOD_LABELS: dict[str, str] = {
    "sda": "SDA only",
    "lida": "SDA + LIDA",
    "calibration": "Calibration",
    "online": "Online",
}

_REF_COLOR = "#2563eb"
_TGT_COLOR = "#ca3c3c"
_MAX_PLOT_POINTS = 5000
_DEFAULT_ZOOM_S = 120.0
_RESAMPLE_HZ = 25.0


def _apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#d4d4d4",
            "axes.labelcolor": "#262626",
            "axes.titlecolor": "#171717",
            "text.color": "#171717",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "grid.color": "#e5e5e5",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )


def _load_sensor_csv(recording_name: str, stage: str, sensor: str) -> Optional[pd.DataFrame]:
    csv_path = recording_stage_dir(recording_name, stage) / f"{sensor}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return None
    return df


def _resampled_norms(
    recording_name: str,
    stage: str,
    sensor: str,
    sample_rate_hz: float,
) -> Optional[pd.DataFrame]:
    df = _load_sensor_csv(recording_name, stage, sensor)
    if df is None or df.empty:
        return None
    df = add_vector_norms(df)
    df = remove_dropouts(df)
    return resample_stream(df, sample_rate_hz)


def _decimate_pair(
    t: np.ndarray,
    y: np.ndarray,
    max_pts: int = _MAX_PLOT_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size == 0:
        return t, y
    if t.size <= max_pts:
        return t, y
    idx = np.linspace(0, t.size - 1, max_pts, dtype=int)
    return t[idx], y[idx]


def _decimate_triple(
    t: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    max_pts: int = _MAX_PLOT_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    m = np.isfinite(t) & np.isfinite(y1) & np.isfinite(y2)
    t, y1, y2 = t[m], y1[m], y2[m]
    if t.size == 0:
        return t, y1, y2
    if t.size <= max_pts:
        return t, y1, y2
    idx = np.linspace(0, t.size - 1, max_pts, dtype=int)
    return t[idx], y1[idx], y2[idx]


def _time_seconds_from_start(ts_ms: np.ndarray, t0_ms: float) -> np.ndarray:
    """Both arguments in milliseconds → seconds relative to *t0_ms*."""
    ts_ms = np.asarray(ts_ms, dtype=float)
    return (ts_ms - t0_ms) / 1000.0


def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_methods_norm_grid(
    recording_name: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    out_dir: Optional[Path] = None,
    zoom_s: float = _DEFAULT_ZOOM_S,
    sample_rate_hz: float = _RESAMPLE_HZ,
) -> Optional[Path]:
    """One figure: rows = methods, cols = ‖acc‖ / ‖gyro‖ / ‖mag‖ (zoomed window)."""
    _apply_plot_style()

    norm_specs = [
        ("acc_norm", "‖a‖ (m/s²)"),
        ("gyro_norm", "‖ω‖ (°/s)"),
        ("mag_norm", "‖m‖ (µT)"),
    ]

    rows: list[tuple[str, str, Optional[pd.DataFrame], Optional[pd.DataFrame]]] = []
    for method in ("sda", "lida", "calibration", "online"):
        stage = _METHOD_STAGES[method]
        ref_df = _resampled_norms(recording_name, stage, reference_sensor, sample_rate_hz)
        tgt_df = _resampled_norms(recording_name, stage, target_sensor, sample_rate_hz)
        if ref_df is None and tgt_df is None:
            continue
        rows.append((method, _METHOD_LABELS[method], ref_df, tgt_df))

    if not rows:
        return None

    n = len(rows)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12.5, 2.35 * n + 0.6),
        sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])

    col_titles = [lbl for _, lbl in norm_specs]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontweight="semibold", pad=6)

    t0_ms: Optional[float] = None
    for _m, _lbl, ref_df, tgt_df in rows:
        for df in (ref_df, tgt_df):
            if df is not None and not df.empty:
                t0_ms = float(df["timestamp"].iloc[0])
                break
        if t0_ms is not None:
            break
    if t0_ms is None:
        plt.close(fig)
        return None

    zoom_ms = zoom_s * 1000.0

    for r, (method, label, ref_df, tgt_df) in enumerate(rows):
        info = _load_sync_info(recording_name, _METHOD_STAGES[method])
        r_full = None
        if info:
            c = (info.get("correlation") or {}).get("offset_and_drift")
            if c is not None:
                r_full = float(c)

        for c, (ncol, _ylab) in enumerate(norm_specs):
            ax = axes[r, c]

            def _plot_one(df: Optional[pd.DataFrame], color: str, name: str) -> None:
                if df is None or df.empty or ncol not in df.columns:
                    return
                ts = df["timestamp"].to_numpy(dtype=float)
                y = df[ncol].to_numpy(dtype=float)
                rel = _time_seconds_from_start(ts, t0_ms)
                mask = (rel >= 0) & (rel <= zoom_s) & np.isfinite(y)
                rel, y = rel[mask], y[mask]
                rel, y = _decimate_pair(rel, y)
                if rel.size == 0:
                    return
                ax.plot(rel, y, color=color, lw=1.1, label=name, alpha=0.92)

            _plot_one(ref_df, _REF_COLOR, reference_sensor)
            _plot_one(tgt_df, _TGT_COLOR, target_sensor)

            ax.set_xlim(0, zoom_s)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=40)
            if c == 0 and r_full is not None:
                ax.text(
                    0.02,
                    0.96,
                    f"r = {r_full:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#525252",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e5e5e5"},
                )

    for c in range(3):
        axes[-1, c].set_xlabel(f"Time (s) · first {zoom_s:.0f} s")

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )

    fig.suptitle(
        f"{recording_name} — per-method alignment (reference vs target)",
        fontsize=11,
        fontweight="semibold",
        y=1.07,
    )

    if out_dir is None:
        out_dir = recordings_root() / recording_name / "synced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sync_methods_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/synced] {out_path.name}")
    return out_path


def plot_method_scores(
    recording_name: str,
    result: Any,
    *,
    comparison: dict[str, Any],
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Horizontal bars: correlation (offset + drift) and drift ppm; winner emphasised."""
    _apply_plot_style()

    if out_dir is None:
        out_dir = recordings_root() / recording_name / "synced"
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.arange(len(_METHOD_ORDER))
    corr_vals: list[float] = []
    drift_vals: list[float] = []
    for m in _METHOD_ORDER:
        q = result.qualities[m]
        if not q.available:
            corr_vals.append(float("nan"))
            drift_vals.append(float("nan"))
            continue
        cmp_m = comparison.get(m)
        cdict = (cmp_m or {}).get("correlation") or {} if isinstance(cmp_m, dict) else {}
        cv = cdict.get("offset_and_drift")
        corr_vals.append(float(cv) if cv is not None else float("nan"))
        dv = q.drift_ppm
        drift_vals.append(float(dv) if dv is not None else float("nan"))

    fig, (ax_r, ax_d) = plt.subplots(
        1,
        2,
        figsize=(10.5, 3.8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    colors = ["#22c55e" if m == result.method else "#cbd5e1" for m in _METHOD_ORDER]
    edgec = ["#14532d" if m == result.method else "#94a3b8" for m in _METHOD_ORDER]
    lw = [1.4 if m == result.method else 0.6 for m in _METHOD_ORDER]
    ax_r.barh(y, corr_vals, height=0.52, color=colors, edgecolor=edgec, linewidth=lw)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([_METHOD_LABELS[m] for m in _METHOD_ORDER])
    ax_r.set_xlabel("Pearson r  (‖acc‖, offset + drift)")
    ax_r.set_title("Which method fits best?", fontsize=10, fontweight="semibold")
    finite_c = [v for v in corr_vals if np.isfinite(v)]
    xmax = max(finite_c + [0.05]) * 1.12 if finite_c else 1.0
    ax_r.set_xlim(0, min(1.05, xmax))
    ax_r.invert_yaxis()
    for i, v in enumerate(corr_vals):
        if np.isfinite(v):
            ax_r.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color="#404040")

    dcolors = ["#22c55e" if m == result.method else "#94a3b8" for m in _METHOD_ORDER]
    ax_d.barh(y, drift_vals, height=0.52, color=dcolors, edgecolor="none")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([_METHOD_LABELS[m] for m in _METHOD_ORDER])
    ax_d.set_xlabel("Drift (ppm)")
    ax_d.axvline(0.0, color="#a3a3a3", lw=0.8, zorder=0)
    ax_d.set_title("Estimated target clock drift", fontsize=10, fontweight="semibold")
    ax_d.invert_yaxis()

    fig.suptitle(
        f"{recording_name}  ·  selected: {_METHOD_LABELS.get(result.method, result.method)}",
        fontsize=11,
        fontweight="semibold",
        y=1.03,
    )

    out_path = out_dir / "sync_method_metrics.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/synced] {out_path.name}")
    return out_path


def _vector_norm(df: pd.DataFrame, sensor_type: str) -> np.ndarray:
    cols = SENSOR_COMPONENTS[sensor_type]
    if not all(c in df.columns for c in cols):
        return np.full(len(df), np.nan)
    a = df[list(cols)].to_numpy(dtype=float)
    return np.sqrt(np.nansum(a * a, axis=1))


def plot_synced_norm_overlay(
    recording_name: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    selected_method_key: str,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Three stacked panels: ‖acc‖, ‖gyro‖, ‖mag‖ for the final ``synced/`` streams."""
    _apply_plot_style()

    try:
        path_r = find_sensor_csv(recording_name, "synced", reference_sensor)
        path_t = find_sensor_csv(recording_name, "synced", target_sensor)
    except (FileNotFoundError, ValueError):
        return None

    df_r = mask_dropout_packets(load_dataframe(path_r))
    df_t = mask_dropout_packets(load_dataframe(path_t))
    if df_r.empty or df_t.empty:
        return None

    ts_r = df_r["timestamp"].astype(float).to_numpy()
    ts_t = df_t["timestamp"].astype(float).to_numpy()
    t0 = min(ts_r.min(), ts_t.min())
    tr = (ts_r - t0) / 1000.0
    order = np.argsort(ts_t)
    ts_t_sorted = ts_t[order]

    specs = [
        ("acc", "Accelerometer", "‖a‖ (m/s²)"),
        ("gyro", "Gyroscope", "‖ω‖ (°/s)"),
        ("mag", "Magnetometer", "‖m‖ (µT)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True, constrained_layout=True)

    for ax, (kind, title, ylab) in zip(axes, specs):
        yr = _vector_norm(df_r, kind)
        yt = _vector_norm(df_t, kind)[order]
        yt_i = np.interp(ts_r, ts_t_sorted, yt, left=np.nan, right=np.nan)
        m = np.isfinite(yr) & np.isfinite(yt_i)
        tr_p, yr_p, yt_p = _decimate_triple(tr[m], yr[m], yt_i[m])
        ax.plot(tr_p, yr_p, color=_REF_COLOR, lw=0.9, label=reference_sensor, alpha=0.9)
        ax.plot(tr_p, yt_p, color=_TGT_COLOR, lw=0.9, label=target_sensor, alpha=0.85)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9, loc="left", color="#525252", pad=2)

    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[-1].set_xlabel("Time (s)")

    method_label = _METHOD_LABELS.get(selected_method_key, selected_method_key)
    fig.suptitle(
        f"{recording_name} — applied sync ({method_label})",
        fontsize=11,
        fontweight="semibold",
    )

    if out_dir is None:
        out_dir = recording_stage_dir(recording_name, "synced")
    out_path = out_dir / "synced_norms_overlay.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/synced] {out_path.name}")
    return out_path
