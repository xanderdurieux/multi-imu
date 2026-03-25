"""Interactive Plotly figures (HTML) for exploration and thesis supplementary material."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import load_dataframe

from ._utils import mask_dropout_packets, mask_valid_plot_x, time_axis_seconds
from .thesis_style import THESIS_COLORS, plotly_template_layout


def _try_import_plotly():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


def write_feature_explorer_html(df: pd.DataFrame, out_path: Path) -> None:
    """Interactive PCA scatter; hover shows section / recording / window time."""
    go, _ = _try_import_plotly()
    from .insight_plots import _numeric_feature_columns, _pca_2d

    feat_cols = _numeric_feature_columns(df, max_cols=28)
    if len(feat_cols) < 3:
        return
    work = df.dropna(subset=feat_cols, how="any")
    if len(work) < 5:
        return

    X = work[feat_cols].to_numpy(dtype=float)
    scores, _loadings, ev1, ev2 = _pca_2d(X)

    def _hover_for_row(row: pd.Series) -> str:
        parts = []
        if "recording" in work.columns:
            parts.append(f"recording={row['recording']}")
        if "section" in work.columns:
            parts.append(f"section={row['section']}")
        if "window_center_s" in work.columns and pd.notna(row.get("window_center_s")):
            parts.append(f"t={float(row['window_center_s']):.2f}s")
        return "<br>".join(parts) if parts else ""

    layout_extra = plotly_template_layout()
    fig = go.Figure()

    if "recording" in work.columns:
        recordings = work["recording"].astype(str)
        for i, rec in enumerate(pd.unique(recordings)):
            m = recordings == rec
            idx = np.nonzero(m.to_numpy())[0]
            hover = [_hover_for_row(work.iloc[j]) for j in idx]
            fig.add_trace(
                go.Scatter(
                    x=scores[idx, 0],
                    y=scores[idx, 1],
                    mode="markers",
                    name=str(rec),
                    marker=dict(size=9, color=THESIS_COLORS[i % len(THESIS_COLORS)], opacity=0.78, line=dict(width=0)),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        title_suffix = ", by recording"
    else:
        hover = [_hover_for_row(r) for _, r in work.iterrows()]
        fig.add_trace(
            go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode="markers",
                marker=dict(size=9, color=THESIS_COLORS[0], opacity=0.78, line=dict(width=0)),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        title_suffix = ""

    fig.update_layout(
        title=f"Feature PCA (PC1={ev1 * 100:.1f}%, PC2={ev2 * 100:.1f}% var.){title_suffix}",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        **layout_extra,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", config={"responsive": True})


def write_parallel_coordinates_html(df: pd.DataFrame, out_path: Path, max_dims: int = 8) -> None:
    """Parallel coordinates for top-variance features (interactive brushing in browser)."""
    go, _ = _try_import_plotly()
    from .insight_plots import _numeric_feature_columns

    feat_cols = _numeric_feature_columns(df, max_cols=max_dims)
    if len(feat_cols) < 3:
        return
    work = df[feat_cols].dropna(how="any")
    if len(work) < 4:
        return

    dims = [
        go.parcoords.Dimension(label=c, values=work[c].to_numpy(dtype=float)) for c in feat_cols
    ]
    color_vals = work[feat_cols[0]].to_numpy(dtype=float)
    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(color=color_vals, colorscale="Viridis", showscale=True),
                dimensions=dims,
            )
        ]
    )
    fig.update_layout(title="Parallel coordinates (feature space)", **plotly_template_layout())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", config={"responsive": True})


def write_sensor_timeseries_html(csv_path: Path, out_path: Path, title: str | None = None) -> None:
    """Tri-axial accelerometer + gyroscope with range slider and dropdown visibility."""
    go, make_subplots = _try_import_plotly()
    df = load_dataframe(csv_path)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return
    df = mask_dropout_packets(df)
    t = time_axis_seconds(df["timestamp"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Accelerometer [m/s²]", "Gyroscope [rad/s or device units]"),
    )
    acc_cols = ["ax", "ay", "az"]
    gyro_cols = ["gx", "gy", "gz"]
    acc_present = [c for c in acc_cols if c in df.columns]
    gyro_present = [c for c in gyro_cols if c in df.columns]
    colors = list(THESIS_COLORS)
    tx = t.to_numpy(dtype=float)
    xm = mask_valid_plot_x(tx)
    for i, c in enumerate(acc_present):
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        yp = np.where(xm, y, np.nan)
        fig.add_trace(
            go.Scatter(x=tx, y=yp, name=c, mode="lines", line=dict(width=1.1, color=colors[i % len(colors)])),
            row=1,
            col=1,
        )
    for i, c in enumerate(gyro_present):
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        yp = np.where(xm, y, np.nan)
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=yp,
                name=c,
                mode="lines",
                line=dict(width=1.1, color=colors[i % len(colors)]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Time [s]", row=2, col=1, rangeslider=dict(visible=True))
    layout = plotly_template_layout()
    fig_title = title or csv_path.stem
    fig.update_layout(title=fig_title, height=720, **layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", config={"responsive": True})


def write_interactive_bundle(df: pd.DataFrame, interactive_dir: Path) -> None:
    """Write HTML dashboard files under ``interactive_dir``."""
    interactive_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        write_feature_explorer_html(df, interactive_dir / "feature_pca.html")
        write_parallel_coordinates_html(df, interactive_dir / "feature_parallel_coords.html")


def write_section_sensor_html(section_dir: Path, interactive_dir: Path) -> None:
    """For a section directory, build interactive sensor plots for each calibrated CSV."""
    cal = section_dir / "calibrated"
    if not cal.is_dir():
        return
    interactive_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(cal.glob("*.csv")):
        if csv_path.suffix.lower() != ".csv":
            continue
        out = interactive_dir / f"sensor_timeseries_{csv_path.stem}.html"
        write_sensor_timeseries_html(csv_path, out, title=f"{section_dir.name} — {csv_path.stem}")
