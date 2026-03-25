"""Plot GPS tracks on a map (matplotlib PNG + optional Plotly HTML)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from parser.gps import load_gps

from .thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style, plotly_template_layout


def _ensure_track_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(f"DataFrame must contain 'latitude' and 'longitude'; got {list(df.columns)}")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    m = lat.notna() & lon.notna()
    if not m.any():
        raise ValueError("No finite latitude/longitude samples to plot.")
    return lat[m], lon[m]


def plot_gps_track_matplotlib(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    """
    Save a PNG of the track in geographic coordinates with a locally correct aspect ratio.
    """
    lat, lon = _ensure_track_columns(df)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_matplotlib_thesis_style()
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.plot(lon.to_numpy(), lat.to_numpy(), color=THESIS_COLORS[0], linewidth=1.4, alpha=0.92)
    ax.scatter(
        [lon.iloc[0]],
        [lat.iloc[0]],
        s=36,
        color=THESIS_COLORS[2],
        zorder=5,
        label="start",
    )
    ax.scatter(
        [lon.iloc[-1]],
        [lat.iloc[-1]],
        s=36,
        color=THESIS_COLORS[3],
        zorder=5,
        label="end",
    )

    mean_lat = float(np.nanmean(lat))
    cos_lat = np.cos(np.radians(mean_lat))
    if abs(cos_lat) > 1e-6:
        ax.set_aspect(1.0 / cos_lat, adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(loc="best", framealpha=0.95)
    ax.set_title(title or "GPS track")
    fig.savefig(out_path)
    plt.close(fig)


def plot_gps_track_plotly_html(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    """
    Write an interactive HTML figure with a geographic basemap (Plotly ``Scattergeo``).
    """
    lat, lon = _ensure_track_columns(df)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = plotly_template_layout()
    layout["title"] = title or "GPS track"

    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=lon.to_numpy(),
                lat=lat.to_numpy(),
                mode="lines",
                line=dict(width=2, color=THESIS_COLORS[0]),
                name="track",
            ),
            go.Scattergeo(
                lon=[float(lon.iloc[0])],
                lat=[float(lat.iloc[0])],
                mode="markers",
                marker=dict(size=8, color=THESIS_COLORS[2]),
                name="start",
            ),
            go.Scattergeo(
                lon=[float(lon.iloc[-1])],
                lat=[float(lat.iloc[-1])],
                mode="markers",
                marker=dict(size=8, color=THESIS_COLORS[3]),
                name="end",
            ),
        ],
        layout=layout,
    )

    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#f0f0f0",
        showocean=True,
        oceancolor="#e8f4fc",
        showlakes=True,
        lakecolor="#e8f4fc",
        coastlinewidth=0.6,
        countrywidth=0.5,
        fitbounds="locations",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_gps_file(
    gps_path: Path | str,
    out_dir: Path | str | None = None,
    *,
    stem: str | None = None,
    title: str | None = None,
) -> tuple[Path, Path]:
    """
    Load a GPS file with :func:`parser.gps.load_gps`, then write ``<stem>_map.png`` and
    ``<stem>_map.html`` into *out_dir* (defaults to the GPS file's parent directory).
    """
    gps_path = Path(gps_path)
    df = load_gps(gps_path)
    if df.empty:
        raise ValueError(f"No GPS points parsed from {gps_path}")

    out_dir = Path(out_dir) if out_dir is not None else gps_path.parent
    base = stem if stem is not None else gps_path.stem
    png_path = out_dir / f"{base}_map.png"
    html_path = out_dir / f"{base}_map.html"

    plot_gps_track_matplotlib(df, png_path, title=title or f"GPS track — {gps_path.name}")
    plot_gps_track_plotly_html(df, html_path, title=title or f"GPS track — {gps_path.name}")
    return png_path, html_path


__all__ = [
    "plot_gps_file",
    "plot_gps_track_matplotlib",
    "plot_gps_track_plotly_html",
]
