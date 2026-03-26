"""Interactive browser labeler: synced IMU (acc, gyro, mag) + session GPS on a map.

Time on the x-axis is **seconds from the first ``sporsa`` sample** in the chosen folder
(same origin as ``labels.parser`` interval ``window_*_s`` for sections).

GPS is clipped to the IMU span plus a few seconds, loaded from
``data/sessions/<session>/*gps*.csv``. The map sits beside the time stack; charts use the browser width (no horizontal
scrollbar on the plot strip). Hover on any sensor panel highlights the same time
(unified hover + crosshair) and moves a marker on the map.

Usage (from ``analysis/``)::

    uv run python -m labels.event_labeler 2026-02-26_r2/synced
    uv run python -m labels.event_labeler 2026-02-26_r2s1 out.html

If the output path is omitted, writes ``event_labeler.html`` next to the CSVs.
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

import plotly
from common import load_dataframe
from common.paths import session_input_dir, recording_stage_dir
from parser.gps import parse_gps_csv
from visualization._utils import mask_dropout_packets, mask_valid_plot_x
from visualization.thesis_style import THESIS_COLORS, plotly_template_layout

log = logging.getLogger(__name__)

ACC = ("ax", "ay", "az")
GYRO = ("gx", "gy", "gz")
MAG = ("mx", "my", "mz")

LABEL_CSV_HEADER = [
    "scope",
    "recording_id",
    "section_id",
    "window_start_s",
    "window_end_s",
    "scenario_label",
    "label_source",
]


def infer_recording_section_ids(data_dir: Path) -> tuple[str, str]:
    p = data_dir.resolve()
    parts = list(p.parts)
    if "sections" in parts:
        i = parts.index("sections")
        if i + 1 < len(parts):
            maybe_section_folder = parts[i + 1]
            from common.paths import parse_section_folder_name

            rec_name, sec_idx = parse_section_folder_name(maybe_section_folder)
            _ = sec_idx
            return rec_name, maybe_section_folder
    return p.parent.name, ""


def session_name_from_recording_id(recording_id: str) -> str | None:
    rid = recording_id.strip()
    m = re.match(r"^(.+)_r(\d+)$", rid)
    return m.group(1) if m else None


def resolve_labeling_data_dir(target: str | Path) -> Path:
    """Resolve ``recording/<stage>`` or a section-folder name or a directory path.

    Supported shorthand inputs (relative to ``analysis/``):
    - ``<recording>/<stage>`` (e.g. ``2026-02-26_r5/parsed``)
    - ``<recording>s<section_idx>`` (e.g. ``2026-02-26_r5s1``)
    """
    if isinstance(target, Path):
        p = target.resolve()
        if p.is_dir() and (p / "sporsa.csv").is_file():
            return p
        raise FileNotFoundError(f"Not a directory with sporsa.csv: {p}")

    s = str(target).strip().rstrip("/").replace("\\", "/")
    parts = s.split("/")

    # Section-folder shorthand: <recording>s<section_idx> (e.g. 2026-02-26_r5s1).
    if len(parts) == 1:
        from common.paths import parse_section_folder_name, sections_root

        try:
            _rec_name, _sec_idx = parse_section_folder_name(parts[0])
        except Exception:
            _rec_name = None
        else:
            d = sections_root() / parts[0]
            if d.is_dir() and (d / "sporsa.csv").is_file():
                return d

    if len(parts) >= 2:
        rec = parts[0]
        stage = "/".join(parts[1:])
        d = recording_stage_dir(rec, stage)
        if d.is_dir() and (d / "sporsa.csv").is_file():
            return d

    p = Path(s)
    if p.is_dir() and (p / "sporsa.csv").is_file():
        return p.resolve()

    raise FileNotFoundError(
        f"Could not resolve {target!r}. Try e.g. '2026-02-26_r5/parsed' or "
        f"'2026-02-26_r5s1' (from analysis/)."
    )



def _nan_where_bad_time_x(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = mask_valid_plot_x(t)
    return np.where(xm, t, np.nan), np.where(xm, y, np.nan)


def _load_imu(
    csv_path: Path,
    t0_ms: float,
    required: tuple[str, ...],
) -> tuple[np.ndarray, pd.DataFrame] | None:
    if not csv_path.is_file():
        return None
    df = load_dataframe(csv_path)
    if df.empty or "timestamp" not in df.columns:
        return None
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if not all(c in df.columns for c in required):
        return None
    df = mask_dropout_packets(df)
    ts = df["timestamp"].to_numpy(dtype=float)
    t_s = (ts - t0_ms) / 1000.0
    return t_s, df


def _gps_epoch_ms_series(time_utc: pd.Series) -> pd.Series:
    s = pd.to_datetime(time_utc, utc=True, errors="coerce")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (s - epoch) / pd.Timedelta(milliseconds=1)


def find_best_gps_csv(session_dir: Path, imu_t0_ms: float, imu_t1_ms: float) -> Path | None:
    if not session_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in session_dir.glob("*.csv") if "gps" in p.name.lower()),
        key=lambda x: x.name,
    )
    if not candidates:
        return None

    best_path: Path | None = None
    best_score = -1.0
    for p in candidates:
        try:
            df = parse_gps_csv(p)
        except (ValueError, OSError) as e:
            log.debug("Skip GPS candidate %s: %s", p, e)
            continue
        if df.empty or "time_utc" not in df.columns:
            continue
        if df["time_utc"].isna().all():
            continue
        tms = _gps_epoch_ms_series(df["time_utc"])
        valid = tms.notna()
        if not valid.any():
            continue
        g0 = float(tms[valid].min())
        g1 = float(tms[valid].max())
        overlap = max(0.0, min(imu_t1_ms, g1) - max(imu_t0_ms, g0))
        if overlap > best_score:
            best_score = overlap
            best_path = p

    return best_path if best_path is not None else None


GPS_CLIP_EXTRA_S = 5.0
MAP_COLUMN_MAX_PX = 380
# Cap points sent to the browser for IMU WebGL traces (keeps pan/zoom/hover responsive).
MAX_IMU_POINTS = 12_000
# Lighter GPS polyline on the map (cursor still uses full-resolution payload).
MAX_GPS_MAP_POINTS = 5_000


def _prepare_gps_track_arrays(
    gps_path: Path | None,
    t0_ms: float,
    imu_t0_ms: float,
    imu_t1_ms: float,
    *,
    extra_s: float = GPS_CLIP_EXTRA_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return ``t_s``, ``latitude``, ``longitude``, ``speed_m_s`` (NaN if absent), and a status note."""
    empty = (
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )

    if gps_path is None:
        return (*empty, "No GPS CSV (filename contains 'gps') found in the session folder.")

    try:
        df = parse_gps_csv(gps_path)
    except (ValueError, OSError) as exc:
        return (*empty, f"GPS read error: {exc}")

    if df.empty:
        return (*empty, f"Empty GPS file: {gps_path.name}")

    if df["time_utc"].isna().all():
        return (*empty, f"GPS file has no parseable times: {gps_path.name}")

    tms = _gps_epoch_ms_series(df["time_utc"])
    m = tms.notna() & df["latitude"].notna() & df["longitude"].notna()
    df = df.loc[m].reset_index(drop=True)
    tms = tms[m]
    if df.empty:
        return (*empty, "No GPS rows with time + lat/lon.")

    margin_ms = float(extra_s) * 1000.0
    keep = (tms.to_numpy(dtype=float) >= imu_t0_ms - margin_ms) & (
        tms.to_numpy(dtype=float) <= imu_t1_ms + margin_ms
    )
    df = df.loc[keep].reset_index(drop=True)
    tms = tms[keep]
    if df.empty:
        return (
            *empty,
            f"GPS times do not overlap this IMU window (within ±{extra_s:g} s).",
        )

    t_s = (tms.to_numpy(dtype=float) - t0_ms) / 1000.0
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    if "speed_m_s" in df.columns:
        spd = pd.to_numeric(df["speed_m_s"], errors="coerce").to_numpy(dtype=float)
    else:
        spd = np.full(len(df), np.nan, dtype=float)
    order = np.argsort(t_s, kind="mergesort")
    t_s = t_s[order]
    lat = lat[order]
    lon = lon[order]
    spd = spd[order]
    note = f"GPS: {gps_path.name} ({len(df)} pts, ±{extra_s:g} s pad)"
    return t_s, lat, lon, spd, note


def _downsample_index(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    step = max(1, int(np.ceil(n / max_points)))
    return np.arange(0, n, step, dtype=int)


def _build_gps_column_figure(
    lat: np.ndarray,
    lon: np.ndarray,
    t_s: np.ndarray,
    speed: np.ndarray,
    *,
    time_stack_height_px: int,
    title: str = "GPS (recording window)",
) -> go.Figure:
    """Map (top) + speed vs time (bottom, WebGL). Geo cursor stays trace index 1."""
    geo_h = max(200, int(time_stack_height_px * 0.42))
    spd_h = max(130, int(time_stack_height_px * 0.20))
    total_h = geo_h + spd_h + 72

    if lat.size == 0:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[geo_h, spd_h],
            vertical_spacing=0.08,
            specs=[[{"type": "scattergeo"}], [{"type": "xy"}]],
            subplot_titles=("GPS track", "GPS speed (m/s)"),
        )
        fig.add_annotation(
            text="No GPS points",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.78,
            showarrow=False,
            row=1,
            col=1,
        )
        fig.add_annotation(
            text="—",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.22,
            showarrow=False,
            row=2,
            col=1,
        )
        fig.update_layout(title=title, autosize=True, height=total_h, margin=dict(t=36, b=24))
        return fig

    idx = _downsample_index(lat.size, MAX_GPS_MAP_POINTS)
    lat_d, lon_d = lat[idx], lon[idx]
    t_d, spd_d = t_s[idx], speed[idx]

    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_span = max(lat_max - lat_min, 1e-5)
    lon_span = max(lon_max - lon_min, 1e-5)
    pad_frac = 0.34
    pad_lat = max(lat_span * pad_frac, 4e-4)
    pad_lon = max(lon_span * pad_frac, 4e-4)
    lat_axis_range = [lat_min - pad_lat, lat_max + pad_lat]
    lon_axis_range = [lon_min - pad_lon, lon_max + pad_lon]
    # Floor so short tracks are not framed too tight (~1.7 km lat, ~1.1 km at 51° lon).
    min_lat_range = 0.015
    min_lon_range = 0.024
    if lat_axis_range[1] - lat_axis_range[0] < min_lat_range:
        mid = 0.5 * (lat_axis_range[0] + lat_axis_range[1])
        h = 0.5 * min_lat_range
        lat_axis_range = [mid - h, mid + h]
    if lon_axis_range[1] - lon_axis_range[0] < min_lon_range:
        mid = 0.5 * (lon_axis_range[0] + lon_axis_range[1])
        h = 0.5 * min_lon_range
        lon_axis_range = [mid - h, mid + h]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[geo_h, spd_h],
        vertical_spacing=0.07,
        specs=[[{"type": "scattergeo"}], [{"type": "xy"}]],
        subplot_titles=("GPS track", "GPS speed (m/s)"),
    )
    fig.add_trace(
        go.Scattergeo(
            lon=lon_d,
            lat=lat_d,
            mode="lines",
            line=dict(width=2, color=THESIS_COLORS[0]),
            name="track",
            hovertemplate="lat=%{lat:.6f}<br>lon=%{lon:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[float(lon[0])],
            lat=[float(lat[0])],
            mode="markers",
            name="time cursor",
            marker=dict(size=14, color="#dc2626", line=dict(width=1, color="#111")),
            hovertemplate="lat=%{lat:.6f}<br>lon=%{lon:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    tx_s, sy = _nan_where_bad_time_x(t_d, spd_d)
    fig.add_trace(
        go.Scattergl(
            x=tx_s,
            y=sy,
            mode="lines",
            name="speed",
            line=dict(width=1.3, color=THESIS_COLORS[1]),
            hovertemplate="%{x:.2f} s<br>%{y:.3f} m/s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    base = plotly_template_layout()
    base.pop("template", None)
    fig.update_layout(
        title=title,
        autosize=True,
        height=total_h,
        margin=dict(l=4, r=8, t=40, b=28),
        showlegend=False,
        hovermode="closest",
        transition_duration=0,
        **{k: v for k, v in base.items() if k not in ("width", "height", "margin", "title", "showlegend", "autosize", "hovermode", "transition_duration")},
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
        lataxis_range=lat_axis_range,
        lonaxis_range=lon_axis_range,
        row=1,
        col=1,
    )
    # Lock speed panel: zoom tools must not rescale time/speed (only the geo map zooms).
    fig.update_xaxes(
        title_text="Time from first sporsa sample (s)",
        showspikes=False,
        fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(fixedrange=True, title_text="m/s", row=2, col=1)
    if not np.isfinite(speed).any():
        fig.add_annotation(
            text="No speed column in GPS CSV",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=11, color="#64748b"),
            row=2,
            col=1,
        )
    return fig


def _downsample_imu_pair(t_s: np.ndarray, df: pd.DataFrame, max_n: int) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(t_s)
    if n <= max_n:
        return t_s, df
    idx = _downsample_index(n, max_n)
    return t_s[idx], df.iloc[idx].reset_index(drop=True)


def _legend_id_for_row(row_1based: int) -> str:
    return "legend" if row_1based == 1 else f"legend{row_1based}"


def _series_or_nan(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return np.full(n, np.nan, dtype=float)


def _imu_xyz_traces(
    *,
    t_s: np.ndarray,
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    col_z: str,
    hover_prefix: str,
    color_offset: int,
    legend_id: str,
    legendgroup_stem: str,
) -> list[go.Scattergl]:
    """Three WebGL line traces; hover shows Timestamp + ``{prefix}_x/y/z`` via ``customdata`` on the first trace only."""
    n = len(t_s)
    vx = _series_or_nan(df, col_x, n)
    vy = _series_or_nan(df, col_y, n)
    vz = _series_or_nan(df, col_z, n)
    tx, xm = _nan_where_bad_time_x(t_s, vx)
    _, ym = _nan_where_bad_time_x(t_s, vy)
    _, zm = _nan_where_bad_time_x(t_s, vz)
    customdata = np.column_stack([xm, ym, zm])
    ht = (
        f"Timestamp:<br>%{{x:.4f}} s<br><br>"
        f"{hover_prefix}_x:<br>%{{customdata[0]:.4f}}<br>"
        f"{hover_prefix}_y:<br>%{{customdata[1]:.4f}}<br>"
        f"{hover_prefix}_z:<br>%{{customdata[2]:.4f}}<extra></extra>"
    )
    return [
        go.Scattergl(
            x=tx,
            y=xm,
            mode="lines",
            name=col_x,
            line=dict(width=1.1, color=THESIS_COLORS[(0 + color_offset) % len(THESIS_COLORS)]),
            legendgroup=f"{legendgroup_stem}-{col_x}",
            legend=legend_id,
            customdata=customdata,
            hovertemplate=ht,
        ),
        go.Scattergl(
            x=tx,
            y=ym,
            mode="lines",
            name=col_y,
            line=dict(width=1.1, color=THESIS_COLORS[(1 + color_offset) % len(THESIS_COLORS)]),
            legendgroup=f"{legendgroup_stem}-{col_y}",
            legend=legend_id,
            hoverinfo="skip",
        ),
        go.Scattergl(
            x=tx,
            y=zm,
            mode="lines",
            name=col_z,
            line=dict(width=1.1, color=THESIS_COLORS[(2 + color_offset) % len(THESIS_COLORS)]),
            legendgroup=f"{legendgroup_stem}-{col_z}",
            legend=legend_id,
            hoverinfo="skip",
        ),
    ]


def _apply_per_subplot_legends(fig: go.Figure, n_rows: int, *, legend_x_paper: float = 1.0) -> None:
    """Place ``legend``, ``legend2``, … at the top-right of each subplot in paper coordinates."""
    updates: dict[str, Any] = {}
    for i in range(1, n_rows + 1):
        yax = fig.layout.yaxis if i == 1 else getattr(fig.layout, f"yaxis{i}", None)
        if yax is None or yax.domain is None:
            continue
        d0, d1 = float(yax.domain[0]), float(yax.domain[1])
        y_top = d0 + (d1 - d0) * 0.94
        key = _legend_id_for_row(i)
        updates[key] = dict(
            x=legend_x_paper,
            xref="paper",
            y=y_top,
            yref="paper",
            yanchor="top",
            xanchor="left",
            orientation="v",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#94a3b8",
            borderwidth=1,
            font=dict(size=9),
            itemsizing="constant",
            itemwidth=30,
            tracegroupgap=0,
        )
    fig.update_layout(**updates)


def build_event_labeler_figure(
    data_dir: Path,
) -> tuple[go.Figure, go.Figure | None, float, str, dict[str, list[float]]]:
    """Return time-series figure, optional map figure, duration (s), GPS note, GPS arrays for JS sync.

    Figures use ``autosize=True`` so the HTML page can fit them to the viewport (no fixed pixel width).
    """
    data_dir = Path(data_dir)
    sp = data_dir / "sporsa.csv"
    ts_b = load_dataframe(sp).dropna(subset=["timestamp"]).sort_values("timestamp")
    if ts_b.empty:
        raise FileNotFoundError(f"No timestamps in {sp}")
    t0_ms = float(ts_b["timestamp"].iloc[0])
    imu_t0_ms = t0_ms
    imu_t1_ms = float(ts_b["timestamp"].iloc[-1])

    t_bike, df_b = _load_imu(sp, t0_ms, ACC)
    if t_bike is None:
        raise FileNotFoundError(f"No usable {sp}")

    ar_path = data_dir / "arduino.csv"
    rider = _load_imu(ar_path, t0_ms, ACC)
    has_rider = rider is not None

    recording_id, _ = infer_recording_section_ids(data_dir)
    session = session_name_from_recording_id(recording_id)
    gps_path: Path | None = None
    if session:
        gps_path = find_best_gps_csv(session_input_dir(session), imu_t0_ms, imu_t1_ms)
        if gps_path is None:
            cand = session_input_dir(session)
            log.info("No overlapping GPS CSV under %s; trying any *gps*.csv name.", cand)
            if cand.is_dir():
                any_gps = sorted(
                    (p for p in cand.glob("*.csv") if "gps" in p.name.lower()),
                    key=lambda x: x.name,
                )
                gps_path = any_gps[0] if any_gps else None

    gps_t_s, gps_lat, gps_lon, gps_speed, gps_note = _prepare_gps_track_arrays(
        gps_path, t0_ms, imu_t0_ms, imu_t1_ms
    )

    subplot_specs: list[tuple[str, list[go.Scattergl]]] = []

    t_b_ds, df_b_ds = _downsample_imu_pair(t_bike, df_b, MAX_IMU_POINTS)
    rider_ds: tuple[np.ndarray, pd.DataFrame] | None = None
    if has_rider:
        t_r0, df_r0 = rider
        rider_ds = _downsample_imu_pair(t_r0, df_r0, MAX_IMU_POINTS)

    def next_legend_id() -> str:
        return _legend_id_for_row(len(subplot_specs) + 1)

    # Order: acc, acc, gyro, gyro, mag, mag (bike then rider per sensor when rider exists).
    subplot_specs.append(
        (
            "Bike (sporsa) — acc (m/s²)",
            _imu_xyz_traces(
                t_s=t_b_ds,
                df=df_b_ds,
                col_x="ax",
                col_y="ay",
                col_z="az",
                hover_prefix="bike_acc",
                color_offset=0,
                legend_id=next_legend_id(),
                legendgroup_stem="Bike (sporsa)-acc",
            ),
        )
    )
    if rider_ds is not None:
        t_r_ds, df_r_ds = rider_ds
        subplot_specs.append(
            (
                "Rider (arduino) — acc (m/s²)",
                _imu_xyz_traces(
                    t_s=t_r_ds,
                    df=df_r_ds,
                    col_x="ax",
                    col_y="ay",
                    col_z="az",
                    hover_prefix="rider_acc",
                    color_offset=2,
                    legend_id=next_legend_id(),
                    legendgroup_stem="Rider (arduino)-acc",
                ),
            )
        )
    subplot_specs.append(
        (
            "Bike (sporsa) — gyro",
            _imu_xyz_traces(
                t_s=t_b_ds,
                df=df_b_ds,
                col_x="gx",
                col_y="gy",
                col_z="gz",
                hover_prefix="bike_gyro",
                color_offset=3,
                legend_id=next_legend_id(),
                legendgroup_stem="Bike (sporsa)-gyro",
            ),
        )
    )
    if rider_ds is not None:
        t_r_ds, df_r_ds = rider_ds
        subplot_specs.append(
            (
                "Rider (arduino) — gyro",
                _imu_xyz_traces(
                    t_s=t_r_ds,
                    df=df_r_ds,
                    col_x="gx",
                    col_y="gy",
                    col_z="gz",
                    hover_prefix="rider_gyro",
                    color_offset=5,
                    legend_id=next_legend_id(),
                    legendgroup_stem="Rider (arduino)-gyro",
                ),
            )
        )
    subplot_specs.append(
        (
            "Bike (sporsa) — mag",
            _imu_xyz_traces(
                t_s=t_b_ds,
                df=df_b_ds,
                col_x="mx",
                col_y="my",
                col_z="mz",
                hover_prefix="bike_mag",
                color_offset=6,
                legend_id=next_legend_id(),
                legendgroup_stem="Bike (sporsa)-mag",
            ),
        )
    )
    if rider_ds is not None:
        t_r_ds, df_r_ds = rider_ds
        subplot_specs.append(
            (
                "Rider (arduino) — mag",
                _imu_xyz_traces(
                    t_s=t_r_ds,
                    df=df_r_ds,
                    col_x="mx",
                    col_y="my",
                    col_z="mz",
                    hover_prefix="rider_mag",
                    color_offset=8,
                    legend_id=next_legend_id(),
                    legendgroup_stem="Rider (arduino)-mag",
                ),
            )
        )

    n_rows = len(subplot_specs)
    titles = [s[0] for s in subplot_specs]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        subplot_titles=titles,
    )
    for i, (_, traces) in enumerate(subplot_specs, start=1):
        for tr in traces:
            fig.add_trace(tr, row=i, col=1)

    duration_s = float(np.nanmax(t_bike) - np.nanmin(t_bike))
    if not np.isfinite(duration_s) or duration_s <= 0:
        duration_s = 1.0
    row_h = 140
    plot_h = 100 + n_rows * row_h

    base = plotly_template_layout()
    base.pop("template", None)
    title = f"Event labeler — {data_dir.parent.name}/{data_dir.name}"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.98, yanchor="top", pad=dict(b=6)),
        autosize=True,
        height=plot_h,
        margin=dict(t=56, r=108, l=56, b=48),
        hovermode="x unified",
        hoverdistance=32,
        spikedistance=-1,
        dragmode="zoom",
        transition_duration=0,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#94a3b8",
            font_size=11,
            namelength=0,
        ),
        **{
            k: v
            for k, v in base.items()
            if k
            not in (
                "legend",
                "width",
                "height",
                "autosize",
                "margin",
                "title",
                "hoverlabel",
                "transition_duration",
            )
        },
    )
    for ri in range(1, n_rows + 1):
        fig.update_xaxes(
            showspikes=True,
            spikecolor="#64748b",
            spikesnap="data",
            spikemode="across",
            spikethickness=1,
            showline=True,
            mirror=True,
            row=ri,
            col=1,
        )
    fig.update_yaxes(fixedrange=True, showline=True, mirror=True)
    fig.update_xaxes(title_text="Time from first sporsa sample in this folder (s)", row=n_rows, col=1)
    _apply_per_subplot_legends(fig, n_rows)

    fig_map: go.Figure | None = None
    gps_payload: dict[str, list[float]] = {"t": [], "lat": [], "lon": []}
    if gps_lat.size > 0:
        fig_map = _build_gps_column_figure(
            gps_lat,
            gps_lon,
            gps_t_s,
            gps_speed,
            time_stack_height_px=plot_h,
        )
        gps_payload = {
            "t": [float(x) for x in gps_t_s],
            "lat": [float(x) for x in gps_lat],
            "lon": [float(x) for x in gps_lon],
        }

    return fig, fig_map, duration_s, gps_note, gps_payload


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "labels"


def _plotly_js_cdn_url() -> str:
    js_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    try:
        head = js_path.read_text(encoding="utf-8", errors="ignore")[:600]
    except OSError:
        head = ""
    m = re.search(r"plotly\.js v(\d+\.\d+\.\d+)", head)
    ver = m.group(1) if m else "3.4.0"
    return f"https://cdn.plot.ly/plotly-{ver}.min.js"


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Event labeler — {title_esc}</title>
  <script src="{plotly_js_url_esc}"></script>
  <style>
    :root {{
      font-family: system-ui, sans-serif;
      --bg: #f8fafc;
      --card: #fff;
      --border: #e2e8f0;
      --accent: #4f46e5;
    }}
    body {{ margin: 0; background: var(--bg); color: #0f172a; }}
    .wrap {{ max-width: 100%; margin: 0 auto; padding: 12px 16px 32px; }}
    h1 {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 8px; }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 12px;
    }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; margin-bottom: 8px; }}
    label {{ font-size: 0.85rem; color: #475569; }}
    input[type="text"], input[type="number"] {{
      padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; min-width: 140px;
    }}
    .hint {{ font-size: 0.8rem; color: #64748b; margin-top: 6px; line-height: 1.4; }}
    button {{
      padding: 8px 14px; border-radius: 8px; border: none;
      background: var(--accent); color: #fff; font-weight: 500; cursor: pointer;
    }}
    button.secondary {{ background: #64748b; }}
    button.danger {{ background: #b91c1c; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    .plot-strip {{
      overflow-x: hidden;
      overflow-y: visible;
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--card);
      margin-bottom: 12px;
    }}
    .plot-strip-inner {{
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      gap: 10px;
      padding: 8px;
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
    }}
    #plot-time {{
      flex: 1 1 0;
      min-width: 0;
      width: 100%;
    }}
    #plot-map-wrap {{
      flex: 0 0 clamp(200px, 26vw, {map_col_max}px);
      max-width: min(100%, {map_col_max}px);
      min-height: 200px;
      align-self: stretch;
      display: flex;
      flex-direction: column;
      background: var(--card);
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    #plot-map {{ width: 100%; height: 100%; min-height: 200px; flex: 1; }}
    #plot-map-wrap.hidden {{ display: none; }}
    @media (max-width: 720px) {{
      .plot-strip-inner {{ flex-direction: column; }}
      #plot-map-wrap {{
        flex: 1 1 auto;
        width: 100%;
        max-width: 100%;
      }}
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 8px; }}
    th, td {{ border: 1px solid var(--border); padding: 6px 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    table input[type="number"], table input[type="text"] {{
      width: 100%;
      max-width: 12rem;
      box-sizing: border-box;
      padding: 4px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.85rem;
    }}
    table td:nth-child(3) input {{ max-width: 100%; }}
    .adj-hint {{ font-size: 0.8rem; color: #64748b; margin: 6px 0 8px 0; line-height: 1.4; }}
    .warn {{ color: #b45309; font-size: 0.8rem; }}
    .mode label {{ margin-right: 12px; cursor: pointer; }}
    .gps-note {{ font-size: 0.8rem; color: #475569; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Event labeler</h1>
    <div class="panel">
      <div class="row">
        <div><label>Recording ID<br/><input type="text" id="recId" value="{recording_id_esc}"/></label></div>
        <div><label>Section ID<br/><input type="text" id="secId" placeholder="2026-02-26_r2s1" value="{section_id_esc}"/></label></div>
        <div><label>Label (scenario)<br/><input type="text" id="scenario" placeholder="e.g. braking_event"/></label></div>
        <div><label>Label source<br/><input type="text" id="labelSrc" value="manual_event_labeler"/></label></div>
      </div>
      <div class="row mode">
        <span><b>Mode:</b></span>
        <label><input type="radio" name="mode" value="interval" checked/> Interval (2 clicks)</label>
        <label><input type="radio" name="mode" value="peak"/> Peak (1 click)</label>
      </div>
      <div class="row">
        <div><label>Peak half-width (s)<br/><input type="number" id="peakHalf" value="0.25" min="0.01" step="0.05"/></label></div>
        <button type="button" class="secondary" id="btnCancel">Cancel pending click</button>
        <button type="button" class="secondary" id="btnLoad">Load CSV…</button>
        <input type="file" id="fileLoad" accept=".csv,text/csv" style="display:none"/>
        <button type="button" id="btnDownload">Download labels.csv</button>
      </div>
      <p class="hint">
        <b>Navigate:</b> default drag is <b>box zoom</b> (shared time axis on all rows); use the mode bar <b>Pan</b> to drag. <b>Scroll wheel</b> zooms in the time plot.
        The strip fits the page width.
        <b>Hover:</b> each row shows Timestamp plus <code>…_x / _y / _z</code> (e.g. <code>bike_acc</code>); spikes align across panels. <b>Y range is fixed</b> — only the time axis zooms. The map cursor follows the hovered time, and a <b>vertical line</b> marks the same time on the speed chart. The right column shows <b>GPS speed</b> (m/s) under the map when the CSV has a speed column. The map is framed with margin so the full track stays in view.
        <b>Interval:</b> two clicks; <b>Peak:</b> one click. Times are seconds from the first sporsa sample in this folder.
      </p>
      <p class="gps-note">{gps_note_esc}</p>
      {section_warn}
    </div>
    <div class="plot-strip" id="plotStrip">
      <div class="plot-strip-inner">
        <div id="plot-time"></div>
        <div id="plot-map-wrap" class="{map_wrap_class}">
          <div id="plot-map"></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <b>Labeled intervals</b> <span id="count">0</span>
      <p class="adj-hint">Edit start, end, or scenario below; when you leave a field, the value is applied and the shaded intervals on the plot update.</p>
      <table>
        <thead><tr><th>start (s)</th><th>end (s)</th><th>scenario_label</th><th></th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
  <script type="application/json" id="payload">{payload_json}</script>
  <script>
(function() {{
  const payload = JSON.parse(document.getElementById('payload').textContent);
  const figTime = payload.figureTime;
  const figMap = payload.figureMap;
  const plotTimeEl = document.getElementById('plot-time');
  const plotMapEl = document.getElementById('plot-map');

  const recEl = document.getElementById('recId');
  const secEl = document.getElementById('secId');
  const scenEl = document.getElementById('scenario');
  const srcEl = document.getElementById('labelSrc');
  const peakHalfEl = document.getElementById('peakHalf');
  const tbody = document.getElementById('tbody');
  const countEl = document.getElementById('count');

  const configTime = {{
    responsive: true,
    scrollZoom: true,
    displayModeBar: true,
  }};
  const configMap = {{
    responsive: true,
    // Wheel zoom on the map column (speed axes are fixedrange in Python).
    scrollZoom: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
  }};

  function resizeBothPlots() {{
    try {{ Plotly.Plots.resize(plotTimeEl); }} catch (e) {{}}
    if (mapGd) {{ try {{ Plotly.Plots.resize(plotMapEl); }} catch (e) {{}} }}
  }}
  window.addEventListener('resize', function() {{
    window.requestAnimationFrame(resizeBothPlots);
  }});

  let mapGd = null;

  /** @type {{ id: number, t0: number, t1: number, label: string }} */
  let rows = [];
  let nextId = 1;
  /** @type {{number|null}} */
  let pendingT = null;
  let shapeBase = [];

  function getMode() {{
    const r = document.querySelector('input[name="mode"]:checked');
    return r ? r.value : 'interval';
  }}

  function fmt4(x) {{
    return Math.round(x * 10000) / 10000;
  }}

  function labelShapes() {{
    const out = shapeBase.slice();
    const palette = ['rgba(79,70,229,0.18)', 'rgba(14,165,233,0.18)', 'rgba(234,88,12,0.18)', 'rgba(22,163,74,0.18)'];
    rows.forEach(function(r, i) {{
      out.push({{
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: r.t0,
        x1: r.t1,
        y0: 0,
        y1: 1,
        fillcolor: palette[i % palette.length],
        line: {{ width: 0 }},
        layer: 'below',
      }});
    }});
    if (pendingT !== null) {{
      out.push({{
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: pendingT,
        x1: pendingT,
        y0: 0,
        y1: 1,
        line: {{ color: '#f59e0b', width: 2 }},
        layer: 'above',
      }});
    }}
    return out;
  }}

  function redrawShapes() {{
    Plotly.relayout(plotTimeEl, {{ shapes: labelShapes() }});
  }}

  function findRow(id) {{
    for (let i = 0; i < rows.length; i++) {{
      if (rows[i].id === id) return rows[i];
    }}
    return null;
  }}

  function revertRowInputs(tr, r) {{
    if (!r || !tr) return;
    const t0In = tr.querySelector('.cell-t0');
    const t1In = tr.querySelector('.cell-t1');
    const labIn = tr.querySelector('.cell-label');
    if (t0In) t0In.value = String(r.t0);
    if (t1In) t1In.value = String(r.t1);
    if (labIn) labIn.value = r.label;
  }}

  function applyRowEdit(id) {{
    const r = findRow(id);
    if (!r) return;
    const tr = tbody.querySelector('tr[data-row-id="' + id + '"]');
    if (!tr) return;
    const t0In = tr.querySelector('.cell-t0');
    const t1In = tr.querySelector('.cell-t1');
    const labIn = tr.querySelector('.cell-label');
    const t0 = parseFloat(t0In.value);
    const t1 = parseFloat(t1In.value);
    const label = labIn.value.trim();
    if (!isFinite(t0) || !isFinite(t1)) {{
      alert('Start and end must be valid numbers.');
      revertRowInputs(tr, r);
      return;
    }}
    if (!(t1 > t0)) {{
      alert('End time must be greater than start time.');
      revertRowInputs(tr, r);
      return;
    }}
    if (!label) {{
      alert('Scenario label cannot be empty.');
      revertRowInputs(tr, r);
      return;
    }}
    r.t0 = t0;
    r.t1 = t1;
    r.label = label;
    redrawShapes();
  }}

  function renderTable() {{
    tbody.innerHTML = '';
    rows.forEach(function(r) {{
      const tr = document.createElement('tr');
      tr.setAttribute('data-row-id', String(r.id));

      function mkNumCell(cls, val) {{
        const td = document.createElement('td');
        const inp = document.createElement('input');
        inp.type = 'number';
        inp.step = 'any';
        inp.className = cls;
        inp.value = String(val);
        inp.addEventListener('change', function() {{ applyRowEdit(r.id); }});
        td.appendChild(inp);
        return td;
      }}

      tr.appendChild(mkNumCell('cell-t0', r.t0));
      tr.appendChild(mkNumCell('cell-t1', r.t1));

      const tdLab = document.createElement('td');
      const labIn = document.createElement('input');
      labIn.type = 'text';
      labIn.className = 'cell-label';
      labIn.value = r.label;
      labIn.addEventListener('change', function() {{ applyRowEdit(r.id); }});
      tdLab.appendChild(labIn);
      tr.appendChild(tdLab);

      const tdRm = document.createElement('td');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'danger';
      btn.setAttribute('data-id', String(r.id));
      btn.textContent = 'Remove';
      tdRm.appendChild(btn);
      tr.appendChild(tdRm);

      tbody.appendChild(tr);
    }});
    countEl.textContent = String(rows.length);
    tbody.querySelectorAll('button[data-id]').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        const id = parseInt(btn.getAttribute('data-id'), 10);
        rows = rows.filter(function(x) {{ return x.id !== id; }});
        redrawShapes();
        renderTable();
      }});
    }});
  }}

  function scenarioFromInput() {{
    return scenEl.value.trim();
  }}

  function addRow(t0, t1) {{
    const lab = scenarioFromInput();
    if (!lab) {{
      alert('Enter a label (scenario) before adding.');
      return;
    }}
    if (!(t1 > t0)) {{
      alert('End time must be greater than start time.');
      return;
    }}
    rows.push({{ id: nextId++, t0: t0, t1: t1, label: lab }});
    pendingT = null;
    redrawShapes();
    renderTable();
  }}

  function nearestGpsIndex(t) {{
    const a = payload.gpsT;
    if (!a || !a.length) return -1;
    let lo = 0, hi = a.length - 1;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (a[mid] < t) lo = mid + 1; else hi = mid;
    }}
    let i = lo;
    if (i > 0 && Math.abs(a[i - 1] - t) <= Math.abs(a[i] - t)) i--;
    return i;
  }}

  function speedHoverLineShape(t) {{
    return {{
      type: 'line',
      xref: 'x',
      x0: t,
      x1: t,
      yref: 'y domain',
      y0: 0,
      y1: 1,
      line: {{ color: '#64748b', width: 1.5, dash: 'solid' }},
      layer: 'above',
    }};
  }}

  function syncMapToTime(t) {{
    if (!mapGd || !payload.gpsT || !payload.gpsT.length) return;
    const i = nearestGpsIndex(t);
    if (i < 0) return;
    Plotly.restyle(mapGd, {{
      lat: [[payload.gpsLat[i]]],
      lon: [[payload.gpsLon[i]]],
    }}, [1]);
    Plotly.relayout(mapGd, {{ shapes: [speedHoverLineShape(t)] }});
  }}

  function clearSpeedHoverLine() {{
    if (!mapGd) return;
    Plotly.relayout(mapGd, {{ shapes: [] }});
  }}

  const plotPromises = [
    Plotly.newPlot(plotTimeEl, figTime.data, figTime.layout, configTime).then(function(gd) {{
      shapeBase = (figTime.layout && figTime.layout.shapes) ? figTime.layout.shapes.slice() : [];
      redrawShapes();

      gd.on('plotly_hover', function(ev) {{
        var x = null;
        if (ev.xvals && ev.xvals.length && typeof ev.xvals[0] === 'number') x = ev.xvals[0];
        else if (ev.points && ev.points.length && typeof ev.points[0].x === 'number') x = ev.points[0].x;
        if (x === null || !isFinite(x)) return;
        syncMapToTime(x);
      }});

      gd.on('plotly_unhover', function() {{
        clearSpeedHoverLine();
      }});

      gd.on('plotly_click', function(ev) {{
        if (!ev.points || !ev.points.length) return;
        const x = ev.points[0].x;
        if (typeof x !== 'number' || !isFinite(x)) return;

        const mode = getMode();
        if (mode === 'peak') {{
          const w = parseFloat(peakHalfEl.value);
          const half = (isFinite(w) && w > 0) ? w : 0.25;
          addRow(x - half, x + half);
          return;
        }}

        if (pendingT === null) {{
          pendingT = x;
          redrawShapes();
          return;
        }}
        const a = pendingT;
        const b = x;
        const t0 = Math.min(a, b);
        const t1 = Math.max(a, b);
        if (t0 === t1) {{
          pendingT = x;
          redrawShapes();
          return;
        }}
        addRow(t0, t1);
      }});
      return gd;
    }}),
  ];

  if (figMap && payload.hasMap) {{
    plotPromises.push(
      Plotly.newPlot(plotMapEl, figMap.data, figMap.layout, configMap).then(function(gd) {{
        mapGd = gd;
        return gd;
      }})
    );
  }}

  Promise.all(plotPromises).then(function() {{
    window.requestAnimationFrame(function() {{
      window.requestAnimationFrame(resizeBothPlots);
    }});
  }});

  document.getElementById('btnCancel').addEventListener('click', function() {{
    pendingT = null;
    redrawShapes();
  }});

  function csvEscape(cell) {{
    const s = String(cell);
    if (/[",\\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }}

  function buildCsv() {{
    const rec = recEl.value.trim();
    const sec = secEl.value.trim();
    const src = srcEl.value.trim() || 'manual';
    const lines = [payload.csvHeader.join(',')];
    rows.forEach(function(r) {{
      lines.push([
        'interval',
        csvEscape(rec),
        csvEscape(sec),
        fmt4(r.t0),
        fmt4(r.t1),
        csvEscape(r.label),
        csvEscape(src),
      ].join(','));
    }});
    return lines.join('\\n');
  }}

  document.getElementById('btnDownload').addEventListener('click', function() {{
    if (!recEl.value.trim()) {{
      alert('Set recording_id before download.');
      return;
    }}
    if (!secEl.value.trim()) {{
      if (!confirm('Section ID is empty. Interval labels require section_id for the feature pipeline. Continue?')) return;
    }}
    const blob = new Blob([buildCsv()], {{ type: 'text/csv;charset=utf-8' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = payload.suggestedFilename;
    a.click();
    URL.revokeObjectURL(a.href);
  }});

  const fileLoad = document.getElementById('fileLoad');
  document.getElementById('btnLoad').addEventListener('click', function() {{ fileLoad.click(); }});
  fileLoad.addEventListener('change', function() {{
    const f = fileLoad.files && fileLoad.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = function() {{
      const text = String(reader.result || '');
      const parsed = parseLabelsCsv(text);
      if (parsed.error) {{
        alert(parsed.error);
        return;
      }}
      if (parsed.recordingId) recEl.value = parsed.recordingId;
      if (parsed.sectionId !== null) secEl.value = parsed.sectionId;
      if (parsed.labelSource) srcEl.value = parsed.labelSource;
      rows = parsed.rows;
      nextId = rows.reduce(function(m, r) {{ return Math.max(m, r.id); }}, 0) + 1;
      pendingT = null;
      redrawShapes();
      renderTable();
    }};
    reader.readAsText(f);
    fileLoad.value = '';
  }});

  function parseLabelsCsv(text) {{
    const lines = text.split(/\\r?\\n/).filter(function(l) {{ return l.trim().length; }});
    if (!lines.length) return {{ error: 'Empty file', rows: [] }};
    const header = lines[0].split(',').map(function(h) {{ return h.trim().toLowerCase(); }});
    const need = ['scope', 'recording_id', 'section_id', 'window_start_s', 'window_end_s', 'scenario_label'];
    const idx = {{}};
    need.forEach(function(k) {{
      const i = header.indexOf(k);
      if (i < 0) return;
      idx[k] = i;
    }});
    if (idx.scope === undefined || idx.window_start_s === undefined || idx.window_end_s === undefined)
      return {{ error: 'CSV must include scope, window_start_s, window_end_s', rows: [] }};
    let recordingId = '';
    let sectionId = null;
    let labelSource = '';
    const out = [];
    let maxId = 0;
    for (let li = 1; li < lines.length; li++) {{
      const cells = splitCsvLine(lines[li]);
      const scope = (cells[idx.scope] || '').trim().toLowerCase();
      if (scope !== 'interval') continue;
      const t0 = parseFloat(cells[idx.window_start_s]);
      const t1 = parseFloat(cells[idx.window_end_s]);
      const lab = (cells[idx.scenario_label] || '').trim();
      if (!isFinite(t0) || !isFinite(t1) || !lab) continue;
      if (!recordingId && idx.recording_id !== undefined) recordingId = (cells[idx.recording_id] || '').trim();
      if (sectionId === null && idx.section_id !== undefined) sectionId = (cells[idx.section_id] || '').trim();
      const si = header.indexOf('label_source');
      if (!labelSource && si >= 0) labelSource = (cells[si] || '').trim();
      maxId += 1;
      out.push({{ id: maxId, t0: t0, t1: t1, label: lab }});
    }}
    return {{ rows: out, recordingId: recordingId, sectionId: sectionId, labelSource: labelSource }};
  }}

  function splitCsvLine(line) {{
    const out = [];
    let cur = '';
    let q = false;
    for (let i = 0; i < line.length; i++) {{
      const c = line[i];
      if (q) {{
        if (c === '"') {{
          if (line[i+1] === '"') {{ cur += '"'; i++; }}
          else q = false;
        }} else cur += c;
      }} else {{
        if (c === '"') q = true;
        else if (c === ',') {{ out.push(cur); cur = ''; }}
        else cur += c;
      }}
    }}
    out.push(cur);
    return out;
  }}
}})();
  </script>
</body>
</html>
"""


def write_event_labeler_html(
    target: str | Path,
    out_path: Path | None = None,
    *,
    pixels_per_second: float | None = None,
) -> Path:
    data_dir = resolve_labeling_data_dir(target)
    if out_path is None:
        out_path = data_dir / "event_labeler.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recording_id, section_id = infer_recording_section_ids(data_dir)
    if pixels_per_second is not None:
        log.warning("pixels_per_second is ignored; event labeler uses viewport width.")
    fig_time, fig_map, duration_s, gps_note, gps_payload = build_event_labeler_figure(data_dir)
    fig_time_json = fig_time.to_plotly_json()
    fig_map_json = fig_map.to_plotly_json() if fig_map is not None else None
    map_wrap_class = "hidden" if fig_map is None else ""

    slug = _slug(recording_id)
    sec_part = f"_{section_id}" if section_id else ""
    suggested = f"labels_intervals_{slug}{sec_part}.csv"

    section_warn = ""
    if not section_id:
        section_warn = (
            '<p class="warn">This folder is not under <code>sections/section_N</code>. '
            "Set Section ID manually so exported interval rows match your feature windows.</p>"
        )

    payload = {
        "figureTime": fig_time_json,
        "figureMap": fig_map_json,
        "hasMap": fig_map is not None,
        "durationS": duration_s,
        "gpsT": gps_payload["t"],
        "gpsLat": gps_payload["lat"],
        "gpsLon": gps_payload["lon"],
        "csvHeader": LABEL_CSV_HEADER,
        "suggestedFilename": suggested,
    }
    payload_json = json.dumps(payload, cls=PlotlyJSONEncoder, separators=(",", ":"))
    payload_json = payload_json.replace("</script>", "<\\/script>")

    title_esc = html.escape(f"{recording_id}/{section_id or data_dir.name}")
    plotly_js_url = _plotly_js_cdn_url()
    log.info("Plotly.js CDN: %s", plotly_js_url)
    page = _PAGE_TEMPLATE.format(
        title_esc=title_esc,
        plotly_js_url_esc=html.escape(plotly_js_url, quote=True),
        recording_id_esc=html.escape(recording_id, quote=True),
        section_id_esc=html.escape(section_id, quote=True),
        gps_note_esc=html.escape(gps_note, quote=True),
        section_warn=section_warn,
        map_wrap_class=map_wrap_class,
        map_col_max=MAP_COLUMN_MAX_PX,
        payload_json=payload_json,
    )
    out_path.write_text(page, encoding="utf-8")
    log.info("Wrote %s (%.1f s span, responsive layout)", out_path, duration_s)
    return out_path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not argv:
        print(__doc__ or "Usage: uv run python -m labels.event_labeler <target> [out.html]")
        sys.exit(1)
    target = argv[0]
    out = Path(argv[1]) if len(argv) > 1 else None
    p = write_event_labeler_html(target, out)
    print(p.resolve())


if __name__ == "__main__":
    main()
