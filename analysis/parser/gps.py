"""Parse GPS tracks from GPX, CSV, or NMEA text logs into a standard DataFrame."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from common.paths import read_csv


GPS_COLUMNS = ("latitude", "longitude", "elevation_m", "time_utc")


def _strip_xml_tag(tag: str) -> str:
    """Return strip xml tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_gpx(path: Path | str) -> pd.DataFrame:
    """Parse GPX 1.0/1.1 track points into ``latitude``, ``longitude``, optional ``elevation_m``, ``time_utc``."""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[dict[str, object]] = []
    for elem in root.iter():
        if _strip_xml_tag(elem.tag) != "trkpt":
            continue
        lat_s = elem.get("lat")
        lon_s = elem.get("lon")
        if lat_s is None or lon_s is None:
            continue
        try:
            lat = float(lat_s)
            lon = float(lon_s)
        except ValueError:
            continue

        elev = pd.NA
        t_utc = pd.NA
        for child in elem:
            tag = _strip_xml_tag(child.tag)
            if tag == "ele" and child.text:
                try:
                    elev = float(child.text.strip())
                except ValueError:
                    elev = pd.NA
            elif tag == "time" and child.text:
                t_utc = pd.to_datetime(child.text.strip(), utc=True, errors="coerce")

        rows.append(
            {
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
                "time_utc": t_utc,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(GPS_COLUMNS))
    return pd.DataFrame(rows)


_NMEA_RMC_RE = re.compile(
    r"""
    ^\$G(?:P|N)RMC,              # GPRMC / GNRMC
    (?P<hms>[^,]*),              # UTC time hhmmss.sss
    (?P<status>[AV]),            # A=valid
    (?P<lat_dm>[^,]*),(?P<ns>[NS]),
    (?P<lon_dm>[^,]*),(?P<ew>[EW]),
    """,
    re.VERBOSE,
)


def _nmea_latlon_to_deg(dm: str, hemi: str) -> float:
    """Return nmea latlon to deg."""
    dm = (dm or "").strip()
    if not dm or dm == "0":
        return float("nan")
    try:
        v = float(dm)
    except ValueError:
        return float("nan")
    deg = int(v // 100)
    minutes = v - deg * 100
    dec = deg + minutes / 60.0
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def parse_nmea(path: Path | str) -> pd.DataFrame:
    """Parse nmea."""
    path = Path(path)
    rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _NMEA_RMC_RE.match(line.strip())
            if not m:
                continue
            if m.group("status") != "A":
                continue
            lat = _nmea_latlon_to_deg(m.group("lat_dm"), m.group("ns"))
            lon = _nmea_latlon_to_deg(m.group("lon_dm"), m.group("ew"))
            if not (pd.notna(lat) and pd.notna(lon)):
                continue
            rows.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "elevation_m": pd.NA,
                    "time_utc": pd.NA,
                }
            )

    if not rows:
        return pd.DataFrame(columns=list(GPS_COLUMNS))
    return pd.DataFrame(rows)


def _normalize_column_name(name: str) -> str:
    """Normalize column name."""
    return name.strip().lower().replace(" ", "_")


def _csv_first_line_is_wrapped_record(path: Path) -> bool:
    """Return csv first line is wrapped record."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                break
        else:
            return False
    parsed = list(csv.reader([line]))
    if not parsed or not parsed[0]:
        return False
    return len(parsed[0]) == 1 and "," in parsed[0][0]


def _read_wrapped_inner_csv(path: Path) -> pd.DataFrame:
    """Parse CSV files where each line is ``\"inner,csv,row\"`` (double CSV encoding)."""
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            outer = list(csv.reader([line]))
            if not outer or len(outer[0]) != 1:
                raise ValueError(
                    f"Expected one outer CSV field per line in {path}; got {len(outer[0]) if outer else 0}."
                )
            inner_rows = list(csv.reader([outer[0][0]]))
            if not inner_rows:
                continue
            rows.append(inner_rows[0])

    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def parse_gps_csv(
    path: Path | str,
    *,
    lat_column: str | None = None,
    lon_column: str | None = None,
) -> pd.DataFrame:
    """Parse gps csv."""
    path = Path(path)
    if _csv_first_line_is_wrapped_record(path):
        df = _read_wrapped_inner_csv(path)
    else:
        df = read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=list(GPS_COLUMNS))

    norm = {_normalize_column_name(c): c for c in df.columns}

    def resolve_lat() -> str | None:
        """Resolve lat."""
        if lat_column:
            return lat_column if lat_column in df.columns else None
        for key in ("latitude", "lat", "y", "gps_lat"):
            if key in norm:
                return norm[key]
        return None

    def resolve_lon() -> str | None:
        """Resolve lon."""
        if lon_column:
            return lon_column if lon_column in df.columns else None
        for key in ("longitude", "lon", "lng", "long", "x", "gps_lon", "gps_lng"):
            if key in norm:
                return norm[key]
        return None

    lc = resolve_lat()
    oc = resolve_lon()
    if lc is None or oc is None:
        raise ValueError(
            f"Could not infer latitude/longitude columns in {path}; "
            f"columns are {list(df.columns)}. Pass lat_column= and lon_column= explicitly."
        )

    out = pd.DataFrame(
        {
            "latitude": pd.to_numeric(df[lc], errors="coerce"),
            "longitude": pd.to_numeric(df[oc], errors="coerce"),
            "elevation_m": pd.NA,
            "time_utc": pd.NA,
            "speed_m_s": pd.NA,
        }
    )

    # Optional elevation / time / speed if present
    for _csv_name, variants, target in (
        ("elevation", ("elevation_m", "elevation", "ele", "alt", "altitude", "altitude_m", "alt_m"), "elevation_m"),
        ("time", ("time_utc", "time", "timestamp", "timestamp_absolute", "timestamp_utc", "datetime"), "time_utc"),
        ("speed", ("speed", "velocity", "ground_speed", "speed_m_s"), "speed_m_s"),
    ):
        col = None
        selected_variant: str | None = None
        for v in variants:
            if v in norm:
                col = norm[v]
                selected_variant = v
                break
        if col is not None:
            if target == "time_utc":
                out[target] = pd.to_datetime(df[col], utc=True, errors="coerce")
            else:
                spd = pd.to_numeric(df[col], errors="coerce")
                # Many GPS exports store speed in km/h while this code exposes it
                # as `speed_m_s`. Convert unless the source column already looks
                # like m/s.
                if target == "speed_m_s" and selected_variant != "speed_m_s":
                    spd = spd / 3.6
                out[target] = spd

    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def write_gps_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write gps csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "time_utc" in out.columns:
        ts = pd.to_datetime(out["time_utc"], utc=True, errors="coerce")
        out["time_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    out.to_csv(path, index=False)


def load_gps(path: Path | str) -> pd.DataFrame:
    """Load gps."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".gpx":
        df = parse_gpx(path)
    elif suf == ".csv":
        df = parse_gps_csv(path)
    elif suf in (".nmea", ".txt"):
        df = parse_nmea(path)
    else:
        raise ValueError(f"Unsupported GPS file type {suf!r} for {path}")

    if df.empty:
        return df

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    for col in ("elevation_m", "time_utc"):
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(GPS_COLUMNS)]
