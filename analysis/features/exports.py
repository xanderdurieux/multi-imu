"""Thesis-ready consolidated feature tables (bike-only, rider-only, fused)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import recordings_root, sections_root
from common.paths import parse_section_folder_name

log = logging.getLogger(__name__)

META_COLS = [
    "recording_id",
    "section_id",
    "window_start_s",
    "window_end_s",
    "window_center_s",
    "sync_method",
    "orientation_method",
    "calibration_quality",
    "label_source",
    "scenario_label",
]


def _collect_feature_frames(sections_root_path: Path) -> pd.DataFrame:
    """Concatenate all ``sections/*/features/features.csv`` under sections root."""
    all_dfs: list[pd.DataFrame] = []
    if not sections_root_path.exists():
        return pd.DataFrame()
    for sec_dir in sorted(sections_root_path.iterdir()):
        if not sec_dir.is_dir():
            continue
        fp = sec_dir / "features" / "features.csv"
        if not fp.is_file():
            continue
        try:
            df = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if "recording_id" not in df.columns or "section_id" not in df.columns:
            # Derive missing metadata from folder name.
            _rec_name, sec_idx = parse_section_folder_name(sec_dir.name)
            if "recording_id" not in df.columns:
                df["recording_id"] = _rec_name
            if "section_id" not in df.columns:
                df["section_id"] = sec_dir.name
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _subset_columns(df: pd.DataFrame, prefix: str | None) -> pd.DataFrame:
    """Keep metadata columns and optionally all columns with *prefix*."""
    meta = [c for c in META_COLS if c in df.columns]
    extra = ["section"] if "section" in df.columns else []
    keep = meta + extra
    if prefix is None:
        feat = [c for c in df.columns if c not in keep]
    else:
        feat = [c for c in df.columns if c.startswith(prefix)]
    cols = keep + feat
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def export_thesis_feature_tables(
    *,
    out_dir: Path | None = None,
    recordings_root_path: Path | None = None,
) -> dict[str, Path]:
    """Write bike-only, rider-only, and fused feature CSVs plus a small manifest.

    Sporsa = bicycle-mounted IMU, Arduino = rider-mounted IMU (thesis convention).

    Returns
    -------
    dict
        Keys ``bike``, ``rider``, ``fused``, ``manifest`` mapping to written paths.
    """
    recordings_root_path = recordings_root_path or recordings_root()
    data_root = recordings_root_path.parent
    out = Path(out_dir) if out_dir is not None else data_root / "exports"
    out.mkdir(parents=True, exist_ok=True)

    full = _collect_feature_frames(data_root / "sections")
    if full.empty:
        log.warning("No feature rows found under %s", data_root / "sections")
        empty = out / "features_bike.csv"
        pd.DataFrame().to_csv(empty, index=False)
        return {
            "bike": empty,
            "rider": out / "features_rider.csv",
            "fused": out / "features_fused.csv",
            "manifest": out / "export_manifest.json",
        }

    bike = _subset_columns(full, "sporsa__")
    rider = _subset_columns(full, "arduino__")
    fused = full.copy()

    paths = {
        "bike": out / "features_bike.csv",
        "rider": out / "features_rider.csv",
        "fused": out / "features_fused.csv",
    }
    bike.to_csv(paths["bike"], index=False)
    rider.to_csv(paths["rider"], index=False)
    fused.to_csv(paths["fused"], index=False)

    manifest: dict[str, Any] = {
        "n_rows": len(full),
        "n_recordings": int(full["recording_id"].nunique()) if "recording_id" in full.columns else 0,
        "bike_columns": list(bike.columns),
        "rider_columns": list(rider.columns),
        "fused_columns": len(fused.columns),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    mpath = out / "export_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = mpath
    log.info("Wrote exports under %s (%d rows)", out, len(full))
    return paths


def export_qc_summaries(
    *,
    out_dir: Path | None = None,
    recordings_root_path: Path | None = None,
) -> Path:
    """Aggregate per-section QC JSON files if present (from validation pipeline)."""
    recordings_root_path = recordings_root_path or recordings_root()
    data_root = recordings_root_path.parent
    out = Path(out_dir) if out_dir is not None else data_root / "exports"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    qc_name = "qc_section.json"
    sec_root = data_root / "sections"
    if not sec_root.exists():
        qcsv = out / "qc_sections_summary.csv"
        pd.DataFrame().to_csv(qcsv, index=False)
        log.info("QC summary: %s (0 sections)", qcsv)
        return qcsv

    for sec_dir in sorted(sec_root.iterdir()):
        if not sec_dir.is_dir():
            continue
        qf = sec_dir / qc_name
        if qf.is_file():
            try:
                data = json.loads(qf.read_text(encoding="utf-8"))
                _rec_name, sec_idx = parse_section_folder_name(sec_dir.name)
                data["recording_id"] = _rec_name
                data["section_id"] = sec_dir.name
                rows.append(data)
            except Exception:
                continue

    qcsv = out / "qc_sections_summary.csv"
    if rows:
        pd.DataFrame(rows).to_csv(qcsv, index=False)
    else:
        pd.DataFrame().to_csv(qcsv, index=False)
    log.info("QC summary: %s (%d sections)", qcsv, len(rows))
    return qcsv
