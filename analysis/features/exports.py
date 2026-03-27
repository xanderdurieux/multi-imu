"""Thesis-ready consolidated feature tables (bike-only, rider-only, fused)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    "orientation_quality",
    "calibration_quality",
    "upstream_confidence_score",
    "upstream_quality_flags",
    "feature_reliability_summary",
    "exclude_from_downstream",
    "label_source",
    "scenario_label",
    "quality_schema_version",
    "section_quality_score",
    "section_usability_category",
    "window_quality_score",
    "window_quality_label",
    "window_usability_category",
    "section_sync_confidence",
    "section_sync_residual_quality",
    "section_interpolation_burden",
    "section_packet_loss_burden",
    "section_frame_estimation_confidence",
    "section_feature_reliability_score",
]


def _collect_feature_frames(sections_root_path: Path) -> pd.DataFrame:
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
            _rec_name, _sec_idx = parse_section_folder_name(sec_dir.name)
            if "recording_id" not in df.columns:
                df["recording_id"] = _rec_name
            if "section_id" not in df.columns:
                df["section_id"] = sec_dir.name
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _subset_columns(df: pd.DataFrame, prefix: str | None) -> pd.DataFrame:
    meta = [c for c in META_COLS if c in df.columns]
    extra = ["section"] if "section" in df.columns else []
    keep = meta + extra
    if prefix is None:
        feat = [c for c in df.columns if c not in keep]
    else:
        feat = [c for c in df.columns if c.startswith(prefix)]
    cols = [c for c in keep + feat if c in df.columns]
    return df[cols].copy()


def _collect_section_quality_rows(sec_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not sec_root.exists():
        return pd.DataFrame()
    for sec_dir in sorted(sec_root.iterdir()):
        if not sec_dir.is_dir():
            continue
        qf = sec_dir / "quality_metadata.json"
        if not qf.is_file():
            continue
        try:
            rows.append(json.loads(qf.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pd.DataFrame(rows)


def _write_quality_plots(section_q: pd.DataFrame, out: Path) -> None:
    if section_q.empty:
        return
    if "overall_usability_category" in section_q.columns:
        counts = section_q["overall_usability_category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color=["#2ca25f", "#fdae6b", "#de2d26"])
        ax.set_title("Section usability histogram")
        ax.set_ylabel("Sections")
        ax.set_xlabel("Usability category")
        fig.tight_layout()
        fig.savefig(out / "quality_usability_hist.png", dpi=140)
        plt.close(fig)

    if "overall_quality_score" in section_q.columns:
        vals = pd.to_numeric(section_q["overall_quality_score"], errors="coerce").dropna()
        if len(vals):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals, bins=12, color="#3182bd", alpha=0.9)
            ax.set_title("Section quality score distribution")
            ax.set_xlabel("Overall quality score")
            ax.set_ylabel("Sections")
            fig.tight_layout()
            fig.savefig(out / "quality_confidence_hist.png", dpi=140)
            plt.close(fig)


def export_thesis_feature_tables(
    *,
    out_dir: Path | None = None,
    recordings_root_path: Path | None = None,
) -> dict[str, Path]:
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
                _rec_name, _sec_idx = parse_section_folder_name(sec_dir.name)
                flat = dict(data)
                meta = flat.pop("quality_metadata", {})
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        flat[f"quality__{k}"] = v
                flat["recording_id"] = _rec_name
                flat["section_id"] = sec_dir.name
                rows.append(flat)
            except Exception:
                continue

    qcsv = out / "qc_sections_summary.csv"
    qjson = out / "qc_sections_summary.json"
    section_quality_csv = out / "section_quality_summary.csv"
    recording_quality_csv = out / "recording_quality_report.csv"
    if rows:
        qdf = pd.DataFrame(rows)
        qdf.to_csv(qcsv, index=False)
        qjson.write_text(qdf.to_json(orient="records", indent=2), encoding="utf-8")

        sq = _collect_section_quality_rows(sec_root)
        if not sq.empty:
            sq.to_csv(section_quality_csv, index=False)
            rec = (
                sq.groupby("recording_id", dropna=False)
                .agg(
                    sections=("section_id", "count"),
                    mean_quality_score=("overall_quality_score", "mean"),
                    usable_sections=("overall_usability_category", lambda s: int((s == "usable").sum())),
                    caution_sections=("overall_usability_category", lambda s: int((s == "caution").sum())),
                    excluded_sections=("overall_usability_category", lambda s: int((s == "exclude").sum())),
                )
                .reset_index()
            )
            rec.to_csv(recording_quality_csv, index=False)
            _write_quality_plots(sq, out)
    else:
        pd.DataFrame().to_csv(qcsv, index=False)
    log.info("QC summary: %s (%d sections)", qcsv, len(rows))
    return qcsv
