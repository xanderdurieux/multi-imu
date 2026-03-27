"""Unified section/window quality metadata helpers for dual-IMU pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_dataframe
from common.paths import parse_section_folder_name, recording_dir


def quality_label_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "low"
    return "poor"


def usability_category_from_score(score: float) -> str:
    if score >= 0.8:
        return "usable"
    if score >= 0.6:
        return "caution"
    return "exclude"


def _quality_tag_score(tag: str) -> float:
    return {
        "good": 1.0,
        "marginal": 0.65,
        "poor": 0.25,
        "unknown": 0.45,
    }.get(str(tag or "unknown"), 0.45)


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _timing_burden(df: pd.DataFrame) -> tuple[float, float]:
    """Return (interpolation_burden, packet_loss_burden) from timestamp gaps."""
    ts = pd.to_numeric(df.get("timestamp"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ts) < 3:
        return np.nan, np.nan
    dt = np.diff(ts)
    med = float(np.nanmedian(dt))
    if not np.isfinite(med) or med <= 0:
        return np.nan, np.nan
    gap_mask = dt > 1.5 * med
    interp_burden = float(np.mean(gap_mask))
    missing = 0.0
    for g in dt[gap_mask]:
        missing += max(0.0, round(g / med) - 1.0)
    packet_burden = float(missing / max(len(ts), 1))
    return interp_burden, packet_burden


def _sync_metadata(recording_id: str, sync_method: str) -> tuple[str, float, float, list[str]]:
    method = sync_method or "unknown"
    flags: list[str] = []
    corr_score = np.nan
    residual_quality = np.nan
    try:
        path = recording_dir(recording_id) / "synced" / "all_methods.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            method = str(sync_method or data.get("selected_method") or "unknown")
            block = data.get(method, {}) if isinstance(data, dict) else {}
            corr = _safe_float(((block.get("correlation") or {}).get("offset_and_drift")))
            if np.isfinite(corr):
                corr_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
                residual_quality = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
            else:
                flags.append("sync_corr_missing")
        else:
            flags.append("sync_all_methods_missing")
    except Exception:
        flags.append("sync_metadata_parse_error")
    return method, corr_score, residual_quality, flags


def build_section_quality_metadata(
    section_path: Path,
    *,
    orientation_variant: str = "complementary_orientation",
    sync_method: str = "",
) -> dict[str, Any]:
    section_path = Path(section_path)
    section_id = section_path.name
    recording_id, _ = parse_section_folder_name(section_id)

    quality_flags: list[str] = []
    chosen_sync_method, sync_conf, sync_resid_q, sync_flags = _sync_metadata(recording_id, sync_method)
    quality_flags.extend(sync_flags)

    interpolation_vals: list[float] = []
    packet_vals: list[float] = []
    calibration_scores: list[float] = []
    calibration_tags: list[str] = []
    frame_scores: list[float] = []
    orientation_scores: list[float] = []
    orientation_tags: list[str] = []

    for sensor in ("sporsa", "arduino"):
        csv_path = section_path / f"{sensor}.csv"
        if csv_path.exists():
            df = load_dataframe(csv_path)
            ib, pb = _timing_burden(df)
            if np.isfinite(ib):
                interpolation_vals.append(ib)
            if np.isfinite(pb):
                packet_vals.append(pb)

    cal_path = section_path / "calibrated" / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        for sensor in ("sporsa", "arduino"):
            block = cal.get(sensor, {}) if isinstance(cal, dict) else {}
            if not isinstance(block, dict):
                continue
            tag = str(block.get("calibration_quality", "unknown"))
            calibration_tags.append(tag)
            calibration_scores.append(_quality_tag_score(tag))
            ff = block.get("forward_frame_meta") or {}
            conf = _safe_float(ff.get("confidence_score"))
            if np.isfinite(conf):
                frame_scores.append(float(np.clip(conf, 0.0, 1.0)))
    else:
        quality_flags.append("calibration_missing")

    orient_path = section_path / "orientation" / "orientation_stats.json"
    if orient_path.exists():
        ost = json.loads(orient_path.read_text(encoding="utf-8"))
        key = f"__{orientation_variant}"
        for sensor in ("sporsa", "arduino"):
            block = ost.get(sensor, {}) if isinstance(ost, dict) else {}
            entry = block.get(key, {}) if isinstance(block, dict) else {}
            if isinstance(entry, dict):
                tag = str(entry.get("quality", "unknown"))
                orientation_tags.append(tag)
                orientation_scores.append(_quality_tag_score(tag))
    else:
        quality_flags.append("orientation_missing")

    feat_path = section_path / "features" / "features.csv"
    reliability_summary = "not_computed"
    feature_rel_score = np.nan
    if feat_path.exists():
        fdf = pd.read_csv(feat_path)
        if len(fdf):
            cols = [c for c in fdf.columns if c.startswith("feature_reliable__")]
            if cols:
                rel = pd.to_numeric(fdf[cols].stack(), errors="coerce").dropna()
                if len(rel):
                    feature_rel_score = float(np.clip(rel.mean(), 0.0, 1.0))
            if "feature_reliability_summary" in fdf.columns:
                top = (
                    fdf["feature_reliability_summary"]
                    .astype(str)
                    .replace("", "ok")
                    .value_counts(dropna=False)
                    .head(3)
                )
                reliability_summary = "|".join(str(x) for x in top.index.tolist())
    
    interp_burden = float(np.nanmean(interpolation_vals)) if interpolation_vals else np.nan
    packet_burden = float(np.nanmean(packet_vals)) if packet_vals else np.nan
    calibration_quality_score = float(np.nanmean(calibration_scores)) if calibration_scores else np.nan
    frame_conf = float(np.nanmean(frame_scores)) if frame_scores else 0.5
    orientation_quality_score = float(np.nanmean(orientation_scores)) if orientation_scores else np.nan

    burden_score = 1.0 - float(np.clip(np.nanmean([interp_burden, packet_burden]), 0.0, 1.0)) if (
        np.isfinite(interp_burden) or np.isfinite(packet_burden)
    ) else 0.5
    combined = np.nanmean([
        sync_conf,
        sync_resid_q,
        calibration_quality_score,
        frame_conf,
        orientation_quality_score,
        burden_score,
        feature_rel_score,
    ])
    overall_score = float(combined) if np.isfinite(combined) else 0.0
    overall_label = quality_label_from_score(overall_score)
    usability = usability_category_from_score(overall_score)

    return {
        "schema_version": "quality_metadata.v1",
        "recording_id": recording_id,
        "section_id": section_id,
        "chosen_sync_method": chosen_sync_method,
        "sync_confidence": sync_conf,
        "sync_residual_quality": sync_resid_q,
        "interpolation_burden": interp_burden,
        "packet_loss_burden": packet_burden,
        "calibration_quality": "unknown" if not calibration_tags else min(calibration_tags, key=_quality_tag_score),
        "calibration_quality_score": calibration_quality_score,
        "frame_estimation_confidence": frame_conf,
        "orientation_quality": "unknown" if not orientation_tags else min(orientation_tags, key=_quality_tag_score),
        "orientation_quality_score": orientation_quality_score,
        "feature_reliability_summary": reliability_summary,
        "feature_reliability_score": feature_rel_score,
        "overall_quality_score": overall_score,
        "overall_quality_label": overall_label,
        "overall_usability_category": usability,
        "quality_flags": sorted(set(quality_flags)),
    }


def write_section_quality_metadata(section_path: Path, **kwargs: Any) -> Path:
    section_path = Path(section_path)
    payload = build_section_quality_metadata(section_path, **kwargs)
    out = section_path / "quality_metadata.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def attach_window_quality_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cross_conf = pd.to_numeric(out.get("feature_confidence__cross_sensor"), errors="coerce")
    energy_conf = pd.to_numeric(out.get("feature_confidence__energy_ratio"), errors="coerce")
    orient_conf = pd.to_numeric(out.get("feature_confidence__orientation_cross_sensor"), errors="coerce")
    grouped_conf = pd.to_numeric(out.get("feature_confidence__grouped"), errors="coerce")
    upstream = pd.to_numeric(out.get("upstream_confidence_score"), errors="coerce")
    win_score = pd.concat([cross_conf, energy_conf, orient_conf, grouped_conf, upstream], axis=1).mean(axis=1, skipna=True)
    out["window_quality_score"] = win_score.clip(lower=0.0, upper=1.0)
    out["window_quality_label"] = out["window_quality_score"].map(lambda x: quality_label_from_score(float(x)) if pd.notna(x) else "poor")
    out["window_usability_category"] = out["window_quality_score"].map(lambda x: usability_category_from_score(float(x)) if pd.notna(x) else "exclude")
    return out


def filter_by_quality(df: pd.DataFrame, *, min_label: str = "medium") -> pd.DataFrame:
    order = {"poor": 0, "low": 1, "medium": 2, "high": 3}
    floor = order.get(min_label, 2)
    labels = df.get("window_quality_label", pd.Series(["poor"] * len(df))).astype(str)
    keep = labels.map(lambda s: order.get(s, 0) >= floor)
    return df.loc[keep].copy()
