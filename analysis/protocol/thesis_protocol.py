from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_locked_split_manifest(path: Path | str) -> dict[str, set[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = data.get("splits", data)
    if not isinstance(splits, dict):
        raise ValueError("Split manifest must be an object with split lists")
    out: dict[str, set[str]] = {}
    for name, vals in splits.items():
        if not isinstance(vals, list):
            raise ValueError(f"Split '{name}' must be a list")
        out[name] = {str(v).strip() for v in vals if str(v).strip()}
    keys = sorted(out)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = out[a] & out[b]
            if overlap:
                raise ValueError(f"Locked split leakage detected between '{a}' and '{b}': {sorted(overlap)}")
    return out


def _section_qc(section_dir: Path) -> dict[str, Any]:
    qc_path = section_dir / "qc_section.json"
    if not qc_path.is_file():
        return {"quality_tier": "unknown", "reasons": ["qc_missing"], "quality_metadata": {}}
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except Exception:
        return {"quality_tier": "unknown", "reasons": ["qc_parse_error"], "quality_metadata": {}}
    if not isinstance(qc, dict):
        return {"quality_tier": "unknown", "reasons": ["qc_invalid"], "quality_metadata": {}}
    return qc


def evaluate_qc_sections(
    section_ids: list[str],
    *,
    sections_root: Path,
    policy: dict[str, Any],
) -> pd.DataFrame:
    allow_tiers = set(policy.get("allow_qc_tiers", ["good", "marginal"]))
    min_cal = float(policy.get("min_calibration_confidence", 0.35))
    min_orient = float(policy.get("min_orientation_confidence", 0.35))
    rows: list[dict[str, Any]] = []
    for section_id in sorted(set(section_ids)):
        qc = _section_qc(sections_root / section_id)
        qmeta = qc.get("quality_metadata") if isinstance(qc.get("quality_metadata"), dict) else {}
        tier = str(qc.get("quality_tier") or "unknown")
        cal_conf = qmeta.get("frame_estimation_confidence")
        orient_conf = qmeta.get("orientation_quality_score")
        try:
            cal_conf_f = float(cal_conf)
        except Exception:
            cal_conf_f = np.nan
        try:
            orient_conf_f = float(orient_conf)
        except Exception:
            orient_conf_f = np.nan

        reasons: list[str] = []
        if tier not in allow_tiers:
            reasons.append(f"qc_tier:{tier}")
        if np.isnan(cal_conf_f):
            reasons.append("calibration_confidence_missing")
        elif cal_conf_f < min_cal:
            reasons.append(f"calibration_confidence_below:{min_cal:.2f}")
        if np.isnan(orient_conf_f):
            reasons.append("orientation_confidence_missing")
        elif orient_conf_f < min_orient:
            reasons.append(f"orientation_confidence_below:{min_orient:.2f}")

        qc_reasons = [str(x).strip() for x in qc.get("reasons", []) if str(x).strip()]
        rows.append(
            {
                "section_id": section_id,
                "include": len(reasons) == 0,
                "exclusion_reasons": "|".join(sorted(set(reasons))),
                "qc_tier": tier,
                "qc_reasons": "|".join(qc_reasons),
                "calibration_confidence": cal_conf_f,
                "orientation_confidence": orient_conf_f,
            }
        )
    return pd.DataFrame(rows)
