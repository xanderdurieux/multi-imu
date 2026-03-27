from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import sections_root


@dataclass
class TableArtifact:
    key: str
    path: Path
    status: str
    note: str


def _sections() -> list[Path]:
    root = sections_root()
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir()])


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(df: pd.DataFrame, out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    md_path = out_stem.with_suffix(".md")
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return out_stem.with_suffix(".csv")


def make_quality_summary_table(out_dir: Path) -> TableArtifact:
    rows = []
    for section in _sections():
        qm = _read_json(section / "quality_metadata.json")
        qc = _read_json(section / "qc_section.json")
        if not qm and not qc:
            continue
        rows.append(
            {
                "section_id": section.name,
                "overall_quality_score": qm.get("overall_quality_score", np.nan),
                "overall_quality_label": qm.get("overall_quality_label", "unknown"),
                "usability": qm.get("overall_usability_category", "unknown"),
                "qc_tier": qc.get("quality_tier", "unknown"),
                "sync_confidence": qm.get("sync_confidence", np.nan),
                "orientation_quality_score": qm.get("orientation_quality_score", np.nan),
            }
        )
    out_stem = out_dir / "quality_summary"
    if not rows:
        df = pd.DataFrame([
            {
                "section_id": "N/A",
                "overall_quality_score": np.nan,
                "overall_quality_label": "no_data",
                "usability": "no_data",
                "qc_tier": "no_data",
                "sync_confidence": np.nan,
                "orientation_quality_score": np.nan,
            }
        ])
        path = _write(df, out_stem)
        return TableArtifact("quality_summary", path, "placeholder", "no quality metadata found")

    df = pd.DataFrame(rows).sort_values("overall_quality_score", ascending=False, na_position="last")
    path = _write(df, out_stem)
    return TableArtifact("quality_summary", path, "ok", f"sections={len(df)}")


def make_sync_method_comparison_table(out_dir: Path) -> TableArtifact:
    rows = []
    for section in _sections():
        qm = _read_json(section / "quality_metadata.json")
        if not qm:
            continue
        rows.append(
            {
                "section_id": section.name,
                "sync_method": qm.get("chosen_sync_method", "unknown"),
                "sync_confidence": qm.get("sync_confidence", np.nan),
                "sync_residual_quality": qm.get("sync_residual_quality", np.nan),
            }
        )
    out_stem = out_dir / "sync_method_comparison"
    if not rows:
        df = pd.DataFrame([{"sync_method": "no_data", "n_sections": 0, "sync_confidence_median": np.nan, "sync_residual_quality_median": np.nan}])
        path = _write(df, out_stem)
        return TableArtifact("sync_method_comparison", path, "placeholder", "no sync metadata")

    sec_df = pd.DataFrame(rows)
    agg = (
        sec_df.groupby("sync_method", as_index=False)
        .agg(
            n_sections=("section_id", "count"),
            sync_confidence_median=("sync_confidence", "median"),
            sync_residual_quality_median=("sync_residual_quality", "median"),
        )
        .sort_values(["n_sections", "sync_confidence_median"], ascending=[False, False])
    )
    path = _write(agg, out_stem)
    return TableArtifact("sync_method_comparison", path, "ok", f"methods={len(agg)}")


def make_single_vs_dual_sensor_table(out_dir: Path) -> TableArtifact:
    rows = []
    for section in _sections():
        feat = section / "features" / "features.csv"
        if not feat.exists():
            continue
        df = pd.read_csv(feat)
        if df.empty or "scenario_label" not in df.columns:
            continue
        y = df["scenario_label"].astype(str)
        if y.nunique() < 2:
            continue

        def group_score(prefixes: tuple[str, ...]) -> float:
            cols = [c for c in df.columns if c.startswith(prefixes)]
            if not cols:
                return np.nan
            vals = pd.to_numeric(df[cols].stack(), errors="coerce").dropna()
            if len(vals) < 5:
                return np.nan
            group_means = df[cols].groupby(y).mean(numeric_only=True)
            inter = float(group_means.var(axis=0, ddof=0).mean())
            intra = float(df[cols].var(axis=0, ddof=0).mean())
            return inter / max(intra, 1e-9)

        rows.append(
            {
                "section_id": section.name,
                "sporsa_only_score": group_score(("sporsa__",)),
                "arduino_only_score": group_score(("arduino__",)),
                "dual_sensor_score": group_score(("cross_sensor__", "feature_confidence__cross_sensor")),
            }
        )
    out_stem = out_dir / "single_vs_dual_sensor_comparison"
    if not rows:
        df = pd.DataFrame([{"comparison": "no_data", "sporsa_only_median": np.nan, "arduino_only_median": np.nan, "dual_sensor_median": np.nan}])
        path = _write(df, out_stem)
        return TableArtifact("single_vs_dual_sensor_comparison", path, "placeholder", "no labeled feature sets")

    sec_df = pd.DataFrame(rows)
    agg = pd.DataFrame(
        [
            {
                "sporsa_only_median": sec_df["sporsa_only_score"].median(),
                "arduino_only_median": sec_df["arduino_only_score"].median(),
                "dual_sensor_median": sec_df["dual_sensor_score"].median(),
                "n_sections": len(sec_df),
            }
        ]
    )
    path = _write(agg, out_stem)
    return TableArtifact("single_vs_dual_sensor_comparison", path, "ok", f"sections={len(sec_df)}")


def make_case_study_table(out_dir: Path) -> TableArtifact:
    rows = []
    for section in _sections():
        qm = _read_json(section / "quality_metadata.json")
        qc = _read_json(section / "qc_section.json")
        if not qm:
            continue
        rows.append(
            {
                "section_id": section.name,
                "overall_quality_score": qm.get("overall_quality_score", np.nan),
                "usability": qm.get("overall_usability_category", "unknown"),
                "quality_label": qm.get("overall_quality_label", "unknown"),
                "qc_tier": qc.get("quality_tier", "unknown"),
                "key_flags": "|".join(qm.get("quality_flags", [])[:3]),
            }
        )
    out_stem = out_dir / "case_studies"
    if len(rows) < 2:
        df = pd.DataFrame([{"section_id": "N/A", "quality_label": "no_data", "key_flags": "insufficient_quality_metadata"}])
        path = _write(df, out_stem)
        return TableArtifact("case_studies", path, "placeholder", "insufficient cases")

    df = pd.DataFrame(rows).dropna(subset=["overall_quality_score"]).sort_values("overall_quality_score")
    subset = pd.concat([df.head(2), df.tail(2)], ignore_index=True)
    path = _write(subset, out_stem)
    return TableArtifact("case_studies", path, "ok", "2 failure + 2 success exemplars")
