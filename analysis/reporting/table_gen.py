from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import sections_root

from .case_mining import mine_success_failure_cases
from .figure_gen import STATUS_MISSING, STATUS_REAL


@dataclass
class TableArtifact:
    key: str
    status: str
    note: str
    path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)


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
        return TableArtifact(
            "quality_summary",
            STATUS_MISSING,
            "No quality metadata found.",
            diagnostics=["Missing prerequisite: sections/*/quality_metadata.json or qc_section.json"],
        )

    df = pd.DataFrame(rows).sort_values("overall_quality_score", ascending=False, na_position="last")
    path = _write(df, out_stem)
    return TableArtifact("quality_summary", STATUS_REAL, f"sections={len(df)}", path)


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
        return TableArtifact(
            "sync_method_comparison",
            STATUS_MISSING,
            "No synchronization metadata found.",
            diagnostics=["Missing prerequisite: sections/*/quality_metadata.json with chosen_sync_method/sync_* fields."],
        )

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
    return TableArtifact("sync_method_comparison", STATUS_REAL, f"methods={len(agg)}", path)


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
        return TableArtifact(
            "single_vs_dual_sensor_comparison",
            STATUS_MISSING,
            "No labeled feature sets found for sensor-source comparison.",
            diagnostics=["Missing prerequisite: sections/*/features/features.csv with scenario_label and source-prefixed feature columns."],
        )

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
    return TableArtifact("single_vs_dual_sensor_comparison", STATUS_REAL, f"sections={len(sec_df)}", path)


def make_case_study_table(out_dir: Path) -> TableArtifact:
    mined = mine_success_failure_cases(_sections(), n_success=2, n_failure=2)
    out_stem = out_dir / "case_studies"
    if mined.cases.empty:
        return TableArtifact(
            "case_studies",
            STATUS_MISSING,
            "Could not mine representative success/failure cases.",
            diagnostics=mined.diagnostics,
        )

    keep_cols = [
        "case_type",
        "section_id",
        "composite_signal_score",
        "overall_quality_score",
        "sync_confidence",
        "orientation_quality_score",
        "downstream_proxy_score",
        "qc_tier",
        "quality_label",
        "usability",
        "key_flags",
    ]
    df = mined.cases[keep_cols].sort_values(["case_type", "composite_signal_score"], ascending=[True, False])
    path = _write(df, out_stem)
    return TableArtifact("case_studies", STATUS_REAL, f"cases={len(df)}", path, diagnostics=mined.diagnostics)
