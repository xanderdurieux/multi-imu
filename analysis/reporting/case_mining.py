from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CaseMiningResult:
    cases: pd.DataFrame
    diagnostics: list[str]


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _downstream_proxy_score(features_path: Path) -> float:
    if not features_path.exists():
        return np.nan
    try:
        df = pd.read_csv(features_path)
    except Exception:
        return np.nan
    if df.empty or "scenario_label" not in df.columns:
        return np.nan
    labels = df["scenario_label"].astype(str).str.strip()
    valid = labels != ""
    df = df.loc[valid].copy()
    labels = labels.loc[valid]
    if df.empty or labels.nunique() < 2:
        return np.nan

    preferred = [
        c
        for c in df.columns
        if c.startswith(("cross_sensor__", "feature_confidence__cross_sensor"))
    ]
    numeric = df[preferred] if preferred else df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.nan

    scores: list[float] = []
    for col in numeric.columns:
        s = pd.to_numeric(numeric[col], errors="coerce")
        if s.notna().sum() < 8:
            continue
        grouped = [s[labels == lab].dropna().to_numpy() for lab in sorted(labels.unique())]
        grouped = [g for g in grouped if len(g) >= 3]
        if len(grouped) < 2:
            continue
        means = np.array([np.mean(g) for g in grouped])
        vars_ = np.array([np.var(g, ddof=0) + 1e-9 for g in grouped])
        scores.append(float(np.var(means, ddof=0) / np.mean(vars_)))
    if not scores:
        return np.nan
    return float(np.nanpercentile(scores, 75))


def mine_success_failure_cases(
    section_dirs: list[Path],
    *,
    n_success: int = 2,
    n_failure: int = 2,
) -> CaseMiningResult:
    rows: list[dict] = []
    diagnostics: list[str] = []
    for section in section_dirs:
        qm = _safe_json(section / "quality_metadata.json")
        qc = _safe_json(section / "qc_section.json")
        if not qm and not qc:
            continue

        qc_tier = str(qc.get("quality_tier") or "unknown").lower()
        tier_map = {"good": 1.0, "marginal": 0.55, "poor": 0.2, "bad": 0.1, "unknown": 0.35}
        qc_score = tier_map.get(qc_tier, 0.35)

        score = pd.to_numeric(pd.Series([qm.get("overall_quality_score", np.nan)]), errors="coerce").iloc[0]
        sync_conf = pd.to_numeric(pd.Series([qm.get("sync_confidence", np.nan)]), errors="coerce").iloc[0]
        orient_conf = pd.to_numeric(pd.Series([qm.get("orientation_quality_score", np.nan)]), errors="coerce").iloc[0]
        downstream_proxy = _downstream_proxy_score(section / "features" / "features.csv")

        components = {
            "overall_quality_score": score,
            "sync_confidence": sync_conf,
            "orientation_quality_score": orient_conf,
            "qc_signal": qc_score,
            "downstream_proxy_score": downstream_proxy,
        }
        valid_values = [float(v) for v in components.values() if pd.notna(v)]
        if not valid_values:
            continue
        composite = float(np.mean(valid_values))

        rows.append(
            {
                "section_id": section.name,
                "composite_signal_score": composite,
                "overall_quality_score": score,
                "sync_confidence": sync_conf,
                "orientation_quality_score": orient_conf,
                "qc_tier": qc_tier,
                "qc_signal": qc_score,
                "downstream_proxy_score": downstream_proxy,
                "quality_label": qm.get("overall_quality_label", "unknown"),
                "usability": qm.get("overall_usability_category", "unknown"),
                "key_flags": "|".join((qm.get("quality_flags") or [])[:3]),
            }
        )

    if not rows:
        diagnostics.append("No sections with usable quality metadata or QC signals were found.")
        return CaseMiningResult(cases=pd.DataFrame(), diagnostics=diagnostics)

    df = pd.DataFrame(rows).sort_values("composite_signal_score")
    if len(df) < max(n_success, n_failure) * 2:
        diagnostics.append(
            f"Only {len(df)} case candidates are available; at least {max(n_success, n_failure) * 2} are recommended for balanced success/failure exemplars."
        )

    worst = df.head(n_failure).assign(case_type="failure")
    best = df.tail(n_success).sort_values("composite_signal_score", ascending=False).assign(case_type="success")
    chosen = pd.concat([worst, best], ignore_index=True)
    return CaseMiningResult(cases=chosen, diagnostics=diagnostics)
