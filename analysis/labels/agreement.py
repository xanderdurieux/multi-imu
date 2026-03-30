"""Inter-rater agreement utilities for thesis-scale doubly labeled subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

KEY_COLS = [
    "scope",
    "recording_id",
    "section_id",
    "window_start_s",
    "window_end_s",
    "event_id",
    "event_type",
    "event_time_s",
    "annotation_id",
]


def _normalize_key(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in KEY_COLS if c in df.columns]
    if "annotation_id" in cols:
        k = df["annotation_id"].astype(str).str.strip()
        if (k != "").all():
            return k
    return df[cols].fillna("").astype(str).agg("::".join, axis=1)


def compute_inter_rater_agreement(
    labels_a_path: Path | str,
    labels_b_path: Path | str,
    *,
    out_dir: Path,
    label_col: str = "scenario_label",
) -> dict[str, Path]:
    a = pd.read_csv(labels_a_path)
    b = pd.read_csv(labels_b_path)
    a = a.copy()
    b = b.copy()
    a["_key"] = _normalize_key(a)
    b["_key"] = _normalize_key(b)

    ma = a[["_key", label_col] + [c for c in ["scope", "recording_id", "section_id"] if c in a.columns]].rename(
        columns={label_col: "label_a"}
    )
    mb = b[["_key", label_col] + [c for c in ["scope", "recording_id", "section_id"] if c in b.columns]].rename(
        columns={label_col: "label_b"}
    )
    m = ma.merge(mb[["_key", "label_b"]], on="_key", how="inner")
    m = m[(m["label_a"].notna()) & (m["label_b"].notna())].copy()
    m["label_a"] = m["label_a"].astype(str)
    m["label_b"] = m["label_b"].astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    if m.empty:
        payload: dict[str, Any] = {"n_overlap": 0, "warning": "No overlapping keys between label files."}
        (out_dir / "inter_rater_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(out_dir / "inter_rater_by_scope.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "inter_rater_disagreements.csv", index=False)
        return {
            "summary": out_dir / "inter_rater_summary.json",
            "by_scope": out_dir / "inter_rater_by_scope.csv",
            "disagreements": out_dir / "inter_rater_disagreements.csv",
        }

    exact = float((m["label_a"] == m["label_b"]).mean())
    kappa = float(cohen_kappa_score(m["label_a"], m["label_b"])) if len(m) > 1 else float("nan")

    by_scope_rows: list[dict[str, Any]] = []
    if "scope" in m.columns:
        for scope, g in m.groupby("scope", dropna=False):
            if len(g) < 2:
                ks = np.nan
            else:
                ks = float(cohen_kappa_score(g["label_a"], g["label_b"]))
            by_scope_rows.append(
                {
                    "scope": scope,
                    "n": int(len(g)),
                    "percent_agreement": float((g["label_a"] == g["label_b"]).mean()),
                    "cohen_kappa": ks,
                }
            )
    by_scope = pd.DataFrame(by_scope_rows)
    disagreements = m[m["label_a"] != m["label_b"]].copy()

    summary = {
        "n_overlap": int(len(m)),
        "percent_agreement": exact,
        "cohen_kappa": kappa,
        "n_disagreements": int(len(disagreements)),
        "label_col": label_col,
        "inputs": {"labels_a": str(labels_a_path), "labels_b": str(labels_b_path)},
    }
    (out_dir / "inter_rater_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_scope.to_csv(out_dir / "inter_rater_by_scope.csv", index=False)
    disagreements.to_csv(out_dir / "inter_rater_disagreements.csv", index=False)
    return {
        "summary": out_dir / "inter_rater_summary.json",
        "by_scope": out_dir / "inter_rater_by_scope.csv",
        "disagreements": out_dir / "inter_rater_disagreements.csv",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compute inter-rater agreement for doubly labeled thesis subset.")
    p.add_argument("--labels-a", type=Path, required=True)
    p.add_argument("--labels-b", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label-col", type=str, default="scenario_label")
    args = p.parse_args()
    out = compute_inter_rater_agreement(
        args.labels_a,
        args.labels_b,
        out_dir=args.out_dir,
        label_col=args.label_col,
    )
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
