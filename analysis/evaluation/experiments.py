"""Compare feature sets and run simple baselines with leave-one-recording-out CV."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _numeric_feature_matrix(
    df: pd.DataFrame,
    *,
    exclude: frozenset[str],
) -> tuple[pd.DataFrame, list[str]]:
    num = df.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if c not in exclude]
    return df[cols], cols


def feature_missingness(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-column NaN fraction."""
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"column": c, "nan_fraction": 1.0})
        else:
            rows.append({"column": c, "nan_fraction": float(df[c].isna().mean())})
    return pd.DataFrame(rows)


def class_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).value_counts(dropna=False)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges-style simple Cohen *d* (pooled std)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v = (
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / max(len(a) + len(b) - 2, 1)
    )
    sp = np.sqrt(v) if v > 0 else float("nan")
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def within_between_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Per-feature within-class and between-class variance of window means."""
    rows = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        sub = df[[label_col, c]].dropna()
        if sub.empty:
            continue
        means = sub.groupby(label_col, observed=False)[c].mean()
        overall = float(sub[c].mean())
        between = float(np.var(means.to_numpy(dtype=float), ddof=1)) if len(means) > 1 else 0.0
        within_w = 0.0
        wsum = 0
        for lab, grp in sub.groupby(label_col, observed=False):
            arr = grp[c].to_numpy(dtype=float)
            if len(arr) < 2:
                continue
            within_w += float((len(arr) - 1) * np.var(arr, ddof=1))
            wsum += len(arr) - 1
        within = float(within_w / wsum) if wsum > 0 else float("nan")
        rows.append(
            {
                "feature": c,
                "var_between_class_means": between,
                "var_within_classes_pooled": within,
                "ratio_between_within": between / within if within and within > 1e-18 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation_report(
    features_fused_csv: Path,
    out_dir: Path,
    *,
    label_col: str = "scenario_label",
    group_col: str = "recording_id",
    min_per_class: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    """Load fused features, compute descriptives, optional LORO classifiers.

    Requires ``scikit-learn`` for classifier metrics; if import fails, only descriptives run.

    Parameters
    ----------
    features_fused_csv:
        Typically ``data/exports/features_fused.csv`` from :mod:`features.exports`.
    out_dir:
        Directory for ``evaluation_summary.json``, CSV tables, etc.
    """
    features_fused_csv = Path(features_fused_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_fused_csv)
    meta = frozenset(
        {
            label_col,
            group_col,
            "section_id",
            "section",
            "window_start_s",
            "window_end_s",
            "window_center_s",
            "sync_method",
            "orientation_method",
            "calibration_quality",
            "label_source",
        }
    )
    _, feat_cols = _numeric_feature_matrix(df, exclude=meta)
    feat_cols = [c for c in feat_cols if not c.startswith("_")]

    miss = feature_missingness(df, feat_cols[:200])
    miss_path = out_dir / "feature_missingness.csv"
    miss.to_csv(miss_path, index=False)

    counts = class_counts(df, label_col)
    counts_path = out_dir / "class_counts.csv"
    counts.to_frame("count").to_csv(counts_path)

    wb = within_between_variance(df, feat_cols[: min(80, len(feat_cols))], label_col)
    wb_path = out_dir / "variance_within_between.csv"
    wb.to_csv(wb_path, index=False)

    classes = [c for c in counts.index.astype(str) if c and str(c).lower() != "nan"]
    effect_rows = []
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            for c in feat_cols[:40]:
                sa = df.loc[df[label_col].astype(str) == a, c].to_numpy(dtype=float)
                sb = df.loc[df[label_col].astype(str) == b, c].to_numpy(dtype=float)
                effect_rows.append(
                    {
                        "class_a": a,
                        "class_b": b,
                        "feature": c,
                        "cohen_d": cohen_d(sa, sb),
                    }
                )
    fx = pd.DataFrame(effect_rows)
    fx_path = out_dir / "effect_sizes_pairs.csv"
    fx.to_csv(fx_path, index=False)

    summary: dict[str, Any] = {
        "n_rows": len(df),
        "n_features_numeric_considered": len(feat_cols),
        "label_col": label_col,
        "group_col": group_col,
        "outputs": {
            "missingness": str(miss_path),
            "class_counts": str(counts_path),
            "variance": str(wb_path),
            "effects": str(fx_path),
        },
        "classifiers": {},
    }

    # --- Feature set definitions for comparison ---
    bike_cols = [c for c in feat_cols if c.startswith("sporsa__")]
    rider_cols = [c for c in feat_cols if c.startswith("arduino__")]
    cross_cols = [
        c
        for c in feat_cols
        if not c.startswith("sporsa__") and not c.startswith("arduino__")
    ]
    sets = {
        "bike_only": bike_cols,
        "rider_only": rider_cols,
        "fused_dual": feat_cols,
        "cross_only": cross_cols,
    }

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("scikit-learn not installed — skipping classifiers")
        summary_path = out_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary_path, "missingness": miss_path}

    y_raw = df[label_col].astype(str)
    valid = y_raw.notna() & (y_raw.str.strip() != "") & (y_raw.str.lower() != "nan")
    dfv = df.loc[valid].copy()
    y = y_raw[valid]
    groups = dfv[group_col].astype(str)

    clf_results: list[dict[str, Any]] = []
    for set_name, cols in sets.items():
        use = [c for c in cols if c in dfv.columns]
        if len(use) < 2:
            continue
        vc = y.value_counts()
        if (vc < min_per_class).any() or len(vc) < 2:
            log.warning("Skipping %s: need ≥2 classes each with count ≥ %d", set_name, min_per_class)
            continue

        X = dfv[use].replace([np.inf, -np.inf], np.nan)
        logo = LeaveOneGroupOut()
        fold_acc_lr: list[float] = []
        fold_acc_rf: list[float] = []

        for train_i, test_i in logo.split(X, y, groups):
            X_tr, X_te = X.iloc[train_i], X.iloc[test_i]
            y_tr, y_te = y.iloc[train_i], y.iloc[test_i]
            if len(np.unique(y_tr)) < 2:
                continue

            pipe_lr = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(max_iter=500, random_state=random_state),
                    ),
                ]
            )
            pipe_rf = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=random_state,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            )
            pipe_lr.fit(X_tr, y_tr)
            pipe_rf.fit(X_tr, y_tr)
            fold_acc_lr.append(float(pipe_lr.score(X_te, y_te)))
            fold_acc_rf.append(float(pipe_rf.score(X_te, y_te)))

        if fold_acc_lr:
            clf_results.append(
                {
                    "feature_set": set_name,
                    "n_features": len(use),
                    "logo_mean_acc_logistic_regression": float(np.mean(fold_acc_lr)),
                    "logo_std_acc_logistic_regression": float(np.std(fold_acc_lr)),
                    "logo_mean_acc_random_forest": float(np.mean(fold_acc_rf)),
                    "logo_std_acc_random_forest": float(np.std(fold_acc_rf)),
                    "n_folds": len(fold_acc_lr),
                }
            )

    clf_df = pd.DataFrame(clf_results)
    clf_path = out_dir / "classifier_loro_summary.csv"
    clf_df.to_csv(clf_path, index=False)
    summary["classifiers"] = clf_results
    summary["outputs"]["classifiers"] = str(clf_path)

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote evaluation summary to %s", summary_path)
    return {
        "summary": summary_path,
        "missingness": miss_path,
        "classifiers": clf_path,
    }
