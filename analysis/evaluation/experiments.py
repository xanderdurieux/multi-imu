"""Config-driven experiment layer for thesis-ready dual-IMU evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "label_col": "scenario_label",
    "group_col": "recording_id",
    "sync_col": "sync_method",
    "orientation_col": "orientation_method",
    "min_samples_per_class": 6,
    "pr_curve_minority_ratio_threshold": 0.33,
    "max_effect_size_features": 60,
    "max_variance_features": 120,
    "max_importance_features": 20,
    "feature_sets": {
        "bike_only": {"include_prefixes": ["sporsa__"]},
        "rider_only": {"include_prefixes": ["arduino__"]},
        "fused": {"include_prefixes": []},
    },
    "evaluation_seed": 42,
    "feature_family_ablation": {
        "enabled": True,
        "families": {
            "bumps": ["bump_"],
            "braking": ["brake_"],
            "cornering": ["corner_"],
            "sprinting": ["sprint_"],
            "disagreement": ["disagree_"],
        },
    },
}


@dataclass(frozen=True)
class EvalContext:
    df: pd.DataFrame
    label_col: str
    group_col: str
    feature_cols: list[str]


def _load_config(config_path: Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config_path is None:
        return cfg
    user_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _numeric_feature_columns(df: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude and not c.startswith("_")]


def _resolve_feature_sets(feature_cols: list[str], cfg: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, rules in cfg["feature_sets"].items():
        prefixes = rules.get("include_prefixes", [])
        excludes = rules.get("exclude_patterns", [])
        if not prefixes:
            cols = list(feature_cols)
        else:
            cols = [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]
        if excludes:
            cols = [c for c in cols if not any(p in c for p in excludes)]
        out[name] = cols
    return out


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / max(
        len(a) + len(b) - 2,
        1,
    )
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _effect_size_table(df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> pd.DataFrame:
    labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    for feat in feature_cols:
        best = 0.0
        best_pair = (None, None)
        for i, a in enumerate(labels):
            vals_a = df.loc[df[label_col].astype(str) == a, feat].to_numpy(dtype=float)
            for b in labels[i + 1 :]:
                vals_b = df.loc[df[label_col].astype(str) == b, feat].to_numpy(dtype=float)
                d = _cohen_d(vals_a, vals_b)
                if np.isfinite(d) and abs(d) > abs(best):
                    best = float(d)
                    best_pair = (a, b)
        rows.append(
            {
                "feature": feat,
                "max_abs_cohen_d": abs(best),
                "signed_cohen_d": best,
                "pair_a": best_pair[0],
                "pair_b": best_pair[1],
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_cohen_d", ascending=False)


def _variance_table(df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        sub = df[[label_col, c]].dropna()
        if sub.empty:
            continue
        class_means = sub.groupby(label_col, observed=False)[c].mean()
        between = float(np.var(class_means.to_numpy(dtype=float), ddof=1)) if len(class_means) > 1 else 0.0
        within_num, within_den = 0.0, 0
        for _, grp in sub.groupby(label_col, observed=False):
            arr = grp[c].to_numpy(dtype=float)
            if len(arr) < 2:
                continue
            within_num += float((len(arr) - 1) * np.var(arr, ddof=1))
            within_den += len(arr) - 1
        within = float(within_num / within_den) if within_den else float("nan")
        ratio = between / within if within and within > 1e-12 else np.nan
        rows.append(
            {
                "feature": c,
                "var_between_class_means": between,
                "var_within_classes_pooled": within,
                "ratio_between_within": ratio,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_between_within", ascending=False)


def _plot_pca_clusters(X: np.ndarray, y: np.ndarray, out_path: Path, *, random_state: int) -> None:
    if X.shape[0] < 3 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in np.unique(y):
        m = y == cls
        ax.scatter(X2[m, 0], X2[m, 1], s=24, alpha=0.7, label=str(cls))
    ax.set_title("PCA separability overview")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_kmeans_overlay(X: np.ndarray, y: np.ndarray, out_path: Path, *, random_state: int) -> None:
    if X.shape[0] < 6 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)
    k = len(np.unique(y))
    if k < 2:
        return
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = km.fit_predict(X2)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap="tab10", s=25, alpha=0.75)
    ax.set_title("KMeans clusters in PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_group_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Pipeline]]:
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=800, random_state=random_state, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    y_true_acc: dict[str, list[int]] = {k: [] for k in models}
    y_pred_acc: dict[str, list[int]] = {k: [] for k in models}

    for fold_idx, (tri, tei) in enumerate(logo.split(X, y, groups), start=1):
        X_tr, X_te = X.iloc[tri], X.iloc[tei]
        y_tr, y_te = y.iloc[tri], y.iloc[tei]
        if y_tr.nunique() < 2:
            continue
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            m = _evaluate_predictions(y_te.to_numpy(), pred)
            fold_rows.append({"model": name, "fold": fold_idx, **m})
            y_true_acc[name].extend(y_te.tolist())
            y_pred_acc[name].extend(pred.tolist())

    fold_df = pd.DataFrame(fold_rows)
    y_true_arr = {k: np.asarray(v, dtype=int) for k, v in y_true_acc.items() if v}
    y_pred_arr = {k: np.asarray(v, dtype=int) for k, v in y_pred_acc.items() if v}
    return fold_df, y_true_arr, y_pred_arr, models


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize=None)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (one-vs-rest minority)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_feature_importance(
    model: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    out_csv: Path,
    max_features: int,
    *,
    random_state: int,
) -> None:
    clf = model.named_steps["clf"]
    imputer = model.named_steps["imputer"]
    Xt = imputer.fit_transform(X)

    if hasattr(clf, "feature_importances_"):
        clf.fit(Xt, y)
        score = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        scaler = model.named_steps.get("scaler")
        Xs = scaler.fit_transform(Xt) if scaler is not None else Xt
        clf.fit(Xs, y)
        coef = np.asarray(clf.coef_, dtype=float)
        score = np.mean(np.abs(coef), axis=0)
    else:
        Xt2 = Xt
        clf.fit(Xt2, y)
        perm = permutation_importance(clf, Xt2, y, n_repeats=8, random_state=random_state)
        score = perm.importances_mean

    imp = pd.DataFrame({"feature": X.columns, "importance": score, "model": model_name})
    imp = imp.sort_values("importance", ascending=False).head(max_features)
    imp.to_csv(out_csv, index=False)


def _recommendation_text(classifier_table: pd.DataFrame, separability_table: pd.DataFrame) -> str:
    if classifier_table.empty:
        return (
            "Labels/splits were too limited for stable recording-aware classification. "
            "Use separability/effect-size analyses as primary evidence and report models as exploratory only."
        )
    best = classifier_table.sort_values("balanced_accuracy", ascending=False).iloc[0]
    if best["balanced_accuracy"] >= 0.7 and best["f1_macro"] >= 0.65:
        return (
            f"Strongest quantitative evidence: {best['comparison']} with {best['model']} "
            f"(balanced accuracy {best['balanced_accuracy']:.3f}, F1 {best['f1_macro']:.3f}). "
            "Include with confusion matrix + top features and caveat group-wise CV scope."
        )
    top_sep = separability_table.iloc[0] if not separability_table.empty else None
    if top_sep is not None and top_sep.get("max_abs_cohen_d", 0.0) >= 0.8:
        return (
            "Classifier performance is moderate; rely on feature-level evidence (large effect sizes/variance ratios) "
            "for thesis claims, and present classifiers as supportive trends."
        )
    return "Evidence is weak-to-moderate; emphasize descriptive trends, uncertainty, and need for more labeled sessions."


def run_evaluation_report(
    features_fused_csv: Path,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    random_state: int = 42,
) -> dict[str, Path]:
    """Run lightweight, config-driven experiments and write thesis-ready artifacts."""
    cfg = _load_config(config_path)
    effective_seed = int(cfg.get("evaluation_seed", random_state)) if random_state is not None else int(cfg.get("evaluation_seed", 42))
    np.random.seed(effective_seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_fused_csv)
    label_col = cfg["label_col"]
    group_col = cfg["group_col"]
    meta_cols = {
        label_col,
        group_col,
        cfg["sync_col"],
        cfg["orientation_col"],
        "section_id",
        "section",
        "window_start_s",
        "window_end_s",
        "window_center_s",
    }

    feature_cols = _numeric_feature_columns(df, exclude=meta_cols)
    feature_sets = _resolve_feature_sets(feature_cols, cfg)

    valid = df[label_col].notna() & df[group_col].notna()
    dfe = df.loc[valid].copy()
    dfe[label_col] = dfe[label_col].astype(str)
    dfe[group_col] = dfe[group_col].astype(str)

    counts = dfe[label_col].value_counts().rename_axis("label").reset_index(name="count")
    counts.to_csv(out_dir / "class_counts.csv", index=False)

    labels_good = counts[counts["count"] >= int(cfg["min_samples_per_class"])]
    dfe = dfe[dfe[label_col].isin(labels_good["label"])].copy()

    variance_df = _variance_table(
        dfe,
        label_col,
        feature_cols[: int(cfg["max_variance_features"])],
    )
    variance_df.to_csv(out_dir / "separability_within_between_variance.csv", index=False)

    effect_df = _effect_size_table(
        dfe,
        label_col,
        feature_cols[: int(cfg["max_effect_size_features"])],
    )
    effect_df.to_csv(out_dir / "separability_effect_size.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    enc = LabelEncoder()
    y = pd.Series(enc.fit_transform(dfe[label_col]), index=dfe.index)
    groups = dfe[group_col]

    comparisons: dict[str, dict[str, list[str]]] = {
        "feature_source": feature_sets,
        "sync_method": {},
        "orientation_method": {},
        "feature_family_ablation": {},
    }

    for sync_method in sorted(dfe[cfg["sync_col"]].dropna().astype(str).unique().tolist()):
        comparisons["sync_method"][sync_method] = feature_sets.get("fused", feature_cols)
    for orient_method in sorted(dfe[cfg["orientation_col"]].dropna().astype(str).unique().tolist()):
        comparisons["orientation_method"][orient_method] = feature_sets.get("fused", feature_cols)

    ablation_cfg = cfg.get("feature_family_ablation", {})
    if ablation_cfg.get("enabled", False):
        fused_cols = feature_sets.get("fused", feature_cols)
        comparisons["feature_family_ablation"]["all_fused"] = fused_cols
        for fam_name, patterns in ablation_cfg.get("families", {}).items():
            remain = [c for c in fused_cols if not any(tok in c for tok in patterns)]
            comparisons["feature_family_ablation"][f"minus_{fam_name}"] = remain

    for comparison, spec in comparisons.items():
        for variant, cols in spec.items():
            use_cols = [c for c in cols if c in dfe.columns]
            if len(use_cols) < 2:
                continue

            mask = pd.Series(True, index=dfe.index)
            if comparison == "sync_method":
                mask &= dfe[cfg["sync_col"]].astype(str) == variant
            elif comparison == "orientation_method":
                mask &= dfe[cfg["orientation_col"]].astype(str) == variant

            dsub = dfe.loc[mask].copy()
            if dsub.empty or dsub[group_col].nunique() < 2 or dsub[label_col].nunique() < 2:
                continue

            usable_cols = [
                c for c in use_cols
                if pd.to_numeric(dsub[c], errors="coerce").notna().sum() >= 2
            ]
            if len(usable_cols) < 2:
                continue

            y_sub = pd.Series(enc.transform(dsub[label_col]), index=dsub.index)
            fold_df, y_true_dict, y_pred_dict, models = _run_group_cv(
                dsub[usable_cols].replace([np.inf, -np.inf], np.nan),
                y_sub,
                dsub[group_col],
                random_state=random_state,
            )
            if fold_df.empty:
                continue

            mean_df = (
                fold_df.groupby("model", as_index=False)[
                    ["balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
                ]
                .mean()
                .sort_values("balanced_accuracy", ascending=False)
            )
            for _, row in mean_df.iterrows():
                comparison_rows.append(
                    {
                        "comparison": comparison,
                        "variant": variant,
                        "model": row["model"],
                        "n_samples": len(dsub),
                        "n_groups": int(dsub[group_col].nunique()),
                        "n_features": len(usable_cols),
                        "balanced_accuracy": float(row["balanced_accuracy"]),
                        "precision_macro": float(row["precision_macro"]),
                        "recall_macro": float(row["recall_macro"]),
                        "f1_macro": float(row["f1_macro"]),
                    }
                )

            labels = enc.classes_.tolist()
            best_model_name = mean_df.iloc[0]["model"]
            y_true = y_true_dict[best_model_name]
            y_pred = y_pred_dict[best_model_name]
            _save_confusion(
                y_true,
                y_pred,
                labels,
                out_dir / f"cm_{comparison}_{variant}_{best_model_name}.png",
            )

            model_full = models[best_model_name]
            _save_feature_importance(
                model_full,
                best_model_name,
                dsub[usable_cols].replace([np.inf, -np.inf], np.nan),
                y_sub,
                out_dir / f"feature_importance_{comparison}_{variant}_{best_model_name}.csv",
                max_features=int(cfg["max_importance_features"]),
                random_state=random_state,
            )

            minority_ratio = dsub[label_col].value_counts(normalize=True).min()
            if minority_ratio < float(cfg["pr_curve_minority_ratio_threshold"]):
                clf = model_full.fit(dsub[usable_cols].replace([np.inf, -np.inf], np.nan), y_sub)
                if hasattr(clf.named_steps["clf"], "predict_proba"):
                    y_prob = clf.predict_proba(dsub[usable_cols].replace([np.inf, -np.inf], np.nan))
                    minority_class = int(np.argmin(dsub[label_col].value_counts().sort_index().to_numpy()))
                    y_bin = (y_sub.to_numpy() == minority_class).astype(int)
                    if y_prob.shape[1] > minority_class:
                        _save_pr_curve(
                            y_bin,
                            y_prob[:, minority_class],
                            out_dir / f"pr_{comparison}_{variant}_{best_model_name}.png",
                        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_dir / "classification_summary.csv", index=False)

    # unsupervised plots from fused set, for limited labels cases
    fused_cols = feature_sets.get("fused", feature_cols)
    usable_fused_cols = [
        c for c in fused_cols
        if c in dfe.columns and pd.to_numeric(dfe[c], errors="coerce").notna().sum() >= 2
    ]
    if len(usable_fused_cols) >= 2 and not dfe.empty:
        Xf = dfe[usable_fused_cols].replace([np.inf, -np.inf], np.nan)
        imp = SimpleImputer(strategy="median")
        Xfi = imp.fit_transform(Xf)
        if Xfi.shape[1] >= 2:
            Xfs = StandardScaler().fit_transform(Xfi)
            _plot_pca_clusters(Xfs, dfe[label_col].to_numpy(), out_dir / "pca_label_scatter.png", random_state=effective_seed)
            _plot_kmeans_overlay(Xfs, dfe[label_col].to_numpy(), out_dir / "pca_kmeans_scatter.png", random_state=effective_seed)

    recommendation = _recommendation_text(comparison_df, effect_df)

    thesis_table = comparison_df.sort_values(
        ["comparison", "variant", "balanced_accuracy"],
        ascending=[True, True, False],
    )
    thesis_table.to_csv(out_dir / "thesis_table_model_metrics.csv", index=False)

    summary = {
        "inputs": {
            "features_csv": str(features_fused_csv),
            "config": str(config_path) if config_path else "default",
            "evaluation_seed": effective_seed,
        },
        "dataset": {
            "rows_after_label_filter": int(len(dfe)),
            "groups": int(dfe[group_col].nunique()) if not dfe.empty else 0,
            "classes": sorted(dfe[label_col].unique().tolist()) if not dfe.empty else [],
        },
        "seeds": {"evaluation_seed": effective_seed},
        "outputs": {
            "classification_summary": str(out_dir / "classification_summary.csv"),
            "thesis_table_model_metrics": str(out_dir / "thesis_table_model_metrics.csv"),
            "effect_size": str(out_dir / "separability_effect_size.csv"),
            "within_between": str(out_dir / "separability_within_between_variance.csv"),
            "class_counts": str(out_dir / "class_counts.csv"),
            "pca_label_scatter": str(out_dir / "pca_label_scatter.png"),
            "pca_kmeans_scatter": str(out_dir / "pca_kmeans_scatter.png"),
        },
        "recommendation": recommendation,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "evaluation_manifest.json").write_text(json.dumps({"evaluation_seed": effective_seed, "inputs": summary["inputs"], "dataset": summary["dataset"]}, indent=2), encoding="utf-8")

    md = [
        "# Evaluation Summary",
        "",
        f"- Rows analyzed: **{summary['dataset']['rows_after_label_filter']}**",
        f"- Recordings/groups: **{summary['dataset']['groups']}**",
        f"- Classes: **{', '.join(summary['dataset']['classes']) if summary['dataset']['classes'] else 'n/a'}**",
        "",
        "## Recommendation",
        recommendation,
        "",
        "## Key files",
        "- `thesis_table_model_metrics.csv`",
        "- `classification_summary.csv`",
        "- `separability_effect_size.csv`",
        "- `separability_within_between_variance.csv`",
    ]
    (out_dir / "THESIS_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    return {
        "summary": out_dir / "evaluation_summary.json",
        "thesis_table": out_dir / "thesis_table_model_metrics.csv",
        "classification": out_dir / "classification_summary.csv",
    }
