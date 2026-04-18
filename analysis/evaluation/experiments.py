"""Model training and evaluation experiments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.paths import project_relative_path, read_csv, write_csv

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["poor", "marginal", "good"]

_META_COLS = {
    # Window identifiers
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    # Labels
    "scenario_label",
    "scenario_label_coarse",
    "scenario_label_binary",
    # Quality summary (section- and window-level)
    "overall_quality_label",
    "overall_quality_score",
    "quality_tier",
    "calibration_quality",
    "sync_confidence",
    # Per-component quality scores (from revised multi-IMU quality scheme)
    "quality_bike",
    "quality_rider",
    "quality_alignment",
    "quality_calibration",
    "quality_cross",
    # Per-sensor signal validity ratios
    "window_valid_ratio_sporsa",
    "window_valid_ratio_arduino",
}

_MODEL_REGISTRY: dict[str, Any] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

_MODEL_DISPLAY: dict[str, str] = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
}

# Human-readable names for each feature set.
_FS_DISPLAY: dict[str, str] = {
    "bike": "Bike only",
    "rider": "Rider only",
    "fused_no_cross": "Fused (no cross)",
    "fused": "Fused",
}

# Canonical display order for comparison tables (simple → complex).
_FS_ORDER: list[str] = ["bike", "rider", "fused_no_cross", "fused"]


def _select_feature_cols(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    """Return columns matching any of the given prefixes, excluding metadata cols."""
    return [
        c
        for c in df.columns
        if c not in _META_COLS
        and any(c.startswith(p) for p in prefixes)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _auto_detect_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return default feature set → prefix mapping based on available columns.

    Four sets form a 2×2 ablation:
      bike             unimodal, frame sensor only
      rider            unimodal, body sensor only
      fused_no_cross   bimodal, no cross-sensor features (baseline fusion)
      fused            bimodal, full feature set including cross-sensor features

    Comparing fused vs fused_no_cross directly quantifies the added value of
    the cross-sensor alignment and disagreement features.
    """
    return {
        "bike": ["bike_", "sporsa_"],
        "rider": ["rider_", "arduino_"],
        "fused_no_cross": ["bike_", "sporsa_", "rider_", "arduino_"],
        "fused": ["bike_", "sporsa_", "rider_", "arduino_", "cross_"],
    }


def _build_model(name: str, seed: int) -> Any:
    cls = _MODEL_REGISTRY[name]
    kwargs: dict[str, Any] = {"random_state": seed}
    if name == "logistic_regression":
        kwargs["max_iter"] = 1000
    return cls(**kwargs)


def _cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Any,
    n_splits: int = 5,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run GroupKFold CV; ALL metrics are derived from out-of-fold predictions.

    Each fold trains on groups unseen by the validation set.  OOF predictions
    are concatenated across folds so that the final confusion matrix and
    per-class classification report reflect held-out performance only — no
    in-sample evaluation anywhere.

    Returns
    -------
    dict with keys:
        accuracy, accuracy_std       – mean / std of per-fold accuracy
        macro_f1, macro_f1_std       – mean / std of per-fold macro-F1
        fold_accuracies, fold_f1s    – per-fold scores (length = n_splits)
        per_class                    – classification_report on OOF predictions
        confusion_matrix             – confusion matrix on OOF predictions
        feature_importances          – mean importances across folds, or None
    """
    cv = GroupKFold(n_splits=n_splits)

    # Pre-allocate OOF arrays — every sample is a validation sample exactly once.
    oof_pred = np.empty(len(y), dtype=y.dtype)

    fold_accuracies: list[float] = []
    fold_f1s: list[float] = []
    fold_importances: list[np.ndarray] = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # clone() creates a fresh, unfitted copy with identical hyperparameters —
        # required so each fold starts from scratch and shares no state.
        # Median imputation is fit on the training fold only to avoid leaking
        # validation-fold statistics into the filled values; replaces the
        # previous fillna(0) which conflated 'missing' with 'zero'.
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clone(model)),
            ]
        )
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_val)

        oof_pred[val_idx] = y_hat

        fold_accuracies.append(float(accuracy_score(y_val, y_hat)))
        fold_f1s.append(float(f1_score(y_val, y_hat, average="macro", zero_division=0)))

        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            fold_importances.append(clf.feature_importances_.copy())

    # --- OOF aggregate metrics -------------------------------------------------
    # classification_report and confusion_matrix are computed on the stacked OOF
    # predictions, which cover all samples exactly once under GroupKFold.
    report = classification_report(
        y, oof_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, oof_pred)

    # Mean feature importances across folds (None for models that lack them).
    importances: list[float] | None = None
    if fold_importances:
        importances = np.mean(fold_importances, axis=0).tolist()

    return {
        "accuracy": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies, ddof=1)),
        "macro_f1": float(np.mean(fold_f1s)),
        "macro_f1_std": float(np.std(fold_f1s, ddof=1)),
        "fold_accuracies": [round(v, 4) for v in fold_accuracies],
        "fold_f1s": [round(v, 4) for v in fold_f1s],
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
    }


def run_evaluation(
    features_path: Path | str,
    *,
    output_dir: Path | str,
    label_col: str = "scenario_label",
    group_col: str = "section_id",
    seed: int = 42,
    min_quality: str = "marginal",
    feature_sets: dict[str, list[str]] | None = None,
    no_plots: bool = False,
) -> dict:
    """Train and evaluate models on feature table.

    Parameters
    ----------
    features_path:
        Path to the CSV feature table (e.g. features_fused.csv).
    output_dir:
        Directory for all output artefacts.
    label_col:
        Column name for the classification target.
    group_col:
        Column for group-based CV splitting (prevents data leakage).
    seed:
        Random seed for reproducibility.
    min_quality:
        Minimum quality label. Rows below this are dropped.
    feature_sets:
        Mapping of set name → list of column prefixes. ``None`` = auto-detect.
    no_plots:
        If ``True``, skip thesis figure generation.

    Returns
    -------
    Evaluation summary dict (also written to evaluation_summary.json).
    """
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info("Loading features from %s", project_relative_path(features_path))
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = read_csv(features_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Apply quality filter
    if "overall_quality_label" in df.columns:
        if min_quality not in _QUALITY_ORDER:
            raise ValueError(f"min_quality must be one of {_QUALITY_ORDER}")
        min_idx = _QUALITY_ORDER.index(min_quality)
        valid = set(_QUALITY_ORDER[min_idx:])
        before = len(df)
        df = df[df["overall_quality_label"].isin(valid)].copy()
        log.info("Quality filter: %d → %d rows", before, len(df))

    # Drop unlabeled rows
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in features.")
    before = len(df)
    df = df[df[label_col].notna() & (df[label_col] != "unlabeled")].copy()
    log.info("Dropped unlabeled rows: %d → %d", before, len(df))

    if len(df) == 0:
        raise ValueError("No labeled rows remain after filtering.")

    classes = sorted(df[label_col].unique().tolist())
    log.info("Classes: %s (%d total windows)", classes, len(df))

    # ------------------------------------------------------------------
    # 2. Determine feature sets
    # ------------------------------------------------------------------
    if feature_sets is None:
        feature_sets = _auto_detect_feature_sets(df)

    # Filter to sets that have at least one column
    active_sets: dict[str, list[str]] = {}
    for fs_name, prefixes in feature_sets.items():
        cols = _select_feature_cols(df, prefixes)
        if cols:
            active_sets[fs_name] = cols
            log.info("Feature set '%s': %d features", fs_name, len(cols))
        else:
            log.warning("Feature set '%s': no columns found, skipping", fs_name)

    if not active_sets:
        raise ValueError("No feature columns found for any feature set.")

    # ------------------------------------------------------------------
    # 3. Encode labels and groups
    # ------------------------------------------------------------------
    le = LabelEncoder()
    le.fit(classes)
    y_all = le.transform(df[label_col].values)

    groups_all: np.ndarray | None = None
    if group_col in df.columns:
        groups_all = df[group_col].values
    else:
        log.warning("Group column '%s' not found; using no-group CV", group_col)

    # ------------------------------------------------------------------
    # 4. Train and evaluate
    # ------------------------------------------------------------------
    all_results: dict[str, dict[str, Any]] = {}
    metrics_rows: list[dict] = []

    for fs_name, feat_cols in active_sets.items():
        # NaN handling happens inside the CV pipeline (SimpleImputer) so that
        # imputation statistics are fit on the training fold only.
        sub = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        all_nan = [c for c in feat_cols if sub[c].isna().all()]
        if all_nan:
            log.warning(
                "Feature set '%s': dropping %d all-NaN columns: %s",
                fs_name, len(all_nan), all_nan,
            )
            feat_cols = [c for c in feat_cols if c not in all_nan]
            if not feat_cols:
                log.warning("Feature set '%s': no usable columns remain, skipping", fs_name)
                continue
            sub = sub[feat_cols]
        X = sub.to_numpy(dtype=float)

        # Use available n_splits (capped at n_groups)
        n_groups = len(np.unique(groups_all)) if groups_all is not None else len(y_all)
        n_splits = min(5, n_groups)
        if n_splits < 2:
            log.warning(
                "Feature set '%s': only %d groups available, skipping CV", fs_name, n_groups
            )
            continue

        for model_name in _MODEL_REGISTRY:
            key = f"{fs_name}__{model_name}"
            log.info("Evaluating %s ...", key)

            model = _build_model(model_name, seed)
            try:
                result = _cv_evaluate(
                    X,
                    y_all,
                    groups=groups_all if groups_all is not None else np.arange(len(y_all)),
                    model=model,
                    n_splits=n_splits,
                    seed=seed,
                    class_names=classes,
                )
            except Exception as exc:
                log.warning(
                    "Evaluation failed for %s (%s): %s",
                    key,
                    exc.__class__.__name__,
                    exc,
                )
                continue

            all_results[key] = result

            metrics_rows.append(
                {
                    "feature_set": fs_name,
                    "n_features": len(feat_cols),
                    "model": model_name,
                    "accuracy": round(result["accuracy"], 4),
                    "accuracy_std": round(result["accuracy_std"], 4),
                    "macro_f1": round(result["macro_f1"], 4),
                    "macro_f1_std": round(result["macro_f1_std"], 4),
                }
            )

            # Write confusion matrix — rows = true class, cols = predicted class.
            # Built from OOF predictions; label index preserved for readability.
            cm_path = output_dir / f"confusion_matrix_{fs_name}_{model_name}.csv"
            cm_df = pd.DataFrame(
                result["confusion_matrix"],
                index=le.classes_,
                columns=le.classes_,
            )
            cm_df.to_csv(cm_path, index=True)
            log.debug("Wrote confusion matrix to %s", project_relative_path(cm_path))

            # Write per-class report (OOF-based)
            per_class_path = output_dir / f"per_class_report_{fs_name}_{model_name}.json"
            per_class_path.write_text(
                json.dumps(result["per_class"], indent=2), encoding="utf-8"
            )

            # Write fold-wise scores so variance is inspectable
            fold_path = output_dir / f"fold_scores_{fs_name}_{model_name}.csv"
            fold_df = pd.DataFrame(
                {
                    "fold": range(1, n_splits + 1),
                    "accuracy": result["fold_accuracies"],
                    "macro_f1": result["fold_f1s"],
                }
            )
            write_csv(fold_df, fold_path)
            log.debug("Wrote fold scores to %s", project_relative_path(fold_path))

            # Write feature importances (mean across folds)
            if result["feature_importances"] is not None:
                fi_path = output_dir / f"feature_importance_{model_name}_{fs_name}.csv"
                fi_df = pd.DataFrame(
                    {
                        "feature": feat_cols,
                        "importance": result["feature_importances"],
                    }
                ).sort_values("importance", ascending=False)
                write_csv(fi_df, fi_path)
                log.debug("Wrote feature importances to %s", project_relative_path(fi_path))

    # ------------------------------------------------------------------
    # 5. Write metrics table
    # ------------------------------------------------------------------
    if metrics_rows:
        metrics_rows = sorted(
            metrics_rows,
            key=lambda r: (-r["macro_f1"], r["macro_f1_std"], -r["accuracy"]),
        )
        metrics_df = pd.DataFrame(metrics_rows).reset_index(drop=True)
        metrics_path = output_dir / "metrics_table.csv"
        write_csv(metrics_df, metrics_path)
        log.info("Wrote metrics table to %s", project_relative_path(metrics_path))

    # ------------------------------------------------------------------
    # 6. Build evaluation summary
    # ------------------------------------------------------------------
    # Rank by macro-F1 (more informative than accuracy for imbalanced classes).
    best_key = max(
        all_results,
        key=lambda k: all_results[k]["macro_f1"],
        default=None,
    )

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "label_col": label_col,
        "n_windows": int(len(df)),
        "n_classes": len(classes),
        "classes": classes,
        "results": {
            k: {
                "accuracy": round(v["accuracy"], 4),
                "accuracy_std": round(v["accuracy_std"], 4),
                "macro_f1": round(v["macro_f1"], 4),
                "macro_f1_std": round(v["macro_f1_std"], 4),
                "fold_accuracies": v["fold_accuracies"],
                "fold_f1s": v["fold_f1s"],
            }
            for k, v in all_results.items()
        },
    }

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote evaluation summary to %s", project_relative_path(summary_path))

    # ------------------------------------------------------------------
    # 7. Write THESIS_SUMMARY.md
    # ------------------------------------------------------------------
    _write_thesis_summary(
        output_dir=output_dir,
        summary=summary,
        metrics_rows=metrics_rows,
        best_key=best_key,
        all_results=all_results,
        label_display=_MODEL_DISPLAY,
        fs_display=_FS_DISPLAY,
    )

    # ------------------------------------------------------------------
    # 8. Generate thesis figures
    # ------------------------------------------------------------------
    if not no_plots:
        try:
            from evaluation.plots import generate_evaluation_figures, plot_per_class_f1
            generate_evaluation_figures(output_dir)

            # Per-class F1 for each feature set that has results
            for fs_name in active_sets:
                out = output_dir / "figures" / f"per_class_f1_{fs_name}.png"
                plot_per_class_f1(
                    pd.DataFrame(metrics_rows),
                    all_results,
                    classes,
                    out,
                    feature_set=fs_name,
                )
        except Exception as exc:
            log.warning("Evaluation figure generation failed: %s", exc)

    return summary


def _ordered_feature_sets(metrics_rows: list[dict]) -> list[str]:
    """Return feature sets in canonical order (simple → complex), unknowns last."""
    present = {r["feature_set"] for r in metrics_rows}
    ordered = [fs for fs in _FS_ORDER if fs in present]
    extras = sorted(fs for fs in present if fs not in _FS_ORDER)
    return ordered + extras


def _pivot_table_lines(
    metrics_rows: list[dict],
    metric: str,
    std_metric: str,
    ordered_fs: list[str],
    label_display: dict[str, str],
    fs_display: dict[str, str],
) -> list[str]:
    """Build markdown lines for a model × feature-set pivot table.

    Rows = models (in _MODEL_REGISTRY insertion order).
    Columns = feature sets (canonical order from _FS_ORDER).
    Cells = ``mean% ± std%``.
    """
    # Column headers
    fs_headers = " | ".join(fs_display.get(fs, fs) for fs in ordered_fs)
    sep = "|---" * (len(ordered_fs) + 1) + "|"
    lines = [f"| Model | {fs_headers} |", sep]

    # One row per model
    for model_name in _MODEL_REGISTRY:
        mdl = label_display.get(model_name, model_name)
        cells: list[str] = []
        for fs in ordered_fs:
            row = next(
                (r for r in metrics_rows if r["feature_set"] == fs and r["model"] == model_name),
                None,
            )
            if row:
                mean_pct = row[metric] * 100
                std_pct = row[std_metric] * 100
                cells.append(f"{mean_pct:.1f} ± {std_pct:.1f}")
            else:
                cells.append("—")
        lines.append(f"| {mdl} | " + " | ".join(cells) + " |")
    return lines


def _write_thesis_summary(
    output_dir: Path,
    summary: dict,
    metrics_rows: list[dict],
    best_key: str | None,
    all_results: dict,
    label_display: dict[str, str],
    fs_display: dict[str, str],
) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ordered_fs = _ordered_feature_sets(metrics_rows)

    lines = [
        "# Evaluation Summary",
        "",
        f"**Date:** {date_str}  "
        f"**Seed:** {summary['seed']}  "
        f"**Windows:** {summary['n_windows']}",
        "",
        "> All metrics are out-of-fold (OOF) estimates from GroupKFold CV.",
        "> Mean ± std are percentages computed over individual folds (ddof=1).",
        "",
        "## Full Results",
        "",
        "| Feature Set | n | Model | Accuracy | Macro-F1 |",
        "|---|---|---|---|---|",
    ]

    for row in sorted(
        metrics_rows,
        key=lambda r: (
            _FS_ORDER.index(r["feature_set"]) if r["feature_set"] in _FS_ORDER else 99,
            list(_MODEL_REGISTRY).index(r["model"]) if r["model"] in _MODEL_REGISTRY else 99,
        ),
    ):
        fs_label = fs_display.get(row["feature_set"], row["feature_set"])
        n_feat = row.get("n_features", "—")
        mdl = label_display.get(row["model"], row["model"])
        acc = f"{row['accuracy'] * 100:.1f} ± {row['accuracy_std'] * 100:.1f}%"
        f1 = f"{row['macro_f1'] * 100:.1f} ± {row['macro_f1_std'] * 100:.1f}%"
        lines.append(f"| {fs_label} | {n_feat} | {mdl} | {acc} | {f1} |")

    # ------------------------------------------------------------------
    # Pivot table 1: macro-F1 (models × feature sets)
    # ------------------------------------------------------------------
    lines += [
        "",
        "## Macro-F1 Comparison — mean ± std % (models × feature sets)",
        "",
    ]
    lines += _pivot_table_lines(
        metrics_rows, "macro_f1", "macro_f1_std", ordered_fs, label_display, fs_display
    )

    # ------------------------------------------------------------------
    # Pivot table 2: accuracy (models × feature sets)
    # ------------------------------------------------------------------
    lines += [
        "",
        "## Accuracy Comparison — mean ± std % (models × feature sets)",
        "",
    ]
    lines += _pivot_table_lines(
        metrics_rows, "accuracy", "accuracy_std", ordered_fs, label_display, fs_display
    )

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    lines += ["", "## Key Findings", ""]

    if best_key and best_key in all_results:
        f1_pct = all_results[best_key]["macro_f1"] * 100
        f1_std_pct = all_results[best_key]["macro_f1_std"] * 100
        fs_part, _, mdl_part = best_key.partition("__")
        mdl_disp = label_display.get(mdl_part, mdl_part)
        fs_disp = fs_display.get(fs_part, fs_part)
        lines.append(
            f"- **Best:** {fs_disp} + {mdl_disp} — "
            f"macro-F1 {f1_pct:.1f} ± {f1_std_pct:.1f}%"
        )

        # Cross-sensor contribution: fused vs fused_no_cross for same best model.
        # This quantifies whether the cross-sensor alignment/disagreement features
        # add predictive value beyond naive sensor fusion.
        if mdl_part:
            fused_key = f"fused__{mdl_part}"
            no_cross_key = f"fused_no_cross__{mdl_part}"
            if fused_key in all_results and no_cross_key in all_results:
                delta_f1 = (
                    all_results[fused_key]["macro_f1"]
                    - all_results[no_cross_key]["macro_f1"]
                ) * 100
                sign = "+" if delta_f1 >= 0 else ""
                lines.append(
                    f"- **Cross-sensor contribution** (fused vs fused_no_cross, {mdl_disp}): "
                    f"{sign}{delta_f1:.1f} pp macro-F1"
                )

            # Sensor contribution: fused_no_cross vs best single-sensor set.
            bike_key = f"bike__{mdl_part}"
            rider_key = f"rider__{mdl_part}"
            if no_cross_key in all_results and bike_key in all_results and rider_key in all_results:
                best_single_f1 = max(
                    all_results[bike_key]["macro_f1"],
                    all_results[rider_key]["macro_f1"],
                )
                best_single_name = (
                    "Bike only"
                    if all_results[bike_key]["macro_f1"] >= all_results[rider_key]["macro_f1"]
                    else "Rider only"
                )
                delta_fusion = (
                    all_results[no_cross_key]["macro_f1"] - best_single_f1
                ) * 100
                sign = "+" if delta_fusion >= 0 else ""
                lines.append(
                    f"- **Sensor fusion gain** (fused_no_cross vs {best_single_name}, {mdl_disp}): "
                    f"{sign}{delta_fusion:.1f} pp macro-F1"
                )
    else:
        lines.append("- No results available.")

    lines += [
        f"- Classes evaluated: {', '.join(summary['classes'])}",
        f"- Total labeled windows: {summary['n_windows']}",
        "",
    ]

    md_path = output_dir / "THESIS_SUMMARY.md"
    md_path.write_text("\n".join(lines))
    log.info("Wrote thesis summary to %s", project_relative_path(md_path))
