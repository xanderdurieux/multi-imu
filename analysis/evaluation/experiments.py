"""Model training and evaluation experiments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.paths import project_relative_path, read_csv, write_csv

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["poor", "marginal", "good"]

_META_COLS = {
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "scenario_label",
    "overall_quality_label",
    "quality_tier",
    "calibration_quality",
    "sync_confidence",
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
    """Return default feature set → prefix mapping based on available columns."""
    return {
        "bike": ["bike_", "sporsa_"],
        "rider": ["rider_", "arduino_"],
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
) -> dict[str, Any]:
    """Run GroupKFold cross-validation and return aggregated metrics."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    cv = GroupKFold(n_splits=n_splits)
    cv_results = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
        return_estimator=True,
    )

    accuracy = float(np.mean(cv_results["test_accuracy"]))
    macro_f1 = float(np.mean(cv_results["test_f1_macro"]))

    # Gather per-class metrics via one full fold re-fit on all data
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    # Extract feature importances from the last estimator if available
    clf = pipe.named_steps["clf"]
    importances: dict[str, float] | None = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_.tolist()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
        "fitted_pipeline": pipe,
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
        X = df[feat_cols].fillna(0).values

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
                )
            except Exception as exc:
                log.warning("Evaluation failed for %s: %s", key, exc)
                continue

            all_results[key] = result

            metrics_rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "accuracy": round(result["accuracy"], 4),
                    "macro_f1": round(result["macro_f1"], 4),
                }
            )

            # Write confusion matrix — use to_csv directly to preserve class-label index
            cm_path = output_dir / f"confusion_matrix_{fs_name}_{model_name}.csv"
            cm_df = pd.DataFrame(
                result["confusion_matrix"],
                index=le.classes_,
                columns=le.classes_,
            )
            cm_df.to_csv(cm_path, index=True)
            log.debug("Wrote confusion matrix to %s", project_relative_path(cm_path))

            # Write feature importances
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
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = output_dir / "metrics_table.csv"
        write_csv(metrics_df, metrics_path)
        log.info("Wrote metrics table to %s", project_relative_path(metrics_path))

    # ------------------------------------------------------------------
    # 6. Build evaluation summary
    # ------------------------------------------------------------------
    best_key = max(
        all_results,
        key=lambda k: all_results[k]["accuracy"],
        default=None,
    )

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_windows": int(len(df)),
        "n_classes": len(classes),
        "classes": classes,
        "results": {
            k: {
                "accuracy": round(v["accuracy"], 4),
                "macro_f1": round(v["macro_f1"], 4),
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


def _write_thesis_summary(
    output_dir: Path,
    summary: dict,
    metrics_rows: list[dict],
    best_key: str | None,
    all_results: dict,
    label_display: dict[str, str],
) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Evaluation Summary",
        "",
        f"**Date:** {date_str}  "
        f"**Seed:** {summary['seed']}  "
        f"**Windows:** {summary['n_windows']}",
        "",
        "## Results by Feature Set",
        "",
        "| Feature Set | Model | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
    ]

    for row in metrics_rows:
        fs = row["feature_set"].capitalize()
        mdl = label_display.get(row["model"], row["model"])
        acc = f"{row['accuracy'] * 100:.1f}%"
        f1 = f"{row['macro_f1'] * 100:.1f}%"
        lines.append(f"| {fs} | {mdl} | {acc} | {f1} |")

    lines += ["", "## Key Findings"]

    if best_key and best_key in all_results:
        acc_pct = all_results[best_key]["accuracy"] * 100
        fs_part, _, mdl_part = best_key.partition("__")
        mdl_display = label_display.get(mdl_part, mdl_part)
        lines.append(
            f"- Best performing: {fs_part} + {mdl_display} ({acc_pct:.1f}% accuracy)"
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
