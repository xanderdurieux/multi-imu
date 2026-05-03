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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.paths import project_relative_path, read_csv, write_csv
from features.label_config import default_label_config
from features.labels import (
    to_activity_label,
    to_binary_label,
    to_coarse_label,
    to_set_based_label,
)

# Single-label fallback mappers when *label_col* is missing and the scheme is
# resolved from the priority-collapsed ``scenario_label`` only.  Set-based
# schemes (registered in labels.default.json) are derived from the pipe-split
# ``scenario_labels`` column instead — see :func:`run_evaluation`.
_DERIVED_LABEL_MAPPERS: dict[str, Any] = {
    "scenario_label_activity": to_activity_label,
    "scenario_label_coarse": to_coarse_label,
    "scenario_label_binary": to_binary_label,
}

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
    "scenario_labels",
    "scenario_label_activity",
    "scenario_label_coarse",
    "scenario_label_binary",
    "scenario_label_riding",
    "scenario_label_cornering",
    "scenario_label_head_motion",
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
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

_MODEL_DISPLAY: dict[str, str] = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
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
    """Return auto detect feature sets."""
    return {
        "bike": ["bike_", "sporsa_"],
        "rider": ["rider_", "arduino_"],
        "fused_no_cross": ["bike_", "sporsa_", "rider_", "arduino_"],
        "fused": ["bike_", "sporsa_", "rider_", "arduino_", "cross_"],
    }


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray | None,
    classes: list[str],
    *,
    positive_class: str = "riding",
) -> dict[str, Any]:
    """Return binary metrics."""
    if len(classes) != 2:
        raise ValueError(f"_binary_metrics expects exactly 2 classes, got {classes!r}")

    if positive_class not in classes:
        raise ValueError(
            f"positive_class={positive_class!r} not in classes={classes!r}"
        )

    pos_idx = classes.index(positive_class)
    neg_idx = 1 - pos_idx

    cm = confusion_matrix(y_true, y_pred, labels=[neg_idx, pos_idx])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=[neg_idx, pos_idx],
        target_names=[classes[neg_idx], classes[pos_idx]],
        output_dict=True,
        zero_division=0,
    )

    out: dict[str, Any] = {
        "positive_class": positive_class,
        "negative_class": classes[neg_idx],
        "support": {
            classes[neg_idx]: int(tn + fp),
            classes[pos_idx]: int(fn + tp),
        },
        "confusion_matrix": {
            "labels": [classes[neg_idx], classes[pos_idx]],
            "matrix": cm.tolist(),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            classes[neg_idx]: {
                "precision": float(report[classes[neg_idx]]["precision"]),
                "recall":    float(report[classes[neg_idx]]["recall"]),
                "f1":        float(report[classes[neg_idx]]["f1-score"]),
                "support":   int(report[classes[neg_idx]]["support"]),
            },
            classes[pos_idx]: {
                "precision": float(report[classes[pos_idx]]["precision"]),
                "recall":    float(report[classes[pos_idx]]["recall"]),
                "f1":        float(report[classes[pos_idx]]["f1-score"]),
                "support":   int(report[classes[pos_idx]]["support"]),
            },
        },
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall":    float(report["macro avg"]["recall"]),
            "f1":        float(report["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall":    float(report["weighted avg"]["recall"]),
            "f1":        float(report["weighted avg"]["f1-score"]),
        },
    }

    # PR-AUC and ROC-AUC require predicted probability for the positive class.
    # Skipped when the model lacks predict_proba or when only one class is
    # present in the OOF labels (sklearn raises in that case).
    if proba_pos is not None and not np.all(np.isnan(proba_pos)):
        finite = np.isfinite(proba_pos)
        n_pos = int((y_true[finite] == pos_idx).sum())
        n_neg = int((y_true[finite] == neg_idx).sum())
        if n_pos > 0 and n_neg > 0:
            out["average_precision"] = float(
                average_precision_score((y_true[finite] == pos_idx).astype(int), proba_pos[finite])
            )
            out["roc_auc"] = float(
                roc_auc_score((y_true[finite] == pos_idx).astype(int), proba_pos[finite])
            )
            # Curve points are useful for downstream PR/ROC plots.
            prec, rec, pr_thr = precision_recall_curve(
                (y_true[finite] == pos_idx).astype(int), proba_pos[finite]
            )
            fpr, tpr, roc_thr = roc_curve(
                (y_true[finite] == pos_idx).astype(int), proba_pos[finite]
            )
            out["pr_curve"] = {
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "thresholds": pr_thr.tolist(),
            }
            out["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thr.tolist(),
            }
        else:
            log.warning(
                "Skipping PR-AUC/ROC-AUC: only one class present in OOF labels "
                "(n_pos=%d, n_neg=%d).", n_pos, n_neg,
            )

    return out


def _export_misclassified(
    df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray | None,
    classes: list[str],
    label_col: str,
    output_path: Path,
    positive_class: str = "riding",
) -> int:
    """Export misclassified."""
    pos_idx = classes.index(positive_class)
    neg_idx = 1 - pos_idx
    inv_class = {neg_idx: classes[neg_idx], pos_idx: classes[pos_idx]}

    wrong_mask = y_true != y_pred
    if not wrong_mask.any():
        log.info("No misclassified windows for %s — nothing to export.", output_path.name)
        return 0

    # Surface the most useful columns for human inspection. Anything optional
    # is included only if present in the source frame.
    candidate_cols = [
        "section_id",
        "recording_name",
        "window_idx",
        "window_start_ms",
        "window_end_ms",
        "window_duration_s",
        # scenario_labels (multi-label set) is the most useful column for
        # label refinement: it shows every annotation that overlapped the
        # window, so the user can see whether the prediction disagrees with
        # a clearly-correct label or with a missing/ambiguous one.
        "scenario_labels",
        "scenario_label",
        "scenario_label_activity",
        "scenario_label_coarse",
        "scenario_label_binary",
        "scenario_label_riding",
        "scenario_label_cornering",
        "scenario_label_head_motion",
        label_col,
        "overall_quality_label",
        "overall_quality_score",
    ]
    present_cols = [c for c in candidate_cols if c in df.columns]

    rows = df.loc[wrong_mask, present_cols].copy()
    rows.insert(0, "true_binary", [inv_class[v] for v in y_true[wrong_mask]])
    rows.insert(1, "pred_binary", [inv_class[v] for v in y_pred[wrong_mask]])
    rows.insert(
        2,
        "error_type",
        np.where(
            y_true[wrong_mask] == pos_idx,
            f"FN_({positive_class}_predicted_as_{classes[neg_idx]})",
            f"FP_({classes[neg_idx]}_predicted_as_{positive_class})",
        ),
    )

    # Confidence score = predicted probability for the *predicted* class. Higher
    # values mark predictions the model was sure about — most likely candidates
    # for label-error inspection rather than genuine model failure.
    if proba_pos is not None and not np.all(np.isnan(proba_pos)):
        proba_pred = np.where(
            y_pred[wrong_mask] == pos_idx,
            proba_pos[wrong_mask],
            1.0 - proba_pos[wrong_mask],
        )
        rows.insert(3, "pred_proba", np.round(proba_pred, 4))
        rows.insert(4, "proba_positive", np.round(proba_pos[wrong_mask], 4))
        rows = rows.sort_values("pred_proba", ascending=False)
    else:
        rows = rows.sort_values(
            ["section_id", "window_idx"] if "window_idx" in rows.columns else ["section_id"]
        )

    write_csv(rows, output_path)
    log.info(
        "Wrote %d misclassified windows → %s",
        len(rows),
        project_relative_path(output_path),
    )
    return int(len(rows))


def _build_model(name: str, seed: int) -> Any:
    """Build model."""
    cls = _MODEL_REGISTRY[name]
    kwargs: dict[str, Any] = {"random_state": seed}
    if name == "logistic_regression":
        kwargs["max_iter"] = 1000
        kwargs["class_weight"] = "balanced"
    if name == "random_forest":
        kwargs["class_weight"] = "balanced"
    if name == "hist_gradient_boosting":
        kwargs["class_weight"] = "balanced"
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
    """Return cv evaluate."""
    cv = GroupKFold(n_splits=n_splits)

    # Pre-allocate OOF arrays — every sample is a validation sample exactly once.
    oof_pred = np.empty(len(y), dtype=y.dtype)
    n_classes = len(np.unique(y))
    oof_proba: np.ndarray | None = np.full((len(y), n_classes), np.nan, dtype=float)

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

        clf = pipe.named_steps["clf"]
        if oof_proba is not None and hasattr(clf, "predict_proba"):
            proba = pipe.predict_proba(X_val)
            # Map per-fold class order onto the global class index so OOF
            # probabilities are comparable across folds even when a fold is
            # missing a class.
            for col_idx, cls in enumerate(clf.classes_):
                global_col = int(np.where(np.unique(y) == cls)[0][0])
                oof_proba[val_idx, global_col] = proba[:, col_idx]
        else:
            oof_proba = None

        fold_accuracies.append(float(accuracy_score(y_val, y_hat)))
        fold_f1s.append(float(f1_score(y_val, y_hat, average="macro", zero_division=0)))

        if hasattr(clf, "feature_importances_"):
            fold_importances.append(clf.feature_importances_.copy())

    # --- OOF aggregate metrics -------------------------------------------------
    # classification_report and confusion_matrix are computed on the stacked OOF
    # predictions, which cover all samples exactly once under GroupKFold.
    labels = np.unique(y)
    report = classification_report(
        y, oof_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, oof_pred, labels=labels)

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
        "labels": labels.tolist(),
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
    }


def run_evaluation(
    features_path: Path | str,
    *,
    output_dir: Path | str,
    label_col: str = "scenario_label_activity",
    group_col: str = "section_id",
    seed: int = 42,
    min_quality: str = "marginal",
    feature_sets: dict[str, list[str]] | None = None,
    exclude_non_riding: bool = False,
    no_plots: bool = False,
    compute_permutation_importance: bool = True,
    permutation_n_repeats: int = 5,
    permutation_models: tuple[str, ...] = ("random_forest",),
) -> dict:
    """Train and evaluate models on a feature table."""
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

    # Derive missing label columns on the fly so older feature tables stay
    # usable without a full re-extraction.
    if label_col not in df.columns:
        cfg_labels = default_label_config()
        if label_col in cfg_labels.set_based_scheme_names:
            if "scenario_labels" not in df.columns:
                raise ValueError(
                    f"Cannot derive {label_col!r}: column 'scenario_labels' missing "
                    "(set-based schemes require the pipe-delimited overlap set)."
                )
            df[label_col] = (
                df["scenario_labels"]
                .fillna("unlabeled")
                .astype(str)
                .map(
                    lambda s: to_set_based_label(
                        label_col,
                        [t for t in s.split("|") if t.strip()],
                        config=cfg_labels,
                    )
                )
            )
            log.info("Derived %s from scenario_labels (multi-label set)", label_col)
        elif label_col in _DERIVED_LABEL_MAPPERS:
            if "scenario_label" not in df.columns:
                raise ValueError(
                    f"Cannot derive {label_col!r}: source column 'scenario_label' missing."
                )
            mapper = _DERIVED_LABEL_MAPPERS[label_col]
            df[label_col] = (
                df["scenario_label"].fillna("unlabeled").astype(str).map(mapper)
            )
            log.info("Derived %s from scenario_label", label_col)

    if exclude_non_riding:
        binary_col = "scenario_label_binary"
        if binary_col not in df.columns:
            if "scenario_label" not in df.columns:
                raise ValueError(
                    "Cannot exclude non-riding windows: source column "
                    "'scenario_label' missing."
                )
            df[binary_col] = (
                df["scenario_label"].fillna("unlabeled").astype(str).map(to_binary_label)
            )
            log.info("Derived %s from scenario_label", binary_col)
        before = len(df)
        df = df[df[binary_col] != "non_riding"].copy()
        log.info("Excluded non-riding rows: %d → %d", before, len(df))

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
            (output_dir / model_name).mkdir(exist_ok=True)

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
            cm_path = output_dir / model_name / f"confusion_matrix_{fs_name}.csv"
            cm_df = pd.DataFrame(
                result["confusion_matrix"],
                index=le.classes_,
                columns=le.classes_,
            )
            cm_df.to_csv(cm_path, index=True)
            log.debug("Wrote confusion matrix to %s", project_relative_path(cm_path))

            # Confusion analysis: hardest classes + dominant confusion targets +
            # top-N off-diagonal pairs.  Cheap (pure NumPy on the matrix);
            # written next to the matrix it derives from.
            try:
                from evaluation.confusion_analysis import write_confusion_analysis
                write_confusion_analysis(
                    cm_df,
                    output_dir / model_name,
                    config_name=fs_name,
                )
            except Exception as exc:
                log.warning("Confusion analysis failed for %s: %s", key, exc)

            # Write per-class report (OOF-based)
            per_class_path = output_dir / model_name /f"per_class_report_{fs_name}.json"
            per_class_path.write_text(
                json.dumps(result["per_class"], indent=2), encoding="utf-8"
            )

            # Binary-only block: full metrics + misclassified-window export.
            # Triggered when the target has exactly 2 classes.  For configured
            # set-based schemes, positive/negative come from labels.default.json
            # so PR-AUC reflects "detect cornering" etc., not alphabetical order.
            if len(classes) == 2:
                cfg_eval = default_label_config()
                sb = cfg_eval.set_based_scheme(label_col)
                if (
                    sb is not None
                    and sb.positive_value in classes
                    and sb.negative_value in classes
                ):
                    positive_class = sb.positive_value
                    negative_class = sb.negative_value
                else:
                    negative_class = next(
                        (c for c in classes if str(c).lower() == "non_riding"),
                        classes[0],
                    )
                    positive_class = next(c for c in classes if c != negative_class)

                proba_pos: np.ndarray | None = None
                proba_full = result.get("oof_proba")
                if proba_full is not None:
                    pos_global_idx = list(le.classes_).index(positive_class)
                    proba_pos = proba_full[:, pos_global_idx]

                try:
                    binary = _binary_metrics(
                        y_true=y_all,
                        y_pred=result["oof_pred"],
                        proba_pos=proba_pos,
                        classes=list(le.classes_),
                        positive_class=positive_class,
                    )
                except Exception as exc:
                    log.warning("Binary metrics failed for %s: %s", key, exc)
                    binary = None

                if binary is not None:
                    binary_path = output_dir / model_name / f"binary_metrics_{fs_name}.json"
                    binary_path.write_text(json.dumps(binary, indent=2), encoding="utf-8")
                    log.info(
                        "Binary metrics %s: balanced_acc=%.3f  AP=%s  ROC-AUC=%s",
                        key,
                        binary["balanced_accuracy"],
                        f"{binary['average_precision']:.3f}" if "average_precision" in binary else "n/a",
                        f"{binary['roc_auc']:.3f}" if "roc_auc" in binary else "n/a",
                    )

                    miscls_path = output_dir / model_name / f"misclassified_{fs_name}.csv"
                    n_miscls = _export_misclassified(
                        df=df,
                        y_true=y_all,
                        y_pred=result["oof_pred"],
                        proba_pos=proba_pos,
                        classes=list(le.classes_),
                        label_col=label_col,
                        output_path=miscls_path,
                        positive_class=positive_class,
                    )

                    # Per-section overlay PNGs: signal streams + annotation
                    # strip + misclassified-window highlights. Skipped under
                    # no_plots; safe to skip if section CSVs are missing
                    # (the helper already warns and continues per section).
                    if not no_plots and n_miscls > 0:
                        try:
                            from evaluation.plots import plot_misclassified_overlay
                            overlay_dir = (
                                output_dir / model_name / f"misclassified_overlays_{fs_name}"
                            )
                            plot_misclassified_overlay(
                                miscls_path,
                                overlay_dir,
                                title_suffix=f"{model_name} / {fs_name}",
                            )
                        except Exception as exc:
                            log.warning(
                                "Misclassified overlay generation failed for %s: %s",
                                key, exc,
                            )

            # Write fold-wise scores so variance is inspectable
            fold_path = output_dir / model_name / f"fold_scores_{fs_name}.csv"
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
                fi_path = output_dir / model_name / f"feature_importance_{fs_name}.csv"
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
    # 5b. IMU contribution: paired feature-set comparison
    # ------------------------------------------------------------------
    # Operates on in-memory `all_results` so it shares the exact GroupKFold
    # splits used during CV — the paired Wilcoxon test is only valid when
    # the same fold indices are used for every feature set.
    imu_summary_path: Path | None = None
    if all_results:
        try:
            from evaluation.imu_contribution import write_imu_contribution
            imu_summary_path, _ = write_imu_contribution(
                all_results,
                classes=classes,
                output_dir=output_dir,
            )
        except Exception as exc:
            log.warning("IMU contribution analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 5c. Permutation feature importance (model-agnostic)
    # ------------------------------------------------------------------
    # Only run for the models listed in `permutation_models` to bound runtime.
    # Output goes alongside the existing impurity-based feature_importance_*.csv
    # so the user can compare the two rankings.
    if compute_permutation_importance and all_results:
        try:
            from evaluation.permutation_importance import (
                compute_permutation_importance_grouped,
                write_permutation_importance,
            )
        except Exception as exc:
            log.warning("Could not import permutation importance helpers: %s", exc)
        else:
            for fs_name, feat_cols in active_sets.items():
                # Re-derive X for this fs: same logic as the CV loop above so
                # we operate on the same column set (after dropping all-NaN
                # columns, which `_select_feature_cols` already handled).
                sub = df[feat_cols].apply(pd.to_numeric, errors="coerce")
                drop = [c for c in feat_cols if sub[c].isna().all()]
                cols = [c for c in feat_cols if c not in drop]
                if not cols:
                    continue
                X = sub[cols].to_numpy(dtype=float)

                for model_name in permutation_models:
                    if f"{fs_name}__{model_name}" not in all_results:
                        continue
                    log.info(
                        "Permutation importance: %s / %s (n_features=%d, n_repeats=%d) ...",
                        fs_name, model_name, len(cols), permutation_n_repeats,
                    )
                    try:
                        perm_df = compute_permutation_importance_grouped(
                            X, y_all,
                            groups=groups_all if groups_all is not None else np.arange(len(y_all)),
                            model=_build_model(model_name, seed),
                            feature_names=cols,
                            n_splits=min(5, len(np.unique(groups_all)) if groups_all is not None else len(y_all)),
                            n_repeats=permutation_n_repeats,
                            scoring="f1_macro",
                            seed=seed,
                        )
                        write_permutation_importance(
                            perm_df,
                            output_dir / model_name,
                            config_name=fs_name,
                        )
                    except Exception as exc:
                        log.warning(
                            "Permutation importance failed for %s/%s: %s",
                            fs_name, model_name, exc,
                        )

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
        "exclude_non_riding": exclude_non_riding,
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

        # IMU contribution plot — requires imu_contribution.csv from §5b.
        if imu_summary_path is not None and imu_summary_path.exists():
            try:
                from evaluation.plots import plot_imu_contribution
                imu_df = pd.read_csv(imu_summary_path)
                for metric in ("macro_f1", "accuracy"):
                    plot_imu_contribution(
                        imu_df,
                        output_dir / "figures" / f"imu_contribution_{metric}.png",
                        metric=metric,
                    )
            except Exception as exc:
                log.warning("IMU contribution plot failed: %s", exc)

        # Permutation importance plots — one bar chart per (model, fs) and
        # one sensor-group totals chart per (model, fs).  Reads the CSVs we
        # wrote in §5c to stay decoupled from the in-memory result dict.
        if compute_permutation_importance:
            try:
                from evaluation.plots import (
                    plot_permutation_importance,
                    plot_sensor_group_contribution,
                )
            except Exception as exc:
                log.warning("Could not import permutation-importance plot helpers: %s", exc)
            else:
                for model_name in permutation_models:
                    model_dir = output_dir / model_name
                    fig_dir = model_dir / "figures"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    for fi_path in sorted(model_dir.glob("permutation_importance_*.csv")):
                        if fi_path.stem.startswith("permutation_importance_by_group_"):
                            continue
                        fs_name = fi_path.stem.removeprefix("permutation_importance_")
                        try:
                            perm_df = pd.read_csv(fi_path)
                            plot_permutation_importance(
                                perm_df,
                                fig_dir / f"{fi_path.stem}.png",
                                title=(
                                    f"Permutation importance — "
                                    f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                                    f"{_FS_DISPLAY.get(fs_name, fs_name)}"
                                ),
                            )
                        except Exception as exc:
                            log.warning(
                                "Permutation importance plot failed for %s: %s",
                                fi_path.name, exc,
                            )
                    for grp_path in sorted(model_dir.glob("permutation_importance_by_group_*.csv")):
                        fs_name = grp_path.stem.removeprefix("permutation_importance_by_group_")
                        try:
                            grp_df = pd.read_csv(grp_path)
                            plot_sensor_group_contribution(
                                grp_df,
                                fig_dir / f"{grp_path.stem}.png",
                                title=(
                                    f"Sensor-group contribution — "
                                    f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                                    f"{_FS_DISPLAY.get(fs_name, fs_name)}"
                                ),
                            )
                        except Exception as exc:
                            log.warning(
                                "Sensor-group plot failed for %s: %s",
                                grp_path.name, exc,
                            )

    return summary


def _ordered_feature_sets(metrics_rows: list[dict]) -> list[str]:
    """Return feature sets in canonical order."""
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
    """Return pivot table lines."""
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
