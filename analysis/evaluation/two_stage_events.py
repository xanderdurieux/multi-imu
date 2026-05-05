"""Two-stage event detection and contrast evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.paths import features_fingerprint, merge_csv, project_relative_path, read_csv, write_csv
from evaluation.experiments import (
    _auto_detect_feature_sets,
    _build_model,
    _select_feature_cols,
    resolve_evaluation_models,
)

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["poor", "marginal", "good"]
_FEATURE_SET_PAIRS: tuple[tuple[str, str], ...] = (
    ("fused", "bike"),
    ("fused", "rider"),
    ("fused", "fused_no_cross"),
    ("rider", "bike"),
)
_CORE_TASKS = ("turning", "deceleration")


@dataclass(frozen=True)
class TwoStageEventTask:
    """Configuration for a broad detector followed by a binary contrast."""

    name: str
    detector_positive_tokens: frozenset[str]
    detector_positive_label: str
    detector_negative_label: str
    normal_token: str
    critical_token: str
    normal_label: str
    critical_label: str
    priority: str = "core"


TWO_STAGE_EVENT_TASKS: dict[str, TwoStageEventTask] = {
    "turning": TwoStageEventTask(
        name="turning",
        detector_positive_tokens=frozenset({"cornering", "swerving"}),
        detector_positive_label="cornering_or_swerving",
        detector_negative_label="non_cornering",
        normal_token="cornering",
        critical_token="swerving",
        normal_label="cornering",
        critical_label="swerving",
    ),
    "deceleration": TwoStageEventTask(
        name="deceleration",
        detector_positive_tokens=frozenset({"slowing", "hard_braking"}),
        detector_positive_label="slowing_or_hard_braking",
        detector_negative_label="non_deceleration",
        normal_token="slowing",
        critical_token="hard_braking",
        normal_label="slowing",
        critical_label="hard_braking",
    ),
    "high_effort": TwoStageEventTask(
        name="high_effort",
        detector_positive_tokens=frozenset({"accelerating", "sprinting"}),
        detector_positive_label="accelerating_or_sprinting",
        detector_negative_label="non_high_effort",
        normal_token="accelerating",
        critical_token="sprinting",
        normal_label="accelerating",
        critical_label="sprinting",
        priority="optional",
    ),
    "posture": TwoStageEventTask(
        name="posture",
        detector_positive_tokens=frozenset({"riding", "riding_standing"}),
        detector_positive_label="riding_or_standing",
        detector_negative_label="non_riding_posture",
        normal_token="riding",
        critical_token="riding_standing",
        normal_label="riding",
        critical_label="riding_standing",
        priority="optional",
    ),
}

DEFAULT_TWO_STAGE_EVENT_TASKS: tuple[TwoStageEventTask, ...] = tuple(
    TWO_STAGE_EVENT_TASKS[name] for name in _CORE_TASKS
)


def _token_set(raw: Any) -> set[str]:
    """Return normalized scenario-label tokens."""
    return {
        token.strip()
        for token in str(raw or "").split("|")
        if token.strip() and token.strip().lower() not in {"nan", "none", "unlabeled"}
    }


def _display_path(path: Path | str) -> str:
    """Return a readable project-relative path when possible."""
    try:
        return project_relative_path(path)
    except ValueError:
        return str(Path(path))


def _apply_quality_filter(df: pd.DataFrame, min_quality: str | None) -> pd.DataFrame:
    """Return rows at or above the requested quality level."""
    if min_quality is None:
        return df.copy()
    if min_quality not in _QUALITY_ORDER:
        raise ValueError(f"min_quality must be one of {_QUALITY_ORDER} or None")
    if "overall_quality_label" not in df.columns:
        log.warning("overall_quality_label missing; running without quality filter")
        return df.copy()
    min_idx = _QUALITY_ORDER.index(min_quality)
    valid = set(_QUALITY_ORDER[min_idx:])
    return df[df["overall_quality_label"].isin(valid)].copy()


def resolve_two_stage_tasks(
    task_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[TwoStageEventTask, ...]:
    """Resolve task names, where ``None`` or ``['core']`` selects core tasks."""
    if task_names is None or list(task_names) == ["core"]:
        return DEFAULT_TWO_STAGE_EVENT_TASKS
    if list(task_names) == ["all"]:
        return tuple(TWO_STAGE_EVENT_TASKS.values())
    out: list[TwoStageEventTask] = []
    seen: set[str] = set()
    for name in task_names:
        if name not in TWO_STAGE_EVENT_TASKS:
            raise ValueError(
                f"unknown two-stage task {name!r}; "
                f"expected one of {sorted(TWO_STAGE_EVENT_TASKS)} plus 'core' or 'all'"
            )
        if name not in seen:
            seen.add(name)
            out.append(TWO_STAGE_EVENT_TASKS[name])
    return tuple(out)


def resolve_two_stage_labels(
    tokens: set[str],
    task: TwoStageEventTask,
) -> tuple[str | None, str | None, bool, bool]:
    """Return detector/contrast labels and inclusion flags for one token set."""
    if not tokens:
        return None, None, False, False

    detector_hit = bool(tokens & task.detector_positive_tokens)
    detector_label = (
        task.detector_positive_label if detector_hit else task.detector_negative_label
    )

    normal_hit = task.normal_token in tokens
    critical_hit = task.critical_token in tokens
    if normal_hit and critical_hit:
        contrast_label = "ambiguous_both_tokens"
        contrast_eligible = False
    elif normal_hit:
        contrast_label = task.normal_label
        contrast_eligible = True
    elif critical_hit:
        contrast_label = task.critical_label
        contrast_eligible = True
    else:
        contrast_label = None
        contrast_eligible = False

    return detector_label, contrast_label, True, contrast_eligible


def build_two_stage_task_table(
    df: pd.DataFrame,
    task: TwoStageEventTask,
) -> pd.DataFrame:
    """Return rows with detector and contrast labels for one task."""
    if "scenario_labels" not in df.columns:
        raise ValueError("features table must contain scenario_labels")

    out = df.copy()
    token_sets = [_token_set(raw) for raw in out["scenario_labels"].fillna("")]
    labels = [resolve_two_stage_labels(tokens, task) for tokens in token_sets]
    out["two_stage_task"] = task.name
    out["contains_normal_token"] = [task.normal_token in tokens for tokens in token_sets]
    out["contains_critical_token"] = [task.critical_token in tokens for tokens in token_sets]
    out["detector_label"] = [item[0] for item in labels]
    out["contrast_label"] = [item[1] for item in labels]
    out["detector_eligible"] = [item[2] for item in labels]
    out["contrast_eligible"] = [item[3] for item in labels]
    return out[out["detector_eligible"]].copy()


def _active_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return usable feature-set columns."""
    active: dict[str, list[str]] = {}
    for fs_name, prefixes in _auto_detect_feature_sets(df).items():
        cols = _select_feature_cols(df, prefixes)
        usable = [
            c
            for c in cols
            if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all()
        ]
        if usable:
            active[fs_name] = usable
    if not active:
        raise ValueError("No usable two-stage event feature columns found")
    return active


def _ensure_recording_groups(df: pd.DataFrame) -> pd.Series:
    """Return recording-level group labels."""
    if "recording_name" in df.columns:
        return df["recording_name"].astype(str)
    if "section_id" in df.columns:
        return df["section_id"].astype(str).map(
            lambda s: s.rsplit("s", 1)[0]
            if len(s.rsplit("s", 1)) == 2 and s.rsplit("s", 1)[1].isdigit()
            else s
        )
    raise ValueError("features table must contain recording_name or section_id")


def _pipeline(model: Any) -> Pipeline:
    """Return the standard evaluation pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", clone(model)),
        ]
    )


def _positive_proba(pipe: Pipeline, X: np.ndarray, positive_class: int = 1) -> np.ndarray:
    """Return positive-class probability or NaNs if unavailable."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "predict_proba"):
        return np.full(len(X), np.nan, dtype=float)
    proba = pipe.predict_proba(X)
    classes = list(clf.classes_)
    if positive_class not in classes:
        return np.full(len(X), np.nan, dtype=float)
    return proba[:, classes.index(positive_class)]


def select_detector_threshold(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    *,
    target_recall: float = 0.90,
) -> float:
    """Select the highest-F2 threshold that reaches target recall when possible."""
    finite = np.isfinite(proba_pos)
    if not finite.any() or len(np.unique(y_true[finite])) < 2:
        return 0.5

    y = y_true[finite].astype(int)
    p = proba_pos[finite].astype(float)
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    rows: list[tuple[float, float, float]] = []
    for thr in thresholds:
        pred = (p >= thr).astype(int)
        recall = recall_score(y, pred, zero_division=0)
        f2 = fbeta_score(y, pred, beta=2.0, zero_division=0)
        rows.append((float(thr), float(recall), float(f2)))

    viable = [row for row in rows if row[1] >= target_recall]
    if viable:
        # Among thresholds that meet recall, maximize F2; on ties prefer the
        # stricter threshold so fewer false positives are routed downstream.
        return max(viable, key=lambda row: (row[2], row[0]))[0]
    return max(rows, key=lambda row: (row[2], row[1], row[0]))[0]


def _binary_metric_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray | None = None,
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Return binary metrics with stable labels and nan-safe AUCs."""
    if len(y_true) == 0:
        return {
            f"{prefix}_support": 0,
            f"{prefix}_positive_support": 0,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_balanced_accuracy": np.nan,
            f"{prefix}_macro_f1": np.nan,
            f"{prefix}_precision": np.nan,
            f"{prefix}_recall": np.nan,
            f"{prefix}_f2": np.nan,
            f"{prefix}_average_precision": np.nan,
            f"{prefix}_roc_auc": np.nan,
        }

    balanced_accuracy = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2
        else np.nan
    )
    out: dict[str, float | int] = {
        f"{prefix}_support": int(len(y_true)),
        f"{prefix}_positive_support": int((y_true == 1).sum()),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": balanced_accuracy,
        f"{prefix}_macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
        ),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        f"{prefix}_average_precision": np.nan,
        f"{prefix}_roc_auc": np.nan,
    }
    if proba_pos is not None:
        finite = np.isfinite(proba_pos)
        if finite.any() and len(np.unique(y_true[finite])) == 2:
            out[f"{prefix}_average_precision"] = float(
                average_precision_score(y_true[finite], proba_pos[finite])
            )
            out[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true[finite], proba_pos[finite]))
    return out


def _support_by_recording(df: pd.DataFrame, task: TwoStageEventTask) -> dict[str, Any]:
    """Return compact support counts by recording."""
    if df.empty:
        return {}
    groups = _ensure_recording_groups(df)
    rows = []
    for rec, sub in df.assign(_recording_group=groups).groupby("_recording_group"):
        rows.append(
            {
                "recording_name": rec,
                "detector_positive": int(
                    (sub["detector_label"] == task.detector_positive_label).sum()
                ),
                "detector_negative": int(
                    (sub["detector_label"] == task.detector_negative_label).sum()
                ),
                "normal": int((sub["contrast_label"] == task.normal_label).sum()),
                "critical": int((sub["contrast_label"] == task.critical_label).sum()),
                "ambiguous": int((sub["contrast_label"] == "ambiguous_both_tokens").sum()),
            }
        )
    return {"recordings": rows}


def _cv_two_stage(
    task_df: pd.DataFrame,
    task: TwoStageEventTask,
    feature_cols: list[str],
    model_id: str,
    *,
    seed: int,
    target_recall: float,
    hop_s: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run recording-level OOF two-stage evaluation for one feature set/model."""
    X_all = task_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    groups = _ensure_recording_groups(task_df).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("at least two recording groups are required")

    detector_y = (task_df["detector_label"] == task.detector_positive_label).to_numpy(dtype=int)
    contrast_y = task_df["contrast_label"].map(
        {task.normal_label: 0, task.critical_label: 1}
    )
    critical_truth = task_df["contains_critical_token"].to_numpy(dtype=int)
    contrast_eligible = task_df["contrast_eligible"].to_numpy(dtype=bool)

    cv = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    model = _build_model(model_id, seed)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_all, detector_y, groups), start=1):
        log.info(
            "    fold %d/%d (train=%d, val=%d) ...",
            fold_idx, n_splits, len(train_idx), len(val_idx),
        )
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if train_groups & val_groups:
            raise ValueError("recording leakage detected across train/validation folds")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        det_y_train, det_y_val = detector_y[train_idx], detector_y[val_idx]
        if len(np.unique(det_y_train)) < 2:
            raise ValueError(f"fold {fold_idx}: detector training has only one class")

        detector = _pipeline(model)
        detector.fit(X_train, det_y_train)
        train_proba = _positive_proba(detector, X_train)
        threshold = select_detector_threshold(
            det_y_train,
            train_proba,
            target_recall=target_recall,
        )
        det_proba_val = _positive_proba(detector, X_val)
        if np.all(np.isnan(det_proba_val)):
            det_pred_val = detector.predict(X_val).astype(int)
        else:
            det_pred_val = (det_proba_val >= threshold).astype(int)

        c_train_mask = contrast_eligible[train_idx]
        c_val_mask = contrast_eligible[val_idx]
        contrast_model: Pipeline | None = None
        if c_train_mask.any():
            c_train_idx = train_idx[c_train_mask]
            y_c_train = contrast_y.iloc[c_train_idx].to_numpy(dtype=int)
            if len(np.unique(y_c_train)) == 2:
                contrast_model = _pipeline(model)
                contrast_model.fit(X_all[c_train_idx], y_c_train)

        oracle_pred = np.full(len(val_idx), -1, dtype=int)
        oracle_proba = np.full(len(val_idx), np.nan, dtype=float)
        if contrast_model is not None and c_val_mask.any():
            local = np.where(c_val_mask)[0]
            X_c_val = X_val[local]
            oracle_pred[local] = contrast_model.predict(X_c_val).astype(int)
            oracle_proba[local] = _positive_proba(contrast_model, X_c_val)

        routed_mask = det_pred_val == 1
        predicted_gated_pred = np.full(len(val_idx), -1, dtype=int)
        predicted_gated_proba = np.full(len(val_idx), np.nan, dtype=float)
        if contrast_model is not None and routed_mask.any():
            local = np.where(routed_mask)[0]
            X_routed = X_val[local]
            predicted_gated_pred[local] = contrast_model.predict(X_routed).astype(int)
            predicted_gated_proba[local] = _positive_proba(contrast_model, X_routed)

        final_critical_pred = (
            (det_pred_val == 1) & (predicted_gated_pred == 1)
        ).astype(int)

        val_task_df = task_df.iloc[val_idx].copy()
        val_task_df["fold"] = fold_idx
        val_task_df["detector_true"] = det_y_val
        val_task_df["detector_proba"] = det_proba_val
        val_task_df["detector_threshold"] = threshold
        val_task_df["detector_pred"] = det_pred_val
        val_task_df["contrast_true"] = contrast_y.iloc[val_idx].to_numpy()
        val_task_df["contrast_proba_critical_oracle"] = oracle_proba
        val_task_df["contrast_pred_oracle"] = oracle_pred
        val_task_df["contrast_proba_critical_routed"] = predicted_gated_proba
        val_task_df["contrast_pred_routed"] = predicted_gated_pred
        val_task_df["final_critical_true"] = critical_truth[val_idx]
        val_task_df["final_critical_pred"] = final_critical_pred
        val_task_df["routing_status"] = np.select(
            [
                (det_pred_val == 0) & c_val_mask,
                (det_pred_val == 1) & c_val_mask,
                (det_pred_val == 1) & ~c_val_mask,
            ],
            ["detector_miss_event", "routed_true_event", "routed_false_positive"],
            default="not_routed",
        )
        val_task_df["final_label"] = np.select(
            [
                det_pred_val == 0,
                (det_pred_val == 1) & (predicted_gated_pred == 1),
                (det_pred_val == 1) & (predicted_gated_pred == 0),
            ],
            ["non_event", task.critical_label, task.normal_label],
            default="event_unclassified",
        )
        pred_frames.append(val_task_df)

        fold_row: dict[str, Any] = {
            "fold": fold_idx,
            "train_recordings": "|".join(sorted(train_groups)),
            "validation_recordings": "|".join(sorted(val_groups)),
            "detector_threshold": threshold,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_routed": int(routed_mask.sum()),
            "n_routed_true_event": int((routed_mask & c_val_mask).sum()),
            "n_routed_false_positive": int((routed_mask & ~c_val_mask).sum()),
            "n_detector_missed_events": int(((det_pred_val == 0) & c_val_mask).sum()),
        }
        fold_row.update(
            _binary_metric_block(
                det_y_val,
                det_pred_val,
                det_proba_val,
                prefix="detector",
            )
        )
        if c_val_mask.any() and np.any(oracle_pred[c_val_mask] >= 0):
            fold_row.update(
                _binary_metric_block(
                    contrast_y.iloc[val_idx].to_numpy(dtype=float)[c_val_mask].astype(int),
                    oracle_pred[c_val_mask],
                    oracle_proba[c_val_mask],
                    prefix="oracle_contrast",
                )
            )
        else:
            fold_row.update(_binary_metric_block(np.array([]), np.array([]), prefix="oracle_contrast"))

        routed_true_event = routed_mask & c_val_mask & (predicted_gated_pred >= 0)
        if routed_true_event.any():
            fold_row.update(
                _binary_metric_block(
                    contrast_y.iloc[val_idx].to_numpy(dtype=float)[routed_true_event].astype(int),
                    predicted_gated_pred[routed_true_event],
                    predicted_gated_proba[routed_true_event],
                    prefix="predicted_gated_contrast",
                )
            )
        else:
            fold_row.update(_binary_metric_block(np.array([]), np.array([]), prefix="predicted_gated_contrast"))

        fold_row.update(
            _binary_metric_block(
                critical_truth[val_idx],
                final_critical_pred,
                np.where(det_pred_val == 1, predicted_gated_proba, 0.0),
                prefix="end_to_end_critical",
            )
        )
        fold_rows.append(fold_row)

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    predictions["merge_max_gap_ms"] = float(hop_s * 1000.0 * 1.5)
    return fold_rows, predictions


def _aggregate_fold_metrics(
    fold_df: pd.DataFrame,
    *,
    task: str,
    feature_set: str,
    model: str,
    n_features: int,
) -> dict[str, Any]:
    """Return one metrics row from fold-level results."""
    row: dict[str, Any] = {
        "task": task,
        "feature_set": feature_set,
        "model": model,
        "n_features": n_features,
        "n_folds": int(len(fold_df)),
    }
    metric_cols = [
        c
        for c in fold_df.columns
        if c.endswith(("_accuracy", "_balanced_accuracy", "_macro_f1", "_precision", "_recall", "_f2", "_average_precision", "_roc_auc"))
    ]
    for col in metric_cols:
        vals = pd.to_numeric(fold_df[col], errors="coerce").to_numpy(dtype=float)
        row[col] = round(float(np.nanmean(vals)), 4) if np.isfinite(vals).any() else np.nan
        row[f"{col}_std"] = (
            round(float(np.nanstd(vals, ddof=1)), 4)
            if np.isfinite(vals).sum() > 1
            else 0.0
        )
    for col in ("n_routed", "n_routed_true_event", "n_routed_false_positive", "n_detector_missed_events"):
        row[col] = int(pd.to_numeric(fold_df[col], errors="coerce").fillna(0).sum())
    return row


def _wilcoxon_pvalue(deltas: np.ndarray) -> float | None:
    """Return one-sided Wilcoxon p-value for positive fold deltas."""
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0 or np.all(finite == 0):
        return None
    try:
        return float(wilcoxon(finite, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def _imu_contribution_rows(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Return paired feature-set deltas from fold scores."""
    rows: list[dict[str, Any]] = []
    if fold_scores.empty:
        return pd.DataFrame()
    metrics = (
        "detector_macro_f1",
        "oracle_contrast_macro_f1",
        "predicted_gated_contrast_macro_f1",
        "end_to_end_critical_f2",
        "end_to_end_critical_recall",
    )
    for (task, model), sub in fold_scores.groupby(["task", "model"]):
        for better, baseline in _FEATURE_SET_PAIRS:
            b = sub[sub["feature_set"] == better].sort_values("fold")
            a = sub[sub["feature_set"] == baseline].sort_values("fold")
            if len(b) == 0 or len(b) != len(a):
                continue
            for metric in metrics:
                if metric not in b.columns or metric not in a.columns:
                    continue
                b_vals = pd.to_numeric(b[metric], errors="coerce").to_numpy(dtype=float)
                a_vals = pd.to_numeric(a[metric], errors="coerce").to_numpy(dtype=float)
                deltas = b_vals - a_vals
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "better": better,
                        "baseline": baseline,
                        "metric": metric,
                        "n_folds": int(len(deltas)),
                        "better_mean": round(float(np.nanmean(b_vals)), 4),
                        "baseline_mean": round(float(np.nanmean(a_vals)), 4),
                        "delta_mean": round(float(np.nanmean(deltas)), 4),
                        "delta_std": round(float(np.nanstd(deltas, ddof=1)), 4)
                        if np.isfinite(deltas).sum() > 1
                        else 0.0,
                        "delta_min": round(float(np.nanmin(deltas)), 4),
                        "delta_max": round(float(np.nanmax(deltas)), 4),
                        "wilcoxon_p_one_sided": _wilcoxon_pvalue(deltas),
                    }
                )
    return pd.DataFrame(rows)


def _candidate_columns(df: pd.DataFrame) -> list[str]:
    """Return useful candidate/prediction columns that exist in the frame."""
    preferred = [
        "task",
        "feature_set",
        "model",
        "recording_name",
        "section_id",
        "window_idx",
        "window_start_ms",
        "window_end_ms",
        "window_type",
        "scenario_labels",
        "detector_label",
        "contrast_label",
        "fold",
        "detector_proba",
        "detector_threshold",
        "detector_pred",
        "contrast_proba_critical_routed",
        "contrast_pred_routed",
        "final_label",
        "routing_status",
        "final_critical_true",
        "final_critical_pred",
    ]
    return [c for c in preferred if c in df.columns]


def _merge_candidate_intervals(candidates: pd.DataFrame, *, hop_s: float) -> pd.DataFrame:
    """Merge adjacent predicted-positive windows into simple intervals."""
    if candidates.empty:
        return pd.DataFrame()
    max_gap_ms = hop_s * 1000.0 * 1.5
    rows: list[dict[str, Any]] = []
    group_cols = [
        c
        for c in ("task", "feature_set", "model", "recording_name", "section_id", "final_label")
        if c in candidates.columns
    ]
    sorted_df = candidates.sort_values(group_cols + ["window_start_ms", "window_end_ms"])
    for group_key, sub in sorted_df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_cols, group_key, strict=False))
        current: dict[str, Any] | None = None
        for row in sub.itertuples(index=False):
            start = float(getattr(row, "window_start_ms"))
            end = float(getattr(row, "window_end_ms"))
            proba = float(getattr(row, "detector_proba", np.nan))
            crit_proba = float(getattr(row, "contrast_proba_critical_routed", np.nan))
            if current is None or start > float(current["end_ms"]) + max_gap_ms:
                if current is not None:
                    rows.append(current)
                current = {
                    **group_values,
                    "start_ms": start,
                    "end_ms": end,
                    "n_windows": 1,
                    "detector_proba_max": proba,
                    "critical_proba_max": crit_proba,
                }
            else:
                current["end_ms"] = max(float(current["end_ms"]), end)
                current["n_windows"] = int(current["n_windows"]) + 1
                current["detector_proba_max"] = float(
                    np.nanmax([current["detector_proba_max"], proba])
                )
                current["critical_proba_max"] = float(
                    np.nanmax([current["critical_proba_max"], crit_proba])
                )
        if current is not None:
            rows.append(current)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["duration_s"] = (out["end_ms"] - out["start_ms"]) / 1000.0
    return out


def _json_default(value: Any) -> Any:
    """Return JSON-friendly values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _task_to_dict(task: TwoStageEventTask) -> dict[str, Any]:
    """Return a JSON-friendly task config."""
    return {
        "name": task.name,
        "detector_positive_tokens": sorted(task.detector_positive_tokens),
        "detector_positive_label": task.detector_positive_label,
        "detector_negative_label": task.detector_negative_label,
        "normal_token": task.normal_token,
        "critical_token": task.critical_token,
        "normal_label": task.normal_label,
        "critical_label": task.critical_label,
        "priority": task.priority,
    }


def run_two_stage_event_evaluation(
    features_path: Path | str,
    *,
    output_dir: Path | str,
    task_names: list[str] | tuple[str, ...] | None = None,
    models: list[str] | tuple[str, ...] | None = None,
    min_quality: str | None = "marginal",
    seed: int = 42,
    target_recall: float = 0.90,
    hop_s: float = 1.0,
    no_plots: bool = False,
) -> dict[str, Any]:
    """Run two-stage event evaluation and write artifacts."""
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")

    root_output_dir = Path(output_dir)
    out_dir = root_output_dir / "two_stage_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = resolve_two_stage_tasks(task_names)
    model_ids = resolve_evaluation_models(
        list(models or ["hist_gradient_boosting", "logistic_regression"])
    )

    log.info("Loading features from %s", _display_path(features_path))
    source_df = _apply_quality_filter(read_csv(features_path), min_quality)
    if source_df.empty:
        raise ValueError("No rows remain after quality filtering")

    metrics_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    support: dict[str, Any] = {}

    n_tasks = len(tasks)
    for ti, task in enumerate(tasks, start=1):
        log.info("── Two-stage task %d/%d: %s ──", ti, n_tasks, task.name)
        task_df = build_two_stage_task_table(source_df, task)
        if task_df.empty:
            skipped.append({"task": task.name, "reason": "no_labeled_rows"})
            continue
        support[task.name] = _support_by_recording(task_df, task)

        detector_counts = task_df["detector_label"].value_counts().to_dict()
        contrast_counts = task_df.loc[task_df["contrast_eligible"], "contrast_label"].value_counts().to_dict()
        if len(detector_counts) < 2:
            skipped.append({"task": task.name, "reason": "detector_has_one_class", "counts": detector_counts})
            continue
        if len(contrast_counts) < 2:
            skipped.append({"task": task.name, "reason": "contrast_has_one_class", "counts": contrast_counts})
            continue

        groups = _ensure_recording_groups(task_df)
        class_group_counts = (
            task_df.assign(_group=groups)
            .groupby("detector_label")["_group"]
            .nunique()
            .to_dict()
        )
        if min(class_group_counts.values()) < 2:
            skipped.append(
                {
                    "task": task.name,
                    "reason": "detector_class_not_in_two_recordings",
                    "class_group_counts": class_group_counts,
                }
            )
            continue

        active_sets = _active_feature_sets(task_df)
        n_unique_groups = len(np.unique(_ensure_recording_groups(task_df)))
        n_splits_task = min(5, n_unique_groups)
        log.info(
            "  detector classes: %s | contrast classes: %s | groups=%d  splits=%d",
            detector_counts, contrast_counts, n_unique_groups, n_splits_task,
        )
        n_fs = len(active_sets)
        n_models = len(model_ids)
        for fsi, (feature_set, feature_cols) in enumerate(active_sets.items(), start=1):
            for mi, model_id in enumerate(model_ids, start=1):
                log.info(
                    "  [%d/%d fs, %d/%d model] %s / %s (%d features) ...",
                    fsi, n_fs, mi, n_models, feature_set, model_id, len(feature_cols),
                )
                try:
                    fold_rows, predictions = _cv_two_stage(
                        task_df,
                        task,
                        feature_cols,
                        model_id,
                        seed=seed,
                        target_recall=target_recall,
                        hop_s=hop_s,
                    )
                except Exception as exc:
                    log.warning(
                        "Two-stage event evaluation failed for %s/%s/%s: %s",
                        task.name,
                        feature_set,
                        model_id,
                        exc,
                    )
                    skipped.append(
                        {
                            "task": task.name,
                            "feature_set": feature_set,
                            "model": model_id,
                            "reason": str(exc),
                        }
                    )
                    continue

                fold_df = pd.DataFrame(fold_rows)
                if not fold_df.empty:
                    det_f1 = pd.to_numeric(fold_df.get("detector_macro_f1"), errors="coerce").mean()
                    e2e_f2 = pd.to_numeric(fold_df.get("end_to_end_critical_f2"), errors="coerce").mean()
                    log.info(
                        "    → det_macro_f1=%.3f  e2e_critical_f2=%.3f",
                        det_f1 if np.isfinite(det_f1) else float("nan"),
                        e2e_f2 if np.isfinite(e2e_f2) else float("nan"),
                    )
                fold_df.insert(0, "task", task.name)
                fold_df.insert(1, "feature_set", feature_set)
                fold_df.insert(2, "model", model_id)
                fold_frames.append(fold_df)

                predictions.insert(0, "task", task.name)
                predictions.insert(1, "feature_set", feature_set)
                predictions.insert(2, "model", model_id)
                prediction_frames.append(predictions)

                metrics_rows.append(
                    _aggregate_fold_metrics(
                        fold_df,
                        task=task.name,
                        feature_set=feature_set,
                        model=model_id,
                        n_features=len(feature_cols),
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "two_stage_event_metrics.csv"
    merge_csv(metrics_df, metrics_path, ["task", "feature_set", "model"])

    fold_scores_df = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    )
    fold_scores_path = out_dir / "two_stage_event_fold_scores.csv"
    merge_csv(fold_scores_df, fold_scores_path, ["task", "feature_set", "model", "fold"])

    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    predictions_path = out_dir / "two_stage_event_predictions.csv"
    write_csv(predictions_df, predictions_path)

    if predictions_df.empty:
        candidates_df = pd.DataFrame()
    else:
        candidates_df = predictions_df[predictions_df["detector_pred"] == 1].copy()
        candidates_df = candidates_df[_candidate_columns(candidates_df)]
    candidates_path = out_dir / "two_stage_event_candidates.csv"
    write_csv(candidates_df, candidates_path)

    intervals_df = _merge_candidate_intervals(candidates_df, hop_s=hop_s)
    intervals_path = out_dir / "two_stage_event_intervals.csv"
    write_csv(intervals_df, intervals_path)

    imu_df = _imu_contribution_rows(fold_scores_df)
    imu_path = out_dir / "two_stage_event_imu_contribution.csv"
    merge_csv(imu_df, imu_path, ["task", "model", "better", "baseline", "metric"])

    fp = features_fingerprint(features_path)
    summary_path = out_dir / "two_stage_event_summary.json"
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("features_fingerprint") != fp:
                log.warning(
                    "Features file has changed since the last two-stage event run "
                    "(fingerprint mismatch). Merged results may be inconsistent."
                )
        except Exception:
            pass

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "features_path": _display_path(features_path),
        "features_fingerprint": fp,
        "output_dir": _display_path(out_dir),
        "tasks": [_task_to_dict(task) for task in tasks],
        "models": list(model_ids),
        "grouping": "recording_name",
        "min_quality": min_quality,
        "seed": seed,
        "target_recall": target_recall,
        "hop_s": hop_s,
        "n_source_windows": int(len(source_df)),
        "n_metric_rows": int(len(metrics_df)),
        "skipped": skipped,
        "support": support,
        "artifacts": {
            "metrics": _display_path(metrics_path),
            "fold_scores": _display_path(fold_scores_path),
            "predictions": _display_path(predictions_path),
            "candidates": _display_path(candidates_path),
            "intervals": _display_path(intervals_path),
            "imu_contribution": _display_path(imu_path),
        },
    }
    if not no_plots:
        try:
            from visualization.plot_eval_events import generate_two_stage_event_figures
            figure_paths = generate_two_stage_event_figures(out_dir)
            summary["artifacts"]["figures"] = [
                _display_path(path) for path in figure_paths
            ]
        except Exception as exc:
            log.warning("Two-stage event figure generation failed: %s", exc)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    log.info(
        "Two-stage event evaluation complete: %d metric row(s) -> %s",
        len(metrics_df),
        _display_path(out_dir),
    )
    return summary
