"""Event contrast evaluation for second-IMU contribution analysis."""

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
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
from evaluation.permutation_importance import (
    aggregate_by_sensor_group,
    compute_permutation_importance_grouped,
)

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["poor", "marginal", "good"]
_FEATURE_SET_PAIRS: tuple[tuple[str, str], ...] = (
    ("fused", "bike"),
    ("fused", "rider"),
    ("fused", "fused_no_cross"),
    ("rider", "bike"),
)

_SENSOR_GROUP_NAME: dict[str, str] = {
    "bike_": "bike",
    "sporsa_": "bike",
    "rider_": "rider",
    "arduino_": "rider",
    "cross_": "cross",
}


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return balanced accuracy when both classes are present, otherwise nan."""
    return (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2
        else np.nan
    )


@dataclass(frozen=True)
class EventContrast:
    """Configuration for a binary event contrast."""

    name: str
    normal_token: str
    critical_token: str
    normal_label: str
    critical_label: str


DEFAULT_EVENT_CONTRASTS: tuple[EventContrast, ...] = (
    EventContrast(
        name="cornering_vs_swerving",
        normal_token="cornering",
        critical_token="swerving",
        normal_label="cornering",
        critical_label="swerving",
    ),
    EventContrast(
        name="slowing_vs_hard_braking",
        normal_token="slowing",
        critical_token="hard_braking",
        normal_label="slowing",
        critical_label="hard_braking",
    ),
)


def _token_set(raw: Any) -> set[str]:
    """Return normalized pipe-delimited scenario labels."""
    return {
        token.strip()
        for token in str(raw or "").split("|")
        if token.strip() and token.strip().lower() not in {"nan", "none", "unlabeled"}
    }


def contrast_label_from_tokens(tokens: set[str], contrast: EventContrast) -> str | None:
    """Return the resolved contrast label, or ``None`` when the row is excluded."""
    normal_hit = contrast.normal_token in tokens
    critical_hit = contrast.critical_token in tokens
    if normal_hit and critical_hit:
        return None
    if normal_hit:
        return contrast.normal_label
    if critical_hit:
        return contrast.critical_label
    return None


def _contrast_status(tokens: set[str], contrast: EventContrast) -> tuple[str, bool]:
    """Return audit label and whether the row should be used for evaluation."""
    normal_hit = contrast.normal_token in tokens
    critical_hit = contrast.critical_token in tokens
    if normal_hit and critical_hit:
        return "ambiguous_both_tokens", False
    label = contrast_label_from_tokens(tokens, contrast)
    if label is None:
        return "unrelated", False
    return label, True


def _sensor_group(feature: str) -> str:
    """Return the feature's sensor group."""
    for prefix, name in _SENSOR_GROUP_NAME.items():
        if feature.startswith(prefix):
            return name
    return "other"


def _wilcoxon_pvalue(deltas: np.ndarray) -> float | None:
    """Return a one-sided Wilcoxon p-value when the paired deltas are usable."""
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0 or np.all(finite == 0):
        return None
    try:
        return float(wilcoxon(finite, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def _apply_quality_filter(df: pd.DataFrame, min_quality: str | None) -> pd.DataFrame:
    """Return rows at or above the requested quality level."""
    if min_quality is None:
        return df.copy()
    if min_quality not in _QUALITY_ORDER:
        raise ValueError(f"min_quality must be one of {_QUALITY_ORDER} or None")
    if "overall_quality_label" not in df.columns:
        log.warning("overall_quality_label missing; event contrasts run without quality filter")
        return df.copy()
    min_idx = _QUALITY_ORDER.index(min_quality)
    valid = set(_QUALITY_ORDER[min_idx:])
    return df[df["overall_quality_label"].isin(valid)].copy()


def build_event_contrast_table(
    df: pd.DataFrame,
    contrast: EventContrast,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return included contrast rows and an audit table for touched windows."""
    if "scenario_labels" not in df.columns:
        raise ValueError("features table must contain scenario_labels")

    rows: list[dict[str, Any]] = []
    labels: list[str | None] = []
    used_mask: list[bool] = []

    tokens_series = df["scenario_labels"].map(_token_set)
    for idx, tokens in tokens_series.items():
        status, used = _contrast_status(tokens, contrast)
        if (
            contrast.normal_token in tokens
            or contrast.critical_token in tokens
            or used
        ):
            source = df.loc[idx]
            rows.append(
                {
                    "contrast": contrast.name,
                    "recording_name": source.get("recording_name", ""),
                    "section_id": source.get("section_id", ""),
                    "window_idx": source.get("window_idx", np.nan),
                    "window_start_ms": source.get("window_start_ms", np.nan),
                    "window_end_ms": source.get("window_end_ms", np.nan),
                    "window_type": source.get("window_type", ""),
                    "scenario_labels": source.get("scenario_labels", ""),
                    "contrast_label": status,
                    "used_in_evaluation": bool(used),
                }
            )
        labels.append(status if used else None)
        used_mask.append(bool(used))

    out = df.loc[used_mask].copy()
    out["event_contrast"] = contrast.name
    out["event_contrast_label"] = [label for label in labels if label is not None]
    audit = pd.DataFrame(rows)
    return out, audit


def _active_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return non-empty event-contrast feature sets."""
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
        raise ValueError("No usable event contrast feature columns found")
    return active


def _ensure_recording_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Return grouping labels, deriving recording_name from section_id if needed."""
    if group_col in df.columns:
        return df[group_col].astype(str)
    if group_col == "recording_name" and "section_id" in df.columns:
        return df["section_id"].astype(str).map(
            lambda s: s.rsplit("s", 1)[0]
            if len(s.rsplit("s", 1)) == 2 and s.rsplit("s", 1)[1].isdigit()
            else s
        )
    raise ValueError(f"group column {group_col!r} not found")


def _cv_binary_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Any,
    *,
    n_splits: int,
) -> dict[str, Any]:
    """Run grouped CV and return OOF binary metrics."""
    cv = GroupKFold(n_splits=n_splits)
    oof_pred = np.empty(len(y), dtype=y.dtype)
    oof_proba_pos = np.full(len(y), np.nan, dtype=float)
    all_labels = np.unique(y)

    fold_rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        log.info("    fold %d/%d (train=%d, val=%d) ...", fold_idx, n_splits, len(train_idx), len(val_idx))
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clone(model)),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[val_idx])
        oof_pred[val_idx] = pred

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(X[val_idx])
            classes = list(pipe.named_steps["clf"].classes_)
            if 1 in classes:
                oof_proba_pos[val_idx] = proba[:, classes.index(1)]

        fold_rows.append(
            {
                "fold": float(fold_idx),
                "accuracy": float(accuracy_score(y[val_idx], pred)),
                "balanced_accuracy": _safe_balanced_accuracy(y[val_idx], pred),
                "macro_f1": float(f1_score(y[val_idx], pred, average="macro", labels=all_labels, zero_division=0)),
            }
        )

    finite_proba = np.isfinite(oof_proba_pos)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y, oof_pred),
        "macro_f1": float(f1_score(y, oof_pred, average="macro", labels=all_labels, zero_division=0)),
        "fold_scores": fold_rows,
        "oof_pred": oof_pred,
        "oof_proba_pos": oof_proba_pos,
    }
    if finite_proba.any() and len(np.unique(y[finite_proba])) == 2:
        out["average_precision"] = float(average_precision_score(y[finite_proba], oof_proba_pos[finite_proba]))
        out["roc_auc"] = float(roc_auc_score(y[finite_proba], oof_proba_pos[finite_proba]))
    else:
        out["average_precision"] = np.nan
        out["roc_auc"] = np.nan
    return out


def summarize_feature_stability(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Any,
    feature_names: list[str],
    *,
    n_splits: int,
    top_k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Return fold-rank stability for model-native or validation permutation scores."""
    cv = GroupKFold(n_splits=n_splits)
    fold_scores: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clone(model)),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            scores = np.asarray(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_, dtype=float)
            scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            result = permutation_importance(
                pipe,
                X[val_idx],
                y[val_idx],
                n_repeats=1,
                random_state=seed + fold_idx,
                scoring="f1_macro",
                n_jobs=1,
            )
            scores = np.asarray(result.importances_mean, dtype=float)
        fold_scores.append(np.nan_to_num(scores, nan=0.0))

    if not fold_scores:
        return pd.DataFrame()

    scores_stack = np.vstack(fold_scores)
    ranks = np.zeros_like(scores_stack, dtype=float)
    for i, scores in enumerate(scores_stack):
        order = np.argsort(-scores, kind="stable")
        rank = np.empty_like(order, dtype=float)
        rank[order] = np.arange(1, len(scores) + 1)
        ranks[i] = rank

    top_n = min(top_k, len(feature_names))
    top_hits = (ranks <= top_n).mean(axis=0)
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "sensor_group": [_sensor_group(f) for f in feature_names],
            "rank_mean": ranks.mean(axis=0),
            "rank_std": ranks.std(axis=0, ddof=1) if ranks.shape[0] > 1 else 0.0,
            "top_k": top_n,
            "top_k_frequency": top_hits,
            "score_mean": scores_stack.mean(axis=0),
            "score_std": scores_stack.std(axis=0, ddof=1) if scores_stack.shape[0] > 1 else 0.0,
        }
    )
    return out.sort_values(["rank_mean", "top_k_frequency"], ascending=[True, False]).reset_index(drop=True)


def _paired_contribution_rows(
    results: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return paired feature-set contribution rows from fold metrics."""
    rows: list[dict[str, Any]] = []
    contrasts = sorted({k[0] for k in results})
    models = sorted({k[2] for k in results})
    for contrast in contrasts:
        for model in models:
            for better, baseline in _FEATURE_SET_PAIRS:
                rb = results.get((contrast, better, model))
                ra = results.get((contrast, baseline, model))
                if rb is None or ra is None:
                    continue
                fb = pd.DataFrame(rb["fold_scores"])
                fa = pd.DataFrame(ra["fold_scores"])
                if len(fb) == 0 or len(fb) != len(fa):
                    continue
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    deltas = fb[metric].to_numpy(dtype=float) - fa[metric].to_numpy(dtype=float)
                    rows.append(
                        {
                            "contrast": contrast,
                            "model": model,
                            "better": better,
                            "baseline": baseline,
                            "metric": metric,
                            "n_folds": int(len(deltas)),
                            "better_mean": round(float(fb[metric].mean()), 4),
                            "baseline_mean": round(float(fa[metric].mean()), 4),
                            "delta_mean": round(float(deltas.mean()), 4),
                            "delta_std": round(float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0, 4),
                            "delta_min": round(float(deltas.min()), 4),
                            "delta_max": round(float(deltas.max()), 4),
                            "wilcoxon_p_one_sided": _wilcoxon_pvalue(deltas),
                        }
                    )
    return rows


def _json_default(value: Any) -> Any:
    """Return JSON-friendly scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _display_path(path: Path | str) -> str:
    """Return a readable project-relative path when possible."""
    try:
        return project_relative_path(path)
    except ValueError:
        return str(Path(path))


_CONTRAST_FIGURE_CSV_KEYS: tuple[tuple[str, list[str]], ...] = (
    ("event_contrast_metrics.csv", ["contrast", "feature_set", "model"]),
    ("event_contrast_imu_contribution.csv", ["contrast", "model", "better", "baseline", "metric"]),
    ("event_contrast_feature_stability.csv", ["contrast", "feature_set", "model", "feature"]),
    ("event_contrast_feature_importance_by_group.csv", ["contrast", "model", "feature_set", "sensor_group"]),
)


def _flush_contrast_figures(contrast_dir: Path, event_output_dir: Path) -> None:
    """Merge per-contrast CSVs into top-level files and regenerate figures."""
    for fname, keys in _CONTRAST_FIGURE_CSV_KEYS:
        src = contrast_dir / fname
        if src.exists():
            try:
                df = read_csv(src)
                if not df.empty:
                    merge_csv(df, event_output_dir / fname, keys)
            except Exception as exc:
                log.warning("Intermediate CSV flush failed for %s: %s", fname, exc)
    try:
        from visualization.plot_eval_events import generate_event_contrast_figures
        generate_event_contrast_figures(event_output_dir)
    except Exception as exc:
        log.warning("Intermediate event contrast figures failed: %s", exc)


def _load_reusable_contrast_summary(
    contrast_dir: Path,
    contrast: EventContrast,
    *,
    fp: str,
    models: tuple[str, ...],
    group_col: str,
    min_quality: str | None,
    seed: int,
    permutation_n_repeats: int,
    stability_top_k: int,
) -> dict[str, Any] | None:
    """Return saved per-contrast summary when it matches the current config; None means rerun."""
    summary_path = contrast_dir / f"event_contrast_{contrast.name}_summary.json"
    if not summary_path.exists():
        return None
    try:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read contrast summary for %s: %s", contrast.name, exc)
        return None
    for key, expected in (
        ("features_fingerprint", fp),
        ("models", list(models)),
        ("group_col", group_col),
        ("min_quality", min_quality),
        ("seed", seed),
        ("permutation_n_repeats", permutation_n_repeats),
        ("stability_top_k", stability_top_k),
        ("contrast", contrast.name),
    ):
        if saved.get(key) != expected:
            return None
    if saved.get("status") == "ok" and not (contrast_dir / "event_contrast_metrics.csv").exists():
        return None
    return saved


def _collect_contrast_artifacts(
    contrast_dir: Path,
    reused_summary: dict[str, Any],
    *,
    metrics_rows: list[dict[str, Any]],
    audit_frames: list[pd.DataFrame],
    stability_frames: list[pd.DataFrame],
    perm_frames: list[pd.DataFrame],
    contribution_frames: list[pd.DataFrame],
    skipped: list[dict[str, Any]],
) -> None:
    """Load per-contrast CSVs into top-level accumulators when reusing a saved contrast."""
    skipped.extend(reused_summary.get("skipped", []))
    if reused_summary.get("status") != "ok":
        return

    def _try_load(filename: str) -> pd.DataFrame:
        p = contrast_dir / filename
        if p.exists():
            try:
                return read_csv(p)
            except Exception as exc:
                log.warning("Could not load %s for reuse: %s", p.name, exc)
        return pd.DataFrame()

    mdf = _try_load("event_contrast_metrics.csv")
    if not mdf.empty:
        metrics_rows.extend(mdf.to_dict("records"))

    wdf = _try_load("event_contrast_windows.csv")
    if not wdf.empty:
        audit_frames.append(wdf)

    sdf = _try_load("event_contrast_feature_stability.csv")
    if not sdf.empty:
        stability_frames.append(sdf)

    pdf = _try_load("event_contrast_feature_importance.csv")
    if not pdf.empty:
        perm_frames.append(pdf)

    cdf = _try_load("event_contrast_imu_contribution.csv")
    if not cdf.empty:
        contribution_frames.append(cdf)


def _write_contrast_artifacts(
    contrast_dir: Path,
    contrast: EventContrast,
    *,
    fp: str,
    models: tuple[str, ...],
    group_col: str,
    min_quality: str | None,
    seed: int,
    permutation_n_repeats: int,
    stability_top_k: int,
    status: str,
    n_windows: int,
    n_groups: int,
    n_splits: int,
    label_counts: dict[str, int],
    contrast_skipped: list[dict[str, Any]],
    contrast_metrics_rows: list[dict[str, Any]],
    contrast_stability_frames: list[pd.DataFrame],
    contrast_perm_frames: list[pd.DataFrame],
    contrast_results: dict[tuple[str, str, str], dict[str, Any]],
    audit: pd.DataFrame,
) -> None:
    """Write per-contrast artifact CSVs and summary JSON to contrast_dir."""
    metrics_df = pd.DataFrame(contrast_metrics_rows)
    if not metrics_df.empty:
        merge_csv(metrics_df, contrast_dir / "event_contrast_metrics.csv", ["feature_set", "model"])

    contribution_rows = _paired_contribution_rows(contrast_results)
    contribution_df = pd.DataFrame(contribution_rows)
    if not contribution_df.empty:
        merge_csv(
            contribution_df,
            contrast_dir / "event_contrast_imu_contribution.csv",
            ["model", "better", "baseline", "metric"],
        )

    stability_df = (
        pd.concat(contrast_stability_frames, ignore_index=True)
        if contrast_stability_frames
        else pd.DataFrame()
    )
    if not stability_df.empty:
        merge_csv(
            stability_df,
            contrast_dir / "event_contrast_feature_stability.csv",
            ["feature_set", "model", "feature"],
        )

    perm_df = (
        pd.concat(contrast_perm_frames, ignore_index=True)
        if contrast_perm_frames
        else pd.DataFrame()
    )
    if not perm_df.empty:
        merge_csv(
            perm_df,
            contrast_dir / "event_contrast_feature_importance.csv",
            ["model", "feature_set", "feature"],
        )
        grouped_frames: list[pd.DataFrame] = []
        for (model_id, fs_name), sub in perm_df.groupby(["model", "feature_set"], sort=False):
            drop_cols = [c for c in ("contrast", "model", "feature_set") if c in sub.columns]
            grp = aggregate_by_sensor_group(sub.drop(columns=drop_cols))
            grp.insert(0, "contrast", contrast.name)
            grp.insert(1, "model", model_id)
            grp.insert(2, "feature_set", fs_name)
            grouped_frames.append(grp)
        if grouped_frames:
            merge_csv(
                pd.concat(grouped_frames, ignore_index=True),
                contrast_dir / "event_contrast_feature_importance_by_group.csv",
                ["model", "feature_set", "sensor_group"],
            )

    if not audit.empty:
        write_csv(audit, contrast_dir / "event_contrast_windows.csv")

    summary_doc: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "contrast": contrast.name,
        "status": status,
        "features_fingerprint": fp,
        "models": list(models),
        "group_col": group_col,
        "min_quality": min_quality,
        "seed": seed,
        "permutation_n_repeats": permutation_n_repeats,
        "stability_top_k": stability_top_k,
        "n_windows": n_windows,
        "n_groups": n_groups,
        "n_splits": n_splits,
        "label_counts": label_counts,
        "skipped": contrast_skipped,
    }
    (contrast_dir / f"event_contrast_{contrast.name}_summary.json").write_text(
        json.dumps(summary_doc, indent=2, default=_json_default), encoding="utf-8"
    )


def _load_reusable_event_summary(
    event_output_dir: Path,
    *,
    features_path: Path,
    contrasts: tuple[EventContrast, ...],
    models: tuple[str, ...],
    group_col: str,
    min_quality: str | None,
    seed: int,
    permutation_n_repeats: int,
    stability_top_k: int,
) -> dict[str, Any] | None:
    """Return a saved event-contrast summary when it matches this request."""
    summary_path = event_output_dir / "event_contrast_summary.json"
    metrics_path = event_output_dir / "event_contrast_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read saved event-contrast summary %s: %s", _display_path(summary_path), exc)
        return None

    if summary.get("features_fingerprint") != features_fingerprint(features_path):
        return None
    if summary.get("models") != list(models):
        return None
    if summary.get("group_col") != group_col:
        return None
    if summary.get("min_quality") != min_quality:
        return None
    if summary.get("seed") != seed:
        return None
    if summary.get("permutation_n_repeats") != permutation_n_repeats:
        return None
    if summary.get("stability_top_k") != stability_top_k:
        return None

    saved_contrasts = [c.get("name") for c in summary.get("contrasts", [])]
    if saved_contrasts != [c.name for c in contrasts]:
        return None
    return summary


def run_event_contrast_evaluation(
    features_path: Path | str,
    *,
    output_dir: Path | str,
    contrasts: tuple[EventContrast, ...] | list[EventContrast] | None = None,
    models: tuple[str, ...] | list[str] | None = None,
    group_col: str = "recording_name",
    min_quality: str | None = "marginal",
    seed: int = 42,
    permutation_n_repeats: int = 5,
    stability_top_k: int = 20,
    no_plots: bool = False,
    force: bool = True,
) -> dict[str, Any]:
    """Run event contrast evaluation and write CSV/JSON artifacts."""
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    root_output_dir = Path(output_dir)
    event_output_dir = root_output_dir / "event_contrasts"
    event_output_dir.mkdir(parents=True, exist_ok=True)

    contrast_configs = tuple(contrasts or DEFAULT_EVENT_CONTRASTS)
    model_ids = resolve_evaluation_models(list(models or ["hist_gradient_boosting", "logistic_regression"]))

    if not force:
        reusable = _load_reusable_event_summary(
            event_output_dir,
            features_path=features_path,
            contrasts=contrast_configs,
            models=model_ids,
            group_col=group_col,
            min_quality=min_quality,
            seed=seed,
            permutation_n_repeats=permutation_n_repeats,
            stability_top_k=stability_top_k,
        )
        if reusable is not None:
            log.info("Reusing saved event-contrast artifacts; regenerating figures")
            if not no_plots:
                try:
                    from visualization.plot_eval_events import generate_event_contrast_figures
                    figure_paths = generate_event_contrast_figures(event_output_dir)
                    reusable.setdefault("artifacts", {})["figures"] = [
                        _display_path(path) for path in figure_paths
                    ]
                except Exception as exc:
                    log.warning("Event contrast figure regeneration failed: %s", exc)
            reusable["reused_existing_artifacts"] = True
            (event_output_dir / "event_contrast_summary.json").write_text(
                json.dumps(reusable, indent=2, default=_json_default),
                encoding="utf-8",
            )
            return reusable

    log.info("Loading features from %s", _display_path(features_path))
    source_df = _apply_quality_filter(read_csv(features_path), min_quality)
    if source_df.empty:
        raise ValueError("No rows remain after quality filtering")

    fp = features_fingerprint(features_path)

    metrics_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    perm_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []

    n_contrasts = len(contrast_configs)
    for ci, contrast in enumerate(contrast_configs, start=1):
        log.info("── Event contrast %d/%d: %s ──", ci, n_contrasts, contrast.name)
        contrast_dir = event_output_dir / contrast.name
        contrast_dir.mkdir(parents=True, exist_ok=True)

        if not force:
            reusable_contrast = _load_reusable_contrast_summary(
                contrast_dir,
                contrast,
                fp=fp,
                models=model_ids,
                group_col=group_col,
                min_quality=min_quality,
                seed=seed,
                permutation_n_repeats=permutation_n_repeats,
                stability_top_k=stability_top_k,
            )
            if reusable_contrast is not None:
                log.info("  Reusing saved contrast artifacts for %s", contrast.name)
                _collect_contrast_artifacts(
                    contrast_dir,
                    reusable_contrast,
                    metrics_rows=metrics_rows,
                    audit_frames=audit_frames,
                    stability_frames=stability_frames,
                    perm_frames=perm_frames,
                    contribution_frames=contribution_frames,
                    skipped=skipped,
                )
                if not no_plots and reusable_contrast.get("status") == "ok":
                    _flush_contrast_figures(contrast_dir, event_output_dir)
                continue

        contrast_df, audit = build_event_contrast_table(source_df, contrast)

        contrast_skipped: list[dict[str, Any]] = []
        contrast_metrics_rows: list[dict[str, Any]] = []
        contrast_stability_frames: list[pd.DataFrame] = []
        contrast_perm_frames: list[pd.DataFrame] = []
        contrast_results: dict[tuple[str, str, str], dict[str, Any]] = {}
        n_windows = 0
        n_groups = 0
        n_splits_used = 0
        label_counts: dict[str, int] = {}

        _write_kwargs = dict(
            fp=fp, models=model_ids, group_col=group_col, min_quality=min_quality,
            seed=seed, permutation_n_repeats=permutation_n_repeats, stability_top_k=stability_top_k,
        )

        if contrast_df.empty:
            contrast_skipped.append({"contrast": contrast.name, "reason": "no_matching_windows"})
            _write_contrast_artifacts(
                contrast_dir, contrast, **_write_kwargs,
                status="skipped", n_windows=0, n_groups=0, n_splits=0,
                label_counts={}, contrast_skipped=contrast_skipped,
                contrast_metrics_rows=[], contrast_stability_frames=[],
                contrast_perm_frames=[], contrast_results={}, audit=audit,
            )
            skipped.extend(contrast_skipped)
            if not audit.empty:
                audit_frames.append(audit)
            continue

        label_counts = contrast_df["event_contrast_label"].value_counts().to_dict()
        if len(label_counts) < 2:
            contrast_skipped.append({
                "contrast": contrast.name,
                "reason": "only_one_class",
                "label_counts": label_counts,
            })
            _write_contrast_artifacts(
                contrast_dir, contrast, **_write_kwargs,
                status="skipped", n_windows=len(contrast_df), n_groups=0, n_splits=0,
                label_counts=label_counts, contrast_skipped=contrast_skipped,
                contrast_metrics_rows=[], contrast_stability_frames=[],
                contrast_perm_frames=[], contrast_results={}, audit=audit,
            )
            skipped.extend(contrast_skipped)
            if not audit.empty:
                audit_frames.append(audit)
            continue

        groups_arr = _ensure_recording_group(contrast_df, group_col).to_numpy()
        n_groups = int(len(np.unique(groups_arr)))
        n_splits_used = min(5, n_groups)
        if n_splits_used < 2:
            contrast_skipped.append({
                "contrast": contrast.name,
                "reason": "not_enough_groups",
                "n_groups": n_groups,
            })
            _write_contrast_artifacts(
                contrast_dir, contrast, **_write_kwargs,
                status="skipped", n_windows=len(contrast_df), n_groups=n_groups, n_splits=0,
                label_counts=label_counts, contrast_skipped=contrast_skipped,
                contrast_metrics_rows=[], contrast_stability_frames=[],
                contrast_perm_frames=[], contrast_results={}, audit=audit,
            )
            skipped.extend(contrast_skipped)
            if not audit.empty:
                audit_frames.append(audit)
            continue

        n_windows = len(contrast_df)
        active_sets = _active_feature_sets(contrast_df)
        y = (
            contrast_df["event_contrast_label"]
            .map({contrast.normal_label: 0, contrast.critical_label: 1})
            .to_numpy(dtype=int)
        )
        window_type_counts = (
            contrast_df["window_type"].value_counts().to_dict()
            if "window_type" in contrast_df.columns
            else {}
        )
        log.info(
            "  %d windows  normal=%s(%d)  critical=%s(%d)  groups=%d  splits=%d",
            n_windows,
            contrast.normal_label, label_counts.get(contrast.normal_label, 0),
            contrast.critical_label, label_counts.get(contrast.critical_label, 0),
            n_groups, n_splits_used,
        )

        n_fs = len(active_sets)
        n_models = len(model_ids)
        for fsi, (fs_name, feat_cols) in enumerate(active_sets.items(), start=1):
            X = contrast_df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            for mi, model_id in enumerate(model_ids, start=1):
                key = (contrast.name, fs_name, model_id)
                log.info(
                    "  [%d/%d fs, %d/%d model] %s / %s (%d features) ...",
                    fsi, n_fs, mi, n_models, fs_name, model_id, len(feat_cols),
                )
                model = _build_model(model_id, seed)
                try:
                    result = _cv_binary_evaluate(
                        X, y, groups_arr, model, n_splits=n_splits_used,
                    )
                except Exception as exc:
                    log.warning("Event contrast CV failed for %s/%s/%s: %s", *key, exc)
                    contrast_skipped.append({
                        "contrast": contrast.name,
                        "feature_set": fs_name,
                        "model": model_id,
                        "reason": f"cv_failed: {exc}",
                    })
                    continue

                contrast_results[key] = result
                log.info(
                    "    → accuracy=%.3f  balanced_acc=%.3f  macro_f1=%.3f  AP=%s  ROC-AUC=%s",
                    result["accuracy"],
                    result["balanced_accuracy"],
                    result["macro_f1"],
                    f"{result['average_precision']:.3f}" if np.isfinite(result["average_precision"]) else "n/a",
                    f"{result['roc_auc']:.3f}" if np.isfinite(result["roc_auc"]) else "n/a",
                )
                metrics_row: dict[str, Any] = {
                    "contrast": contrast.name,
                    "feature_set": fs_name,
                    "model": model_id,
                    "n_windows": int(n_windows),
                    "n_groups": n_groups,
                    "n_splits": n_splits_used,
                    "normal_label": contrast.normal_label,
                    "critical_label": contrast.critical_label,
                    "support_normal": int((y == 0).sum()),
                    "support_critical": int((y == 1).sum()),
                    "n_sliding": int(window_type_counts.get("sliding", 0)),
                    "n_event_aligned": int(window_type_counts.get("event_aligned", 0)),
                    "n_features": len(feat_cols),
                    "accuracy": round(float(result["accuracy"]), 4),
                    "balanced_accuracy": round(float(result["balanced_accuracy"]), 4),
                    "macro_f1": round(float(result["macro_f1"]), 4),
                    "average_precision": round(float(result["average_precision"]), 4)
                    if np.isfinite(result["average_precision"])
                    else np.nan,
                    "roc_auc": round(float(result["roc_auc"]), 4)
                    if np.isfinite(result["roc_auc"])
                    else np.nan,
                }
                contrast_metrics_rows.append(metrics_row)
                merge_csv(
                    pd.DataFrame([metrics_row]),
                    contrast_dir / "event_contrast_metrics.csv",
                    ["feature_set", "model"],
                )

                log.info("    feature stability (%d splits) ...", n_splits_used)
                try:
                    stability = summarize_feature_stability(
                        X, y, groups_arr, _build_model(model_id, seed), feat_cols,
                        n_splits=n_splits_used, top_k=stability_top_k, seed=seed,
                    )
                    if not stability.empty:
                        stability.insert(0, "contrast", contrast.name)
                        stability.insert(1, "feature_set", fs_name)
                        stability.insert(2, "model", model_id)
                        contrast_stability_frames.append(stability)
                        merge_csv(
                            stability,
                            contrast_dir / "event_contrast_feature_stability.csv",
                            ["feature_set", "model", "feature"],
                        )
                except Exception as exc:
                    log.warning("Feature stability failed for %s/%s/%s: %s", *key, exc)

                log.info(
                    "    permutation importance (%d splits, %d repeats) ...",
                    n_splits_used, permutation_n_repeats,
                )
                try:
                    perm_df = compute_permutation_importance_grouped(
                        X, y, groups_arr, _build_model(model_id, seed), feat_cols,
                        n_splits=n_splits_used, n_repeats=permutation_n_repeats,
                        scoring="f1_macro", seed=seed,
                    )
                    if not perm_df.empty:
                        perm_df.insert(0, "contrast", contrast.name)
                        perm_df.insert(1, "model", model_id)
                        perm_df.insert(2, "feature_set", fs_name)
                        contrast_perm_frames.append(perm_df)
                        merge_csv(
                            perm_df,
                            contrast_dir / "event_contrast_feature_importance.csv",
                            ["model", "feature_set", "feature"],
                        )
                        drop_cols = [c for c in ("contrast", "model", "feature_set") if c in perm_df.columns]
                        grp = aggregate_by_sensor_group(perm_df.drop(columns=drop_cols))
                        grp.insert(0, "contrast", contrast.name)
                        grp.insert(1, "model", model_id)
                        grp.insert(2, "feature_set", fs_name)
                        merge_csv(
                            grp,
                            contrast_dir / "event_contrast_feature_importance_by_group.csv",
                            ["model", "feature_set", "sensor_group"],
                        )
                except Exception as exc:
                    log.warning("Permutation importance failed for %s/%s/%s: %s", *key, exc)

        _write_contrast_artifacts(
            contrast_dir, contrast, **_write_kwargs,
            status="ok", n_windows=n_windows, n_groups=n_groups, n_splits=n_splits_used,
            label_counts=label_counts, contrast_skipped=contrast_skipped,
            contrast_metrics_rows=contrast_metrics_rows,
            contrast_stability_frames=contrast_stability_frames,
            contrast_perm_frames=contrast_perm_frames,
            contrast_results=contrast_results,
            audit=audit,
        )
        skipped.extend(contrast_skipped)
        if not audit.empty:
            audit_frames.append(audit)
        metrics_rows.extend(contrast_metrics_rows)
        stability_frames.extend(contrast_stability_frames)
        perm_frames.extend(contrast_perm_frames)
        cdf_path = contrast_dir / "event_contrast_imu_contribution.csv"
        if cdf_path.exists():
            try:
                cdf = read_csv(cdf_path)
                if not cdf.empty:
                    contribution_frames.append(cdf)
            except Exception as exc:
                log.warning("Could not load contribution CSV for %s: %s", contrast.name, exc)
        if not no_plots:
            _flush_contrast_figures(contrast_dir, event_output_dir)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = event_output_dir / "event_contrast_metrics.csv"
    merge_csv(metrics_df, metrics_path, ["contrast", "feature_set", "model"])

    contribution_df = (
        pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame()
    )
    contribution_path = event_output_dir / "event_contrast_imu_contribution.csv"
    merge_csv(contribution_df, contribution_path, ["contrast", "model", "better", "baseline", "metric"])

    stability_df = (
        pd.concat(stability_frames, ignore_index=True)
        if stability_frames
        else pd.DataFrame()
    )
    stability_path = event_output_dir / "event_contrast_feature_stability.csv"
    merge_csv(stability_df, stability_path, ["contrast", "feature_set", "model", "feature"])

    perm_df_all = (
        pd.concat(perm_frames, ignore_index=True) if perm_frames else pd.DataFrame()
    )
    perm_path = event_output_dir / "event_contrast_feature_importance.csv"
    merge_csv(perm_df_all, perm_path, ["contrast", "model", "feature_set", "feature"])

    if not perm_df_all.empty:
        grouped_perm_frames = []
        for (contrast_name, model_id, fs_name), sub in perm_df_all.groupby(
            ["contrast", "model", "feature_set"], sort=False
        ):
            grouped = aggregate_by_sensor_group(sub.drop(columns=["contrast", "model", "feature_set"]))
            grouped.insert(0, "contrast", contrast_name)
            grouped.insert(1, "model", model_id)
            grouped.insert(2, "feature_set", fs_name)
            grouped_perm_frames.append(grouped)
        grouped_perm_df = pd.concat(grouped_perm_frames, ignore_index=True) if grouped_perm_frames else pd.DataFrame()
        merge_csv(grouped_perm_df, event_output_dir / "event_contrast_feature_importance_by_group.csv", ["contrast", "model", "feature_set", "sensor_group"])

    audit_df = (
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    )
    audit_path = event_output_dir / "event_contrast_windows.csv"
    write_csv(audit_df, audit_path)

    summary_path = event_output_dir / "event_contrast_summary.json"
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("features_fingerprint") != fp:
                log.warning(
                    "Features file has changed since the last event-contrast run "
                    "(fingerprint mismatch). Merged results may be inconsistent."
                )
        except Exception:
            pass

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "features_path": str(_display_path(features_path)),
        "features_fingerprint": fp,
        "output_dir": str(_display_path(event_output_dir)),
        "contrasts": [c.__dict__ for c in contrast_configs],
        "models": list(model_ids),
        "group_col": group_col,
        "min_quality": min_quality,
        "seed": seed,
        "force": force,
        "permutation_n_repeats": permutation_n_repeats,
        "stability_top_k": stability_top_k,
        "n_source_windows": int(len(source_df)),
        "n_metric_rows": int(len(metrics_df)),
        "skipped": skipped,
        "artifacts": {
            "metrics": str(_display_path(metrics_path)),
            "imu_contribution": str(_display_path(contribution_path)),
            "feature_stability": str(_display_path(stability_path)),
            "windows": str(_display_path(audit_path)),
        },
    }
    figures_dir = event_output_dir / "figures"
    if not no_plots and figures_dir.exists():
        summary["artifacts"]["figures"] = [
            _display_path(p) for p in sorted(figures_dir.glob("*.png"))
        ]
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    log.info(
        "Event contrast evaluation complete: %d metric row(s) -> %s",
        len(metrics_df),
        _display_path(event_output_dir),
    )
    return summary
