"""Permutation importance helpers for train models and build evaluation outputs from exported features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.paths import project_relative_path, write_csv

log = logging.getLogger(__name__)

# Feature-name prefix → sensor group label.  ``sporsa_`` and ``arduino_`` are
# legacy prefixes that may still appear in older feature CSVs; ``cross_`` is
# the dedicated cross-sensor block.
_SENSOR_GROUP_NAME: dict[str, str] = {
    "bike_": "bike",
    "sporsa_": "bike",
    "rider_": "rider",
    "arduino_": "rider",
    "cross_": "cross",
}


def _sensor_group(feature: str) -> str:
    """Return sensor group."""
    for prefix, name in _SENSOR_GROUP_NAME.items():
        if feature.startswith(prefix):
            return name
    return "other"


def compute_permutation_importance_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Any,
    feature_names: list[str],
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    scoring: str = "f1_macro",
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance grouped."""
    n_unique_groups = int(len(np.unique(groups)))
    n_splits = min(n_splits, n_unique_groups)
    if n_splits < 2:
        log.warning(
            "Only %d groups available — cannot compute grouped permutation importance",
            n_unique_groups,
        )
        return pd.DataFrame(
            columns=[
                "feature", "sensor_group",
                "perm_importance_mean", "perm_importance_std",
                "perm_importance_min", "perm_importance_max",
            ]
        )

    cv = GroupKFold(n_splits=n_splits)
    fold_means: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clone(model)),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        try:
            r = permutation_importance(
                pipe,
                X[val_idx],
                y[val_idx],
                n_repeats=n_repeats,
                random_state=seed + fold_idx,
                scoring=scoring,
                n_jobs=1,
            )
            fold_means.append(r.importances_mean)
        except Exception as exc:
            log.warning("Permutation importance failed on fold %d: %s", fold_idx, exc)

    if not fold_means:
        return pd.DataFrame(
            columns=[
                "feature", "sensor_group",
                "perm_importance_mean", "perm_importance_std",
                "perm_importance_min", "perm_importance_max",
            ]
        )

    stack = np.vstack(fold_means)
    means = stack.mean(axis=0)
    stds = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(means)
    mins = stack.min(axis=0)
    maxs = stack.max(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "sensor_group": [_sensor_group(f) for f in feature_names],
                "perm_importance_mean": means,
                "perm_importance_std": stds,
                "perm_importance_min": mins,
                "perm_importance_max": maxs,
            }
        )
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_by_sensor_group(perm_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by sensor group."""
    if perm_df.empty:
        return pd.DataFrame(
            columns=["sensor_group", "total_importance", "n_features", "mean_per_feature"]
        )
    grouped = (
        perm_df.groupby("sensor_group")["perm_importance_mean"]
        .agg(total_importance="sum", n_features="size", mean_per_feature="mean")
        .reset_index()
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["total_importance"] = grouped["total_importance"].round(6)
    grouped["mean_per_feature"] = grouped["mean_per_feature"].round(6)
    return grouped


def write_permutation_importance(
    perm_df: pd.DataFrame,
    output_dir: Path,
    *,
    config_name: str,
) -> tuple[Path, Path]:
    """Write per-feature CSV + sensor-group aggregate CSV."""
    feat_path = output_dir / f"permutation_importance_{config_name}.csv"
    write_csv(perm_df, feat_path)

    grouped = aggregate_by_sensor_group(perm_df)
    grp_path = output_dir / f"permutation_importance_by_group_{config_name}.csv"
    write_csv(grouped, grp_path)

    if not perm_df.empty:
        log.info(
            "Permutation importance (%s): top=%s; group totals: %s",
            config_name,
            perm_df.iloc[0]["feature"],
            grouped.set_index("sensor_group")["total_importance"].to_dict()
            if not grouped.empty
            else {},
        )
    log.debug(
        "Wrote %s, %s",
        project_relative_path(feat_path),
        project_relative_path(grp_path),
    )
    return feat_path, grp_path
