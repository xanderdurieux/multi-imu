"""Lag feature post-processing for sliding-window feature DataFrames.

For each window, adds the feature values from the N previous windows within
the same section.  This gives sequence classifiers context about what happened
before the current window — critical for events whose signature spans multiple
windows:

* Hard braking: the preceding window has a high acc_norm that decays in the
  current window.  A single-window view sees only the tail and misses the
  braking onset.
* Cornering: the preceding windows show a build-up in yaw_rate before the apex.

Only a focused subset of columns is lagged (signal means, stds, and all
temporal shape features) to avoid blowing up the feature matrix with 600+
near-redundant columns.

Usage
-----
    from analysis.features.lag_features import add_lag_features

    features_df = add_lag_features(features_df, n_lags=2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Substrings that identify lag-able feature columns.
# Temporal features are always included; signal aggregates are kept selective.
_LAG_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "_temporal_",            # all temporal shape features (new)
    "_acc_norm_mean",
    "_acc_norm_std",
    "_acc_norm_max",
    "_jerk_norm_mean",
    "_jerk_norm_std",
    "_jerk_norm_max",
    "_gyro_norm_mean",
    "_gyro_norm_std",
    "_gyro_norm_max",
    "_acc_vertical_mean",
    "_acc_hf_mean",
    "_energy_acc_mean",
    "cross_disagree_score_mean",
    "cross_disagree_score_std",
    "cross_acc_diff_mean",
    "cross_gyro_diff_mean",
    "_yaw_rate_mean_abs",
    "_yaw_rate_std",
    "_yaw_rate_p95_abs",
    "_yaw_rate_max_abs",
    "_pitch_mean",
    "_roll_mean",
    "_pitch_std",
    "_roll_std",
)

# Columns that are always excluded regardless of substring matches.
_EXCLUDE_EXACT: frozenset[str] = frozenset({
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "window_type",
    "window_n_samples_sporsa",
    "window_n_samples_arduino",
    "window_valid_ratio_sporsa",
    "window_valid_ratio_arduino",
})

# Substrings that always exclude a column (label/quality/metadata columns).
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "scenario_label",
    "_label",
    "label_",
    "calibration_quality",
    "synchronization_",
    "orientation_quality_",
    "gravity_residual_",
    "magnetometer_",
    "dropout_rate_",
    "quality_tier",
    "overall_quality_",
    "quality_bike",
    "quality_rider",
    "quality_align",
    "quality_calibration",
    "quality_cross",
    "sync_confidence",
    "yaw_conf_",
    "acc_saturation_",
    "gyro_saturation_",
)


def _is_lag_col(col: str) -> bool:
    """Return True if ``col`` should be included in lag features."""
    if col in _EXCLUDE_EXACT:
        return False
    if any(s in col for s in _EXCLUDE_SUBSTRINGS):
        return False
    return any(s in col for s in _LAG_INCLUDE_SUBSTRINGS)


def add_lag_features(
    df: pd.DataFrame,
    n_lags: int = 2,
    group_col: str = "section_id",
    sort_col: str = "window_start_ms",
    sliding_only: bool = True,
) -> pd.DataFrame:
    """Return ``df`` with lagged feature columns appended.

    For each selected feature column ``c``, adds ``c_lag1``, ``c_lag2``, …,
    ``c_lag{n_lags}`` containing values from the 1st, 2nd, … previous window
    *within the same section*.  The first ``n_lags`` windows of each section
    will have NaN for the respective lags.

    Parameters
    ----------
    df:
        Features DataFrame as produced by
        :func:`~analysis.features.pipeline.extract_features_for_section`.
    n_lags:
        Number of lag steps (default 2 → previous and 2-back window).
    group_col:
        Column identifying section boundaries (default ``"section_id"``).
    sort_col:
        Column for ordering windows within each section
        (default ``"window_start_ms"``).
    sliding_only:
        If True (default), lags are only computed for sliding windows
        (``window_type == "sliding"``).  Event-aligned windows are left
        without lag context because they are not sequential.

    Returns
    -------
    DataFrame with lag columns added.  The original row order is restored
    after sorting.
    """
    if n_lags <= 0:
        return df

    df = df.copy()

    has_group = group_col in df.columns
    has_sort = sort_col in df.columns

    if has_group and has_sort:
        original_index = df.index.copy()
        df = df.sort_values([group_col, sort_col]).reset_index(drop=True)
    else:
        original_index = None

    lag_cols = [c for c in df.columns if _is_lag_col(c)]
    if not lag_cols:
        return df

    if sliding_only and "window_type" in df.columns:
        mask = df["window_type"] == "sliding"
    else:
        mask = pd.Series(True, index=df.index)

    work = df.loc[mask, [group_col] + lag_cols].copy() if has_group else df.loc[mask, lag_cols].copy()

    new_cols: dict[str, np.ndarray] = {}
    for lag in range(1, n_lags + 1):
        if has_group:
            shifted = work.groupby(group_col, sort=False)[lag_cols].shift(lag)
        else:
            shifted = work[lag_cols].shift(lag)
        for col in lag_cols:
            new_cols[f"{col}_lag{lag}"] = np.full(len(df), np.nan)
            new_cols[f"{col}_lag{lag}"][mask.to_numpy()] = shifted[col].to_numpy()

    lag_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, lag_df], axis=1)

    if original_index is not None:
        df.index = original_index

    return df


def lag_column_names(df: pd.DataFrame, n_lags: int = 2) -> list[str]:
    """Return the names of lag columns that would be added by :func:`add_lag_features`.

    Useful for selecting or excluding lag features before fitting a model.
    """
    lag_cols = [c for c in df.columns if _is_lag_col(c)]
    return [f"{c}_lag{lag}" for c in lag_cols for lag in range(1, n_lags + 1)]
