"""Quantify the marginal value of the second IMU.

For a fixed (label_col, min_quality, model), compares the four feature sets
``{bike, rider, fused_no_cross, fused}`` on paired CV folds and reports:

  * fold-paired Δ accuracy / Δ macro-F1 (mean ± std)
  * one-sided Wilcoxon signed-rank p-value that ``Δ > 0``
  * per-class F1 deltas computed from the OOF per-class reports

The paired test is valid because :func:`evaluation.experiments._cv_evaluate`
re-uses the same :class:`sklearn.model_selection.GroupKFold` split index for
every feature set within a single :func:`run_evaluation` call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common.paths import project_relative_path, write_csv

log = logging.getLogger(__name__)

_FS_PAIRS: list[tuple[str, str]] = [
    ("fused", "bike"),
    ("fused", "rider"),
    ("fused", "fused_no_cross"),
    ("rider", "bike"),
]


def _wilcoxon_pvalue(deltas: np.ndarray) -> float | None:
    """One-sided Wilcoxon test that ``median(delta) > 0``.

    Returns ``None`` if all deltas are zero or no finite samples remain — the
    test is undefined in those degenerate cases.
    """
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0 or np.all(finite == 0):
        return None
    try:
        result = wilcoxon(finite, alternative="greater", zero_method="wilcox")
        return float(result.pvalue)
    except ValueError:
        return None


def compute_imu_contribution(
    all_results: dict[str, Any],
    *,
    feature_set_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a long-form table of paired feature-set comparisons.

    Parameters
    ----------
    all_results:
        Mapping ``{<fs>__<model>: result_dict}`` produced by
        :func:`evaluation.experiments._cv_evaluate`.  Each value must contain
        ``fold_accuracies`` and ``fold_f1s`` lists of equal length per model.
    feature_set_pairs:
        Pairs ``(better, baseline)`` to evaluate ``Δ = better − baseline``.
        Defaults to the canonical four pairs.
    """
    pairs = feature_set_pairs or _FS_PAIRS
    models = sorted({k.split("__", 1)[1] for k in all_results if "__" in k})

    rows: list[dict] = []
    for model in models:
        for better, baseline in pairs:
            key_b = f"{better}__{model}"
            key_a = f"{baseline}__{model}"
            if key_b not in all_results or key_a not in all_results:
                continue
            r_b = all_results[key_b]
            r_a = all_results[key_a]

            for metric, key in (("accuracy", "fold_accuracies"), ("macro_f1", "fold_f1s")):
                d_b = np.asarray(r_b.get(key, []), dtype=float)
                d_a = np.asarray(r_a.get(key, []), dtype=float)
                if d_b.size == 0 or d_a.size == 0 or d_b.size != d_a.size:
                    log.warning(
                        "Fold-count mismatch (%s vs %s) for %s — skipping",
                        key_b, key_a, metric,
                    )
                    continue
                deltas = d_b - d_a
                rows.append(
                    {
                        "model": model,
                        "better": better,
                        "baseline": baseline,
                        "metric": metric,
                        "n_folds": int(d_b.size),
                        "better_mean": round(float(d_b.mean()), 4),
                        "baseline_mean": round(float(d_a.mean()), 4),
                        "delta_mean": round(float(deltas.mean()), 4),
                        "delta_std": round(
                            float(deltas.std(ddof=1)) if d_b.size > 1 else 0.0, 4
                        ),
                        "delta_min": round(float(deltas.min()), 4),
                        "delta_max": round(float(deltas.max()), 4),
                        "wilcoxon_p_one_sided": _wilcoxon_pvalue(deltas),
                    }
                )
    return pd.DataFrame(rows)


def per_class_f1_deltas(
    all_results: dict[str, Any],
    classes: list[str],
    *,
    better: str,
    baseline: str,
    model: str,
) -> pd.DataFrame:
    """Return per-class F1 differences between two feature sets for one model.

    Empty DataFrame when either side is missing.  ``support`` is taken from
    the *better* side, since both sides see the same OOF rows under
    GroupKFold and supports therefore match.
    """
    key_b = f"{better}__{model}"
    key_a = f"{baseline}__{model}"
    if key_b not in all_results or key_a not in all_results:
        return pd.DataFrame()

    pc_b = all_results[key_b].get("per_class", {})
    pc_a = all_results[key_a].get("per_class", {})

    rows = []
    for cls in classes:
        f1_b = float(pc_b.get(cls, {}).get("f1-score", 0.0))
        f1_a = float(pc_a.get(cls, {}).get("f1-score", 0.0))
        support = int(pc_b.get(cls, {}).get("support", 0))
        rows.append(
            {
                "class": cls,
                f"f1_{better}": round(f1_b, 4),
                f"f1_{baseline}": round(f1_a, 4),
                "delta_f1": round(f1_b - f1_a, 4),
                "support": support,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("delta_f1", ascending=False)
        .reset_index(drop=True)
    )


def write_imu_contribution(
    all_results: dict[str, Any],
    classes: list[str],
    output_dir: Path,
) -> tuple[Path | None, list[Path]]:
    """Run all comparisons and write CSVs.

    Returns ``(summary_path, per_class_delta_paths)``.  Summary path is
    ``None`` when no comparable feature-set pairs were available.
    """
    contribution_df = compute_imu_contribution(all_results)
    if contribution_df.empty:
        log.info("IMU contribution: no comparable feature-set pairs available")
        return None, []

    summary_path = output_dir / "imu_contribution.csv"
    write_csv(contribution_df, summary_path)
    log.info(
        "IMU contribution → %s (%d rows)",
        project_relative_path(summary_path),
        len(contribution_df),
    )

    delta_paths: list[Path] = []
    pair_models = (
        contribution_df[["better", "baseline", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for better, baseline, model in pair_models:
        df = per_class_f1_deltas(
            all_results, classes, better=better, baseline=baseline, model=model
        )
        if df.empty:
            continue
        path = (
            output_dir
            / f"imu_contribution_per_class_{better}_vs_{baseline}__{model}.csv"
        )
        write_csv(df, path)
        delta_paths.append(path)

    return summary_path, delta_paths
