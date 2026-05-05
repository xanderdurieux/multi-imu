"""Label-grid evaluation: run scenario evaluation across label schemes and quality filters."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import features_fingerprint, merge_csv, project_relative_path, read_csv, write_csv
from evaluation.experiments import resolve_evaluation_models, run_evaluation

log = logging.getLogger(__name__)

# Quality levels available for filtering.
DEFAULT_QUALITIES: tuple[str, ...] = ("poor", "marginal", "good")

# Multi-class label columns: classify which activity is happening.
MULTICLASS_LABEL_COLS: tuple[str, ...] = (
    "scenario_label_activity",
    "scenario_label_coarse",
)

# Binary detection label columns: detect presence/absence of a specific phenomenon.
BINARY_LABEL_COLS: tuple[str, ...] = (
    "scenario_label_binary",
    "scenario_label_riding",
    "scenario_label_cornering",
    "scenario_label_head_motion",
)

# Full default set evaluated in a label-grid run.
ALL_LABEL_COLS: tuple[str, ...] = (*MULTICLASS_LABEL_COLS, *BINARY_LABEL_COLS)


def run_label_grid_evaluation(
    features_path: Path | str,
    *,
    output_dir: Path | str,
    label_cols: list[str] | tuple[str, ...] | None = None,
    qualities: list[str] | tuple[str, ...] | None = None,
    primary_quality: str = "marginal",
    seed: int = 42,
    exclude_non_riding: bool = False,
    no_plots: bool = False,
    evaluation_models: tuple[str, ...] | None = None,
    permutation_models: tuple[str, ...] = ("random_forest",),
) -> dict[str, Any]:
    """Run scenario evaluation for each (label_col, quality) combination.

    Each combination gets its own subdirectory under ``output_dir``.  Aggregated
    metrics and IMU contribution tables are written to the top-level directory
    alongside a JSON summary and label-grid heatmap figures.
    """
    features_path = Path(features_path)
    label_cols = list(label_cols) if label_cols else list(ALL_LABEL_COLS)
    qualities = list(qualities) if qualities else list(DEFAULT_QUALITIES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    cv_models = resolve_evaluation_models(
        list(evaluation_models) if evaluation_models is not None else ["auto"]
    )
    perm_models = resolve_evaluation_models(permutation_models)

    if primary_quality not in qualities:
        log.warning(
            "primary_quality=%r not in qualities=%s — no permutation importance "
            "will be computed",
            primary_quality, qualities,
        )

    grid_metrics: list[pd.DataFrame] = []
    grid_imu: list[pd.DataFrame] = []
    runs_meta: list[dict[str, Any]] = []

    for label_col in label_cols:
        for quality in qualities:
            run_name = f"{label_col}__q-{quality}"
            run_out = output_dir / run_name
            log.info("================== label-grid run: %s ==================", run_name)
            try:
                run_summary = run_evaluation(
                    features_path,
                    output_dir=run_out,
                    label_col=label_col,
                    seed=seed,
                    min_quality=quality,
                    exclude_non_riding=exclude_non_riding,
                    no_plots=no_plots,
                    compute_permutation_importance=(quality == primary_quality),
                    evaluation_models=cv_models,
                    permutation_models=perm_models,
                )
            except (FileNotFoundError, ValueError) as exc:
                log.warning("label-grid run %s skipped: %s", run_name, exc)
                runs_meta.append(
                    {
                        "label_col": label_col,
                        "min_quality": quality,
                        "ok": False,
                        "error": str(exc),
                        "output_dir": str(project_relative_path(run_out)),
                    }
                )
                continue
            except Exception as exc:
                log.error("label-grid run %s failed: %s", run_name, exc)
                runs_meta.append(
                    {
                        "label_col": label_col,
                        "min_quality": quality,
                        "ok": False,
                        "error": str(exc),
                        "output_dir": str(project_relative_path(run_out)),
                    }
                )
                continue

            runs_meta.append(
                {
                    "label_col": label_col,
                    "min_quality": quality,
                    "ok": True,
                    "n_windows": int(run_summary["n_windows"]),
                    "n_classes": int(run_summary["n_classes"]),
                    "classes": run_summary["classes"],
                    "output_dir": str(project_relative_path(run_out)),
                }
            )

            metrics_path = run_out / "metrics_table.csv"
            if metrics_path.exists():
                mdf = read_csv(metrics_path)
                mdf.insert(0, "label_col", label_col)
                mdf.insert(1, "min_quality", quality)
                mdf.insert(2, "n_windows", int(run_summary["n_windows"]))
                mdf.insert(3, "n_classes", int(run_summary["n_classes"]))
                grid_metrics.append(mdf)

            imu_path = run_out / "imu_contribution.csv"
            if imu_path.exists():
                idf = read_csv(imu_path)
                idf.insert(0, "label_col", label_col)
                idf.insert(1, "min_quality", quality)
                grid_imu.append(idf)

    grid_metrics_df = (
        pd.concat(grid_metrics, ignore_index=True) if grid_metrics else pd.DataFrame()
    )
    grid_imu_df = (
        pd.concat(grid_imu, ignore_index=True) if grid_imu else pd.DataFrame()
    )

    if not grid_metrics_df.empty:
        metrics_path = output_dir / "label_grid_metrics.csv"
        merge_csv(grid_metrics_df, metrics_path, ["label_col", "min_quality", "feature_set", "model"])
        log.info(
            "Wrote label-grid metrics → %s (%d rows)",
            project_relative_path(metrics_path),
            len(grid_metrics_df),
        )

    if not grid_imu_df.empty:
        imu_path = output_dir / "label_grid_imu_contribution.csv"
        merge_csv(grid_imu_df, imu_path, ["label_col", "min_quality", "metric", "better", "baseline", "model"])
        log.info(
            "Wrote label-grid IMU contribution → %s (%d rows)",
            project_relative_path(imu_path),
            len(grid_imu_df),
        )

    if not no_plots and not grid_metrics_df.empty:
        try:
            from visualization.plot_eval_scenario import (
                plot_label_grid_heatmap,
                plot_label_grid_quality_grid,
            )
            plot_label_grid_heatmap(
                grid_metrics_df,
                output_dir / "figures" / "label_grid_heatmap_macro_f1.png",
                metric="macro_f1",
            )
            plot_label_grid_heatmap(
                grid_metrics_df,
                output_dir / "figures" / "label_grid_heatmap_accuracy.png",
                metric="accuracy",
            )
            plot_label_grid_quality_grid(
                grid_metrics_df,
                output_dir / "figures" / "label_grid_quality_grid_macro_f1.png",
                metric="macro_f1",
            )
        except Exception as exc:
            log.warning("Label-grid figure generation failed: %s", exc)

    fp = features_fingerprint(features_path)
    summary_path = output_dir / "label_grid_summary.json"
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev.get("features_fingerprint") != fp:
                log.warning(
                    "Features file has changed since the last label-grid run "
                    "(fingerprint mismatch). Merged results may be inconsistent."
                )
        except Exception:
            pass

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_fingerprint": fp,
        "label_cols": label_cols,
        "qualities": qualities,
        "primary_quality": primary_quality,
        "evaluation_models": list(cv_models),
        "permutation_models": list(perm_models),
        "exclude_non_riding": exclude_non_riding,
        "n_runs": len(runs_meta),
        "n_runs_ok": int(sum(1 for r in runs_meta if r.get("ok"))),
        "runs": runs_meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote label-grid summary → %s", project_relative_path(summary_path))

    return summary
