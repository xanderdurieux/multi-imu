"""Sweep helpers for train models and build evaluation outputs from exported features."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import project_relative_path, read_csv, write_csv
from evaluation.experiments import resolve_evaluation_models, run_evaluation

log = logging.getLogger(__name__)

DEFAULT_QUALITIES: tuple[str, ...] = ("poor", "marginal", "good")
DEFAULT_LABEL_COLS: tuple[str, ...] = (
    "scenario_label",
    "scenario_label_activity",
    "scenario_label_coarse",
    "scenario_label_binary",
    "scenario_label_riding",
    "scenario_label_cornering",
    "scenario_label_head_motion",
)


def run_evaluation_sweep(
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
    """Run evaluation across label and quality settings."""
    label_cols = list(label_cols) if label_cols else list(DEFAULT_LABEL_COLS)
    qualities = list(qualities) if qualities else list(DEFAULT_QUALITIES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    cv_models = resolve_evaluation_models(
        list(evaluation_models) if evaluation_models is not None else ["auto"]
    )
    perm_models = resolve_evaluation_models(permutation_models)

    if primary_quality not in qualities:
        # Honour the user's choice but warn — without a primary run, no
        # permutation importance is produced.
        log.warning(
            "primary_quality=%r not in qualities=%s — no permutation importance "
            "will be computed",
            primary_quality, qualities,
        )

    sweep_metrics: list[pd.DataFrame] = []
    sweep_imu: list[pd.DataFrame] = []
    runs_meta: list[dict[str, Any]] = []

    for label_col in label_cols:
        for quality in qualities:
            run_name = f"{label_col}__q-{quality}"
            run_out = output_dir / run_name
            log.info("================== sweep run: %s ==================", run_name)
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
                # FileNotFoundError comes from a missing features file (fatal
                # for the whole sweep); ValueError typically means the label
                # scheme produced no rows under this quality filter — that
                # combination is just absent from the aggregate, not fatal.
                log.warning("sweep run %s skipped: %s", run_name, exc)
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
                log.error("sweep run %s failed: %s", run_name, exc)
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
                sweep_metrics.append(mdf)

            imu_path = run_out / "imu_contribution.csv"
            if imu_path.exists():
                idf = read_csv(imu_path)
                idf.insert(0, "label_col", label_col)
                idf.insert(1, "min_quality", quality)
                sweep_imu.append(idf)

    sweep_metrics_df = (
        pd.concat(sweep_metrics, ignore_index=True)
        if sweep_metrics
        else pd.DataFrame()
    )
    sweep_imu_df = (
        pd.concat(sweep_imu, ignore_index=True) if sweep_imu else pd.DataFrame()
    )

    if not sweep_metrics_df.empty:
        sweep_path = output_dir / "evaluation_sweep.csv"
        write_csv(sweep_metrics_df, sweep_path)
        log.info(
            "Wrote sweep metrics → %s (%d rows)",
            project_relative_path(sweep_path),
            len(sweep_metrics_df),
        )

    if not sweep_imu_df.empty:
        imu_path = output_dir / "sweep_imu_contribution.csv"
        write_csv(sweep_imu_df, imu_path)
        log.info(
            "Wrote sweep IMU contribution → %s (%d rows)",
            project_relative_path(imu_path),
            len(sweep_imu_df),
        )

    if not no_plots and not sweep_metrics_df.empty:
        try:
            from evaluation.plots import plot_sweep_heatmap, plot_sweep_label_quality_grid
            plot_sweep_heatmap(
                sweep_metrics_df,
                output_dir / "figures" / "sweep_heatmap_macro_f1.png",
                metric="macro_f1",
            )
            plot_sweep_heatmap(
                sweep_metrics_df,
                output_dir / "figures" / "sweep_heatmap_accuracy.png",
                metric="accuracy",
            )
            plot_sweep_label_quality_grid(
                sweep_metrics_df,
                output_dir / "figures" / "sweep_grid_macro_f1.png",
                metric="macro_f1",
            )
        except Exception as exc:
            log.warning("Sweep figure generation failed: %s", exc)

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote sweep summary → %s", project_relative_path(summary_path))

    return summary
