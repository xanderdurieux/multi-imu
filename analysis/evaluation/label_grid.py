"""Label-grid evaluation: run scenario evaluation across label schemes and quality filters."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import features_fingerprint, merge_csv, project_relative_path, read_csv, write_csv
from evaluation.experiments import (
    resolve_evaluation_models,
    resolve_optional_evaluation_models,
    run_evaluation,
)
from features.label_config import default_label_config

log = logging.getLogger(__name__)

# Quality levels available for filtering.
DEFAULT_QUALITIES: tuple[str, ...] = ("poor", "marginal", "good")

# Multiclass label columns that classify which activity/state is happening.
MULTICLASS_LABEL_COLS: tuple[str, ...] = (
    "scenario_label_fine",
    "scenario_label_activity",
    "scenario_label_coarse",
    "scenario_label_safety",
)


def _binary_label_cols() -> tuple[str, ...]:
    """Return all configured binary (set-based) label schemes from the default config.

    Adding a new entry to the ``set_based_binary_schemes`` section in
    ``labels.default.json`` automatically includes it in label-grid runs.
    """
    return default_label_config().set_based_scheme_names


def all_label_cols() -> tuple[str, ...]:
    """Return the full default set evaluated in a label-grid run."""
    return (*MULTICLASS_LABEL_COLS, *_binary_label_cols())


def _is_binary_col(label_col: str) -> bool:
    return label_col in _binary_label_cols()


def resolve_label_cols(specs: list[str]) -> list[str]:
    """Expand a list of label-column specs into concrete column names.

    Each entry can be:
    - ``"auto"``        → all configured label columns
    - ``"multiclass"``  → multiclass schemes only
    - ``"binary"``      → set-based binary schemes only
    - a concrete column name (``"scenario_label_activity"``, etc.)

    Duplicates are removed while preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(col: str) -> None:
        if col not in seen:
            seen.add(col)
            out.append(col)

    for spec in specs:
        if spec == "auto":
            for col in all_label_cols():
                _add(col)
        elif spec == "multiclass":
            for col in MULTICLASS_LABEL_COLS:
                _add(col)
        elif spec == "binary":
            for col in _binary_label_cols():
                _add(col)
        else:
            _add(spec)

    return out


def _run_name(label_col: str, quality: str) -> str:
    """Return the subdirectory name for one label × quality run."""
    prefix = "binary__" if _is_binary_col(label_col) else ""
    return f"{prefix}{label_col}__q-{quality}"


def _display_path(path: Path | str) -> str:
    """Return a readable project-relative path when possible."""
    try:
        return project_relative_path(path)
    except ValueError:
        return str(Path(path))


def _safe_plot_token(value: object) -> str:
    """Return a compact filename token for a label-grid value."""
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace("|", "-")
    )


def _log_feature_table_overview(features_path: Path) -> None:
    """Log high-level feature table context for a label-grid run."""
    log.info("Loading label-grid features from %s", _display_path(features_path))
    df = read_csv(features_path)
    log.info("Loaded %d rows, %d columns for label-grid evaluation", len(df), len(df.columns))

    if "session" in df.columns:
        log.info("  sessions=%d: %s", df["session"].nunique(), sorted(df["session"].dropna().astype(str).unique()))
    if "recording_name" in df.columns:
        log.info("  recordings=%d", df["recording_name"].nunique())
    if "overall_quality_label" in df.columns:
        log.info("  quality counts: %s", df["overall_quality_label"].value_counts(dropna=False).to_dict())
    if "window_type" in df.columns:
        log.info("  window types: %s", df["window_type"].value_counts(dropna=False).to_dict())


def _collect_per_class_imu(
    run_out: Path,
    label_col: str,
    quality: str,
) -> list[pd.DataFrame]:
    """Collect per-class IMU contribution CSVs from a single run directory."""
    frames: list[pd.DataFrame] = []
    consolidated = run_out / "imu_contribution_per_class.csv"
    if consolidated.exists():
        try:
            df = read_csv(consolidated)
            df.insert(0, "label_col", label_col)
            df.insert(1, "min_quality", quality)
            return [df]
        except Exception as exc:
            log.warning("Could not read per-class IMU CSV %s: %s", consolidated.name, exc)

    for path in run_out.glob("imu_contribution_per_class_*_vs_*__*.csv"):
        try:
            df = read_csv(path)
            stem = path.stem.removeprefix("imu_contribution_per_class_")
            pair, _, model = stem.partition("__")
            better, _, baseline = pair.partition("_vs_")
            df.insert(0, "label_col", label_col)
            df.insert(1, "min_quality", quality)
            if "better" not in df.columns:
                df.insert(2, "better", better)
            if "baseline" not in df.columns:
                df.insert(3, "baseline", baseline)
            if "model" not in df.columns:
                df.insert(4, "model", model)
            frames.append(df)
        except Exception as exc:
            log.warning("Could not read per-class IMU CSV %s: %s", path.name, exc)
    return frames


def _collect_model_artifact_csv(
    run_out: Path,
    filename: str,
    *,
    label_col: str,
    quality: str,
) -> list[pd.DataFrame]:
    """Collect a per-model artifact CSV and add label-grid context."""
    frames: list[pd.DataFrame] = []
    for path in sorted(run_out.glob(f"*/{filename}")):
        model = path.parent.name
        try:
            df = read_csv(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", _display_path(path), exc)
            continue
        if df.empty:
            continue
        df.insert(0, "label_col", label_col)
        df.insert(1, "min_quality", quality)
        df.insert(2, "is_binary", _is_binary_col(label_col))
        df.insert(3, "model", model)
        frames.append(df)
    return frames


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
    save_trained_models: bool = False,
) -> dict[str, Any]:
    """Run scenario evaluation for each (label_col, quality) combination.

    Each combination gets its own subdirectory under ``output_dir``. Aggregated
    metrics, IMU contribution tables, and a JSON summary are written to the
    top-level directory. Report-stage plotting renders summary figures from
    these tables.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    label_cols = list(label_cols) if label_cols else list(all_label_cols())
    qualities = list(qualities) if qualities else list(DEFAULT_QUALITIES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_models = resolve_evaluation_models(
        list(evaluation_models) if evaluation_models is not None else ["auto"]
    )
    perm_models = resolve_optional_evaluation_models(permutation_models)
    total_runs = len(label_cols) * len(qualities)

    _log_feature_table_overview(features_path)
    log.info(
        "Label-grid configuration: labels=%s qualities=%s primary_quality=%s "
        "models=%s permutation_models=%s exclude_non_riding=%s plots=%s total_runs=%d",
        label_cols,
        qualities,
        primary_quality,
        list(cv_models),
        list(perm_models),
        exclude_non_riding,
        not no_plots,
        total_runs,
    )

    if primary_quality not in qualities:
        log.warning(
            "primary_quality=%r not in qualities=%s — no permutation importance "
            "will be computed",
            primary_quality, qualities,
        )

    grid_metrics: list[pd.DataFrame] = []
    grid_imu: list[pd.DataFrame] = []
    grid_imu_per_class: list[pd.DataFrame] = []
    grid_feature_importance: list[pd.DataFrame] = []
    grid_feature_importance_by_group: list[pd.DataFrame] = []
    grid_permutation_importance: list[pd.DataFrame] = []
    grid_permutation_importance_by_group: list[pd.DataFrame] = []
    runs_meta: list[dict[str, Any]] = []

    for label_idx, label_col in enumerate(label_cols, start=1):
        log.info("── Label-grid target %d/%d: %s ──", label_idx, len(label_cols), label_col)
        for quality_idx, quality in enumerate(qualities, start=1):
            run_name = _run_name(label_col, quality)
            run_out = output_dir / run_name
            run_idx = (label_idx - 1) * len(qualities) + quality_idx
            compute_perm = quality == primary_quality and bool(perm_models)
            run_summary: dict[str, Any] | None = None
            log.info(
                "  [%d/%d run, quality %d/%d] %s  permutation=%s -> %s",
                run_idx,
                total_runs,
                quality_idx,
                len(qualities),
                run_name,
                compute_perm,
                _display_path(run_out),
            )
            try:
                run_summary = run_evaluation(
                    features_path,
                    output_dir=run_out,
                    label_col=label_col,
                    seed=seed,
                    min_quality=quality,
                    exclude_non_riding=exclude_non_riding,
                    no_plots=no_plots,
                    compute_permutation_importance=compute_perm,
                    evaluation_models=cv_models,
                    permutation_models=perm_models,
                    save_trained_models=save_trained_models,
                )
            except (FileNotFoundError, ValueError) as exc:
                log.warning(
                    "  label-grid run %s skipped (%d/%d): %s",
                    run_name,
                    run_idx,
                    total_runs,
                    exc,
                )
                runs_meta.append(
                    {
                        "label_col": label_col,
                        "min_quality": quality,
                        "is_binary": _is_binary_col(label_col),
                        "ok": False,
                        "error": str(exc),
                        "output_dir": str(project_relative_path(run_out)),
                    }
                )
                continue
            except Exception as exc:
                log.error(
                    "  label-grid run %s failed (%d/%d): %s",
                    run_name,
                    run_idx,
                    total_runs,
                    exc,
                )
                runs_meta.append(
                    {
                        "label_col": label_col,
                        "min_quality": quality,
                        "is_binary": _is_binary_col(label_col),
                        "ok": False,
                        "error": str(exc),
                        "output_dir": str(project_relative_path(run_out)),
                    }
                )
                continue

            log.info(
                "  completed %s: windows=%d classes=%d class_labels=%s",
                run_name,
                int(run_summary["n_windows"]),
                int(run_summary["n_classes"]),
                run_summary["classes"],
            )

            runs_meta.append(
                {
                    "label_col": label_col,
                    "min_quality": quality,
                    "is_binary": _is_binary_col(label_col),
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
                log.info("  collected metrics for %s: %d rows", run_name, len(mdf))
                mdf.insert(0, "label_col", label_col)
                mdf.insert(1, "min_quality", quality)
                mdf.insert(2, "is_binary", _is_binary_col(label_col))
                mdf.insert(3, "n_windows", int(run_summary["n_windows"]))
                mdf.insert(4, "n_classes", int(run_summary["n_classes"]))
                grid_metrics.append(mdf)

            imu_path = run_out / "imu_contribution.csv"
            if imu_path.exists():
                idf = read_csv(imu_path)
                log.info("  collected IMU contribution for %s: %d rows", run_name, len(idf))
                idf.insert(0, "label_col", label_col)
                idf.insert(1, "min_quality", quality)
                grid_imu.append(idf)
            else:
                log.debug("  no IMU contribution table for %s", run_name)

            per_class_frames = _collect_per_class_imu(run_out, label_col, quality)
            if per_class_frames:
                grid_imu_per_class.extend(per_class_frames)
                log.debug(
                    "  collected %d per-class IMU frames for %s",
                    len(per_class_frames),
                    run_name,
                )

            artifact_specs = (
                ("feature_importance.csv", grid_feature_importance),
                ("feature_importance_by_group.csv", grid_feature_importance_by_group),
                ("permutation_importance.csv", grid_permutation_importance),
                ("permutation_importance_by_group.csv", grid_permutation_importance_by_group),
            )
            for filename, sink in artifact_specs:
                frames = _collect_model_artifact_csv(
                    run_out,
                    filename,
                    label_col=label_col,
                    quality=quality,
                )
                if frames:
                    sink.extend(frames)
                    log.debug("  collected %d %s frame(s) for %s", len(frames), filename, run_name)

    grid_metrics_df = (
        pd.concat(grid_metrics, ignore_index=True) if grid_metrics else pd.DataFrame()
    )
    grid_imu_df = (
        pd.concat(grid_imu, ignore_index=True) if grid_imu else pd.DataFrame()
    )
    grid_imu_per_class_df = (
        pd.concat(grid_imu_per_class, ignore_index=True) if grid_imu_per_class else pd.DataFrame()
    )
    grid_feature_importance_df = (
        pd.concat(grid_feature_importance, ignore_index=True)
        if grid_feature_importance
        else pd.DataFrame()
    )
    grid_feature_importance_by_group_df = (
        pd.concat(grid_feature_importance_by_group, ignore_index=True)
        if grid_feature_importance_by_group
        else pd.DataFrame()
    )
    grid_permutation_importance_df = (
        pd.concat(grid_permutation_importance, ignore_index=True)
        if grid_permutation_importance
        else pd.DataFrame()
    )
    grid_permutation_importance_by_group_df = (
        pd.concat(grid_permutation_importance_by_group, ignore_index=True)
        if grid_permutation_importance_by_group
        else pd.DataFrame()
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

    if not grid_imu_per_class_df.empty:
        imu_pc_path = output_dir / "label_grid_imu_contribution_per_class.csv"
        merge_csv(
            grid_imu_per_class_df,
            imu_pc_path,
            ["label_col", "min_quality", "class", "better", "baseline", "model"],
        )
        log.info(
            "Wrote label-grid per-class IMU contribution → %s (%d rows)",
            project_relative_path(imu_pc_path),
            len(grid_imu_per_class_df),
        )

    if not grid_feature_importance_df.empty:
        fi_path = output_dir / "label_grid_feature_importance.csv"
        merge_csv(
            grid_feature_importance_df,
            fi_path,
            ["label_col", "min_quality", "model", "feature_set", "feature"],
        )
        log.info(
            "Wrote label-grid feature importance → %s (%d rows)",
            project_relative_path(fi_path),
            len(grid_feature_importance_df),
        )

    if not grid_feature_importance_by_group_df.empty:
        fig_path = output_dir / "label_grid_feature_importance_by_group.csv"
        merge_csv(
            grid_feature_importance_by_group_df,
            fig_path,
            ["label_col", "min_quality", "model", "feature_set", "sensor_group"],
        )
        log.info(
            "Wrote label-grid feature importance by group → %s (%d rows)",
            project_relative_path(fig_path),
            len(grid_feature_importance_by_group_df),
        )

    if not grid_permutation_importance_df.empty:
        perm_path = output_dir / "label_grid_permutation_importance.csv"
        merge_csv(
            grid_permutation_importance_df,
            perm_path,
            ["label_col", "min_quality", "model", "feature_set", "feature"],
        )
        log.info(
            "Wrote label-grid permutation importance → %s (%d rows)",
            project_relative_path(perm_path),
            len(grid_permutation_importance_df),
        )

    if not grid_permutation_importance_by_group_df.empty:
        perm_grp_path = output_dir / "label_grid_permutation_importance_by_group.csv"
        merge_csv(
            grid_permutation_importance_by_group_df,
            perm_grp_path,
            ["label_col", "min_quality", "model", "feature_set", "sensor_group"],
        )
        log.info(
            "Wrote label-grid permutation importance by group → %s (%d rows)",
            project_relative_path(perm_grp_path),
            len(grid_permutation_importance_by_group_df),
        )

    # ------------------------------------------------------------------
    # Summary figures
    # ------------------------------------------------------------------
    if not no_plots and not grid_metrics_df.empty:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        try:
            from visualization.plot_eval_scenario import (
                plot_label_grid_heatmap,
                plot_label_grid_quality_grid,
            )
            for metric in ("macro_f1", "accuracy"):
                for fs in ("fused", "fused_no_cross"):
                    plot_label_grid_heatmap(
                        grid_metrics_df,
                        figures_dir / f"label_grid_heatmap_{metric}_{fs}.png",
                        metric=metric,
                        feature_set=fs,
                    )
                plot_label_grid_quality_grid(
                    grid_metrics_df,
                    figures_dir / f"label_grid_quality_grid_{metric}.png",
                    metric=metric,
                )
        except Exception as exc:
            log.warning("Label-grid summary figures failed: %s", exc)

        if not grid_imu_df.empty:
            try:
                from visualization._eval_common import plot_imu_contribution
                primary_imu = grid_imu_df[grid_imu_df["min_quality"] == primary_quality]
                if not primary_imu.empty:
                    for metric in ("macro_f1", "accuracy"):
                        plot_imu_contribution(
                            primary_imu,
                            figures_dir / f"label_grid_imu_contribution_{metric}.png",
                            metric=metric,
                            title=f"IMU contribution — label grid (q={primary_quality})",
                        )
            except Exception as exc:
                log.warning("Label-grid IMU contribution figure failed: %s", exc)

        if (
            not grid_feature_importance_df.empty
            or not grid_feature_importance_by_group_df.empty
            or not grid_permutation_importance_df.empty
            or not grid_permutation_importance_by_group_df.empty
        ):
            try:
                from visualization._eval_common import (
                    plot_permutation_importance,
                    plot_sensor_group_contribution,
                )
                from visualization.plot_eval_scenario import plot_feature_importance

                fi_figures_dir = figures_dir / "feature_importance"
                fi_figures_dir.mkdir(exist_ok=True)

                if not grid_feature_importance_df.empty:
                    for (label, quality, model, fs), sub in grid_feature_importance_df.groupby(
                        ["label_col", "min_quality", "model", "feature_set"],
                        sort=False,
                    ):
                        stem = "__".join(
                            _safe_plot_token(v) for v in (label, f"q-{quality}", model, fs)
                        )
                        plot_feature_importance(
                            sub.drop(
                                columns=[
                                    "label_col",
                                    "min_quality",
                                    "is_binary",
                                    "model",
                                    "feature_set",
                                ],
                                errors="ignore",
                            ),
                            fi_figures_dir / f"label_grid_feature_importance__{stem}.png",
                            title=f"Feature importance — {label} / q={quality} / {model} / {fs}",
                        )

                if not grid_feature_importance_by_group_df.empty:
                    for (label, quality, model, fs), sub in grid_feature_importance_by_group_df.groupby(
                        ["label_col", "min_quality", "model", "feature_set"],
                        sort=False,
                    ):
                        stem = "__".join(
                            _safe_plot_token(v) for v in (label, f"q-{quality}", model, fs)
                        )
                        plot_sensor_group_contribution(
                            sub.drop(
                                columns=[
                                    "label_col",
                                    "min_quality",
                                    "is_binary",
                                    "model",
                                    "feature_set",
                                ],
                                errors="ignore",
                            ),
                            fi_figures_dir / f"label_grid_feature_importance_by_group__{stem}.png",
                            title=(
                                "Feature importance by sensor group — "
                                f"{label} / q={quality} / {model} / {fs}"
                            ),
                        )

                perm_figures_dir = figures_dir / "permutation_importance"
                perm_figures_dir.mkdir(exist_ok=True)

                if not grid_permutation_importance_df.empty:
                    for (label, quality, model, fs), sub in grid_permutation_importance_df.groupby(
                        ["label_col", "min_quality", "model", "feature_set"],
                        sort=False,
                    ):
                        stem = "__".join(
                            _safe_plot_token(v) for v in (label, f"q-{quality}", model, fs)
                        )
                        plot_permutation_importance(
                            sub.sort_values("perm_importance_mean", ascending=False).drop(
                                columns=[
                                    "label_col",
                                    "min_quality",
                                    "is_binary",
                                    "model",
                                    "feature_set",
                                ],
                                errors="ignore",
                            ),
                            perm_figures_dir / f"label_grid_permutation_importance__{stem}.png",
                            title=(
                                "Permutation importance — "
                                f"{label} / q={quality} / {model} / {fs}"
                            ),
                        )

                if not grid_permutation_importance_by_group_df.empty:
                    for (label, quality, model, fs), sub in grid_permutation_importance_by_group_df.groupby(
                        ["label_col", "min_quality", "model", "feature_set"],
                        sort=False,
                    ):
                        stem = "__".join(
                            _safe_plot_token(v) for v in (label, f"q-{quality}", model, fs)
                        )
                        plot_sensor_group_contribution(
                            sub.drop(
                                columns=[
                                    "label_col",
                                    "min_quality",
                                    "is_binary",
                                    "model",
                                    "feature_set",
                                ],
                                errors="ignore",
                            ),
                            perm_figures_dir / f"label_grid_permutation_importance_by_group__{stem}.png",
                            title=(
                                "Permutation importance by sensor group — "
                                f"{label} / q={quality} / {model} / {fs}"
                            ),
                        )
            except Exception as exc:
                log.warning("Label-grid feature-importance figures failed: %s", exc)

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
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
    log.info(
        "Wrote label-grid summary → %s (ok=%d skipped_or_failed=%d)",
        project_relative_path(summary_path),
        int(summary["n_runs_ok"]),
        int(summary["n_runs"] - summary["n_runs_ok"]),
    )

    return summary
