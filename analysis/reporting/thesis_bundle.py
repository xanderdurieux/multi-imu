"""Thesis bundle helpers for build report tables, figures, and thesis bundles."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from common.paths import (
    data_root,
    evaluation_root,
    exports_root,
    project_relative_path,
    read_csv,
    resolve_evaluation_dir,
    sections_root,
    write_csv,
)

log = logging.getLogger(__name__)

_DPI = 200

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load json."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return {}


def _load_optional_csv(path: Path) -> pd.DataFrame:
    """Load optional csv."""
    if not path.exists():
        log.debug("Optional CSV not found: %s", path)
        return pd.DataFrame()
    try:
        return read_csv(path)
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return pd.DataFrame()


def _copy_dir(src: Path, dst: Path) -> int:
    """Copy all PNGs from src into dst; return count."""
    if not src.exists():
        log.warning("Source directory not found: %s", src)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.png")):
        shutil.copy2(f, dst / f.name)
        n += 1
    return n


def _copy_file(src: Path, dst: Path) -> bool:
    """Return copy file."""
    if not src.exists():
        log.warning("Source file not found: %s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_recording_section_counts(
    features_df: pd.DataFrame,
    sections_root_dir: Path,
) -> pd.DataFrame:
    """Return per-recording counts: sections, windows, labeled windows, duration."""
    rows = []

    recordings: list[str] = []
    if "section_id" in features_df.columns:
        # Derive recording names from section IDs (e.g. "2026-02-26_r1s1" → "2026-02-26_r1")
        for sid in features_df["section_id"].dropna().unique():
            # Strip the trailing "s<N>" part
            parts = str(sid).rsplit("s", 1)
            rec = parts[0] if len(parts) == 2 and parts[1].isdigit() else str(sid)
            if rec not in recordings:
                recordings.append(rec)

    for rec in sorted(recordings):
        rec_sections = [
            d for d in sorted(sections_root_dir.iterdir())
            if d.is_dir() and d.name.startswith(rec)
        ] if sections_root_dir.exists() else []

        n_sections = len(rec_sections)
        rec_mask = features_df["section_id"].astype(str).str.startswith(rec) if "section_id" in features_df.columns else pd.Series([], dtype=bool)
        rec_df = features_df[rec_mask]

        n_windows = len(rec_df)
        n_labeled = int((rec_df["scenario_label"].notna() & (rec_df["scenario_label"] != "unlabeled")).sum()) if "scenario_label" in rec_df.columns else 0

        # Duration: sum of (window_end_ms - window_start_ms) / 1000
        if "window_start_ms" in rec_df.columns and "window_end_ms" in rec_df.columns:
            duration_s = round(float((rec_df["window_end_ms"] - rec_df["window_start_ms"]).sum() / 1000.0), 1)
        else:
            duration_s = float("nan")

        rows.append(
            {
                "recording": rec,
                "n_sections": n_sections,
                "n_windows": n_windows,
                "n_windows_labeled": n_labeled,
                "duration_s": duration_s,
            }
        )

    return pd.DataFrame(rows)


def _build_dataset_composition(features_df: pd.DataFrame) -> pd.DataFrame:
    """Window counts per scenario class, with quality tier breakdown."""
    if "scenario_label" not in features_df.columns:
        return pd.DataFrame()

    labeled = features_df[
        features_df["scenario_label"].notna() & (features_df["scenario_label"] != "unlabeled")
    ]

    if labeled.empty:
        return pd.DataFrame()

    rows = []
    for cls in sorted(labeled["scenario_label"].unique()):
        cls_df = labeled[labeled["scenario_label"] == cls]
        row: dict = {"class": cls, "total": len(cls_df)}

        if "quality_tier" in cls_df.columns:
            for tier in ("good", "marginal", "poor"):
                row[f"quality_{tier}"] = int((cls_df["quality_tier"] == tier).sum())

        rows.append(row)

    total_row: dict = {"class": "TOTAL", "total": len(labeled)}
    if "quality_tier" in labeled.columns:
        for tier in ("good", "marginal", "poor"):
            total_row[f"quality_{tier}"] = int((labeled["quality_tier"] == tier).sum())
    rows.append(total_row)

    return pd.DataFrame(rows)


def _build_sync_summary(exports_dir: Path) -> pd.DataFrame:
    """Load sync_params.csv from exports and summarise per recording."""
    sync_path = exports_dir / "sync_params.csv"
    if not sync_path.exists():
        log.warning("sync_params.csv not found at %s", sync_path)
        return pd.DataFrame()

    sync_df = _load_optional_csv(sync_path)
    if sync_df.empty:
        return pd.DataFrame()

    cols_needed = {"recording", "method", "drift_ppm"}
    missing = cols_needed - set(sync_df.columns)
    if missing:
        log.warning("sync_params.csv missing columns: %s", missing)
        return sync_df

    summary = (
        sync_df.groupby("recording")
        .agg(
            method=("method", lambda s: s.mode().iloc[0] if not s.empty else "unknown"),
            drift_ppm_mean=("drift_ppm", "mean"),
            drift_ppm_std=("drift_ppm", "std"),
        )
        .reset_index()
    )
    summary["drift_ppm_mean"] = summary["drift_ppm_mean"].round(2)
    summary["drift_ppm_std"] = summary["drift_ppm_std"].round(2)
    return summary


# ---------------------------------------------------------------------------
# Figure re-rendering helpers
# ---------------------------------------------------------------------------

def _render_confusion_matrix(
    evaluation_dir: Path,
    best_fs: str,
    best_model: str,
    output_path: Path,
) -> bool:
    """Read the CSV confusion matrix for best_fs/best_model and render as PNG."""
    csv_candidates = [
        evaluation_dir / best_model / f"confusion_matrix_{best_fs}.csv",
        evaluation_dir / f"confusion_matrix_{best_fs}_{best_model}.csv",
    ]
    cm_csv = next((path for path in csv_candidates if path.exists()), None)
    if cm_csv is None:
        log.warning(
            "Confusion matrix CSV not found for feature set %s and model %s",
            best_fs,
            best_model,
        )
        png_candidates = [
            evaluation_dir / best_model / "figures" / f"confusion_matrix_{best_fs}.png",
            evaluation_dir / "figures" / f"confusion_matrix_{best_fs}_{best_model}.png",
        ]
        for cm_png in png_candidates:
            if _copy_file(cm_png, output_path):
                return True
        return False

    try:
        cm_df = pd.read_csv(cm_csv, index_col=0)
    except Exception as exc:
        log.warning("Failed to load confusion matrix CSV: %s", exc)
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        classes = cm_df.index.tolist()
        cm = cm_df.values.astype(float)

        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        n = len(classes)
        cell_size = max(0.75, 5.5 / n)
        fig, ax = plt.subplots(figsize=(n * cell_size + 1.5, n * cell_size))

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                raw = int(cm[i, j])
                cell_text = f"{val:.2f}\n({raw})"
                text_color = "white" if val > 0.55 else "black"
                fs = max(6, min(10, 11 - n))
                ax.text(j, i, cell_text, ha="center", va="center", fontsize=fs, color=text_color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        model_disp = best_model.replace("_", " ").title()
        fs_disp = best_fs.replace("_", " ").title()
        plt.colorbar(im, ax=ax, shrink=0.8, label="Recall")
        ax.set_title(f"Confusion Matrix — {model_disp} / {fs_disp} (OOF)", fontsize=11)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Rendered confusion matrix → %s", project_relative_path(output_path))
        return True
    except Exception as exc:
        log.warning("Failed to render confusion matrix: %s", exc)
        return False


def _render_feature_importances(
    evaluation_dir: Path,
    best_model: str,
    best_fs: str,
    output_path: Path,
    top_n: int = 20,
) -> bool:
    """Read feature_importance CSV and render top-N importance bar chart."""
    csv_candidates = [
        evaluation_dir / best_model / f"feature_importance_{best_fs}.csv",
        evaluation_dir / f"feature_importance_{best_model}_{best_fs}.csv",
    ]
    fi_csv = next((path for path in csv_candidates if path.exists()), None)
    if fi_csv is None:
        log.warning(
            "Feature importance CSV not found for model %s and feature set %s",
            best_model,
            best_fs,
        )
        png_candidates = [
            evaluation_dir / best_model / "figures" / f"feature_importance_{best_fs}.png",
            evaluation_dir / "figures" / f"feature_importance_{best_model}_{best_fs}.png",
        ]
        for fi_png in png_candidates:
            if _copy_file(fi_png, output_path):
                return True
        return False

    try:
        fi_df = _load_optional_csv(fi_csv)
    except Exception as exc:
        log.warning("Failed to load feature importance CSV: %s", exc)
        return False

    if fi_df.empty or "feature" not in fi_df.columns or "importance" not in fi_df.columns:
        log.warning("Feature importance CSV is empty or missing required columns")
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _GROUP_COLORS = {
            "bike_": "#3498db",
            "rider_": "#e74c3c",
            "cross_": "#2ecc71",
            "events_": "#f39c12",
            "spectral_": "#9b59b6",
            "orientation_": "#1abc9c",
        }

        def _group_color(feature: str) -> str:
            """Return group color."""
            for prefix, color in _GROUP_COLORS.items():
                if feature.startswith(prefix):
                    return color
            return "#95a5a6"

        fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)
        features = fi_df["feature"].tolist()[::-1]
        importances = fi_df["importance"].tolist()[::-1]
        colors = [_group_color(f) for f in features]

        fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(features) + 1.2)))
        ax.barh(features, importances, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Importance")
        model_disp = best_model.replace("_", " ").title()
        fs_disp = best_fs.replace("_", " ").title()
        ax.set_title(f"Top {top_n} Features — {model_disp} / {fs_disp}")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                color=color,
                label=prefix.rstrip("_").replace("_", " ").capitalize(),
            )
            for prefix, color in _GROUP_COLORS.items()
            if any(f.startswith(prefix) for f in features)
        ]
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Rendered feature importances → %s", project_relative_path(output_path))
        return True
    except Exception as exc:
        log.warning("Failed to render feature importances: %s", exc)
        return False


def _render_feature_set_comparison(
    evaluation_dir: Path,
    output_path: Path,
) -> bool:
    """Read metrics_table.csv and render the model comparison figure."""
    metrics_path = evaluation_dir / "metrics_table.csv"
    if not metrics_path.exists():
        log.warning("metrics_table.csv not found: %s", metrics_path)
        # Fall back to copying existing figure
        fig_src = evaluation_dir / "figures" / "model_comparison.png"
        return _copy_file(fig_src, output_path)

    try:
        metrics_df = _load_optional_csv(metrics_path)
        if metrics_df.empty:
            return False
        from evaluation.plots import plot_model_comparison
        plot_model_comparison(
            metrics_df,
            output_path,
            title="Feature Set Comparison (GroupKFold OOF)",
        )
        log.info("Rendered feature set comparison → %s", project_relative_path(output_path))
        return True
    except Exception as exc:
        log.warning("Failed to render feature set comparison: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def _write_manifest(
    bundle_dir: Path,
    best_key: str,
    best_fs: str,
    best_model: str,
    best_accuracy: float,
    best_f1: float,
    n_windows: int,
    n_classes: int,
    classes: list[str],
    artefact_notes: dict[str, str],
) -> Path:
    """Write manifest."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# Thesis Bundle",
        "",
        f"**Generated:** {date_str}",
        "",
        "## Best Model",
        "",
        f"- **Key:** `{best_key}`",
        f"- **Feature set:** {best_fs}",
        f"- **Model:** {best_model.replace('_', ' ').title()}",
        f"- **OOF Accuracy:** {best_accuracy*100:.1f}%",
        f"- **OOF Macro-F1:** {best_f1*100:.1f}%",
        "",
        "## Dataset",
        "",
        f"- **Windows:** {n_windows}",
        f"- **Classes ({n_classes}):** {', '.join(classes)}",
        "",
        "## Artefacts",
        "",
        "### Tables (`tables/`)",
        "| File | Description | Thesis section |",
        "|------|-------------|----------------|",
        "| `recording_section_counts.csv` | Per-recording sections, windows, labeled windows, duration | Dataset / Data collection |",
        "| `dataset_composition.csv` | Window counts per class + quality tier breakdown | Dataset / Class distribution |",
        "| `feature_set_comparison.csv` | Accuracy and macro-F1 per model × feature set | Evaluation / Feature ablation |",
        "| `sync_summary.csv` | Per-recording sync method and clock drift | Methods / Synchronisation |",
        "",
        "### Sync Figures (`figures/sync_summary/`)",
        "| File | Description | Thesis section |",
        "|------|-------------|----------------|",
        "| `sync_method_selection.png` | Which sync method was selected per recording | Methods / Synchronisation |",
        "| `sync_correlation_comparison.png` | Correlation score per method | Methods / Synchronisation |",
        "| `sync_drift.png` | Clock drift distribution | Methods / Synchronisation |",
        "",
        "### Evaluation Figures (`figures/`)",
        "| File | Description | Thesis section |",
        "|------|-------------|----------------|",
        "| `feature_set_comparison.png` | Grouped bar chart: accuracy and macro-F1 per model × feature set | Evaluation / Feature ablation |",
        "| `best_model_confusion_matrix.png` | Row-normalised OOF confusion matrix for best model | Evaluation / Results |",
        "| `top_feature_importances.png` | Top-20 features for best model, colour-coded by sensor group | Evaluation / Feature analysis |",
        "",
    ]

    if artefact_notes:
        lines += ["### Generation notes", ""]
        for key, note in artefact_notes.items():
            lines.append(f"- **{key}:** {note}")
        lines.append("")

    md_path = bundle_dir / "THESIS_BUNDLE.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote THESIS_BUNDLE.md → %s", project_relative_path(md_path))
    return md_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_thesis_bundle(
    *,
    output_dir: Optional[Path] = None,
    features_path: Optional[Path] = None,
    evaluation_dir: Optional[Path] = None,
    exports_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    top_n_features: int = 20,
) -> dict:
    """Run thesis bundle."""
    if output_dir is None:
        output_dir = data_root() / "report" / "thesis_bundle"
    if features_path is None:
        features_path = exports_root() / "features_fused.csv"
    if evaluation_dir is None:
        evaluation_dir = evaluation_root()
    if exports_dir is None:
        exports_dir = exports_root()
    if report_dir is None:
        report_dir = data_root() / "report"

    bundle_dir = Path(output_dir)
    tables_dir = bundle_dir / "tables"
    figures_dir = bundle_dir / "figures"
    sync_fig_dir = figures_dir / "sync_summary"

    for d in (bundle_dir, tables_dir, figures_dir, sync_fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    artefacts: dict[str, str] = {}
    artefact_notes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Load feature table
    # ------------------------------------------------------------------
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {features_path}. "
            "Run the 'exports' stage first."
        )

    log.info("Loading features from %s", project_relative_path(features_path))
    features_df = read_csv(features_path)
    log.info("Loaded %d rows, %d columns", len(features_df), len(features_df.columns))

    # ------------------------------------------------------------------
    # 2. Identify best model from evaluation_summary.json
    # ------------------------------------------------------------------
    requested_evaluation_dir = Path(evaluation_dir)
    evaluation_dir = resolve_evaluation_dir(requested_evaluation_dir)
    if evaluation_dir != requested_evaluation_dir:
        log.info(
            "Resolved evaluation directory from %s to %s",
            requested_evaluation_dir,
            evaluation_dir,
        )
    summary = _load_json(evaluation_dir / "evaluation_summary.json")
    results = summary.get("results", {})

    best_key = ""
    best_fs = ""
    best_model = ""
    best_accuracy = 0.0
    best_f1 = 0.0
    n_windows = summary.get("n_windows", len(features_df))
    n_classes = summary.get("n_classes", 0)
    classes: list[str] = summary.get("classes", [])

    if results:
        best_key = max(results, key=lambda k: results[k].get("macro_f1", 0))
        best_res = results[best_key]
        best_accuracy = float(best_res.get("accuracy", 0))
        best_f1 = float(best_res.get("macro_f1", 0))
        fs_part, _, model_part = best_key.partition("__")
        best_fs = fs_part
        best_model = model_part
        log.info(
            "Best model: %s (accuracy=%.3f, macro_f1=%.3f)",
            best_key, best_accuracy, best_f1,
        )
    else:
        log.warning(
            "evaluation_summary.json not found or empty — best model unknown. "
            "Run the 'evaluation' stage first."
        )
        artefact_notes["best_model"] = "evaluation_summary.json not found; best model figures skipped"

    # ------------------------------------------------------------------
    # 3. Tables
    # ------------------------------------------------------------------

    # 3a. Recording / section counts
    sec_root = sections_root()
    counts_df = _build_recording_section_counts(features_df, sec_root)
    if not counts_df.empty:
        counts_path = tables_dir / "recording_section_counts.csv"
        write_csv(counts_df, counts_path)
        artefacts["recording_section_counts"] = str(counts_path)
        log.info("Wrote recording_section_counts.csv (%d rows)", len(counts_df))
    else:
        artefact_notes["recording_section_counts"] = "No section_id column in features — table skipped"

    # 3b. Dataset composition
    comp_df = _build_dataset_composition(features_df)
    if not comp_df.empty:
        comp_path = tables_dir / "dataset_composition.csv"
        write_csv(comp_df, comp_path)
        artefacts["dataset_composition"] = str(comp_path)
        log.info("Wrote dataset_composition.csv (%d classes)", len(comp_df) - 1)
    else:
        artefact_notes["dataset_composition"] = "No scenario_label column — table skipped"

    # 3c. Feature set comparison table
    metrics_path = evaluation_dir / "metrics_table.csv"
    if metrics_path.exists():
        metrics_df = _load_optional_csv(metrics_path)
        if not metrics_df.empty:
            fs_comp_path = tables_dir / "feature_set_comparison.csv"
            write_csv(metrics_df, fs_comp_path)
            artefacts["feature_set_comparison_table"] = str(fs_comp_path)
            log.info("Copied metrics_table.csv → feature_set_comparison.csv")
    else:
        artefact_notes["feature_set_comparison_table"] = "metrics_table.csv not found — run 'evaluation' stage first"

    # 3d. Sync summary
    sync_df = _build_sync_summary(exports_dir)
    if not sync_df.empty:
        sync_path = tables_dir / "sync_summary.csv"
        write_csv(sync_df, sync_path)
        artefacts["sync_summary_table"] = str(sync_path)
        log.info("Wrote sync_summary.csv (%d recordings)", len(sync_df))
    else:
        artefact_notes["sync_summary_table"] = "sync_params.csv not found or missing columns"

    # ------------------------------------------------------------------
    # 4. Sync summary figures
    # ------------------------------------------------------------------
    # Prefer pre-rendered figures from report/figures/sync/ (already exist after
    # the 'report' stage).  Fall back to re-rendering from exports if needed.
    report_sync_dir = report_dir / "figures" / "sync"
    n_sync = _copy_dir(report_sync_dir, sync_fig_dir)
    if n_sync > 0:
        artefacts["sync_figures"] = f"{n_sync} figures copied from {project_relative_path(report_sync_dir)}"
        log.info("Copied %d sync figures", n_sync)
    else:
        # Try to generate them directly
        try:
            from reporting.stage_summaries import generate_sync_summary
            sync_csv = exports_dir / "sync_params.csv"
            if sync_csv.exists():
                sync_raw_df = _load_optional_csv(sync_csv)
                if not sync_raw_df.empty:
                    paths = generate_sync_summary(sync_raw_df, sync_fig_dir)
                    n_sync = len(paths)
                    artefacts["sync_figures"] = f"{n_sync} figures re-rendered from sync_params.csv"
                    log.info("Rendered %d sync figures", n_sync)
                else:
                    artefact_notes["sync_figures"] = "sync_params.csv is empty"
            else:
                artefact_notes["sync_figures"] = "sync_params.csv not found"
        except Exception as exc:
            log.warning("Failed to generate sync figures: %s", exc)
            artefact_notes["sync_figures"] = f"rendering failed: {exc}"

    # ------------------------------------------------------------------
    # 5. Feature set comparison figure
    # ------------------------------------------------------------------
    fs_cmp_out = figures_dir / "feature_set_comparison.png"
    ok = _render_feature_set_comparison(evaluation_dir, fs_cmp_out)
    if ok:
        artefacts["feature_set_comparison_figure"] = str(fs_cmp_out)
    else:
        artefact_notes["feature_set_comparison_figure"] = "rendering failed — check evaluation stage"

    # ------------------------------------------------------------------
    # 6. Best-model confusion matrix
    # ------------------------------------------------------------------
    if best_key:
        cm_out = figures_dir / "best_model_confusion_matrix.png"
        ok = _render_confusion_matrix(evaluation_dir, best_fs, best_model, cm_out)
        if ok:
            artefacts["best_model_confusion_matrix"] = str(cm_out)
        else:
            artefact_notes["best_model_confusion_matrix"] = (
                f"confusion_matrix_{best_fs}_{best_model}.csv not found"
            )

    # ------------------------------------------------------------------
    # 7. Top feature importances
    # ------------------------------------------------------------------
    if best_key:
        fi_out = figures_dir / "top_feature_importances.png"
        ok = _render_feature_importances(
            evaluation_dir, best_model, best_fs, fi_out, top_n=top_n_features
        )
        if ok:
            artefacts["top_feature_importances"] = str(fi_out)
        else:
            artefact_notes["top_feature_importances"] = (
                f"feature_importance_{best_model}_{best_fs}.csv not found"
            )

    # ------------------------------------------------------------------
    # 8. Manifest
    # ------------------------------------------------------------------
    _write_manifest(
        bundle_dir=bundle_dir,
        best_key=best_key,
        best_fs=best_fs,
        best_model=best_model,
        best_accuracy=best_accuracy,
        best_f1=best_f1,
        n_windows=n_windows,
        n_classes=n_classes,
        classes=classes,
        artefact_notes=artefact_notes,
    )
    artefacts["manifest"] = str(bundle_dir / "THESIS_BUNDLE.md")

    result = {
        "bundle_dir": str(bundle_dir),
        "best_key": best_key,
        "best_fs": best_fs,
        "best_model": best_model,
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "artefacts": artefacts,
        "notes": artefact_notes,
    }

    total = len(artefacts)
    skipped = len(artefact_notes)
    log.info(
        "Thesis bundle complete: %d artefacts written, %d skipped → %s",
        total,
        skipped,
        project_relative_path(bundle_dir),
    )
    return result
