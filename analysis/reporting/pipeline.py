"""Reporting pipeline orchestrator.

Entry point: run_report()

Reads artefacts produced by earlier pipeline stages and generates
thesis-quality figures, tables, and a markdown summary.

Output layout:
    data/report/
    ├── figures/
    │   ├── dataset/          class_distribution, quality_breakdown,
    │   │                     session_timeline, recording_summary_table
    │   ├── signals/          signal_example_<scenario>.png,
    │   │                     cross_sensor_<scenario>.png
    │   ├── calibration/      gyro bias, gravity residuals, quality overview
    │   ├── sync/             method selection, correlation comparison, drift
    │   ├── orientation/      method selection, gravity alignment, angle stability
    │   └── evaluation/       copies from data/evaluation/figures/
    ├── tables/               CSV tables for appendix
    └── REPORT.md
"""

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
    sections_root,
)

log = logging.getLogger(__name__)


def _copy_evaluation_figures(evaluation_dir: Path, report_eval_dir: Path) -> int:
    """Copy PNG figures from evaluation/figures/ into report/figures/evaluation/."""
    src = evaluation_dir / "figures"
    if not src.exists():
        log.warning("Evaluation figures directory not found: %s", src)
        return 0

    report_eval_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(src.glob("*.png")):
        dest = report_eval_dir / png.name
        shutil.copy2(png, dest)
        n += 1
    log.info("Copied %d evaluation figures to %s", n, project_relative_path(report_eval_dir))
    return n


def _write_report_md(
    output_dir: Path,
    n_windows: int,
    n_classes: int,
    classes: list[str],
    n_signal_plots: int,
    n_eval_figures: int,
    n_stage_figures: dict[str, int],
    evaluation_dir: Path,
) -> Path:
    """Write REPORT.md to output_dir."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    best_line = ""
    summary_path = evaluation_dir / "evaluation_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            results = summary.get("results", {})
            if results:
                best_key = max(results, key=lambda k: results[k].get("accuracy", 0))
                best = results[best_key]
                fs, _, model = best_key.partition("__")
                best_line = (
                    f"\n**Best model:** {fs} + {model.replace('_', ' ').title()} — "
                    f"accuracy {best['accuracy']*100:.1f}%, "
                    f"macro-F1 {best['macro_f1']*100:.1f}%\n"
                )
        except Exception:
            pass

    lines = [
        "# Thesis Report",
        "",
        f"**Generated:** {date_str}",
        "",
        "## Dataset",
        "",
        f"- **Windows:** {n_windows}",
        f"- **Classes:** {n_classes}",
        f"- **Class list:** {', '.join(classes)}",
        best_line,
        "## Figures",
        "",
        "### Dataset Overview (`figures/dataset/`)",
        "- `class_distribution.png` — window count per scenario class",
        "- `quality_breakdown.png` — quality tiers per recording",
        "- `session_timeline.png` — windows coloured by scenario label over time",
        "- `recording_summary_table.png` — per-recording statistics",
        "",
        "### Signal Examples (`figures/signals/`)",
        f"- {n_signal_plots} figures: `signal_example_<scenario>.png` and `cross_sensor_<scenario>.png`",
        "",
        "### Calibration Summary (`figures/calibration/`)",
        f"- {n_stage_figures.get('calibration', 0)} figures:",
        "  - `calibration_gyro_bias.png` — gyro bias distribution per axis per sensor",
        "  - `calibration_acc_bias.png` — accelerometer bias distribution (rider)",
        "  - `calibration_gravity_residuals.png` — gravity residual across sections",
        "  - `calibration_quality_overview.png` — quality pie + protocol detection rate",
        "",
        "### Synchronisation Summary (`figures/sync/`)",
        f"- {n_stage_figures.get('sync', 0)} figures:",
        "  - `sync_method_selection.png` — which method was chosen per recording",
        "  - `sync_correlation_comparison.png` — correlation score per method per recording",
        "  - `sync_drift.png` — clock drift (ppm) per recording",
        "",
        "### Orientation Summary (`figures/orientation/`)",
        f"- {n_stage_figures.get('orientation', 0)} figures:",
        "  - `orientation_method_selection.png` — method chosen per section",
        "  - `orientation_gravity_alignment.png` — gravity alignment quality per sensor",
        "  - `orientation_angle_stability.png` — pitch/roll std across sections",
        "  - `orientation_quality_per_section.png` — per-section quality overview",
        "",
        "### Evaluation (`figures/evaluation/`)",
        f"- {n_eval_figures} figures copied from evaluation stage",
        "",
        "## Tables (`tables/`)",
        "- `class_counts.csv` — window counts per class",
        "- `recording_stats.csv` — per-recording statistics",
        "- `quality_counts.csv` — quality tier distribution",
        "",
    ]

    md_path = output_dir / "REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote REPORT.md → %s", project_relative_path(md_path))
    return md_path


def run_report(
    *,
    output_dir: Optional[Path] = None,
    features_path: Optional[Path] = None,
    evaluation_dir: Optional[Path] = None,
    context_s: float = 5.0,
    scenarios: Optional[list[str]] = None,
) -> dict:
    """Generate all thesis reporting artefacts.

    Parameters
    ----------
    output_dir:
        Root directory for report output. Defaults to ``data/report/``.
    features_path:
        Path to the fused feature CSV. Defaults to ``data/exports/features_fused.csv``.
    evaluation_dir:
        Path to the evaluation output directory. Defaults to ``data/evaluation/``.
    context_s:
        Seconds of context shown on each side of the feature window in signal
        example plots.
    scenarios:
        Subset of scenario labels to process for signal examples. None = all.
    """
    if output_dir is None:
        output_dir = data_root() / "report"
    if features_path is None:
        features_path = exports_root() / "features_fused.csv"
    if evaluation_dir is None:
        evaluation_dir = evaluation_root()

    output_dir    = Path(output_dir)
    features_path = Path(features_path)
    evaluation_dir = Path(evaluation_dir)

    dataset_dir = output_dir / "figures" / "dataset"
    signals_dir = output_dir / "figures" / "signals"
    eval_dir    = output_dir / "figures" / "evaluation"
    stages_dir  = output_dir / "figures"
    tables_dir  = output_dir / "tables"

    for d in (dataset_dir, signals_dir, eval_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load feature table
    # ------------------------------------------------------------------
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {features_path}. "
            "Run the 'exports' stage first."
        )

    log.info("Loading features from %s", project_relative_path(features_path))
    df = read_csv(features_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    classes = (
        sorted(df["scenario_label"].dropna().unique().tolist())
        if "scenario_label" in df.columns
        else []
    )
    n_windows = len(df)
    n_classes = len(classes)

    # ------------------------------------------------------------------
    # 2. Dataset tables
    # ------------------------------------------------------------------
    log.info("Generating dataset tables …")
    from reporting.dataset_stats import generate_dataset_tables
    generate_dataset_tables(df, output_dir)

    # ------------------------------------------------------------------
    # 3. Dataset overview figures
    # ------------------------------------------------------------------
    log.info("Generating dataset overview figures …")
    from reporting.dataset_stats import (
        plot_class_distribution,
        plot_quality_breakdown,
        plot_recording_summary_table,
        plot_session_timeline,
    )

    plot_class_distribution(df, dataset_dir / "class_distribution.png")
    plot_quality_breakdown(df, dataset_dir / "quality_breakdown.png")
    plot_session_timeline(df, dataset_dir / "session_timeline.png")
    plot_recording_summary_table(df, dataset_dir / "recording_summary_table.png")

    # ------------------------------------------------------------------
    # 4. Signal examples
    # ------------------------------------------------------------------
    log.info("Generating signal example figures …")
    from reporting.signal_examples import generate_all_signal_examples
    signal_paths = generate_all_signal_examples(
        df, signals_dir, context_s=context_s, scenarios=scenarios
    )
    n_signal_plots = len(signal_paths)

    # ------------------------------------------------------------------
    # 5. Stage summaries (calibration / sync / orientation)
    # ------------------------------------------------------------------
    log.info("Generating stage summary figures …")
    from reporting.stage_summaries import generate_all_stage_summaries
    stage_results = generate_all_stage_summaries(
        exports_dir=exports_root(),
        sections_root_dir=sections_root(),
        output_dir=stages_dir,
    )
    n_stage_figures = {k: len(v) for k, v in stage_results.items()}

    # ------------------------------------------------------------------
    # 6. Copy evaluation figures
    # ------------------------------------------------------------------
    n_eval_figures = _copy_evaluation_figures(evaluation_dir, eval_dir)

    # ------------------------------------------------------------------
    # 7. Write REPORT.md
    # ------------------------------------------------------------------
    _write_report_md(
        output_dir,
        n_windows=n_windows,
        n_classes=n_classes,
        classes=classes,
        n_signal_plots=n_signal_plots,
        n_eval_figures=n_eval_figures,
        n_stage_figures=n_stage_figures,
        evaluation_dir=evaluation_dir,
    )

    summary = {
        "n_windows": n_windows,
        "n_classes": n_classes,
        "classes": classes,
        "n_signal_plots": n_signal_plots,
        "n_stage_figures": n_stage_figures,
        "n_eval_figures": n_eval_figures,
        "output_dir": str(output_dir),
    }
    total_stage = sum(n_stage_figures.values())
    log.info(
        "Report complete: %d signal plots, %d stage summary figures, %d eval figures → %s",
        n_signal_plots,
        total_stage,
        n_eval_figures,
        project_relative_path(output_dir),
    )
    return summary
