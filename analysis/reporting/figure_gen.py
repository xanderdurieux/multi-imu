from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import sections_root
from visualization.thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style

from .case_mining import mine_success_failure_cases

STATUS_REAL = "real_result"
STATUS_SKIPPED = "skipped"
STATUS_MISSING = "missing_prerequisite"
STATUS_FAILED = "failed"


@dataclass
class FigureArtifact:
    key: str
    status: str
    note: str
    path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)


CORE_FIGURE_KEYS = [
    "pipeline_overview",
    "orientation_filter_comparison",
    "event_centered_bike_vs_rider",
    "feature_separability",
    "success_failure_case_studies",
]

CORE_FIGURE_FILENAMES = {
    "pipeline_overview": "thesis_core_01_pipeline_overview",
    "orientation_filter_comparison": "thesis_core_02_orientation_filter_comparison",
    "event_centered_bike_vs_rider": "thesis_core_03_event_centered_bike_vs_rider",
    "feature_separability": "thesis_core_04_feature_separability",
    "success_failure_case_studies": "thesis_core_05_success_failure_case_studies",
}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _section_dirs(limit: int | None = None) -> list[Path]:
    root = sections_root()
    if not root.exists():
        return []
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    return dirs[:limit] if limit else dirs


def make_pipeline_overview_figure(out_dir: Path, *, filename: str = "pipeline_overview") -> FigureArtifact:
    apply_matplotlib_thesis_style()
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.axis("off")
    stages = [
        "Raw Session Logs",
        "Synchronization",
        "Section Split",
        "Calibration + Orientation",
        "Events + Features",
        "Thesis Reporting",
    ]
    xpos = np.linspace(0.06, 0.94, len(stages))
    for i, (x, label) in enumerate(zip(xpos, stages, strict=True)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": THESIS_COLORS[i % len(THESIS_COLORS)], "alpha": 0.18},
        )
        if i < len(stages) - 1:
            ax.annotate("", xy=(xpos[i + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("Pipeline Overview for Thesis Reporting")
    stem = out_dir / filename
    _save(fig, stem)
    return FigureArtifact("pipeline_overview", STATUS_REAL, "Generated from static pipeline schematic.", stem.with_suffix(".pdf"))


def make_orientation_filter_comparison_figure(
    out_dir: Path, *, max_sections: int = 24, filename: str = "orientation_filter_comparison"
) -> FigureArtifact:
    apply_matplotlib_thesis_style()
    rows: list[pd.DataFrame] = []
    for section in _section_dirs(limit=max_sections):
        metric_path = section / "orientation" / "comparison_dynamic" / "sporsa_per_filter_metrics.csv"
        df = _safe_csv(metric_path)
        if df.empty:
            continue
        df["section_id"] = section.name
        rows.append(df)

    if not rows:
        return FigureArtifact(
            "orientation_filter_comparison",
            STATUS_MISSING,
            "No orientation comparison metrics available.",
            diagnostics=["Missing prerequisite: sections/*/orientation/comparison_dynamic/sporsa_per_filter_metrics.csv"],
        )

    all_df = pd.concat(rows, ignore_index=True)
    agg = (
        all_df.groupby("variant", as_index=False)
        .agg(
            event_separability_index=("event_separability_index", "median"),
            smoothness_responsiveness_ratio=("smoothness_responsiveness_ratio", "median"),
        )
        .dropna()
    )
    if agg.empty:
        return FigureArtifact(
            "orientation_filter_comparison",
            STATUS_MISSING,
            "Metrics were found but required numeric fields are empty.",
            diagnostics=["Required columns: variant, event_separability_index, smoothness_responsiveness_ratio"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(
        agg["smoothness_responsiveness_ratio"],
        agg["event_separability_index"],
        s=55,
        c=THESIS_COLORS[0],
        alpha=0.85,
    )
    for _, row in agg.iterrows():
        ax.text(row["smoothness_responsiveness_ratio"], row["event_separability_index"], f" {row['variant']}", fontsize=8)
    ax.set_xlabel("Smoothness/Responsiveness ratio (median)")
    ax.set_ylabel("Event separability index (median)")
    ax.set_title("Orientation Filter Trade-off Across Sections")
    stem = out_dir / filename
    _save(fig, stem)
    return FigureArtifact("orientation_filter_comparison", STATUS_REAL, f"variants={len(agg)}", stem.with_suffix(".pdf"))


def make_event_centered_plot(out_dir: Path, *, window_s: float = 2.0, filename: str = "event_centered_bike_vs_rider") -> FigureArtifact:
    apply_matplotlib_thesis_style()
    if not _section_dirs():
        return FigureArtifact(
            "event_centered_bike_vs_rider",
            STATUS_MISSING,
            "No processed sections found.",
            diagnostics=["Missing prerequisite: data/sections/* generated by workflow."],
        )

    for section in _section_dirs():
        events = _safe_csv(section / "events" / "event_candidates.csv")
        bike = _safe_csv(section / "sporsa.csv")
        rider = _safe_csv(section / "arduino.csv")
        if events.empty or bike.empty or rider.empty or "time_s" not in events.columns:
            continue
        top = events.sort_values("confidence", ascending=False).iloc[0]
        et = float(top.get("time_s", np.nan))
        if not np.isfinite(et):
            continue

        def prep(df: pd.DataFrame) -> pd.DataFrame:
            if "timestamp" not in df.columns:
                return pd.DataFrame()
            out = df.copy()
            out["time_s"] = (pd.to_numeric(out["timestamp"], errors="coerce") - pd.to_numeric(out["timestamp"], errors="coerce").iloc[0]) / 1000.0
            cols = [c for c in ("ax", "ay", "az") if c in out.columns]
            if len(cols) < 3:
                return pd.DataFrame()
            out["acc_norm"] = np.sqrt(np.square(out[cols]).sum(axis=1))
            return out

        bike = prep(bike)
        rider = prep(rider)
        if bike.empty or rider.empty:
            continue
        bm = bike[(bike["time_s"] >= et - window_s) & (bike["time_s"] <= et + window_s)]
        rm = rider[(rider["time_s"] >= et - window_s) & (rider["time_s"] <= et + window_s)]
        if bm.empty or rm.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.plot(bm["time_s"] - et, bm["acc_norm"], label="Bike IMU (Sporsa)", color=THESIS_COLORS[0])
        ax.plot(rm["time_s"] - et, rm["acc_norm"], label="Rider IMU (Arduino)", color=THESIS_COLORS[1])
        ax.axvline(0.0, color="#555", linestyle="--", linewidth=1.0, label="Detected event center")
        ax.set_xlabel("Time relative to event (s)")
        ax.set_ylabel("Acceleration norm (m/s²)")
        ax.set_title(f"Event-centered bike vs rider response ({section.name})")
        ax.legend(loc="best")
        stem = out_dir / filename
        _save(fig, stem)
        return FigureArtifact("event_centered_bike_vs_rider", STATUS_REAL, f"section={section.name}", stem.with_suffix(".pdf"))

    return FigureArtifact(
        "event_centered_bike_vs_rider",
        STATUS_MISSING,
        "No section satisfied event + synchronized dual-IMU prerequisites.",
        diagnostics=[
            "Need one section containing events/event_candidates.csv with confidence+time_s and both sporsa.csv/arduino.csv with timestamp,ax,ay,az."
        ],
    )


def make_feature_separability_plot(out_dir: Path, *, filename: str = "feature_separability") -> FigureArtifact:
    apply_matplotlib_thesis_style()
    frames: list[pd.DataFrame] = []
    for section in _section_dirs():
        df = _safe_csv(section / "features" / "features.csv")
        if df.empty or "scenario_label" not in df.columns:
            continue
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            continue
        numeric["scenario_label"] = df["scenario_label"].astype(str)
        frames.append(numeric)
    if not frames:
        return FigureArtifact(
            "feature_separability",
            STATUS_MISSING,
            "No labeled feature tables were found.",
            diagnostics=["Missing prerequisite: sections/*/features/features.csv with scenario_label + numeric feature columns."],
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["scenario_label"].str.strip() != ""]
    if all_df.empty:
        return FigureArtifact(
            "feature_separability",
            STATUS_MISSING,
            "Scenario labels are present but empty.",
            diagnostics=["Provide non-empty scenario_label annotations for at least two classes."],
        )

    y = all_df["scenario_label"]
    feature_cols = [c for c in all_df.columns if c != "scenario_label"]
    scores: list[tuple[str, float]] = []
    for col in feature_cols:
        s = pd.to_numeric(all_df[col], errors="coerce")
        if s.isna().all():
            continue
        grouped = [s[y == lab].dropna().to_numpy() for lab in sorted(y.unique())]
        grouped = [g for g in grouped if len(g) >= 4]
        if len(grouped) < 2:
            continue
        means = np.array([np.mean(g) for g in grouped])
        vars_ = np.array([np.var(g) + 1e-9 for g in grouped])
        score = float(np.var(means) / np.mean(vars_))
        scores.append((col, score))
    if len(scores) < 2:
        return FigureArtifact(
            "feature_separability",
            STATUS_MISSING,
            "Not enough informative features for separability plotting.",
            diagnostics=["Need >=2 numeric features with per-class support (>=4 samples per class across >=2 classes)."],
        )

    top = [name for name, _ in sorted(scores, key=lambda t: t[1], reverse=True)[:2]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, label in enumerate(sorted(y.unique())[:6]):
        mask = y == label
        ax.scatter(all_df.loc[mask, top[0]], all_df.loc[mask, top[1]], s=18, alpha=0.68, label=label, color=THESIS_COLORS[i % len(THESIS_COLORS)])
    ax.set_xlabel(top[0])
    ax.set_ylabel(top[1])
    ax.set_title("Feature separability across labeled scenarios")
    ax.legend(loc="best", ncols=2, fontsize=8)
    stem = out_dir / filename
    _save(fig, stem)
    return FigureArtifact("feature_separability", STATUS_REAL, f"features={top[0]}, {top[1]}", stem.with_suffix(".pdf"))


def make_success_failure_case_plot(out_dir: Path, *, filename: str = "success_failure_case_studies") -> FigureArtifact:
    apply_matplotlib_thesis_style()
    mined = mine_success_failure_cases(_section_dirs(), n_success=2, n_failure=2)
    if mined.cases.empty:
        return FigureArtifact(
            "success_failure_case_studies",
            STATUS_MISSING,
            "No representative case-study candidates could be mined.",
            diagnostics=mined.diagnostics,
        )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ordered = mined.cases.sort_values(["case_type", "composite_signal_score"], ascending=[True, True]).reset_index(drop=True)
    xpos = np.arange(len(ordered))
    bars = ax.bar(xpos, ordered["composite_signal_score"], color=[THESIS_COLORS[3] if t == "failure" else THESIS_COLORS[2] for t in ordered["case_type"]])
    ax.set_xticks(xpos)
    ax.set_xticklabels(ordered["section_id"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Composite quality signal score")
    ax.set_title("Representative success/failure sections (QC + confidence + downstream proxy)")
    for bar, ctype in zip(bars, ordered["case_type"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, ctype, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, max(1.02, float(np.nanmax(ordered["composite_signal_score"])) + 0.08))
    stem = out_dir / filename
    _save(fig, stem)
    return FigureArtifact("success_failure_case_studies", STATUS_REAL, f"cases={len(ordered)}", stem.with_suffix(".pdf"), diagnostics=mined.diagnostics)


def generate_core_thesis_figures(out_dir: Path) -> list[FigureArtifact]:
    builders: list[tuple[str, Callable[..., FigureArtifact]]] = [
        ("pipeline_overview", make_pipeline_overview_figure),
        ("orientation_filter_comparison", make_orientation_filter_comparison_figure),
        ("event_centered_bike_vs_rider", make_event_centered_plot),
        ("feature_separability", make_feature_separability_plot),
        ("success_failure_case_studies", make_success_failure_case_plot),
    ]
    artifacts: list[FigureArtifact] = []
    for key, fn in builders:
        try:
            artifacts.append(fn(out_dir, filename=CORE_FIGURE_FILENAMES[key]))
        except Exception as exc:
            artifacts.append(FigureArtifact(key, STATUS_FAILED, f"Unhandled error: {exc}"))
    return artifacts
