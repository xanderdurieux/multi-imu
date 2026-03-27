from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import sections_root
from visualization.thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style


@dataclass
class FigureArtifact:
    key: str
    path: Path
    status: str
    note: str


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


def _placeholder_figure(title: str, message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    return fig


def _section_dirs(limit: int | None = None) -> list[Path]:
    root = sections_root()
    if not root.exists():
        return []
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    return dirs[:limit] if limit else dirs


def make_pipeline_overview_figure(out_dir: Path) -> FigureArtifact:
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
    stem = out_dir / "pipeline_overview"
    _save(fig, stem)
    return FigureArtifact("pipeline_overview", stem.with_suffix(".pdf"), "ok", "")


def make_orientation_filter_comparison_figure(out_dir: Path, *, max_sections: int = 24) -> FigureArtifact:
    apply_matplotlib_thesis_style()
    rows: list[pd.DataFrame] = []
    for section in _section_dirs(limit=max_sections):
        metric_path = section / "orientation" / "comparison_dynamic" / "sporsa_per_filter_metrics.csv"
        df = _safe_csv(metric_path)
        if df.empty:
            continue
        df["section_id"] = section.name
        rows.append(df)

    stem = out_dir / "orientation_filter_comparison"
    if not rows:
        fig = _placeholder_figure("Orientation Filter Comparison", "No comparison_dynamic metrics found in processed section folders.")
        _save(fig, stem)
        return FigureArtifact("orientation_filter_comparison", stem.with_suffix(".pdf"), "placeholder", "no metrics")

    all_df = pd.concat(rows, ignore_index=True)
    agg = (
        all_df.groupby("variant", as_index=False)
        .agg(
            event_separability_index=("event_separability_index", "median"),
            smoothness_responsiveness_ratio=("smoothness_responsiveness_ratio", "median"),
        )
        .dropna()
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
    _save(fig, stem)
    return FigureArtifact("orientation_filter_comparison", stem.with_suffix(".pdf"), "ok", f"variants={len(agg)}")


def make_event_centered_plot(out_dir: Path, *, window_s: float = 2.0) -> FigureArtifact:
    apply_matplotlib_thesis_style()
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
        stem = out_dir / "event_centered_bike_vs_rider"
        _save(fig, stem)
        return FigureArtifact("event_centered_bike_vs_rider", stem.with_suffix(".pdf"), "ok", section.name)

    stem = out_dir / "event_centered_bike_vs_rider"
    fig = _placeholder_figure("Event-centered Bike vs Rider Plot", "No section had both event candidates and synchronized bike/rider streams.")
    _save(fig, stem)
    return FigureArtifact("event_centered_bike_vs_rider", stem.with_suffix(".pdf"), "placeholder", "no eligible section")


def make_feature_separability_plot(out_dir: Path) -> FigureArtifact:
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
    stem = out_dir / "feature_separability"
    if not frames:
        fig = _placeholder_figure("Feature Separability", "No feature tables with scenario labels were found.")
        _save(fig, stem)
        return FigureArtifact("feature_separability", stem.with_suffix(".pdf"), "placeholder", "no labeled features")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["scenario_label"].str.strip() != ""]
    if all_df.empty:
        fig = _placeholder_figure("Feature Separability", "Scenario labels are missing/empty across loaded features.")
        _save(fig, stem)
        return FigureArtifact("feature_separability", stem.with_suffix(".pdf"), "placeholder", "empty labels")

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
        fig = _placeholder_figure("Feature Separability", "Not enough informative features for separability plotting.")
        _save(fig, stem)
        return FigureArtifact("feature_separability", stem.with_suffix(".pdf"), "placeholder", "insufficient informative features")

    top = [name for name, _ in sorted(scores, key=lambda t: t[1], reverse=True)[:2]]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, label in enumerate(sorted(y.unique())[:6]):
        mask = y == label
        ax.scatter(all_df.loc[mask, top[0]], all_df.loc[mask, top[1]], s=18, alpha=0.68, label=label, color=THESIS_COLORS[i % len(THESIS_COLORS)])
    ax.set_xlabel(top[0])
    ax.set_ylabel(top[1])
    ax.set_title("Feature separability across labeled scenarios")
    ax.legend(loc="best", ncols=2, fontsize=8)
    _save(fig, stem)
    return FigureArtifact("feature_separability", stem.with_suffix(".pdf"), "ok", f"features={top[0]}, {top[1]}")


def make_success_failure_case_plot(out_dir: Path) -> FigureArtifact:
    apply_matplotlib_thesis_style()
    rows = []
    for section in _section_dirs():
        qpath = section / "quality_metadata.json"
        if not qpath.exists():
            continue
        try:
            q = pd.read_json(qpath, typ="series")
        except Exception:
            continue
        rows.append({"section": section.name, "score": float(q.get("overall_quality_score", np.nan))})
    stem = out_dir / "success_failure_case_studies"
    if len(rows) < 2:
        fig = _placeholder_figure("Success / Failure Case Studies", "Need at least two sections with quality metadata to compare outcomes.")
        _save(fig, stem)
        return FigureArtifact("success_failure_case_studies", stem.with_suffix(".pdf"), "placeholder", "insufficient sections")

    df = pd.DataFrame(rows).dropna().sort_values("score")
    chosen = pd.concat([df.head(1), df.tail(1)], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(["Failure exemplar", "Success exemplar"], chosen["score"], color=[THESIS_COLORS[3], THESIS_COLORS[2]])
    for b, sec in zip(bars, chosen["section"], strict=True):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02, sec, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overall quality score")
    ax.set_title("Representative success/failure section case studies")
    _save(fig, stem)
    return FigureArtifact("success_failure_case_studies", stem.with_suffix(".pdf"), "ok", "quality-ranked extremes")
