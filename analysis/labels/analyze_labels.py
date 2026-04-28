"""Summarize recording-level label files and generate label-distribution plots.

Examples
--------
    uv run python -m labels.analyze_labels
    uv run python -m labels.analyze_labels --label-set v2
    uv run python -m labels.analyze_labels --workflow-config data/_configs/workflow.train.json
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import (
    default_label_set,
    label_set_dir,
    labels_root,
    project_relative_path,
    read_csv,
    recording_stage_dir,
    write_csv,
    write_json_file,
)
from labels.parser import load_labels
from visualization._utils import QUALITATIVE_PALETTE

log = logging.getLogger(__name__)

_DPI = 160
_EMPTY_COLUMNS = [
    "recording_id",
    "section_id",
    "label_name",
    "scope",
    "start_ms",
    "end_ms",
    "start_s",
    "end_s",
    "duration_s",
    "label_source",
    "annotator",
    "confidence",
    "ambiguous",
    "notes",
    "label_level",
    "label_file",
]


def _display_path(path: Path) -> str:
    try:
        return project_relative_path(path)
    except ValueError:
        return str(path)


def _save_plot(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", _display_path(path))
    return path


def _iter_label_files(label_set: str | None = None) -> list[Path]:
    root = label_set_dir(label_set or default_label_set())
    paths = sorted(root.glob("labels_intervals_*.*")) if root.exists() else []
    if not paths:
        paths = sorted(labels_root().glob("labels_intervals_*.*"))
    return [path for path in paths if path.suffix.lower() in {".csv", ".json"}]


def _infer_recording_id(label_file: Path) -> str:
    match = re.match(r"^labels_intervals_(.+)\.(csv|json)$", label_file.name)
    return match.group(1) if match else ""


def _load_label_rows(label_set: str | None = None) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []

    for label_file in _iter_label_files(label_set):
        rows = load_labels(label_file)
        if not rows:
            continue

        inferred_recording = _infer_recording_id(label_file)

        for row in rows:
            label_name = (row.scenario_label or row.label).strip()
            duration_s = (float(row.end_ms) - float(row.start_ms)) / 1000.0
            if not label_name or not np.isfinite(duration_s) or duration_s <= 0:
                continue

            rows_out.append(
                {
                    "recording_id": row.recording_id.strip() or inferred_recording,
                    "section_id": row.section_id.strip(),
                    "label_name": label_name,
                    "scope": row.scope,
                    "start_ms": float(row.start_ms),
                    "end_ms": float(row.end_ms),
                    "start_s": row.start_s,
                    "end_s": row.end_s,
                    "duration_s": duration_s,
                    "label_source": row.label_source,
                    "annotator": row.annotator,
                    "confidence": float(row.confidence),
                    "ambiguous": bool(row.ambiguous),
                    "notes": row.notes,
                    "label_level": "recording",
                    "label_file": str(label_file),
                }
            )

    if not rows_out:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    df = pd.DataFrame(rows_out)
    df = df.sort_values(["recording_id", "start_ms", "end_ms"]).reset_index(drop=True)
    return df


def _duration_label(seconds: float) -> str:
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.1f} s"


def _label_colors(labels: list[str]) -> dict[str, str]:
    return {
        label: QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        for idx, label in enumerate(labels)
    }


def _recording_duration_s(recording_id: str, cache: dict[str, float | None]) -> float | None:
    if recording_id in cache:
        return cache[recording_id]

    csv_path = recording_stage_dir(recording_id, "synced") / "sporsa.csv"
    if not csv_path.exists():
        cache[recording_id] = None
        return None

    try:
        df = read_csv(csv_path)
    except Exception as exc:
        log.warning("Failed to read %s: %s", csv_path, exc)
        cache[recording_id] = None
        return None

    if "timestamp" not in df.columns or df.empty:
        cache[recording_id] = None
        return None

    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        cache[recording_id] = None
        return None

    duration_s = float(ts.iloc[-1] - ts.iloc[0]) / 1000.0
    cache[recording_id] = duration_s if np.isfinite(duration_s) and duration_s > 0 else None
    return cache[recording_id]


def build_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "label_name",
                "n_intervals",
                "total_duration_s",
                "median_duration_s",
                "mean_duration_s",
                "n_recordings",
                "n_sections",
                "ambiguous_count",
                "mean_confidence",
            ]
        )

    summary = (
        df.groupby("label_name", dropna=False)
        .agg(
            n_intervals=("label_name", "size"),
            total_duration_s=("duration_s", "sum"),
            median_duration_s=("duration_s", "median"),
            mean_duration_s=("duration_s", "mean"),
            n_recordings=("recording_id", "nunique"),
            n_sections=("section_id", lambda s: s.replace("", pd.NA).dropna().nunique()),
            ambiguous_count=("ambiguous", "sum"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["total_duration_s", "n_intervals"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def build_recording_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "recording_id",
                "n_intervals",
                "n_unique_labels",
                "total_labeled_duration_s",
                "recording_duration_s",
                "cumulative_labeled_share_pct",
                "ambiguous_count",
            ]
        )

    duration_cache: dict[str, float | None] = {}
    summary = (
        df.groupby("recording_id", dropna=False)
        .agg(
            n_intervals=("label_name", "size"),
            n_unique_labels=("label_name", "nunique"),
            total_labeled_duration_s=("duration_s", "sum"),
            ambiguous_count=("ambiguous", "sum"),
        )
        .reset_index()
    )
    summary["recording_duration_s"] = summary["recording_id"].map(
        lambda rec: _recording_duration_s(rec, duration_cache)
    )
    summary["cumulative_labeled_share_pct"] = (
        100.0 * summary["total_labeled_duration_s"] / summary["recording_duration_s"]
    )
    summary = summary.sort_values(
        ["total_labeled_duration_s", "n_intervals"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return summary


def build_recording_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["recording_id", "label_name", "n_intervals", "total_duration_s"]
        )

    summary = (
        df.groupby(["recording_id", "label_name"], dropna=False)
        .agg(
            n_intervals=("label_name", "size"),
            total_duration_s=("duration_s", "sum"),
        )
        .reset_index()
        .sort_values(["recording_id", "total_duration_s"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def _plot_label_total_duration(summary: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.48 * len(summary) + 1.5)))
    plot_df = summary.sort_values("total_duration_s", ascending=True)
    color_map = _label_colors(plot_df["label_name"].tolist())
    colors = [color_map[label] for label in plot_df["label_name"]]
    bars = ax.barh(plot_df["label_name"], plot_df["total_duration_s"], color=colors, edgecolor="white", linewidth=0.5)

    max_val = float(plot_df["total_duration_s"].max()) if not plot_df.empty else 1.0
    for bar, seconds in zip(bars, plot_df["total_duration_s"], strict=False):
        ax.text(
            bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            _duration_label(float(seconds)),
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Total labeled duration (s)")
    ax.set_ylabel("Label")
    ax.set_title("Label summary: cumulative duration per label")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max_val * 1.16 if max_val > 0 else 1.0)
    return _save_plot(fig, output_path)


def _plot_label_interval_count(summary: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.48 * len(summary) + 1.5)))
    plot_df = summary.sort_values("n_intervals", ascending=True)
    color_map = _label_colors(plot_df["label_name"].tolist())
    colors = [color_map[label] for label in plot_df["label_name"]]
    bars = ax.barh(plot_df["label_name"], plot_df["n_intervals"], color=colors, edgecolor="white", linewidth=0.5)

    max_val = int(plot_df["n_intervals"].max()) if not plot_df.empty else 1
    for bar, count in zip(bars, plot_df["n_intervals"], strict=False):
        ax.text(
            bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            str(int(count)),
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Number of intervals")
    ax.set_ylabel("Label")
    ax.set_title("Label summary: interval count per label")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max_val * 1.16 if max_val > 0 else 1.0)
    return _save_plot(fig, output_path)


def _plot_recording_total_duration(summary: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.45 * len(summary) + 1.5)))
    plot_df = summary.sort_values("total_labeled_duration_s", ascending=True)
    bars = ax.barh(
        plot_df["recording_id"],
        plot_df["total_labeled_duration_s"],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )

    max_val = float(plot_df["total_labeled_duration_s"].max()) if not plot_df.empty else 1.0
    for bar, total_s, share_pct in zip(
        bars,
        plot_df["total_labeled_duration_s"],
        plot_df["cumulative_labeled_share_pct"],
        strict=False,
    ):
        label = _duration_label(float(total_s))
        if np.isfinite(share_pct):
            label += f" ({share_pct:.1f}%)"
        ax.text(
            bar.get_width() + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Cumulative labeled duration (s)")
    ax.set_ylabel("Recording")
    ax.set_title("Label summary: cumulative labeled duration per recording")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1.0)
    return _save_plot(fig, output_path)


def _plot_recording_label_heatmap(summary: pd.DataFrame, output_path: Path) -> Path | None:
    if summary.empty:
        return None

    matrix = summary.pivot(
        index="label_name",
        columns="recording_id",
        values="total_duration_s",
    ).fillna(0.0)
    if matrix.empty:
        return None

    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    matrix = matrix[matrix.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.1 * len(matrix.columns) + 2.0), max(4.5, 0.55 * len(matrix.index) + 1.8))
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    ax.set_title("Label summary: duration heatmap by label and recording")
    ax.set_xlabel("Recording")
    ax.set_ylabel("Label")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Duration (s)")
    fig.tight_layout()
    return _save_plot(fig, output_path)


def _write_overview_json(
    *,
    df: pd.DataFrame,
    by_label: pd.DataFrame,
    by_recording: pd.DataFrame,
    output_path: Path,
    label_set: str,
    labels_dir: Path,
) -> Path:
    payload = {
        "source": "recording",
        "label_set": label_set,
        "labels_dir": _display_path(labels_dir),
        "n_rows": int(len(df)),
        "n_recordings": int(df["recording_id"].replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "n_sections": int(df["section_id"].replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "n_labels": int(df["label_name"].nunique()) if not df.empty else 0,
        "total_labeled_duration_s": float(df["duration_s"].sum()) if not df.empty else 0.0,
        "ambiguous_count": int(df["ambiguous"].sum()) if not df.empty else 0,
        "top_labels_by_duration": by_label.head(5).to_dict(orient="records"),
        "top_recordings_by_duration": by_recording.head(5).to_dict(orient="records"),
    }
    write_json_file(output_path, payload, indent=2)
    log.info("Wrote %s", _display_path(output_path))
    return output_path


def analyze_labels(
    *,
    output_dir: Path | None = None,
    label_set: str | None = None,
) -> Path:
    resolved_label_set = label_set or default_label_set()
    labels_dir = label_set_dir(resolved_label_set)
    if output_dir is None:
        output_dir = labels_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_label_rows(resolved_label_set)
    if df.empty:
        raise FileNotFoundError(
            f"No recording-level label rows found in {labels_dir!s}"
        )

    by_label = build_label_summary(df)
    by_recording = build_recording_summary(df)
    by_recording_label = build_recording_label_summary(df)

    write_csv(df, output_dir / "label_rows.csv")
    write_csv(by_label, output_dir / "summary_by_label.csv")
    write_csv(by_recording, output_dir / "summary_by_recording.csv")
    write_csv(by_recording_label, output_dir / "summary_by_recording_label.csv")

    duration_matrix = by_recording_label.pivot(
        index="label_name",
        columns="recording_id",
        values="total_duration_s",
    ).fillna(0.0)
    count_matrix = by_recording_label.pivot(
        index="label_name",
        columns="recording_id",
        values="n_intervals",
    ).fillna(0)
    write_csv(duration_matrix.reset_index(), output_dir / "matrix_label_recording_duration_s.csv")
    write_csv(count_matrix.reset_index(), output_dir / "matrix_label_recording_interval_count.csv")

    _plot_label_total_duration(by_label, output_dir / "plot_label_total_duration_s.png")
    _plot_label_interval_count(by_label, output_dir / "plot_label_interval_count.png")
    _plot_recording_total_duration(by_recording, output_dir / "plot_recording_total_labeled_duration_s.png")
    _plot_recording_label_heatmap(by_recording_label, output_dir / "plot_label_recording_duration_heatmap.png")
    _write_overview_json(
        df=df,
        by_label=by_label,
        by_recording=by_recording,
        output_path=output_dir / "overview.json",
        label_set=resolved_label_set,
        labels_dir=labels_dir,
    )
    return output_dir


def _label_set_from_workflow_config(path: Path | None) -> str | None:
    if path is None:
        return None
    from workflow.config import load_workflow_config

    return load_workflow_config(path).label_set


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m labels.analyze_labels",
        description=(
            "Collect recording-level label files from data/_labels, write summary "
            "tables, and generate formal duration/count plots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON summaries and plots. Defaults to data/_labels/<label-set>/summary.",
    )
    parser.add_argument(
        "--label-set",
        default=None,
        help=(
            "Label set directory under data/_labels. Overrides --workflow-config; "
            "default: MULTI_IMU_LABEL_SET or v1."
        ),
    )
    parser.add_argument(
        "--workflow-config",
        "--config",
        dest="workflow_config",
        type=Path,
        default=None,
        help="Workflow config JSON whose label_set should be used.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    label_set = args.label_set or _label_set_from_workflow_config(args.workflow_config)
    output_dir = analyze_labels(output_dir=args.output_dir, label_set=label_set)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
