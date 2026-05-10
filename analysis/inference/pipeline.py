"""Inference pipeline: apply trained models to unlabelled recordings.

Sliding-window features are loaded from each section, the trained model
applied, and the resulting per-window predictions decoded into continuous
time intervals via Viterbi HMM decoding.  Output is written as
``labels_inferred_<label_col>.csv`` in each section's ``labels/`` directory —
the same schema as manual annotations so that approved predictions can feed
back into training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from collections import defaultdict

from common.paths import iter_sections_for_recording, project_relative_path, read_csv, write_json_file
from evaluation.trained_model import TrainedModel, load_trained_model
from labels.parser import LabelRow, load_labels, write_labels

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------

def viterbi_decode(
    log_proba: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> np.ndarray:
    """Return the most probable state sequence via log-space Viterbi.

    Parameters
    ----------
    log_proba : (T, K) per-window per-class log emission probabilities
    log_trans  : (K, K) log transition probabilities; ``[i, j]`` = log P(j | i)
    log_init   : (K,)   log initial state probabilities
    """
    T, K = log_proba.shape
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_init + log_proba[0]
    for t in range(1, T):
        # trans_scores[i, j] = best cumulative score arriving at state j via i
        trans_scores = delta[t - 1, :, None] + log_trans          # (K, K)
        psi[t] = np.argmax(trans_scores, axis=0)                   # (K,)
        delta[t] = trans_scores[psi[t], np.arange(K)] + log_proba[t]

    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ---------------------------------------------------------------------------
# Window sequence → label intervals
# ---------------------------------------------------------------------------

def _path_to_label_rows(
    path: np.ndarray,
    proba_df: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    section_id: str,
    recording_id: str,
    label_col: str,
    model_name: str,
    time_origin_ms: float,
) -> list[LabelRow]:
    """Run-length encode a per-window class path into LabelRow intervals.

    *time_origin_ms* is the section start timestamp used to compute
    section-relative ``start_s``/``end_s`` values, matching the convention
    used by the event_labeler.
    """
    classes = proba_df.columns.tolist()
    annotator = f"{label_col}/{model_name}"
    rows: list[LabelRow] = []
    i = 0
    while i < len(path):
        label = classes[path[i]]
        j = i + 1
        while j < len(path) and path[j] == path[i]:
            j += 1
        start_ms = float(windows.iloc[i]["window_start_ms"])
        end_ms = float(windows.iloc[j - 1]["window_end_ms"])
        mean_prob = float(proba_df.iloc[i:j][label].mean())
        rows.append(
            LabelRow(
                start_ms=start_ms,
                end_ms=end_ms,
                start_s=(start_ms - time_origin_ms) / 1000.0,
                end_s=(end_ms - time_origin_ms) / 1000.0,
                label=label,
                scenario_label=label,
                scope="interval",
                recording_id=recording_id,
                section_id=section_id,
                label_source="model_inferred",
                annotator=annotator,
                confidence=round(mean_prob, 4),
                ambiguous=mean_prob < 0.65,
                notes="",
            )
        )
        i = j
    return rows


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def _write_inference_summary(
    label_rows: list[LabelRow],
    out_path: Path,
    *,
    label_col: str,
    section_id: str,
    model_name: str,
) -> None:
    """Write per-class duration and confidence statistics as a JSON summary."""
    durations: dict[str, float] = defaultdict(float)
    conf_sums: dict[str, float] = defaultdict(float)
    ambig_counts: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for lr in label_rows:
        dur = (lr.end_ms - lr.start_ms) / 1000.0
        durations[lr.label] += dur
        conf_sums[lr.label] += lr.confidence
        ambig_counts[lr.label] += int(lr.ambiguous)
        counts[lr.label] += 1

    classes = {
        label: {
            "n_intervals": counts[label],
            "total_duration_s": round(durations[label], 3),
            "mean_confidence": round(conf_sums[label] / counts[label], 4),
            "frac_ambiguous": round(ambig_counts[label] / counts[label], 4),
        }
        for label in sorted(counts)
    }
    summary = {
        "label_col": label_col,
        "section_id": section_id,
        "model_name": model_name,
        "n_intervals": len(label_rows),
        "total_duration_s": round(sum(durations.values()), 3),
        "classes": classes,
    }
    write_json_file(out_path, summary)


# ---------------------------------------------------------------------------
# Section and recording inference
# ---------------------------------------------------------------------------

def infer_section(
    section_dir: Path,
    tm: TrainedModel,
    *,
    force: bool = False,
) -> list[LabelRow]:
    """Apply *tm* to a section's sliding-window features and write inferred labels.

    Returns the inferred LabelRows, or an empty list if the section is skipped.
    """
    section_id = section_dir.name
    recording_id = section_id.rsplit("s", 1)[0]
    out_path = section_dir / "labels" / f"labels_inferred_{tm.label_col}.csv"

    if not force and out_path.exists():
        log.info("Skipping %s — already exists", project_relative_path(out_path))
        return load_labels(out_path)

    features_path = section_dir / "features" / "features.csv"
    if not features_path.exists():
        log.warning("No features.csv for %s — skipping inference", section_id)
        return []

    df = read_csv(features_path)
    if "window_type" in df.columns:
        df = df[df["window_type"] == "sliding"].copy()
    if df.empty:
        log.warning("No sliding windows in %s — skipping", section_id)
        return []
    df = df.sort_values("window_start_ms").reset_index(drop=True)

    proba_df = tm.predict_proba(df)
    log_proba = np.log(proba_df.values.clip(1e-12))

    if tm.transition_matrix is not None:
        log_trans = np.log(np.array(tm.transition_matrix).clip(1e-12))
        log_init = np.full(len(tm.classes), -np.log(len(tm.classes)))
        path = viterbi_decode(log_proba, log_trans, log_init)
    else:
        path = np.argmax(log_proba, axis=1)

    time_origin_ms = float(df["window_start_ms"].min())
    label_rows = _path_to_label_rows(
        path,
        proba_df,
        df,
        section_id=section_id,
        recording_id=recording_id,
        label_col=tm.label_col,
        model_name=tm.model_name,
        time_origin_ms=time_origin_ms,
    )
    write_labels(label_rows, out_path)
    _write_inference_summary(
        label_rows,
        out_path.with_name(out_path.stem + "_summary.json"),
        label_col=tm.label_col,
        section_id=section_id,
        model_name=tm.model_name,
    )
    log.info(
        "Inferred %d interval(s) [%s] → %s",
        len(label_rows),
        tm.label_col,
        project_relative_path(out_path),
    )
    return label_rows


def infer_recording(
    recording_name: str,
    tm: TrainedModel,
    *,
    force: bool = False,
) -> list[LabelRow]:
    """Apply *tm* to every section of *recording_name* and return all label rows."""
    all_rows: list[LabelRow] = []
    for sec_dir in iter_sections_for_recording(recording_name):
        try:
            rows = infer_section(sec_dir, tm, force=force)
            all_rows.extend(rows)
        except Exception as exc:
            log.error("Inference failed for %s (%s): %s", sec_dir.name, tm.label_col, exc)
    return all_rows


# ---------------------------------------------------------------------------
# Stage-level entry point
# ---------------------------------------------------------------------------

def run_inference(
    recordings: list[str],
    model_paths: list[Path | str],
    *,
    force: bool = False,
    no_plots: bool = False,
) -> dict[str, int]:
    """Apply each trained model to each recording and write inferred label files."""
    counts: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}

    models: list[TrainedModel] = []
    for path in model_paths:
        try:
            models.append(load_trained_model(path))
        except Exception as exc:
            log.error("Could not load model %s: %s", path, exc)
            counts["failed"] += 1

    if not models:
        log.warning("No models loaded — nothing to infer")
        return counts

    for tm in models:
        for recording in recordings:
            try:
                rows = infer_recording(recording, tm, force=force)
                log.info(
                    "Inference complete: %s × %s → %d intervals",
                    recording,
                    tm.label_col,
                    len(rows),
                )
                counts["ok"] += 1
            except Exception as exc:
                log.error("Inference failed for %s (%s): %s", recording, tm.label_col, exc)
                counts["failed"] += 1

    if not no_plots:
        from visualization.stage_plots import plot_section_pipeline_stage
        for recording in recordings:
            for sec_dir in iter_sections_for_recording(recording):
                try:
                    plot_section_pipeline_stage(sec_dir, "inference")
                except Exception as exc:
                    log.warning("Inference plots failed for %s: %s", sec_dir.name, exc)

    return counts
