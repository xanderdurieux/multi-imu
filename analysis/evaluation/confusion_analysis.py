"""Per-run confusion analysis: hardest classes + top-N confused class pairs.

Operates on a square confusion-matrix DataFrame (true × predicted) — typically
the OOF matrix written by :func:`evaluation.experiments._cv_evaluate`. Output
is two CSVs that surface *what* is being confused with *what*, which feeds
the substantive interpretation called for in the thesis evaluation goals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import project_relative_path, write_csv

log = logging.getLogger(__name__)


def analyze_confusion_matrix(
    cm_df: pd.DataFrame,
    *,
    top_pairs: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(per_class_df, top_pairs_df)`` derived from the confusion matrix.

    Parameters
    ----------
    cm_df:
        Square DataFrame with rows = true labels, columns = predicted labels,
        values = counts.
    top_pairs:
        Number of off-diagonal ``(true, predicted)`` pairs to surface.

    Outputs
    -------
    per_class_df:
        One row per class — support, recall, precision, F1, dominant
        confusion target, and the share of the row spent on that target.
        Sorted ascending by recall so the hardest classes appear first.
    top_pairs_df:
        Off-diagonal cells with non-zero count, ranked by absolute count
        (and ``share_of_true`` as tie-breaker).
    """
    classes = list(cm_df.index)
    cm = cm_df.values.astype(float)
    n = len(classes)

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)

    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(row_sum > 0, diag / row_sum, np.nan)
        precision = np.where(col_sum > 0, diag / col_sum, np.nan)
        f1 = np.where(
            (recall + precision) > 0,
            2 * recall * precision / (recall + precision),
            np.nan,
        )

    per_class: list[dict] = []
    for i, cls in enumerate(classes):
        row = cm[i].copy()
        row[i] = -1.0  # mask diagonal so argmax picks the dominant confusion
        j = int(np.argmax(row))
        cnt = int(cm[i, j]) if row[j] > 0 else 0
        share = float(cnt / row_sum[i]) if row_sum[i] > 0 else 0.0
        per_class.append(
            {
                "class": cls,
                "support": int(row_sum[i]),
                "recall": round(float(recall[i]) if not np.isnan(recall[i]) else 0.0, 4),
                "precision": round(float(precision[i]) if not np.isnan(precision[i]) else 0.0, 4),
                "f1": round(float(f1[i]) if not np.isnan(f1[i]) else 0.0, 4),
                "dominant_confusion_with": classes[j] if cnt > 0 else "",
                "dominant_confusion_count": cnt,
                "dominant_confusion_share": round(share, 4),
            }
        )
    per_class_df = (
        pd.DataFrame(per_class)
        .sort_values("recall", ascending=True)
        .reset_index(drop=True)
    )

    pairs: list[dict] = []
    for i in range(n):
        for j in range(n):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append(
                {
                    "true": classes[i],
                    "predicted": classes[j],
                    "count": int(cm[i, j]),
                    "share_of_true": round(
                        float(cm[i, j] / row_sum[i]) if row_sum[i] > 0 else 0.0,
                        4,
                    ),
                }
            )
    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values(["count", "share_of_true"], ascending=[False, False])
        .head(top_pairs)
        .reset_index(drop=True)
    )

    return per_class_df, pairs_df


def write_confusion_analysis(
    cm_df: pd.DataFrame,
    output_dir: Path,
    *,
    config_name: str,
    top_pairs: int = 10,
) -> tuple[Path, Path]:
    """Run analysis on *cm_df* and write per-class + top-pairs CSVs.

    Files written:
        ``confusion_per_class_<config_name>.csv``
        ``confusion_top_pairs_<config_name>.csv``
    """
    per_class_df, pairs_df = analyze_confusion_matrix(cm_df, top_pairs=top_pairs)

    pc_path = output_dir / f"confusion_per_class_{config_name}.csv"
    tp_path = output_dir / f"confusion_top_pairs_{config_name}.csv"
    write_csv(per_class_df, pc_path)
    write_csv(pairs_df, tp_path)

    hardest = per_class_df.iloc[0]["class"] if not per_class_df.empty else "n/a"
    log.info(
        "Confusion analysis (%s): hardest class=%s; %d off-diagonal pair(s) surfaced",
        config_name,
        hardest,
        len(pairs_df),
    )
    log.debug(
        "Wrote %s, %s",
        project_relative_path(pc_path),
        project_relative_path(tp_path),
    )
    return pc_path, tp_path
