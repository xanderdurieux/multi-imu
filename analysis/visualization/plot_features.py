"""Plot feature-stage outputs from section-level ``features/features.csv``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import read_csv, resolve_data_dir
from visualization._utils import filter_valid_plot_xy, save_figure

log = logging.getLogger(__name__)

_EXCLUDE_COLS = {
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "section_id",
    "label",
}
_DEFAULT_PREFIXES = ("bike_", "rider_", "cross_")


def _resolve_features_csv(target: str | Path) -> Path:
    """Resolve features csv."""
    base = resolve_data_dir(target)
    if base.name == "features" and (base / "features.csv").exists():
        return base / "features.csv"
    maybe = base / "features" / "features.csv"
    if maybe.exists():
        return maybe
    direct = Path(str(target)).expanduser()
    if direct.is_file() and direct.suffix.lower() == ".csv":
        return direct.resolve()
    raise FileNotFoundError(f"Could not resolve features CSV from: {target}")


def _default_x_axis(df: pd.DataFrame) -> np.ndarray:
    """Return the default x axis."""
    if "window_idx" in df.columns:
        x = pd.to_numeric(df["window_idx"], errors="coerce").to_numpy(dtype=float)
    elif "window_start_ms" in df.columns:
        x = pd.to_numeric(df["window_start_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    else:
        x = np.arange(len(df), dtype=float)
    return x


def _select_feature_columns(df: pd.DataFrame, top_n: int) -> list[str]:
    """Select feature columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = [c for c in numeric_cols if c not in _EXCLUDE_COLS]
    if not candidates:
        return []

    prefixed = [c for c in candidates if c.startswith(_DEFAULT_PREFIXES)]
    if prefixed:
        candidates = prefixed

    variances = (
        df[candidates]
        .apply(pd.to_numeric, errors="coerce")
        .var(axis=0, skipna=True)
        .sort_values(ascending=False)
    )
    return [c for c in variances.index.tolist() if np.isfinite(variances[c])][:top_n]


def plot_features_stage(
    target: str | Path,
    *,
    top_n: int = 8,
) -> Path | None:
    """Generate a compact multi-panel trend plot for selected feature columns."""
    features_csv = _resolve_features_csv(target)
    df = read_csv(features_csv)
    if df.empty:
        log.warning("Features CSV is empty: %s", features_csv)
        return None

    x = _default_x_axis(df)
    feat_cols = _select_feature_columns(df, top_n=max(1, int(top_n)))
    if not feat_cols:
        log.warning("No numeric feature columns found in %s", features_csv)
        return None

    rows = len(feat_cols)
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3, 2.1 * rows)), sharex=True)
    if rows == 1:
        axes = [axes]

    for idx, col in enumerate(feat_cols):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x_plot, y_plot = filter_valid_plot_xy(x, y)
        axes[idx].plot(x_plot, y_plot, lw=0.8, color="#1f77b4")
        axes[idx].set_ylabel(col, fontsize=8)
        axes[idx].grid(alpha=0.2, lw=0.4)

    if "window_idx" in df.columns:
        axes[-1].set_xlabel("Window index")
    elif "window_start_ms" in df.columns:
        axes[-1].set_xlabel("Window start (s)")
    else:
        axes[-1].set_xlabel("Row")

    if "section_id" in df.columns and len(df["section_id"]) > 0:
        section_name = str(df["section_id"].iloc[0])
    else:
        section_name = features_csv.parent.parent.name
    fig.suptitle(f"{section_name} — features overview (top {len(feat_cols)} by variance)")
    fig.tight_layout()

    out_path = features_csv.parent / "features_overview.png"
    return save_figure(fig, out_path)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_features")
    parser.add_argument("target", help="Section directory, features dir, or features CSV")
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of feature series to plot (default: 8).",
    )
    args = parser.parse_args(argv)

    try:
        out = plot_features_stage(args.target, top_n=args.top_n)
    except Exception as exc:
        log.error("Failed to plot features stage: %s", exc)
        return

    if out is None:
        print("No features plot generated.")
        return
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
