"""Plot overlaid comparison of bike and rider IMU signals.

CLI usage::

    python -m visualization.plot_comparison <recording>/<stage> [--norm]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import analysis_root, sections_root, recording_stage_dir

log = logging.getLogger(__name__)

SENSORS = ["sporsa", "arduino"]
COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}


def _resolve_stage_dir(stage_ref: str) -> Path:
    data_root = analysis_root() / "data"
    # Try sections root
    sec = sections_root() / stage_ref
    if sec.exists():
        return sec
    # Try sections/<last_part>
    parts = stage_ref.split("/")
    if len(parts) >= 2:
        sec2 = sections_root() / parts[-1]
        if sec2.exists():
            return sec2
        # Try recording/stage
        rec, stage = parts[0], "/".join(parts[1:])
        rd = recording_stage_dir(rec, stage)
        if rd.exists():
            return rd
    return data_root / stage_ref


def plot_comparison_data(
    stage_dir: Path,
    *,
    norm_only: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Plot accelerometer norms (and optionally gyro) for all sensors in stage_dir."""
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in SENSORS:
        csv = stage_dir / f"{sensor}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            sensor_dfs[sensor] = df

    if not sensor_dfs:
        log.warning("No sensor CSVs found in %s", stage_dir)
        return stage_dir / "comparison.png"

    suffix = "_norm" if norm_only else ""
    if output_path is None:
        output_path = stage_dir / f"comparison{suffix}.png"

    fig, axes = plt.subplots(2 if not norm_only else 1, 1, figsize=(12, 6 if not norm_only else 3), sharex=True)
    if norm_only:
        axes = [axes]

    for sensor, df in sensor_dfs.items():
        ts = pd.to_numeric(df.get("timestamp", pd.Series()), errors="coerce").to_numpy(dtype=float)
        if ts.size == 0:
            continue
        ts_s = (ts - ts[0]) / 1000.0
        color = COLORS.get(sensor, None)

        acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
        gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]

        if norm_only:
            if acc_cols:
                acc_norm = np.sqrt(np.nansum(df[acc_cols].to_numpy(dtype=float) ** 2, axis=1))
                axes[0].plot(ts_s, acc_norm, lw=0.8, color=color, label=f"{sensor} |acc|", alpha=0.8)
        else:
            if acc_cols:
                acc_norm = np.sqrt(np.nansum(df[acc_cols].to_numpy(dtype=float) ** 2, axis=1))
                axes[0].plot(ts_s, acc_norm, lw=0.8, color=color, label=f"{sensor} |acc|", alpha=0.8)
            if gyro_cols:
                gyro_norm = np.sqrt(np.nansum(df[gyro_cols].to_numpy(dtype=float) ** 2, axis=1))
                axes[1].plot(ts_s, gyro_norm, lw=0.8, color=color, label=f"{sensor} |gyro|", alpha=0.8)

    axes[0].set_ylabel("|acc| (m/s²)")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"{stage_dir.parent.name}/{stage_dir.name} — sensor comparison")

    if not norm_only:
        axes[1].set_ylabel("|gyro|")
        axes[1].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Saved comparison plot → %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_comparison")
    parser.add_argument("stage_ref", help="<recording>/<stage> or section folder name")
    parser.add_argument("--norm", action="store_true", help="Plot norms only")
    parser.add_argument("-o", "--output", help="Output PNG path (auto-derived if omitted)")
    args = parser.parse_args(argv)

    stage_dir = _resolve_stage_dir(args.stage_ref)
    out = Path(args.output) if args.output else None
    try:
        saved = plot_comparison_data(stage_dir, norm_only=args.norm, output_path=out)
        print(f"Saved → {saved}")
    except Exception as exc:
        log.error("Failed to plot comparison: %s", exc)


if __name__ == "__main__":
    main()
