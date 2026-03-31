"""Plot section orientation outputs (yaw/pitch/roll) for one or more sensors."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import project_relative_path, read_csv, resolve_data_dir
from visualization._utils import filter_valid_plot_xy, timestamps_to_relative_seconds

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_ANGLES = ("yaw_deg", "pitch_deg", "roll_deg")
_COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}


def _resolve_orientation_dir(target: str | Path) -> Path:
    base = resolve_data_dir(target)
    if base.name == "orientation":
        return base
    orient = base / "orientation"
    if orient.is_dir():
        return orient
    raise FileNotFoundError(f"No orientation directory found for: {target}")


def _collect_orientation_files(
    orient_dir: Path,
    *,
    sensors: list[str] | None = None,
    variants: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    wanted_sensors = set(sensors or _SENSORS)
    wanted_variants = set(variants) if variants else None

    mapping: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(orient_dir.glob("*.csv")):
        if "__" not in csv_path.stem:
            continue
        sensor, variant = csv_path.stem.split("__", 1)
        if sensor not in wanted_sensors:
            continue
        if wanted_variants is not None and variant not in wanted_variants:
            continue
        mapping.setdefault(variant, {})[sensor] = csv_path
    return mapping


def plot_orientation_stage(
    target: str | Path,
    *,
    sensors: list[str] | None = None,
    variants: list[str] | None = None,
) -> list[Path]:
    """Generate orientation plots for all selected variant/sensor outputs."""
    orient_dir = _resolve_orientation_dir(target)
    file_map = _collect_orientation_files(orient_dir, sensors=sensors, variants=variants)
    if not file_map:
        log.warning("No orientation CSV files matched in %s", orient_dir)
        return []

    out_paths: list[Path] = []
    for variant, sensor_map in sorted(file_map.items()):
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        any_line = False
        for sensor, csv_path in sorted(sensor_map.items()):
            df = read_csv(csv_path)
            if "timestamp" not in df.columns:
                continue
            ts_s = timestamps_to_relative_seconds(df["timestamp"])
            color = _COLORS.get(sensor, None)
            for idx, angle_col in enumerate(_ANGLES):
                if angle_col not in df.columns:
                    continue
                y = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
                x_plot, y_plot = filter_valid_plot_xy(ts_s, y)
                if x_plot.size == 0:
                    continue
                axes[idx].plot(x_plot, y_plot, lw=0.8, alpha=0.9, color=color, label=sensor)
                any_line = True

        for idx, angle_col in enumerate(_ANGLES):
            axes[idx].set_ylabel(angle_col.replace("_deg", " (deg)"))
            handles, labels = axes[idx].get_legend_handles_labels()
            if handles:
                seen: set[str] = set()
                uniq_h = []
                uniq_l = []
                for h, l in zip(handles, labels):
                    if l in seen:
                        continue
                    seen.add(l)
                    uniq_h.append(h)
                    uniq_l.append(l)
                axes[idx].legend(uniq_h, uniq_l, fontsize=8, loc="upper right")
            axes[idx].grid(alpha=0.2, lw=0.4)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{orient_dir.parent.name} — orientation ({variant})")
        fig.tight_layout()

        out_path = orient_dir / f"orientation_{variant}.png"
        if any_line:
            fig.savefig(out_path, dpi=120)
            out_paths.append(out_path)
            log.info("Plot written: %s", project_relative_path(out_path))
        plt.close(fig)

    return out_paths


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="python -m visualization.plot_orientation")
    parser.add_argument("target", help="Section directory reference or orientation dir")
    parser.add_argument(
        "--sensor",
        action="append",
        choices=list(_SENSORS),
        help="Limit to one or more sensors (repeat option).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Limit to one or more orientation variants (repeat option).",
    )
    args = parser.parse_args(argv)

    try:
        paths = plot_orientation_stage(
            args.target,
            sensors=args.sensor,
            variants=args.variant,
        )
    except Exception as exc:
        log.error("Failed to plot orientation: %s", exc)
        return

    if not paths:
        print("No orientation plots generated.")
    for p in paths:
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
