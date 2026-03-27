"""Validation plots/checks for derived signals across one or more sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_single(section_path: Path) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ddir = section_path / "derived"
    bike = pd.read_csv(ddir / "sporsa_signals.csv")
    rider = pd.read_csv(ddir / "arduino_signals.csv")
    cross = pd.read_csv(ddir / "cross_sensor_signals.csv")
    metrics = {
        "vertical_residual_std_m_s2": float(np.nanstd(cross["residual_vertical_m_s2"])),
        "median_shock_transmission_gain": float(np.nanmedian(cross["shock_transmission_gain"])),
        "p95_shock_transmission_gain": float(np.nanpercentile(cross["shock_transmission_gain"], 95)),
    }
    return metrics, bike, rider, cross


def validate_sections_derived(section_paths: list[Path]) -> dict[str, dict[str, float]]:
    if not section_paths:
        raise ValueError("Provide at least one section")

    summaries: dict[str, dict[str, float]] = {}
    sample_sections = section_paths[:3]

    fig, axs = plt.subplots(len(sample_sections), 3, figsize=(15, 4 * len(sample_sections)), sharex=False)
    if len(sample_sections) == 1:
        axs = np.array([axs])

    for row, section_path in enumerate(sample_sections):
        section_path = Path(section_path)
        metrics, bike, rider, cross = _validate_single(section_path)
        summaries[section_path.name] = metrics

        axs[row, 0].plot(bike["time_s"], bike["acc_vertical_m_s2"], lw=0.9, label="Bike")
        axs[row, 0].plot(rider["time_s"], rider["acc_vertical_m_s2"], lw=0.9, alpha=0.85, label="Rider")
        axs[row, 0].set_title(f"{section_path.name}: vertical acc")
        axs[row, 0].grid(alpha=0.3)
        if row == 0:
            axs[row, 0].legend()

        axs[row, 1].plot(cross["time_s"], cross["residual_vertical_m_s2"], color="#2c7fb8", lw=0.9)
        axs[row, 1].axhline(0.0, color="k", lw=0.8, alpha=0.4)
        axs[row, 1].set_title("Vertical residual (rider-bike)")
        axs[row, 1].grid(alpha=0.3)

        axs[row, 2].plot(cross["time_s"], cross["shock_transmission_gain"], color="#d95f0e", lw=0.9)
        axs[row, 2].axhline(1.0, color="k", lw=0.8, alpha=0.4, ls="--")
        axs[row, 2].set_title("Shock transmission gain")
        axs[row, 2].grid(alpha=0.3)

        for c in range(3):
            axs[row, c].set_xlabel("time (s)")

    out_dir = section_paths[0] / "derived"
    fig.tight_layout()
    fig.savefig(out_dir / "derived_validation_overview.png", dpi=160)
    plt.close(fig)

    (out_dir / "derived_validation_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m derived.validate")
    ap.add_argument("sections", nargs="+", help="One or more section folders")
    args = ap.parse_args()
    summaries = validate_sections_derived([Path(s) for s in args.sections])
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
