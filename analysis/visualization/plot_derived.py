"""Plot physically interpretable derived dual-IMU signals for a section.

Run from ``analysis/``:

    uv run python -m visualization.plot_derived <section>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from derived.compute import derive_section_signals


def _resolve_section_path(section_arg: str) -> Path:
    p = Path(section_arg)
    if p.is_dir():
        return p.resolve()
    from common.paths import sections_root

    cand = sections_root() / section_arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(f"Section not found: {section_arg}")


def plot_derived_signals(
    section_path: Path,
    *,
    orientation_variant: str = "complementary_orientation",
    recompute: bool = False,
) -> Path:
    """Create a section-level overview figure for derived signals."""
    section_path = _resolve_section_path(str(section_path))
    ddir = section_path / "derived"

    needed = [
        ddir / "sporsa_signals.csv",
        ddir / "arduino_signals.csv",
        ddir / "cross_sensor_signals.csv",
        ddir / "derived_signals_meta.json",
    ]
    if recompute or not all(p.exists() for p in needed):
        derive_section_signals(section_path, orientation_variant=orientation_variant, include_normalized=True)

    bike = pd.read_csv(ddir / "sporsa_signals.csv")
    rider = pd.read_csv(ddir / "arduino_signals.csv")
    cross = pd.read_csv(ddir / "cross_sensor_signals.csv")
    meta = json.loads((ddir / "derived_signals_meta.json").read_text(encoding="utf-8"))

    fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axs[0].plot(bike["time_s"], bike["acc_vertical_m_s2"], label="Bike vertical", lw=1.0, color="#e05c44")
    axs[0].plot(rider["time_s"], rider["acc_vertical_m_s2"], label="Rider vertical", lw=1.0, color="#4c9be8", alpha=0.9)
    axs[0].set_ylabel("m/s²")
    axs[0].set_title("Gravity-compensated vertical acceleration")
    axs[0].grid(alpha=0.3)
    axs[0].legend(loc="upper right")

    axs[1].plot(cross["time_s"], cross["residual_vertical_m_s2"], lw=1.0, color="#2c7fb8", label="Vertical residual")
    if "residual_longitudinal_m_s2" in cross.columns:
        axs[1].plot(cross["time_s"], cross["residual_longitudinal_m_s2"], lw=0.9, color="#fd8d3c", alpha=0.8, label="Longitudinal residual")
    if "residual_lateral_m_s2" in cross.columns:
        axs[1].plot(cross["time_s"], cross["residual_lateral_m_s2"], lw=0.9, color="#31a354", alpha=0.8, label="Lateral residual")
    axs[1].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    axs[1].set_ylabel("m/s²")
    axs[1].set_title("Rider-minus-bike residual motion")
    axs[1].grid(alpha=0.3)
    axs[1].legend(loc="upper right")

    axs[2].plot(cross["time_s"], cross["shock_transmission_gain"], color="#d95f0e", lw=1.0)
    axs[2].axhline(1.0, color="k", lw=0.8, alpha=0.4, ls="--")
    axs[2].set_ylabel("gain")
    axs[2].set_title("Shock transmission (bike → rider, vertical RMS ratio)")
    axs[2].grid(alpha=0.3)

    axs[3].plot(bike["time_s"], bike["tilt_roll_rate_rad_s"], color="#756bb1", lw=0.9, label="Bike roll-rate")
    axs[3].plot(rider["time_s"], rider["tilt_roll_rate_rad_s"], color="#9e9ac8", lw=0.9, label="Rider roll-rate")
    axs[3].plot(bike["time_s"], bike["tilt_pitch_rate_rad_s"], color="#006d2c", lw=0.9, label="Bike pitch-rate")
    axs[3].plot(rider["time_s"], rider["tilt_pitch_rate_rad_s"], color="#74c476", lw=0.9, label="Rider pitch-rate")
    axs[3].set_ylabel("rad/s")
    axs[3].set_xlabel("Section time (s)")
    axs[3].set_title("Tilt-rate derivatives from orientation")
    axs[3].grid(alpha=0.3)
    axs[3].legend(loc="upper right", ncol=2, fontsize=9)

    fa = bool(meta.get("full_horizontal_alignment_available", False))
    note = (
        f"alignment={meta.get('frame_alignment', 'unknown')} | "
        f"full_horizontal_alignment_available={fa}"
    )
    fig.suptitle(f"Derived signal overview — {section_path.name}\n{note}", fontsize=12)

    out_path = ddir / "derived_signals_overview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m visualization.plot_derived")
    ap.add_argument("section", help="Section folder path or section name")
    ap.add_argument("--orientation-variant", default="complementary_orientation")
    ap.add_argument("--recompute", action="store_true", help="Recompute derived signals before plotting")
    args = ap.parse_args()

    out = plot_derived_signals(
        Path(args.section),
        orientation_variant=args.orientation_variant,
        recompute=args.recompute,
    )
    print(str(out))


if __name__ == "__main__":
    main()
