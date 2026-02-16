from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt

from common.csv_schema import load_dataframe

if TYPE_CHECKING:
    import pandas as pd


def plot_dataframe(df: "pd.DataFrame", title: str, output: Path | None) -> None:
    """Plot one processed IMU CSV stream in acceleration/gyro/magnetometer panels."""
    t = df["timestamp"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    # Acceleration: plot only samples where each component is not NaN
    mask_a = (~df["ax"].isna()) & (~df["ay"].isna()) & (~df["az"].isna())
    axes[0].plot(t[mask_a], df["ax"][mask_a], label="ax")
    axes[0].plot(t[mask_a], df["ay"][mask_a], label="ay")
    axes[0].plot(t[mask_a], df["az"][mask_a], label="az")
    axes[0].set_title("Acceleration")
    axes[0].set_ylabel("m/s^2")
    axes[0].legend(loc="upper right")

    # Gyroscope
    mask_g = (~df["gx"].isna()) & (~df["gy"].isna()) & (~df["gz"].isna())
    axes[1].plot(t[mask_g], df["gx"][mask_g], label="gx")
    axes[1].plot(t[mask_g], df["gy"][mask_g], label="gy")
    axes[1].plot(t[mask_g], df["gz"][mask_g], label="gz")
    axes[1].set_title("Gyroscope")
    axes[1].set_ylabel("deg/s")
    axes[1].legend(loc="upper right")

    # Magnetometer
    mask_m = (~df["mx"].isna()) & (~df["my"].isna()) & (~df["mz"].isna())
    axes[2].plot(t[mask_m], df["mx"][mask_m], label="mx")
    axes[2].plot(t[mask_m], df["my"][mask_m], label="my")
    axes[2].plot(t[mask_m], df["mz"][mask_m], label="mz")
    axes[2].set_title("Magnetometer")
    axes[2].set_ylabel("uT")
    axes[2].set_xlabel("Timestamp")
    axes[2].legend(loc="upper right")

    fig.suptitle(title)
    fig.savefig(output, dpi=150)

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for single-stream plotting."""
    parser = argparse.ArgumentParser(
        prog="python -m plot.plot_device",
        description="Plot one processed IMU CSV file.",
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    csv_path = args.csv_file
    df = load_dataframe(csv_path)
    png_path = args.output or csv_path.with_suffix(".png")
    plot_title = args.title or csv_path.name
    plot_dataframe(df, title=plot_title, output=png_path)


if __name__ == "__main__":
    main()
