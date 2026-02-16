from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.csv_schema import load_dataframe

if TYPE_CHECKING:
    pass


def _vector_magnitude(df: "pd.DataFrame", cols: tuple[str, str, str]) -> "pd.Series":
    """Compute vector magnitude and keep NaN when any component is missing."""
    valid = (~df[cols[0]].isna()) & (~df[cols[1]].isna()) & (~df[cols[2]].isna())
    mag = np.full(len(df), np.nan, dtype=float)
    xyz = df.loc[valid, list(cols)].to_numpy(dtype=float)
    mag[valid.to_numpy()] = np.sqrt(np.sum(xyz * xyz, axis=1))
    return pd.Series(mag, index=df.index)


def plot_dataframe(
    df: "pd.DataFrame",
    title: str,
    output: Path | None,
    *,
    magnitudes: bool = False,
) -> None:
    """Plot one processed IMU stream using components or vector magnitudes."""
    t = df["timestamp"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    if magnitudes:
        acc_mag = _vector_magnitude(df, ("ax", "ay", "az"))
        gyro_mag = _vector_magnitude(df, ("gx", "gy", "gz"))
        mag_mag = _vector_magnitude(df, ("mx", "my", "mz"))

        axes[0].plot(t[~acc_mag.isna()], acc_mag[~acc_mag.isna()], label="|acc|")
        axes[0].set_title("Acceleration Magnitude")
        axes[0].set_ylabel("m/s^2")
        axes[0].legend(loc="upper right")

        axes[1].plot(t[~gyro_mag.isna()], gyro_mag[~gyro_mag.isna()], label="|gyro|")
        axes[1].set_title("Gyroscope Magnitude")
        axes[1].set_ylabel("deg/s")
        axes[1].legend(loc="upper right")

        axes[2].plot(t[~mag_mag.isna()], mag_mag[~mag_mag.isna()], label="|mag|")
        axes[2].set_title("Magnetometer Magnitude")
        axes[2].set_ylabel("uT")
    else:
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
    parser.add_argument(
        "--magnitudes",
        action="store_true",
        help="Plot |x,y,z| magnitudes instead of separate axis components.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    csv_path = args.csv_file
    df = load_dataframe(csv_path)
    png_path = args.output or csv_path.with_suffix(".png")
    plot_title = args.title or csv_path.name
    plot_dataframe(df, title=plot_title, output=png_path, magnitudes=args.magnitudes)


if __name__ == "__main__":
    main()
