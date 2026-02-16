"""Plotting utilities for comparing two IMU streams side-by-side."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe

if TYPE_CHECKING:
    pass


def _prepare_time(df: "pd.DataFrame", *, relative: bool) -> "pd.Series":
    """Return absolute timestamps or relative timestamps (zero at stream start)."""
    t = df["timestamp"].copy()
    if relative and not t.empty:
        t = t - float(t.iloc[0])
    return t


def _plot_triplet(
    ax: Any,
    t_a: "pd.Series",
    df_a: "pd.DataFrame",
    t_b: "pd.Series",
    df_b: "pd.DataFrame",
    *,
    cols: tuple[str, str, str],
    unit: str,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    """Plot x/y/z components in one panel for one sensor type."""
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    axis_names = ("x", "y", "z")

    for col, axis_name in zip(cols, axis_names):
        mask_a = ~df_a[col].isna()
        mask_b = ~df_b[col].isna()

        ax.plot(
            t_a[mask_a],
            df_a[col][mask_a],
            color=colors[axis_name],
            linewidth=1.0,
            alpha=0.8,
            label=f"{label_a} {col}",
        )
        ax.plot(
            t_b[mask_b],
            df_b[col][mask_b],
            color=colors[axis_name],
            linewidth=1.0,
            alpha=0.9,
            linestyle="--",
            label=f"{label_b} {col}",
        )

    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)


def _plot_single_axis(
    ax: Any,
    t_a: "pd.Series",
    df_a: "pd.DataFrame",
    t_b: "pd.Series",
    df_b: "pd.DataFrame",
    *,
    col: str,
    unit: str,
    label_a: str,
    label_b: str,
    title: str,
    color: str,
) -> None:
    """Plot one axis component in a dedicated panel."""
    mask_a = ~df_a[col].isna()
    mask_b = ~df_b[col].isna()

    ax.plot(
        t_a[mask_a],
        df_a[col][mask_a],
        color=color,
        linewidth=1.0,
        alpha=0.8,
        label=f"{label_a} {col}",
    )
    ax.plot(
        t_b[mask_b],
        df_b[col][mask_b],
        color=color,
        linewidth=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"{label_b} {col}",
    )
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def _vector_magnitude(df: "pd.DataFrame", cols: tuple[str, str, str]) -> "pd.Series":
    """Compute vector magnitude and keep NaN when any component is missing."""
    valid = (~df[cols[0]].isna()) & (~df[cols[1]].isna()) & (~df[cols[2]].isna())
    mag = np.full(len(df), np.nan, dtype=float)
    xyz = df.loc[valid, list(cols)].to_numpy(dtype=float)
    mag[valid.to_numpy()] = np.sqrt(np.sum(xyz * xyz, axis=1))
    return pd.Series(mag, index=df.index)


def _plot_magnitude(
    ax: Any,
    t_a: "pd.Series",
    df_a: "pd.DataFrame",
    t_b: "pd.Series",
    df_b: "pd.DataFrame",
    *,
    cols: tuple[str, str, str],
    unit: str,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    """Plot vector magnitudes for one sensor type."""
    mag_a = _vector_magnitude(df_a, cols)
    mag_b = _vector_magnitude(df_b, cols)

    mask_a = ~mag_a.isna()
    mask_b = ~mag_b.isna()

    ax.plot(t_a[mask_a], mag_a[mask_a], color="tab:blue", linewidth=1.0, alpha=0.85, label=label_a)
    ax.plot(
        t_b[mask_b],
        mag_b[mask_b],
        color="tab:orange",
        linewidth=1.0,
        alpha=0.9,
        linestyle="--",
        label=label_b,
    )
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def plot_stream_comparison(
    df_a: "pd.DataFrame",
    df_b: "pd.DataFrame",
    *,
    label_a: str,
    label_b: str,
    title: str,
    output: Path,
    relative_time: bool = False,
    split_axes: bool = False,
    magnitudes: bool = False,
) -> None:
    """
    Overlay two IMU streams for direct visual comparison.

    When `split_axes=True`, every x/y/z component gets its own panel to reduce clutter.
    """
    t_a = _prepare_time(df_a, relative=relative_time)
    t_b = _prepare_time(df_b, relative=relative_time)

    if magnitudes:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True, sharex=True)
        _plot_magnitude(
            axes[0],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("ax", "ay", "az"),
            unit="m/s^2",
            label_a=label_a,
            label_b=label_b,
            title="Acceleration Magnitude",
        )
        _plot_magnitude(
            axes[1],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("gx", "gy", "gz"),
            unit="deg/s",
            label_a=label_a,
            label_b=label_b,
            title="Gyroscope Magnitude",
        )
        _plot_magnitude(
            axes[2],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("mx", "my", "mz"),
            unit="uT",
            label_a=label_a,
            label_b=label_b,
            title="Magnetometer Magnitude",
        )
        x_label = "Timestamp (ms from start)" if relative_time else "Timestamp (ms)"
        axes[-1].set_xlabel(x_label)
    elif not split_axes:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

        _plot_triplet(
            axes[0],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("ax", "ay", "az"),
            unit="m/s^2",
            label_a=label_a,
            label_b=label_b,
            title="Acceleration",
        )
        _plot_triplet(
            axes[1],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("gx", "gy", "gz"),
            unit="deg/s",
            label_a=label_a,
            label_b=label_b,
            title="Gyroscope",
        )
        _plot_triplet(
            axes[2],
            t_a,
            df_a,
            t_b,
            df_b,
            cols=("mx", "my", "mz"),
            unit="uT",
            label_a=label_a,
            label_b=label_b,
            title="Magnetometer",
        )

        x_label = "Timestamp (ms from start)" if relative_time else "Timestamp (ms)"
        axes[2].set_xlabel(x_label)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    else:
        fig, axes = plt.subplots(9, 1, figsize=(14, 22), constrained_layout=True, sharex=True)
        specs = [
            ("ax", "m/s^2", "Acceleration X", "tab:red"),
            ("ay", "m/s^2", "Acceleration Y", "tab:green"),
            ("az", "m/s^2", "Acceleration Z", "tab:blue"),
            ("gx", "deg/s", "Gyroscope X", "tab:red"),
            ("gy", "deg/s", "Gyroscope Y", "tab:green"),
            ("gz", "deg/s", "Gyroscope Z", "tab:blue"),
            ("mx", "uT", "Magnetometer X", "tab:red"),
            ("my", "uT", "Magnetometer Y", "tab:green"),
            ("mz", "uT", "Magnetometer Z", "tab:blue"),
        ]
        for ax, (col, unit, panel_title, color) in zip(axes, specs):
            _plot_single_axis(
                ax,
                t_a,
                df_a,
                t_b,
                df_b,
                col=col,
                unit=unit,
                label_a=label_a,
                label_b=label_b,
                title=panel_title,
                color=color,
            )
        x_label = "Timestamp (ms from start)" if relative_time else "Timestamp (ms)"
        axes[-1].set_xlabel(x_label)

    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for comparing two CSV streams."""
    parser = argparse.ArgumentParser(
        prog="python -m plot.compare_streams",
        description="Plot two processed IMU CSV streams in one figure.",
    )
    parser.add_argument("csv_a", type=Path, help="First stream CSV.")
    parser.add_argument("csv_b", type=Path, help="Second stream CSV.")
    parser.add_argument("--label-a", default=None, help="Legend label for csv_a.")
    parser.add_argument("--label-b", default=None, help="Legend label for csv_b.")
    parser.add_argument("--title", default=None, help="Figure title.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--relative-time",
        action="store_true",
        help="Use time since each stream start on the x-axis.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--split-axes",
        action="store_true",
        help="Use dedicated panels for x/y/z components.",
    )
    mode_group.add_argument(
        "--magnitudes",
        action="store_true",
        help="Plot |x,y,z| magnitudes instead of individual axis components.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    df_a = load_dataframe(args.csv_a)
    df_b = load_dataframe(args.csv_b)

    label_a = args.label_a or args.csv_a.stem
    label_b = args.label_b or args.csv_b.stem
    output = args.output or args.csv_a.with_name(f"{args.csv_a.stem}__vs__{args.csv_b.stem}.png")
    title = args.title or f"{label_a} vs {label_b}"

    plot_stream_comparison(
        df_a,
        df_b,
        label_a=label_a,
        label_b=label_b,
        title=title,
        output=output,
        relative_time=args.relative_time,
        split_axes=args.split_axes,
        magnitudes=args.magnitudes,
    )

    print(f"comparison_plot={output}")


if __name__ == "__main__":
    main()
