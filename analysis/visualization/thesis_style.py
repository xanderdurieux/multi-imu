"""Publication-oriented style defaults for thesis figures (matplotlib + Plotly)."""

from __future__ import annotations

import matplotlib as mpl

# Cohesive palette: colorblind-friendly, works in print (grayscale-friendly hues).
THESIS_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def apply_matplotlib_thesis_style() -> None:
    """Configure matplotlib rcParams for clean, print-ready figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "text.color": "#111111",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#cccccc",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Bitstream Vera Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plotly_template_layout() -> dict:
    """Shared layout dict for Plotly figures (thesis-like white background, crisp fonts)."""
    return {
        "font": {"family": "Georgia, Times New Roman, serif", "size": 13, "color": "#111111"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "#fafafa",
        "margin": {"l": 60, "r": 24, "t": 56, "b": 52},
    }
