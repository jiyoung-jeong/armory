"""Shared matplotlib style for paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette — accessible, distinct, works in grayscale
COLORS = [
    "#37A3D2",  # blue
    "#F94144",  # red-orange
    "#F7B801",  # yellow
    "#8073AC",  # purple
    "#F4A582",  # light salmon
    "#92C5DE",  # light blue
]

# Model colors
MODEL_COLORS: dict[str, str] = {
    "pi05": "#ffd23c",
    "gr00t": "#76B900",
}

# Model display names
MODEL_LABELS: dict[str, str] = {
    "pi05": r"$\pi_{0.5}$",
    "gr00t": "GR00T-N1.7",
}

# Hardware display names
HW_LABELS: dict[str, str] = {
    "l40s": "L40S",
}


def apply() -> None:
    """Apply the paper figure style globally."""
    mpl.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            # Grid
            "axes.grid": True,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            # Lines & markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Legend
            "legend.frameon": False,
            "legend.borderpad": 0.4,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Ticks
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def yonly_grid(ax: plt.Axes) -> None:
    """Show grid lines on y-axis only (suppress x grid lines)."""
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)


def color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def model_color(model: str, idx: int = 0) -> str:
    return MODEL_COLORS.get(model, color(idx))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def hw_label(hw: str) -> str:
    return HW_LABELS.get(hw, hw)
