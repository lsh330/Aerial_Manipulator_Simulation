"""Unified plotting style for publication-quality figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl


def apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "text.usetex": False,  # Set True if LaTeX available
        "font.family": "serif",
    })


# Standard color palette
COLORS = {
    "x": "#1f77b4",       # blue
    "y": "#ff7f0e",       # orange
    "z": "#2ca02c",       # green
    "roll": "#d62728",    # red
    "pitch": "#9467bd",   # purple
    "yaw": "#8c564b",     # brown
    "q1": "#e377c2",      # pink
    "q2": "#7f7f7f",      # gray
    "reference": "#333333",  # dark gray (dashed)
    "motor1": "#1f77b4",
    "motor2": "#ff7f0e",
    "motor3": "#2ca02c",
    "motor4": "#d62728",
}

LINE_STYLES = {
    "actual": "-",
    "reference": "--",
    "error": ":",
}
