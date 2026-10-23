"""Shared matplotlib styling for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (blue, orange, aqua), then text and grid inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "legend.frameon": False,
        }
    )


def save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def hbar(ax, labels, values, color=SERIES[0], fmt="{:.1%}", reference=None, reference_label=None):
    """Horizontal bars, largest at the top, with a value label at each bar end."""
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.6)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=MUTED)
    if reference is not None:
        ax.axvline(reference, color=MUTED, linestyle="--", linewidth=1)
        if reference_label:
            ax.text(reference, -0.8, f" {reference_label}", color=MUTED, fontsize=8, va="bottom")
    ax.margins(x=0.15)
