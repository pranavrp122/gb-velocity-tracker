"""Shared matplotlib styling and the two standard plots.

Series colours follow a fixed categorical order (blue, orange, aqua, yellow)
so a temperature or alloy keeps its colour across every figure. Text stays in
neutral ink; colour only marks identity.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .velocity import KB_EV_PER_K  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
MARKER_RED = "#e34948"


def style_axes(ax: plt.Axes) -> None:
    """Recessive grid and spines, neutral text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def new_figure(nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (7, 4.5),
               **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """Figure on the house surface colour."""
    return plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kwargs)


def plot_distance_fit(ax: plt.Axes, time_ps: np.ndarray, distance_A: np.ndarray,
                      slope: float, intercept: float, color: str, label: str) -> None:
    """Measured distances as points plus the fitted line in the same colour."""
    ax.plot(time_ps, distance_A, "o", ms=6, color=color, mec=SURFACE, mew=1.2, label=label)
    t = np.array([np.nanmin(time_ps), np.nanmax(time_ps)])
    ax.plot(t, slope * t + intercept, "-", lw=2, color=color)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("boundary distance (angstrom)")


def plot_arrhenius(ax: plt.Axes, temps_K: np.ndarray, velocities: np.ndarray,
                   ea_eV: float, prefactor: float, color: str, label: str) -> None:
    """ln v against 1000/T with the fitted Arrhenius line."""
    inv = 1000.0 / np.asarray(temps_K, float)
    ax.plot(inv, np.log(velocities), "o", ms=8, color=color, mec=SURFACE, mew=1.2,
            label=f"{label}  (Ea = {ea_eV:.3f} eV)")
    x = np.linspace(inv.min(), inv.max(), 50)
    ax.plot(x, np.log(prefactor) - ea_eV / KB_EV_PER_K * x / 1000.0, "-", lw=2, color=color)
    ax.set_xlabel("1000 / T  (1/K)")
    ax.set_ylabel("ln v  (v in angstrom/ps)")


def save(fig: plt.Figure, path) -> None:
    """Tight layout, write a PNG, release the figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
