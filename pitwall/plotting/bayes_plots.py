"""Phase 16 plots: posterior car and driver strengths with their uncertainty."""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .colours import team_colour


def strength_intervals(strengths: pd.DataFrame, kind: str = "car", top: int | None = None,
                       ax: Axes | None = None) -> Axes:
    """Posterior mean and interval per team (kind='car') or driver (kind='driver').

    Values are % gaps to the typical car: LOWER = faster, so the fastest is on top.
    Rookies are drawn hollow: their wider intervals are the model saying "not sure yet".
    """
    rows = strengths[strengths["Kind"] == kind].sort_values("Mean", ascending=False)
    if top is not None:
        rows = rows.tail(top)
    if ax is None:
        fig = plt.figure(figsize=(7, 0.3 * len(rows) + 1.2))
        ax = fig.add_subplot()
    y = np.arange(len(rows))
    colours = [team_colour(i) if kind == "car" else "#4c72b0" for i in rows["Id"]]
    ax.hlines(y, rows["Low"], rows["High"], color=colours, linewidth=3, alpha=0.6)
    rookie = rows["Rookie"].to_numpy(bool)
    ax.scatter(rows["Mean"][~rookie], y[~rookie], color=np.array(colours, dtype=object)[~rookie],
               zorder=3, label="estimate")
    if rookie.any():
        ax.scatter(rows["Mean"][rookie], y[rookie], facecolors="none", edgecolors="black",
                   zorder=3, label="rookie")
    ax.set_yticks(y, rows["Id"])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("% gap to the typical car (lower = faster), with interval")
    ax.set_title("Car pace" if kind == "car" else "Driver skill")
    ax.legend(frameon=False, fontsize=8)
    return ax
