"""Trajectory visualisation utilities for Current Rider environments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


def plot_trajectories(
    trajectories: list[dict],
    arena_size: float,
    title: str = "AUV Trajectories",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot AUV trajectories on a 2-D arena.

    Each entry in `trajectories` must contain:
        "positions" : (N, 2) array of (x, y) at each step
        "goal"      : (2,) array — goal position
        "start"     : (2,) array — starting position
        "label"     : str — legend label

    Args:
        trajectories: list of episode dicts (see above).
        arena_size:   total side length of the square arena (metres).
        title:        figure title.
        save_path:    if given, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    half = arena_size / 2.0

    fig, ax = plt.subplots(figsize=(7, 7))

    # Arena boundary
    arena_rect = plt.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=2, edgecolor="black", facecolor="#f5f5f5", zorder=0,
    )
    ax.add_patch(arena_rect)

    for i, ep in enumerate(trajectories):
        # Allow per-trajectory colour override (e.g. green=success, red=failure)
        colour = ep.get("colour", colours[i % len(colours)])
        pos = np.asarray(ep["positions"])  # (N, 2)
        start = np.asarray(ep["start"])
        goal  = np.asarray(ep["goal"])

        # Trajectory as a line with alpha fading from start (dim) to end (solid)
        points = pos[:, np.newaxis, :]           # (N, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
        alphas = np.linspace(0.3, 1.0, max(len(segs), 1))
        rgb    = plt.matplotlib.colors.to_rgb(colour)
        rgba   = [rgb + (a,) for a in alphas]
        lc = LineCollection(segs, colors=rgba, linewidths=1.5, zorder=2)
        ax.add_collection(lc)

        # Direction arrow near the end of the trajectory
        if len(pos) >= 2:
            tip  = pos[-1]
            base = pos[max(0, len(pos) - max(2, len(pos) // 10))]
            dx, dy = tip - base
            if np.hypot(dx, dy) > 1e-6:
                ax.annotate(
                    "", xy=tip, xytext=base,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
                    zorder=3,
                )

        # Start marker — circle
        ax.plot(*start, "o", color=colour, markersize=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5,
                label=ep.get("label", f"Episode {i+1}"))

        # Goal marker — star
        ax.plot(*goal, "*", color=colour, markersize=14, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)

    # Legend entries for shared markers
    start_patch = mpatches.Patch(color="none", label="● start")
    goal_patch  = mpatches.Patch(color="none", label="★ goal")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [start_patch, goal_patch],
              labels  + ["● start", "★ goal"],
              loc="upper right", fontsize=9)

    ax.set_xlim(-half - 0.5, half + 0.5)
    ax.set_ylim(-half - 0.5, half + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")

    return fig
