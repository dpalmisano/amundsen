"""Trajectory visualisation utilities for Current Rider environments."""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _draw_episode_panel(ax, ep: dict, arena_size: float, colour: str) -> None:
    """
    Render a single episode into an existing Axes object.

    Draws: arena boundary, current flow field, fading trajectory,
    direction arrow, start/goal markers, and a physics annotation box.
    """
    half = arena_size / 2.0
    physics = ep.get("physics")

    # ── Arena boundary ────────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=1.5, edgecolor="black", facecolor="#f5f5f5", zorder=0,
    ))

    # ── Current flow field ────────────────────────────────────────────────────
    if physics is not None:
        cx, cy = physics["current"][0], physics["current"][1]
        mag = math.hypot(cx, cy)
        if mag > 1e-6:
            grid_step = arena_size / 5.0
            xs = np.arange(-half + grid_step / 2, half, grid_step)
            ys = np.arange(-half + grid_step / 2, half, grid_step)
            X, Y = np.meshgrid(xs, ys)
            # Arrow length proportional to magnitude; max ~12% of arena size
            arrow_len = (mag / 0.5) * arena_size * 0.12
            u = (cx / mag) * arrow_len
            v = (cy / mag) * arrow_len
            ax.quiver(
                X.ravel(), Y.ravel(),
                np.full(X.size, u), np.full(X.size, v),
                color="#aec6e8", alpha=0.6,
                scale=1, scale_units="xy",
                width=0.004, headwidth=4, headlength=5,
                zorder=1,
            )

    # ── Trajectory ────────────────────────────────────────────────────────────
    pos = np.asarray(ep["positions"])
    if len(pos) > 1:
        points = pos[:, np.newaxis, :]
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.3, 1.0, max(len(segs), 1))
        rgb    = plt.matplotlib.colors.to_rgb(colour)
        lc = LineCollection(segs, colors=[rgb + (a,) for a in alphas],
                            linewidths=1.5, zorder=2)
        ax.add_collection(lc)

        tip  = pos[-1]
        base = pos[max(0, len(pos) - max(2, len(pos) // 10))]
        dx, dy = tip - base
        if math.hypot(dx, dy) > 1e-6:
            ax.annotate("", xy=tip, xytext=base,
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
                        zorder=3)

    # ── Start & goal markers ──────────────────────────────────────────────────
    ax.plot(*ep["start"], "o", color=colour, markersize=8, zorder=4,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(*ep["goal"],  "*", color=colour, markersize=13, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)

    # ── Physics annotation ────────────────────────────────────────────────────
    if physics is not None:
        cx, cy = physics["current"][0], physics["current"][1]
        lines = [
            f"mass  = {physics['mass']:.1f} kg",
            f"drag  = {physics['drag_coeff']:.2f}",
            f"curr  = ({cx:+.2f}, {cy:+.2f})",
            f"noise = {physics['thrust_noise_scale']*100:.1f}%",
            f"gps σ = {physics['goal_noise_std']:.2f} m",
        ]
        ax.text(
            -half + 0.3, -half + 0.3, "\n".join(lines),
            fontsize=6.5, family="monospace", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.75, edgecolor="none"),
            zorder=5,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-half - 0.5, half + 0.5)
    ax.set_ylim(-half - 0.5, half + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_title(ep.get("label", ""), fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=7)


def plot_episodes_grid(
    episodes: list[dict],
    arena_size: float,
    title: str = "AUV Episodes",
    n_cols: int = 3,
    save_path: str | None = None,
    axes: list | None = None,
) -> plt.Figure | None:
    """
    Plot one panel per episode showing trajectory, flow field, and physics.

    Each entry in `episodes` must contain:
        "positions" : (N, 2) array of (x, y) at each step
        "start"     : (2,) array
        "goal"      : (2,) array
        "label"     : str — panel subtitle
        "colour"    : str (optional)
        "physics"   : dict (optional) — mass, drag_coeff, current,
                      thrust_noise_scale, goal_noise_std

    Args:
        axes: if provided, render into this flat list of Axes instead of
              creating a new figure. Useful for embedding into a larger
              figure via subfigures. `save_path` and the return value are
              ignored when axes is given.
    """
    if axes is not None:
        # Render into caller-supplied axes — no figure creation or saving
        for idx, ep in enumerate(episodes):
            colour = ep.get("colour", _PALETTE[idx % len(_PALETTE)])
            _draw_episode_panel(axes[idx], ep, arena_size, colour)
        return None

    # ── Standalone figure ─────────────────────────────────────────────────────
    n      = len(episodes)
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, ax_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ep in enumerate(episodes):
        row, col = divmod(idx, n_cols)
        colour = ep.get("colour", _PALETTE[idx % len(_PALETTE)])
        _draw_episode_panel(ax_grid[row][col], ep, arena_size, colour)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax_grid[row][col].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")

    return fig


def plot_trajectories(
    trajectories: list[dict],
    arena_size: float,
    title: str = "AUV Trajectories",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot all trajectories on a single shared arena (used in the 2×2
    comparison figure where space is tight).

    Each entry in `trajectories` must contain:
        "positions" : (N, 2) array
        "start"     : (2,) array
        "goal"      : (2,) array
        "label"     : str
        "colour"    : str (optional)
    """
    half = arena_size / 2.0
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.add_patch(plt.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=2, edgecolor="black", facecolor="#f5f5f5", zorder=0,
    ))

    for i, ep in enumerate(trajectories):
        colour = ep.get("colour", _PALETTE[i % len(_PALETTE)])
        pos   = np.asarray(ep["positions"])
        start = np.asarray(ep["start"])
        goal  = np.asarray(ep["goal"])

        points = pos[:, np.newaxis, :]
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.3, 1.0, max(len(segs), 1))
        rgb    = plt.matplotlib.colors.to_rgb(colour)
        lc = LineCollection(segs, colors=[rgb + (a,) for a in alphas],
                            linewidths=1.5, zorder=2)
        ax.add_collection(lc)

        if len(pos) >= 2:
            tip  = pos[-1]
            base = pos[max(0, len(pos) - max(2, len(pos) // 10))]
            dx, dy = tip - base
            if np.hypot(dx, dy) > 1e-6:
                ax.annotate("", xy=tip, xytext=base,
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
                            zorder=3)

        ax.plot(*start, "o", color=colour, markersize=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5,
                label=ep.get("label", f"Episode {i+1}"))
        ax.plot(*goal, "*", color=colour, markersize=14, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)

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
