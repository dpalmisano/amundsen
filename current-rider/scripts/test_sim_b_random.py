"""
Sub-Step 4.2 — Sim B sanity check: random actions in AUVComplexEnv.

Runs 3 episodes with random actions, plots per-episode trajectories with
the spatially-varying gyre current field overlaid as a quiver background.
Saves to outputs/sim_b_random_actions.png.
"""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_complex import AUVComplexEnv, ARENA_SIZE
from current_rider.utils.visualise import plot_episodes_grid

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
SAVE_PATH = os.path.join(BASE_DIR, "outputs", "sim_b_random_actions.png")

NUM_EPISODES = 3


# ── Pre-compute the gyre current field for the quiver overlay ─────────────────
# Same grid used for all panels — the field is a fixed function of position.
def _make_gyre_quiver_data(arena_size: float, n_grid: int = 10):
    half = arena_size / 2.0
    xs = np.linspace(-half + 0.5, half - 0.5, n_grid)
    ys = np.linspace(-half + 0.5, half - 0.5, n_grid)
    X, Y = np.meshgrid(xs, ys)

    env = AUVComplexEnv()
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(n_grid):
        for j in range(n_grid):
            c = env._get_current(np.array([X[i, j], Y[i, j]], dtype=np.float32))
            U[i, j] = c[0]
            V[i, j] = c[1]
    env.close()
    return X, Y, U, V


def _add_gyre_field(ax, X, Y, U, V):
    """Overlay the gyre quiver field onto an existing Axes."""
    mag = np.sqrt(U**2 + V**2)
    ax.quiver(
        X, Y, U, V,
        mag,                         # colour by magnitude
        cmap="Blues",
        alpha=0.55,
        scale=4.0,                   # tune arrow length visually
        width=0.003,
        headwidth=4, headlength=5,
        zorder=1,
    )


# ── Run episodes ──────────────────────────────────────────────────────────────
env      = AUVComplexEnv()
episodes = []

print("Sim B — Random Actions")
print("=" * 60)

for ep in range(NUM_EPISODES):
    obs, _ = env.reset(seed=ep * 13)
    start  = env._pos.copy()
    goal   = env._goal.copy()

    positions    = [start.copy()]
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env._pos.copy())
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nEpisode {ep + 1}")
    print(f"  steps        = {info['steps']}")
    print(f"  final_dist   = {info['distance']:.2f} m")
    print(f"  goal_reached = {info['goal_reached']}")
    print(f"  total_reward = {total_reward:.2f}")

    episodes.append({
        "positions": np.array(positions),
        "start":     start,
        "goal":      goal,
        "label":     f"Ep {ep+1} ({'✓' if info['goal_reached'] else '✗'})",
    })

env.close()
print()

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
fig.suptitle("Sim B — Random Actions (Complex Physics)", fontsize=13, fontweight="bold")

# Add a note explaining Sim B effects
fig.text(
    0.5, 0.01,
    "Sim B effects: spatially-varying gyre current (quiver) · quadratic drag · "
    "2-step thruster delay · position noise · rotational inertia",
    ha="center", fontsize=8, color="#555555",
)

# One row of panels
axes = fig.subplots(1, NUM_EPISODES)
plot_episodes_grid(episodes, ARENA_SIZE, axes=list(axes))

# ── Overlay gyre field onto every panel ───────────────────────────────────────
X, Y, U, V = _make_gyre_quiver_data(ARENA_SIZE)
for ax in axes:
    _add_gyre_field(ax, X, Y, U, V)

fig.tight_layout(rect=[0, 0.04, 1, 1])   # leave room for the bottom note

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved → {SAVE_PATH}")
