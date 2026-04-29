"""
Sub-Step 4.3 — Transfer test: Sim A policies deployed in Sim B.

Both agents (baseline and domain-randomised) are evaluated zero-shot in
AUVComplexEnv — no retraining. Same seeds ensure both face identical episodes.

This is the core result of the project: does domain randomization help
bridge the reality gap?
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_complex import AUVComplexEnv, ARENA_SIZE

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
SAVE_PATH = os.path.join(BASE_DIR, "outputs", "transfer_test.png")

N_EVAL  = 30
N_TRAJ  = 5    # trajectories shown per panel
SEEDS   = list(range(N_EVAL))
GREEN   = "#2ca02c"
RED     = "#d62728"
HALF    = ARENA_SIZE / 2.0

# ── Load models ───────────────────────────────────────────────────────────────
baseline   = PPO.load(os.path.join(BASE_DIR, "models", "baseline_fixed_physics"))
randomised = PPO.load(os.path.join(BASE_DIR, "models", "randomised_physics"))
print("Models loaded.\n")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, seeds: list) -> list[dict]:
    env     = AUVComplexEnv()
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start  = env._pos.copy()
        goal   = env._goal.copy()
        positions    = [start.copy()]
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            positions.append(env._pos.copy())
            total_reward += reward
            if terminated or truncated:
                break
        results.append({
            "positions":    np.array(positions),
            "start":        start,
            "goal":         goal,
            "goal_reached": info["goal_reached"],
            "steps":        info["steps"],
            "final_dist":   info["distance"],
            "total_reward": total_reward,
        })
    env.close()
    return results


print(f"Evaluating baseline in Sim B ({N_EVAL} episodes)…")
res_base = evaluate(baseline, SEEDS)

print(f"Evaluating randomised agent in Sim B ({N_EVAL} episodes)…")
res_rand = evaluate(randomised, SEEDS)


# ── Summary table ─────────────────────────────────────────────────────────────
def summarise(results, label):
    succ = [r for r in results if r["goal_reached"]]
    n    = len(results)
    return {
        "label":      label,
        "n":          n,
        "success":    len(succ),
        "avg_steps":  np.mean([r["steps"] for r in succ]) if succ else float("nan"),
        "avg_dist":   np.mean([r["final_dist"] for r in results]),
        "avg_reward": np.mean([r["total_reward"] for r in results]),
    }

s_b = summarise(res_base, "Baseline (fixed)")
s_r = summarise(res_rand, "Randomised")

print()
print("Transfer Test: Sim A agents → Sim B (complex 'real' environment)")
print("─" * 66)
print(f"{'':24s}  {'Baseline (fixed)':>16s}  {'Randomised':>12s}")
print(f"  Success rate:         {s_b['success']:>4d}/{N_EVAL} "
      f"({100*s_b['success']/N_EVAL:.0f}%)      "
      f"{s_r['success']:>4d}/{N_EVAL} ({100*s_r['success']/N_EVAL:.0f}%)")
print(f"  Avg steps (success):  {s_b['avg_steps']:>16.1f}  {s_r['avg_steps']:>12.1f}")
print(f"  Avg final distance:   {s_b['avg_dist']:>14.2f} m  {s_r['avg_dist']:>10.2f} m")
print(f"  Avg total reward:     {s_b['avg_reward']:>16.2f}  {s_r['avg_reward']:>12.2f}")
print("─" * 66)


# ── Pre-compute gyre quiver field ─────────────────────────────────────────────
def _gyre_field(arena_size: float, n_grid: int = 12):
    half = arena_size / 2.0
    xs = np.linspace(-half + 0.5, half - 0.5, n_grid)
    ys = np.linspace(-half + 0.5, half - 0.5, n_grid)
    X, Y = np.meshgrid(xs, ys)
    env = AUVComplexEnv()
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(n_grid):
        for j in range(n_grid):
            c = env._get_current(np.array([X[i, j], Y[i, j]], dtype=np.float32))
            U[i, j], V[i, j] = c[0], c[1]
    env.close()
    return X, Y, U, V

X, Y, U, V = _gyre_field(ARENA_SIZE)


# ── Trajectory panel helper ───────────────────────────────────────────────────
def _draw_panel(ax, results, n, title):
    """Draw n trajectories onto ax with gyre field background."""

    # Arena
    ax.add_patch(plt.Rectangle((-HALF, -HALF), ARENA_SIZE, ARENA_SIZE,
                               linewidth=1.5, edgecolor="black",
                               facecolor="#f5f5f5", zorder=0))

    # Gyre quiver
    mag = np.sqrt(U**2 + V**2)
    ax.quiver(X, Y, U, V, mag, cmap="Blues", alpha=0.45,
              scale=4.0, width=0.003, headwidth=4, headlength=5, zorder=1)

    palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    for i, r in enumerate(results[:n]):
        colour = GREEN if r["goal_reached"] else RED
        traj   = palette[i % len(palette)]   # distinct line colour per episode
        pos    = r["positions"]

        # Fading trajectory line
        if len(pos) > 1:
            points = pos[:, np.newaxis, :]
            segs   = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.25, 1.0, max(len(segs), 1))
            rgb    = plt.matplotlib.colors.to_rgb(traj)
            lc = LineCollection(segs, colors=[rgb + (a,) for a in alphas],
                                linewidths=1.5, zorder=2)
            ax.add_collection(lc)

            # Direction arrow
            tip  = pos[-1]
            base = pos[max(0, len(pos) - max(2, len(pos) // 10))]
            dx, dy = tip - base
            if np.hypot(dx, dy) > 1e-6:
                ax.annotate("", xy=tip, xytext=base,
                            arrowprops=dict(arrowstyle="->", color=traj, lw=1.5),
                            zorder=3)

        # Start: circle (traj colour), Goal: star (green=success / red=fail)
        marker = "✓" if r["goal_reached"] else "✗"
        ax.plot(*r["start"], "o", color=traj, markersize=8, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"Ep {i+1} ({marker})")
        ax.plot(*r["goal"], "*", color=colour, markersize=14, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlim(-HALF - 0.5, HALF + 0.5)
    ax.set_ylim(-HALF - 0.5, HALF + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Success / failure count annotation
    n_succ = sum(r["goal_reached"] for r in results[:n])
    ax.text(-HALF + 0.4, HALF - 1.0,
            f"{n_succ}/{n} shown reached goal",
            fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="none"))


# ── 1×2 Figure ────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Transfer Test — Sim A Policies in Sim B", fontsize=14, fontweight="bold")

_draw_panel(ax_l, res_base, N_TRAJ,
            f"Baseline Agent  ({s_b['success']}/{N_EVAL} success, "
            f"avg dist {s_b['avg_dist']:.2f} m)")
_draw_panel(ax_r, res_rand, N_TRAJ,
            f"Randomised Agent  ({s_r['success']}/{N_EVAL} success, "
            f"avg dist {s_r['avg_dist']:.2f} m)")

# Shared legend for goal markers
succ_patch = mpatches.Patch(color=GREEN, label="★ goal reached")
fail_patch  = mpatches.Patch(color=RED,   label="★ goal not reached")
fig.legend(handles=[succ_patch, fail_patch], loc="lower center",
           ncol=2, fontsize=9, framealpha=0.9)

fig.tight_layout(rect=[0, 0.04, 1, 1])

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {SAVE_PATH}")
