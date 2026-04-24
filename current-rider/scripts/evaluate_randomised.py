"""
Sub-Step 3.4 — Head-to-head evaluation: baseline vs domain-randomised agent.

Both models are tested on AUVSimpleEnv(randomise=True) with identical physics
draws (same seeds) so the comparison is fair. Produces a summary table and a
2×2 comparison figure saved to outputs/baseline_vs_randomised_simA.png.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_simple import AUVSimpleEnv, ARENA_SIZE

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL      = 50   # episodes for head-to-head comparison
N_TRAJ      = 3    # trajectories shown per agent in bottom panels
ROLL_WINDOW = 50
SEEDS       = list(range(N_EVAL))   # shared seeds — both agents face same physics

# ── Load models ───────────────────────────────────────────────────────────────
baseline   = PPO.load(os.path.join(BASE_DIR, "models", "baseline_fixed_physics"))
randomised = PPO.load(os.path.join(BASE_DIR, "models", "randomised_physics"))
print("Models loaded.\n")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, env_kwargs: dict, seeds: list, record_traj: bool = False):
    """Run deterministic episodes and return per-episode result dicts."""
    env = AUVSimpleEnv(**env_kwargs)
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start = env._pos.copy()
        goal  = env._goal.copy()
        positions = [start.copy()] if record_traj else None
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if record_traj:
                positions.append(env._pos.copy())
            total_reward += reward
            if terminated or truncated:
                break
        results.append({
            "goal_reached":  info["goal_reached"],
            "steps":         info["steps"],
            "final_dist":    info["distance"],
            "total_reward":  total_reward,
            "positions":     np.array(positions) if record_traj else None,
            "start":         start,
            "goal":          goal,
        })
    env.close()
    return results


# ── Head-to-head on randomised env ───────────────────────────────────────────
print(f"Evaluating both agents on randomised env ({N_EVAL} episodes, same seeds)…")
res_base_rand = evaluate(baseline,   {"randomise": True},  SEEDS, record_traj=True)
res_rand_rand = evaluate(randomised, {"randomise": True},  SEEDS, record_traj=True)

# ── Success rates on fixed env (for bar chart) ────────────────────────────────
print("Evaluating both agents on fixed env (20 episodes)…")
res_base_fixed = evaluate(baseline,   {"randomise": False}, SEEDS[:20])
res_rand_fixed = evaluate(randomised, {"randomise": False}, SEEDS[:20])


# ── Summary table ─────────────────────────────────────────────────────────────
def summarise(results, label):
    succ = [r for r in results if r["goal_reached"]]
    n    = len(results)
    return {
        "label":       label,
        "n":           n,
        "success":     len(succ),
        "avg_steps":   np.mean([r["steps"] for r in succ]) if succ else float("nan"),
        "avg_dist":    np.mean([r["final_dist"] for r in results]),
        "avg_reward":  np.mean([r["total_reward"] for r in results]),
    }

s_br = summarise(res_base_rand, "Baseline (fixed)")
s_rr = summarise(res_rand_rand, "Randomised")

print()
print(f"Evaluation on Randomised Sim A — {N_EVAL} episodes (same seeds)")
print("─" * 58)
print(f"{'':24s}  {'Baseline (fixed)':>16s}  {'Randomised':>12s}")
print(f"  Success rate:         {s_br['success']:>4d}/{N_EVAL} ({100*s_br['success']/N_EVAL:.0f}%)  "
      f"  {s_rr['success']:>4d}/{N_EVAL} ({100*s_rr['success']/N_EVAL:.0f}%)")
print(f"  Avg steps (success):  {s_br['avg_steps']:>16.1f}  {s_rr['avg_steps']:>12.1f}")
print(f"  Avg final distance:   {s_br['avg_dist']:>14.2f} m  {s_rr['avg_dist']:>10.2f} m")
print(f"  Avg total reward:     {s_br['avg_reward']:>16.2f}  {s_rr['avg_reward']:>12.2f}")
print("─" * 58)


# ── 2×2 Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Baseline vs Domain-Randomised Agent — Sim A", fontsize=14, fontweight="bold")

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
HALF   = ARENA_SIZE / 2.0


# ── Top-left: learning curves ─────────────────────────────────────────────────
ax = axes[0, 0]
for csv_path, colour, label in [
    (os.path.join(BASE_DIR, "logs", "monitor", "train.monitor.csv"),            BLUE,   "Baseline (fixed physics)"),
    (os.path.join(BASE_DIR, "logs", "monitor_randomised", "train.monitor.csv"), ORANGE, "Randomised physics"),
]:
    df = pd.read_csv(csv_path, comment="#")
    df.columns = ["reward", "length", "time"]
    episodes = np.arange(1, len(df) + 1)
    roll = df["reward"].rolling(ROLL_WINDOW, min_periods=1).mean()
    ax.plot(episodes, df["reward"], color=colour, alpha=0.2, linewidth=0.5)
    ax.plot(episodes, roll, color=colour, linewidth=2.0, label=label)

ax.set_title("Learning Curves", fontweight="bold")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)


# ── Top-right: success rate bar chart ────────────────────────────────────────
ax = axes[0, 1]
categories = ["Baseline\n(fixed env)", "Randomised\n(fixed env)",
              "Baseline\n(rand env)",  "Randomised\n(rand env)"]
n_fixed = 20
values = [
    100 * sum(r["goal_reached"] for r in res_base_fixed) / n_fixed,
    100 * sum(r["goal_reached"] for r in res_rand_fixed) / n_fixed,
    100 * s_br["success"] / N_EVAL,
    100 * s_rr["success"] / N_EVAL,
]
colours_bar = [BLUE, ORANGE, BLUE, ORANGE]
bars = ax.bar(categories, values, color=colours_bar, alpha=0.8, edgecolor="white", linewidth=1.2)
ax.axhline(80, color="black", linestyle="--", linewidth=1, label="80% target")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_title("Success Rate by Agent & Environment", fontweight="bold")
ax.set_ylabel("Success rate (%)")
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
patch_base = mpatches.Patch(color=BLUE,   label="Baseline")
patch_rand = mpatches.Patch(color=ORANGE, label="Randomised")
ax.legend(handles=[patch_base, patch_rand], fontsize=9)


# ── Trajectory plotting helper for bottom panels ──────────────────────────────
def plot_traj_panel(ax, results, n, title):
    ax.add_patch(plt.Rectangle((-HALF, -HALF), ARENA_SIZE, ARENA_SIZE,
                               linewidth=2, edgecolor="black", facecolor="#f5f5f5", zorder=0))
    palette = [GREEN, BLUE, ORANGE]
    for i, r in enumerate(results[:n]):
        colour = palette[i % len(palette)]
        pos    = r["positions"]
        points = pos[:, np.newaxis, :]
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.3, 1.0, max(len(segs), 1))
        rgb    = plt.matplotlib.colors.to_rgb(colour)
        lc = LineCollection(segs, colors=[rgb + (a,) for a in alphas],
                            linewidths=1.5, zorder=2)
        ax.add_collection(lc)
        # direction arrow
        if len(pos) >= 2:
            tip  = pos[-1]
            base = pos[max(0, len(pos) - max(2, len(pos) // 10))]
            dx, dy = tip - base
            if np.hypot(dx, dy) > 1e-6:
                ax.annotate("", xy=tip, xytext=base,
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1.5), zorder=3)
        marker = "✓" if r["goal_reached"] else "✗"
        ax.plot(*r["start"], "o", color=colour, markersize=8, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"Ep {i+1} ({marker})")
        ax.plot(*r["goal"], "*", color=colour, markersize=12, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlim(-HALF - 0.5, HALF + 0.5)
    ax.set_ylim(-HALF - 0.5, HALF + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)


plot_traj_panel(axes[1, 0], res_base_rand, N_TRAJ,
                "Baseline Agent — Randomised Physics")
plot_traj_panel(axes[1, 1], res_rand_rand, N_TRAJ,
                "Randomised Agent — Randomised Physics")

fig.tight_layout()
save_path = os.path.join(BASE_DIR, "outputs", "baseline_vs_randomised_simA.png")
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {save_path}")
