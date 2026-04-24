"""
Head-to-head evaluation: baseline vs domain-randomised agent.

Both models are tested on AUVSimpleEnv(randomise=True) with identical physics
draws (same seeds) so the comparison is fair. Produces a summary table and a
comparison figure saved to outputs/baseline_vs_randomised_simA.png.

Figure layout (one file):
  Top-left:     Learning curves (baseline vs randomised)
  Top-right:    Success rate bar chart (fixed env vs randomised env, both agents)
  Bottom-left:  3 baseline trajectories on randomised physics (with flow field)
  Bottom-right: 3 randomised-agent trajectories on the same episodes
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_simple import AUVSimpleEnv, ARENA_SIZE
from current_rider.utils.visualise import plot_episodes_grid

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

N_EVAL      = 50
N_TRAJ      = 3
ROLL_WINDOW = 50
SEEDS       = list(range(N_EVAL))
BLUE        = "#1f77b4"
ORANGE      = "#ff7f0e"
GREEN       = "#2ca02c"
RED         = "#d62728"

# ── Load models ───────────────────────────────────────────────────────────────
baseline   = PPO.load(os.path.join(BASE_DIR, "models", "baseline_fixed_physics"))
randomised = PPO.load(os.path.join(BASE_DIR, "models", "randomised_physics"))
print("Models loaded.\n")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, env_kwargs: dict, seeds: list, record_traj: bool = False):
    env = AUVSimpleEnv(**env_kwargs)
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start = env._pos.copy()
        goal  = env._goal.copy()
        physics = {
            "mass":               env.mass,
            "drag_coeff":         env.drag_coeff,
            "current":            env.current.tolist(),
            "thrust_noise_scale": env.thrust_noise_scale,
            "goal_noise_std":     env.goal_noise_std,
        }
        positions    = [start.copy()] if record_traj else None
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
            "physics":       physics,
        })
    env.close()
    return results


# ── Run evaluations ───────────────────────────────────────────────────────────
print(f"Evaluating both agents on randomised env ({N_EVAL} episodes, same seeds)…")
res_base_rand = evaluate(baseline,   {"randomise": True},  SEEDS, record_traj=True)
res_rand_rand = evaluate(randomised, {"randomise": True},  SEEDS, record_traj=True)

print("Evaluating both agents on fixed env (20 episodes)…")
res_base_fixed = evaluate(baseline,   {"randomise": False}, SEEDS[:20])
res_rand_fixed = evaluate(randomised, {"randomise": False}, SEEDS[:20])


# ── Summary table ─────────────────────────────────────────────────────────────
def summarise(results):
    succ = [r for r in results if r["goal_reached"]]
    n    = len(results)
    return {
        "n":          n,
        "success":    len(succ),
        "avg_steps":  np.mean([r["steps"] for r in succ]) if succ else float("nan"),
        "avg_dist":   np.mean([r["final_dist"] for r in results]),
        "avg_reward": np.mean([r["total_reward"] for r in results]),
    }

s_br = summarise(res_base_rand)
s_rr = summarise(res_rand_rand)

print()
print(f"Evaluation on Randomised Sim A — {N_EVAL} episodes (same seeds)")
print("─" * 58)
print(f"{'':24s}  {'Baseline (fixed)':>16s}  {'Randomised':>12s}")
print(f"  Success rate:         {s_br['success']:>4d}/{N_EVAL} "
      f"({100*s_br['success']/N_EVAL:.0f}%)  "
      f"  {s_rr['success']:>4d}/{N_EVAL} ({100*s_rr['success']/N_EVAL:.0f}%)")
print(f"  Avg steps (success):  {s_br['avg_steps']:>16.1f}  {s_rr['avg_steps']:>12.1f}")
print(f"  Avg final distance:   {s_br['avg_dist']:>14.2f} m  {s_rr['avg_dist']:>10.2f} m")
print(f"  Avg total reward:     {s_br['avg_reward']:>16.2f}  {s_rr['avg_reward']:>12.2f}")
print("─" * 58)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 15))
fig.suptitle("Baseline vs Domain-Randomised Agent — Sim A",
             fontsize=14, fontweight="bold")

# Split figure into top strip and bottom strip
top_sf, bot_sf = fig.subfigures(2, 1, height_ratios=[1, 1.4])

# ── Top-left: learning curves ─────────────────────────────────────────────────
ax_lc, ax_bar = top_sf.subplots(1, 2)

for csv_path, colour, label in [
    (os.path.join(BASE_DIR, "logs", "monitor", "train.monitor.csv"),            BLUE,   "Baseline (fixed physics)"),
    (os.path.join(BASE_DIR, "logs", "monitor_randomised", "train.monitor.csv"), ORANGE, "Randomised physics"),
]:
    df = pd.read_csv(csv_path, comment="#")
    df.columns = ["reward", "length", "time"]
    episodes = np.arange(1, len(df) + 1)
    roll = df["reward"].rolling(ROLL_WINDOW, min_periods=1).mean()
    ax_lc.plot(episodes, df["reward"], color=colour, alpha=0.2, linewidth=0.5)
    ax_lc.plot(episodes, roll, color=colour, linewidth=2.0, label=label)

ax_lc.set_title("Learning Curves", fontweight="bold")
ax_lc.set_xlabel("Episode")
ax_lc.set_ylabel("Total reward")
ax_lc.legend(fontsize=9)
ax_lc.grid(True, linestyle="--", alpha=0.4)

# ── Top-right: success rate bar chart ────────────────────────────────────────
n_fixed = 20
categories = ["Baseline\n(fixed env)", "Randomised\n(fixed env)",
              "Baseline\n(rand env)",  "Randomised\n(rand env)"]
values = [
    100 * sum(r["goal_reached"] for r in res_base_fixed) / n_fixed,
    100 * sum(r["goal_reached"] for r in res_rand_fixed) / n_fixed,
    100 * s_br["success"] / N_EVAL,
    100 * s_rr["success"] / N_EVAL,
]
bars = ax_bar.bar(categories, values,
                  color=[BLUE, ORANGE, BLUE, ORANGE],
                  alpha=0.8, edgecolor="white", linewidth=1.2)
ax_bar.axhline(80, color="black", linestyle="--", linewidth=1, label="80% target")
for bar, val in zip(bars, values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_bar.set_ylim(0, 115)
ax_bar.set_title("Success Rate by Agent & Environment", fontweight="bold")
ax_bar.set_ylabel("Success rate (%)")
ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)
patch_base = mpatches.Patch(color=BLUE,   label="Baseline")
patch_rand = mpatches.Patch(color=ORANGE, label="Randomised")
ax_bar.legend(handles=[patch_base, patch_rand], fontsize=9)

# ── Bottom: trajectory grids with physics ─────────────────────────────────────
bot_l, bot_r = bot_sf.subfigures(1, 2)

# Bottom-left: baseline agent on randomised physics
bot_l.suptitle("Baseline Agent — Randomised Physics", fontsize=11, fontweight="bold")
axes_left = bot_l.subplots(1, N_TRAJ)
episodes_base = [
    {
        "positions": r["positions"],
        "start":     r["start"],
        "goal":      r["goal"],
        "label":     f"Ep {i+1} ({'✓' if r['goal_reached'] else '✗'})",
        "colour":    GREEN if r["goal_reached"] else RED,
        "physics":   r["physics"],
    }
    for i, r in enumerate(res_base_rand[:N_TRAJ])
]
plot_episodes_grid(episodes_base, ARENA_SIZE, axes=list(axes_left))

# Bottom-right: randomised agent on the same episodes
bot_r.suptitle("Randomised Agent — Randomised Physics", fontsize=11, fontweight="bold")
axes_right = bot_r.subplots(1, N_TRAJ)
episodes_rand = [
    {
        "positions": r["positions"],
        "start":     r["start"],
        "goal":      r["goal"],
        "label":     f"Ep {i+1} ({'✓' if r['goal_reached'] else '✗'})",
        "colour":    GREEN if r["goal_reached"] else RED,
        "physics":   r["physics"],
    }
    for i, r in enumerate(res_rand_rand[:N_TRAJ])
]
plot_episodes_grid(episodes_rand, ARENA_SIZE, axes=list(axes_right))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
save_path = os.path.join(BASE_DIR, "outputs", "baseline_vs_randomised_simA.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {save_path}")
