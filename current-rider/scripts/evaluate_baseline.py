"""
Evaluate the trained PPO baseline.

Loads models/baseline_fixed_physics.zip, runs 20 deterministic episodes,
prints a summary table, and saves a trajectory plot.
"""

import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_simple import AUVSimpleEnv, ARENA_SIZE
from current_rider.utils.visualise import plot_episodes_grid

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_fixed_physics.zip")
SAVE_PATH  = os.path.join(BASE_DIR, "outputs", "baseline_trajectories.png")

NUM_EPISODES  = 20
PLOT_EPISODES = 5   # first N episodes shown in the trajectory plot

# ── Load model ────────────────────────────────────────────────────────────────
model = PPO.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
env = AUVSimpleEnv()
results = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()   # fixed seeds for reproducibility
    start = env._pos.copy()
    goal  = env._goal.copy()

    positions    = [start.copy()]
    total_reward = 0.0

    while True:
        # deterministic=True: use policy mean, no exploration noise
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
        "total_reward": total_reward,
        "steps":        info["steps"],
        "goal_reached": info["goal_reached"],
        "final_dist":   info["distance"],
        "physics": {
            "mass":               env.mass,
            "drag_coeff":         env.drag_coeff,
            "current":            env.current.tolist(),
            "thrust_noise_scale": env.thrust_noise_scale,
            "goal_noise_std":     env.goal_noise_std,
        },
    })

env.close()

# ── Summary table ─────────────────────────────────────────────────────────────
successes     = [r for r in results if r["goal_reached"]]
n_success     = len(successes)
avg_steps     = np.mean([r["steps"] for r in successes]) if successes else float("nan")
avg_dist      = np.mean([r["final_dist"] for r in results])
avg_reward    = np.mean([r["total_reward"] for r in results])

print("Baseline Evaluation (Fixed Physics Sim A) — 20 episodes")
print("─" * 57)
print(f"Success rate:        {n_success}/{NUM_EPISODES} ({100*n_success/NUM_EPISODES:.0f}%)")
print(f"Avg steps (success): {avg_steps:.1f}")
print(f"Avg final distance:  {avg_dist:.2f} m")
print(f"Avg total reward:    {avg_reward:.2f}")
print("─" * 57)

# ── Trajectory plot ───────────────────────────────────────────────────────────
episodes = []
for i, r in enumerate(results[:PLOT_EPISODES]):
    episodes.append({
        "positions": r["positions"],
        "start":     r["start"],
        "goal":      r["goal"],
        "label":     f"Ep {i+1} ({'✓' if r['goal_reached'] else '✗'})",
        "colour":    "#2ca02c" if r["goal_reached"] else "#d62728",
        "physics":   r["physics"],
    })

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
plot_episodes_grid(
    episodes=episodes,
    arena_size=ARENA_SIZE,
    title="Baseline Agent — Fixed Physics",
    save_path=SAVE_PATH,
)
