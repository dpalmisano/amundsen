"""
Run a sanity check simulation with fixed physics and a policy that chooses a random action
at every step.

Just use is to confirm the environment works. 
"""

import os
import sys

import numpy as np
from gymnasium.utils.env_checker import check_env

# Ensure the src layout is importable when run directly with `uv run`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_simple import (
    AUVSimpleEnv, ARENA_SIZE,
    DEFAULT_MASS, DEFAULT_DRAG, DEFAULT_CURRENT,
    DEFAULT_THRUST_NOISE, DEFAULT_GOAL_NOISE, MAX_STEPS, DT,
)
from current_rider.utils.visualise import plot_episodes_grid
print("=" * 60)
print("FIXED PHYSICS (randomise=False)")
print(f"  mass             = {DEFAULT_MASS} kg")
print(f"  drag_coeff       = {DEFAULT_DRAG}")
print(f"  current          = {DEFAULT_CURRENT} m/s²")
print(f"  thrust_noise     = {DEFAULT_THRUST_NOISE}")
print(f"  goal_noise_std   = {DEFAULT_GOAL_NOISE} m")
print(f"  max_steps        = {MAX_STEPS}  ({MAX_STEPS * DT:.0f}s simulated per episode)")
print("  Physics are identical every run — no randomisation.")
print("=" * 60)
print()

# ── 1. Env check ──────────────────────────────────────────────────────────────
print("=" * 60)
print("check_env …")
_check_env = AUVSimpleEnv()
check_env(_check_env)
_check_env.close()
print("check_env passed.\n")

# ── 2. Run episodes ───────────────────────────────────────────────────────────
NUM_EPISODES = 3
env = AUVSimpleEnv()
trajectories = []

print(f"Running {NUM_EPISODES} episodes with random actions …")
print("-" * 60)

for ep in range(NUM_EPISODES):
    obs, _ = env.reset(seed=ep * 100)

    # Grab start / goal from internal state right after reset
    start = env._pos.copy()
    goal  = env._goal.copy()

    positions = [start.copy()]
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env._pos.copy())
        total_reward += reward

        if terminated or truncated:
            break

    trajectories.append({
        "positions": np.array(positions),
        "start":     start,
        "goal":      goal,
        "label":     f"Episode {ep + 1}  ({'✓' if info['goal_reached'] else '✗'})",
        "physics": {
            "mass":               env.mass,
            "drag_coeff":         env.drag_coeff,
            "current":            env.current.tolist(),
            "thrust_noise_scale": env.thrust_noise_scale,
            "goal_noise_std":     env.goal_noise_std,
        },
    })

    # Summary stats
    final_dist  = info["distance"]
    steps_taken = info["steps"]
    reached     = info["goal_reached"]
    print(
        f"  Episode {ep + 1}: "
        f"steps={steps_taken:>3d}  "
        f"final_dist={final_dist:.2f} m  "
        f"goal_reached={reached}  "
        f"total_reward={total_reward:.2f}"
    )

env.close()
print("-" * 60)

# ── 3. Plot & save ────────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "random_actions_test.png")

plot_episodes_grid(
    episodes=trajectories,
    arena_size=ARENA_SIZE,
    title="Sim A — Random Actions (Fixed Physics)",
    save_path=save_path,
)
