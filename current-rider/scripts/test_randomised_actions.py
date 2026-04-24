"""
Run a sanity check simulation with random physics and a policy that chooses a random action
at every step.

Just use is to confirm that the random physics it gets applied to the environment correctly.

Physics variables:

  - Mass (3–5 kg) — how heavy the AUV is
  - Drag (0.5–3.0) — water resistance
  - Current (±0.3 m/s² per axis) — a constant push in a random direction
  - Thrust noise (0–15%) — imperfect thrusters, each command fires at slightly the wrong force
  - Goal noise std (0–0.5 m) — the AUV's positioning sensor is noisy, so it doesn't know exactly where the goal is

The key thing to observe in the plot is that the trajectories should look noticeably different from each other: some drifting with current, some sluggish, some twitchy.

Those trajectories should confirms the randomisation is actually doing something.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_simple import AUVSimpleEnv, ARENA_SIZE
from current_rider.utils.visualise import plot_trajectories

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
SAVE_PATH = os.path.join(BASE_DIR, "outputs", "randomised_random_actions.png")

NUM_EPISODES = 5
env = AUVSimpleEnv(randomise=True)
trajectories = []

print("Random actions — Randomised Physics")
print("=" * 60)

for ep in range(NUM_EPISODES):
    obs, info = env.reset(seed=ep * 17)

    start = env._pos.copy()
    goal  = env._goal.copy()

    print(f"\nEpisode {ep + 1}")
    print(f"  mass             = {info['mass']:.2f} kg")
    print(f"  drag_coeff       = {info['drag_coeff']:.2f}")
    print(f"  current          = ({info['current'][0]:+.3f}, {info['current'][1]:+.3f}) m/s²")
    print(f"  thrust_noise     = {info['thrust_noise_scale']:.3f}")
    print(f"  goal_noise_std   = {info['goal_noise_std']:.3f} m")

    positions = [start.copy()]
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env._pos.copy())
        if terminated or truncated:
            break

    print(f"  steps={info['steps']}  final_dist={info['distance']:.2f} m  goal_reached={info['goal_reached']}")

    trajectories.append({
        "positions": np.array(positions),
        "start":     start,
        "goal":      goal,
        "label":     f"Ep {ep + 1}",
    })

env.close()
print()

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
plot_trajectories(
    trajectories=trajectories,
    arena_size=ARENA_SIZE,
    title="Sim A (Randomised Physics) — Random Actions",
    save_path=SAVE_PATH,
)
