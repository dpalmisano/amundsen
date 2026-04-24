"""
Train PPO baseline on fixed-physics.

Trains for 200k timesteps and saves the model to models/baseline_fixed_physics.zip.
The Monitor wrapper writes per-episode stats to logs/monitor/ for the learning curve.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from current_rider.envs.auv_simple import AUVSimpleEnv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MONITOR_DIR = os.path.join(BASE_DIR, "logs", "monitor")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "baseline_fixed_physics")

os.makedirs(MONITOR_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
# Monitor logs episode reward and length to a CSV in MONITOR_DIR
env = Monitor(AUVSimpleEnv(), filename=os.path.join(MONITOR_DIR, "train"))

# ── Model ─────────────────────────────────────────────────────────────────────
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,     # SB3 default; good starting point for continuous control
    n_steps=2048,           # steps collected per rollout before each update
    batch_size=64,          # minibatch size for gradient steps
    n_epochs=10,            # passes over each rollout buffer
    gamma=0.99,             # discount: agent values future reward nearly as much as immediate
    gae_lambda=0.95,        # GAE lambda: bias-variance tradeoff in advantage estimation
    clip_range=0.2,         # PPO clipping: prevents destructively large policy updates
    verbose=1,
    seed=42,
)

print("Training PPO baseline on fixed-physics")
print(f"Monitor log → {MONITOR_DIR}/train.monitor.csv")
print(f"Model will be saved → {MODEL_PATH}.zip\n")

model.learn(total_timesteps=200_000)
model.save(MODEL_PATH)

print(f"\nDone. Model saved to {MODEL_PATH}.zip")
env.close()
