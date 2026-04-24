"""
Train PPO on the domain randomised environment.
Physics here changes.

Same hyperparameters as the baseline, but 300k timesteps (harder problem)
and AUVSimpleEnv(randomise=True). Saves to models/randomised_physics.zip.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from current_rider.envs.auv_simple import AUVSimpleEnv

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MONITOR_DIR = os.path.join(BASE_DIR, "logs", "monitor_randomised")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "randomised_physics")

os.makedirs(MONITOR_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

env = Monitor(AUVSimpleEnv(randomise=True), filename=os.path.join(MONITOR_DIR, "train"))

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    seed=42,
)

print("Training PPO on randomised-physics Sim A …")
print(f"Monitor log → {MONITOR_DIR}/train.monitor.csv")
print(f"Model will be saved → {MODEL_PATH}.zip\n")

model.learn(total_timesteps=300_000)
model.save(MODEL_PATH)

print(f"\nDone. Model saved to {MODEL_PATH}.zip")
env.close()
