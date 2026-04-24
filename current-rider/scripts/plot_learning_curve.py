"""
Sub-Step 2.2 — Plot the PPO baseline learning curve from Monitor CSV.

Reads logs/monitor/train.monitor.csv and saves a two-panel plot to
outputs/baseline_learning_curve.png.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MONITOR_CSV = os.path.join(BASE_DIR, "logs", "monitor", "train.monitor.csv")
SAVE_PATH   = os.path.join(BASE_DIR, "outputs", "baseline_learning_curve.png")
ROLL_WINDOW = 50   # episodes for rolling average
SUCCESS_REF = 10.0 # approximate reward level for a successful episode

# ── Load ──────────────────────────────────────────────────────────────────────
# Monitor CSV has a JSON comment on line 1; skip it with comment='#'
df = pd.read_csv(MONITOR_CSV, comment="#")
df.columns = ["reward", "length", "time"]
df["episode"] = np.arange(1, len(df) + 1)

print(f"Loaded {len(df)} episodes from {MONITOR_CSV}")
print(f"  First episode reward : {df['reward'].iloc[0]:.1f}")
print(f"  Last  episode reward : {df['reward'].iloc[-1]:.1f}")
print(f"  Best  episode reward : {df['reward'].max():.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(
    "Baseline Training (Fixed Physics) — Learning Curve",
    fontsize=13, fontweight="bold",
)

episodes = df["episode"]
roll_r = df["reward"].rolling(ROLL_WINDOW, min_periods=1).mean()
roll_l = df["length"].rolling(ROLL_WINDOW, min_periods=1).mean()

# ── Top panel: episode reward ─────────────────────────────────────────────────
ax_r.plot(episodes, df["reward"], color="#aec6e8", linewidth=0.6,
          alpha=0.7, label="Episode reward")
ax_r.plot(episodes, roll_r, color="#1f77b4", linewidth=2.0,
          label=f"Rolling avg (n={ROLL_WINDOW})")
ax_r.axhline(SUCCESS_REF, color="green", linestyle="--", linewidth=1.2,
             label=f"Success level (~{SUCCESS_REF:.0f})")
ax_r.set_ylabel("Total reward")
ax_r.legend(loc="upper left", fontsize=9)
ax_r.grid(True, linestyle="--", alpha=0.4)

# ── Bottom panel: episode length ──────────────────────────────────────────────
ax_l.plot(episodes, df["length"], color="#f5c193", linewidth=0.6,
          alpha=0.7, label="Episode length")
ax_l.plot(episodes, roll_l, color="#ff7f0e", linewidth=2.0,
          label=f"Rolling avg (n={ROLL_WINDOW})")
ax_l.set_ylabel("Steps")
ax_l.set_xlabel("Episode")
ax_l.legend(loc="upper right", fontsize=9)
ax_l.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved → {SAVE_PATH}")
