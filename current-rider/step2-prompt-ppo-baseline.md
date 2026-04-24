# Prompt: Current Rider — Step 2: PPO Training on Fixed Physics (Baseline)

## Your Role

You are a Senior Research Engineer. You write clean, well-commented code. You implement one piece at a time, stop after each piece, show results, and wait for approval before proceeding.

## Context

This is Step 2 of 6 in the "Current Rider" project. In Step 1 we built `AUVSimpleEnv` — a Gymnasium environment implementing a 2D AUV navigating to a waypoint with fixed physics (mass=10, drag=1.5, no current, no thrust noise).

In this step we:
1. Train a PPO agent on the fixed-physics environment
2. Evaluate the trained agent
3. Visualise the learning curve and trained trajectories

This trained agent becomes our **"no randomization" baseline**. In later steps we'll compare it against a domain-randomized agent when both are deployed in the complex Sim B. We expect this baseline to work well in Sim A but fail in Sim B — demonstrating the reality gap.

**Important:** Do NOT modify the environment or reward function in this step. If training doesn't converge, we'll tune hyperparameters, not the reward. If the reward truly needs changing, flag it and stop — we'll discuss before changing anything.

## Tech Stack

Everything installed in Step 1:
- **Stable-Baselines3** — PPO implementation
- **Matplotlib** — plotting
- **NumPy** — numerical work

## Deliverables for This Step

1. A training script that trains PPO and saves the model
2. An evaluation script that loads the model and tests it
3. Three outputs:
   - A learning curve plot (reward vs. training steps)
   - A trained-agent trajectory plot (multiple episodes in Sim A)
   - Console output with success rate and average metrics

## Sub-Steps

### Sub-Step 2.1 — Training Script

Create `scripts/train_baseline.py` that:

1. Creates the `AUVSimpleEnv` environment
2. Wraps it in a `Monitor` wrapper (from `stable_baselines3.common.monitor`) to log episode rewards and lengths
3. Trains PPO with the following starting hyperparameters:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",                    # Multi-layer perceptron policy (fully connected NN)
    env,
    learning_rate=3e-4,             # SB3 default, good starting point
    n_steps=2048,                   # Steps per rollout buffer collection
    batch_size=64,                  # Minibatch size for SGD updates
    n_epochs=10,                    # Number of passes over the rollout buffer per update
    gamma=0.99,                     # Discount factor — how much the agent values future reward
    gae_lambda=0.95,                # GAE parameter — bias-variance tradeoff in advantage estimation
    clip_range=0.2,                 # PPO clipping — prevents large policy updates
    verbose=1,                      # Print training progress
    tensorboard_log="./logs/",      # Optional, for later inspection
)

model.learn(total_timesteps=200_000)
model.save("models/baseline_fixed_physics")
```

**Why these hyperparameters?** They're SB3's well-tested defaults for continuous control. 200k timesteps is a reasonable starting budget — our environment is simple (6D state, 2D action, fast physics) so this should be enough. If it's not converging, we'll increase to 500k before touching anything else.

4. After training, save the model to `models/baseline_fixed_physics.zip`

**Stop after this sub-step.** Show me the training output (the SB3 verbose logs). We're looking for the `ep_rew_mean` to trend upward over time. Don't worry if it's noisy — RL training curves are always noisy.

---

### Sub-Step 2.2 — Learning Curve Plot

Create `scripts/plot_learning_curve.py` (or add to an existing utils module) that:

1. Reads the episode rewards and lengths from the Monitor CSV log file (it will be in the directory where the monitored env writes — typically alongside the script or in a `monitor` folder)
2. Plots two subplots stacked vertically:
   - **Top:** Episode reward (y) vs. episode number (x), with a rolling average (window=50) overlaid as a bold line
   - **Bottom:** Episode length (y) vs. episode number (x), with the same rolling average
3. Adds a horizontal dashed line on the reward plot at `+10.0` (the goal-reached bonus minus a few step penalties — approximate "success" level)
4. Title: "Baseline Training (Fixed Physics) — Learning Curve"
5. Saves to `outputs/baseline_learning_curve.png`

**What to look for:**
- Reward should climb from very negative (random flailing) toward a plateau near the success level
- Episode length should decrease as the agent learns to reach the goal faster
- If reward plateaus at a very negative value, the agent isn't learning — we'll need to debug

**Stop after this sub-step.** Show me the plot.

---

### Sub-Step 2.3 — Evaluation & Trained Trajectory Plot

Create `scripts/evaluate_baseline.py` that:

1. Loads the trained model from `models/baseline_fixed_physics.zip`
2. Runs 20 evaluation episodes with `deterministic=True` (uses the mean of the policy distribution, not a sample — gives the agent's "best guess" behaviour)
3. Records for each episode:
   - Full trajectory `(x, y)` at each step
   - Total reward
   - Number of steps
   - Whether the goal was reached (terminated=True)
   - Final distance to goal
4. Prints a summary table:

```
Baseline Evaluation (Fixed Physics Sim A) — 20 episodes
─────────────────────────────────────────────────────────
Success rate:       XX/20 (XX%)
Avg steps (success): XX.X
Avg final distance:  XX.XX m
Avg total reward:    XX.XX
```

5. Plots the first 5 trajectories on one figure (same style as Step 1's random-actions plot):
   - Arena boundary
   - Each trajectory as a coloured line with direction arrows
   - Start position (circle), goal position (star)
   - Colour-code: green for successful episodes, red for failures
   - Title: "Baseline Agent — Fixed Physics Sim A"
6. Saves to `outputs/baseline_trajectories.png`

**What success looks like:**
- We want **>80% success rate** on the fixed-physics environment. This is the easy case — the physics don't change between training and testing. If the agent can't solve this, something is wrong with the environment, reward, or hyperparameters.
- Trajectories should look purposeful — roughly straight or gently curved paths to the goal, not spirals or oscillations.

**What failure looks like and what to do:**
- **0% success, reward never improves:** Check that the observation and reward are computed correctly. Print a few `(obs, action, reward)` tuples to sanity-check.
- **Low success, oscillating reward:** Try increasing `total_timesteps` to 500k. The agent might need more experience.
- **Agent reaches vicinity but overshoots/orbits:** The drag might be too low relative to thrust, causing the agent to overshoot. But don't change physics yet — flag it and we'll discuss.

**Stop after this sub-step.** Show me the summary table and trajectory plot.

---

## File Organisation After This Step

```
current-rider/
├── models/
│   └── baseline_fixed_physics.zip    # Trained PPO model
├── outputs/
│   ├── random_actions_test.png       # From Step 1
│   ├── baseline_learning_curve.png   # New
│   └── baseline_trajectories.png     # New
├── logs/                             # Tensorboard logs (optional)
├── scripts/
│   ├── test_random_actions.py        # From Step 1
│   ├── train_baseline.py             # New
│   ├── plot_learning_curve.py        # New
│   └── evaluate_baseline.py          # New
├── src/
│   └── current_rider/
│       ├── envs/
│       │   └── auv_simple.py         # Unchanged from Step 1
│       └── utils/
│           └── visualise.py
└── pyproject.toml
```

## Important Notes

- Do NOT modify `auv_simple.py` in this step. The environment is frozen. We're only writing training and evaluation code.
- Use `deterministic=True` during evaluation, `deterministic=False` (default) during training. Training needs exploration; evaluation needs the agent's best behaviour.
- Ensure the random seed is set for reproducibility: `model = PPO(..., seed=42)` and use a seeded eval env.
- If training takes more than ~5 minutes on your machine, that's fine — this is normal for 200k steps. If it takes more than 15 minutes, something might be wrong (the environment step might be too slow).
