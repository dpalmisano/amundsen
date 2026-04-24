# Prompt: Current Rider — Step 3: Domain Randomization

## Your Role

You are a Senior Research Engineer. You write clean, well-commented code. You implement one piece at a time, stop after each piece, show results, and wait for approval before proceeding.

## Context

This is Step 3 of 6 in the "Current Rider" project. We're applying Tobin et al.'s domain randomization principle to RL control.

In Step 1 we built `AUVSimpleEnv` with fixed physics. In Step 2 we trained a PPO baseline that reliably reaches waypoints in the fixed-physics environment. That baseline is our "no randomization" control — analogous to Tobin et al.'s detector trained on a single texture.

In this step we modify the environment to **randomise physics parameters at the start of each episode**, then retrain PPO. The key insight: by training across a distribution of physics, the agent can't memorise a single dynamics model. It must learn a reactive strategy that works regardless of mass, drag, or current — making it robust to the "reality gap" when we deploy it in the complex Sim B (Step 4).

**Critical design constraint:** The observation space must NOT change. The agent still sees only `(relative_pos, velocity, goal_pos)` — never the physics parameters. This is what forces invariance. If the agent could see the mass, it would learn a mass-conditional strategy that breaks when mass behaves differently in Sim B (e.g., effective mass changes with added mass from entrained water). Note that goal sensor noise does NOT add a new observation dimension — it perturbs the existing goal position values, modelling a noisy positioning sensor. The agent sees the same 6 numbers as before, just with less trustworthy goal coordinates.

## Deliverables for This Step

1. Modified `AUVSimpleEnv` supporting both fixed and randomised physics via a flag
2. A trained domain-randomized PPO agent
3. Three outputs:
   - Learning curve comparison (baseline vs. randomised)
   - Trajectory plot of the randomised agent in Sim A
   - Console evaluation summary

## Sub-Steps

### Sub-Step 3.1 — Add Randomization to the Environment

Modify `src/current_rider/envs/auv_simple.py` to add a `randomise` parameter:

```python
class AUVSimpleEnv(gymnasium.Env):
    def __init__(self, randomise: bool = False):
        self.randomise = randomise
        # ... rest of init unchanged
```

In the `reset()` method, **if `self.randomise` is True**, sample physics parameters from these ranges:

```python
# Randomised per episode (sampled once at reset, held constant for entire episode)
self.mass       = np.random.uniform(5.0, 20.0)       # kg
self.drag_coeff = np.random.uniform(0.5, 3.0)        # dimensionless
self.current    = np.random.uniform(-0.5, 0.5, size=2)  # m/s² — constant within episode
self.thrust_noise_scale = np.random.uniform(0.0, 0.15)  # fraction of thrust
self.goal_noise_std     = np.random.uniform(0.0, 0.5)   # metres — sensor noise on goal position
```

If `self.randomise` is False, use the original fixed values (mass=10, drag=1.5, current=[0,0], noise=0, goal_noise_std=0).

In the `step()` method, apply thrust noise:

```python
# Before computing acceleration:
if self.thrust_noise_scale > 0:
    noise = np.random.uniform(
        1.0 - self.thrust_noise_scale,
        1.0 + self.thrust_noise_scale,
        size=2
    )
    raw_force = action * MAX_THRUST * noise    # multiplicative noise on thrust
else:
    raw_force = action * MAX_THRUST
```

When building the observation (in both `reset()` and `step()`), apply goal sensor noise:

```python
# The true goal is self.goal_position (fixed for the episode).
# What the agent "sees" has noise re-sampled every step,
# modelling a noisy positioning sensor (e.g., acoustic USBL).
if self.goal_noise_std > 0:
    observed_goal = self.goal_position + np.random.normal(0, self.goal_noise_std, size=2)
else:
    observed_goal = self.goal_position

# Use observed_goal (not self.goal_position) when computing the observation:
observation = np.array([
    (self.position[0] - observed_goal[0]) / ARENA_SIZE,
    (self.position[1] - observed_goal[1]) / ARENA_SIZE,
    self.velocity[0] / MAX_SPEED,
    self.velocity[1] / MAX_SPEED,
    observed_goal[0] / ARENA_SIZE,
    observed_goal[1] / ARENA_SIZE,
], dtype=np.float32)
```

**Critical:** The reward and termination condition must still use `self.goal_position` (the TRUE goal), not `observed_goal`. The agent's sensor is noisy, but reality isn't — the AUV actually arrives when it reaches the real goal, regardless of what its sensor says.

Store the sampled parameters in the `info` dict returned by `reset()` so we can inspect them during debugging:

```python
info = {
    "mass": self.mass,
    "drag_coeff": self.drag_coeff,
    "current": self.current.tolist(),
    "thrust_noise_scale": self.thrust_noise_scale,
    "goal_noise_std": self.goal_noise_std,
}
```

**Important implementation details:**
- The `observation_space` and `action_space` must remain IDENTICAL. No new dimensions.
- The randomisation ranges are chosen so the fixed-physics defaults (mass=10, drag=1.5) sit roughly in the middle of each range. This means the fixed-physics case is "one draw" from the randomised distribution.
- Current is applied as a constant acceleration term in the physics update — same equation as before, just now the current vector is non-zero.
- Thrust noise is multiplicative (scaling), not additive. This models thruster efficiency variation, not sensor noise.
- Goal noise is additive Gaussian, re-sampled every step. The *standard deviation* is fixed per episode (sampled at reset), but the actual noise value jitters each step. This models a noisy positioning sensor — the sensor's *quality* is fixed hardware, but each reading fluctuates. The noise std range [0.0, 0.5] means some episodes have perfect goal knowledge and others have up to ~0.5m uncertainty — roughly the accuracy range of real acoustic positioning systems.
- Goal noise applies to the observation only. Reward and termination always use the true goal position.

**Stop after this sub-step.** Show me the modified environment code. Run `gymnasium.utils.env_checker.check_env()` on both `AUVSimpleEnv(randomise=False)` and `AUVSimpleEnv(randomise=True)` to confirm neither is broken.

---

### Sub-Step 3.2 — Quick Sanity Check: Random Actions with Randomised Physics

Create or modify a script to run 5 episodes with random actions in the randomised environment. For each episode, print the sampled physics parameters and plot the trajectory. Save as `outputs/randomised_random_actions.png`.

**What to look for:**
- Trajectories should look noticeably different from each other — some drift with current, some are sluggish (high mass), some are twitchy (low mass, low drag).
- Episodes with high `goal_noise_std` may show slightly wobbly final approaches as the agent chases a jittering target position. This is expected.
- If all trajectories look the same despite randomisation, the randomisation isn't actually being applied — check the `reset()` logic.

**Stop after this sub-step.** Show me the plot and the printed parameters.

---

### Sub-Step 3.3 — Train the Domain-Randomized Agent

Create `scripts/train_randomised.py` that:

1. Creates `AUVSimpleEnv(randomise=True)`
2. Wraps in `Monitor`
3. Trains PPO with the **same hyperparameters** as the baseline:

```python
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
    tensorboard_log="./logs/",
)

model.learn(total_timesteps=300_000)
model.save("models/randomised_physics")
```

**Note: 300k steps instead of 200k.** The randomised environment is a harder problem — the agent needs to generalise across a distribution of dynamics, not just solve one. Expect the learning curve to be noisier and converge more slowly. This is normal and expected. If it hasn't converged by 300k, increase to 500k.

4. Save the model to `models/randomised_physics.zip`

**Stop after this sub-step.** Show me the training logs.

---

### Sub-Step 3.4 — Evaluation & Comparison

Create `scripts/evaluate_randomised.py` that:

1. Loads BOTH models: `baseline_fixed_physics.zip` and `randomised_physics.zip`
2. Evaluates each on **the randomised environment** (`AUVSimpleEnv(randomise=True)`) for 50 episodes with `deterministic=True`, using the same random seed so both face identical physics draws
3. Prints a comparison table:

```
Evaluation on Randomised Sim A — 50 episodes (same seeds)
──────────────────────────────────────────────────────────
                        Baseline (fixed)    Randomised
Success rate:           XX/50 (XX%)         XX/50 (XX%)
Avg steps (success):    XX.X                XX.X
Avg final distance:     XX.XX m             XX.XX m
Avg total reward:       XX.XX               XX.XX
```

4. Plots a 2x2 figure:
   - **Top-left:** Learning curve — baseline (blue) vs. randomised (orange), rolling average reward over episodes
   - **Top-right:** Success rate bar chart — both agents on fixed env, both on randomised env (4 bars)
   - **Bottom-left:** 3 example trajectories from the baseline agent on the randomised env
   - **Bottom-right:** 3 example trajectories from the randomised agent on the randomised env (same episodes/seeds)
5. Save to `outputs/baseline_vs_randomised_simA.png`

**What to expect:**
- The baseline agent should still do okay on the randomised env for episodes where the sampled physics happen to be close to (mass=10, drag=1.5, no current). But it should struggle or fail on episodes with strong currents or very different mass/drag.
- The randomised agent should perform consistently across all physics draws, though its average performance on any single configuration might be slightly worse than the specialist baseline on the fixed configuration. This is the **generalist tax** — the cost of robustness.
- If the baseline actually outperforms the randomised agent on the randomised env, the randomised agent hasn't trained long enough. Increase timesteps.

**Stop after this sub-step.** Show me the comparison table and the 2x2 plot.

---

## File Organisation After This Step

```
current-rider/
├── models/
│   ├── baseline_fixed_physics.zip     # From Step 2
│   └── randomised_physics.zip         # New
├── outputs/
│   ├── random_actions_test.png        # From Step 1
│   ├── baseline_learning_curve.png    # From Step 2
│   ├── baseline_trajectories.png      # From Step 2
│   ├── randomised_random_actions.png  # New
│   └── baseline_vs_randomised_simA.png # New
├── scripts/
│   ├── test_random_actions.py
│   ├── train_baseline.py
│   ├── plot_learning_curve.py
│   ├── evaluate_baseline.py
│   ├── train_randomised.py            # New
│   └── evaluate_randomised.py         # New
├── src/
│   └── current_rider/
│       ├── envs/
│       │   └── auv_simple.py          # Modified (randomise flag added)
│       └── utils/
│           └── visualise.py
└── pyproject.toml
```

## Important Notes

- The randomised agent should be evaluated with `deterministic=True` — we want its learned strategy, not exploration noise.
- When comparing the two agents on the randomised env, seed the environment identically so both face the exact same physics draws. This makes the comparison fair.
- Do NOT retrain or modify the baseline agent. It stays as-is from Step 2 — it's our control condition.
- The observation space is unchanged. If you find yourself adding physics parameters to the observation, STOP — that defeats the purpose of domain randomization.
