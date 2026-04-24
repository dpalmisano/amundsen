# Current Rider

2D AUV waypoint navigation — domain randomization sim-to-real transfer demo.

## Scripts

| Script | Physics | Policy | Purpose |
|---|---|---|---|
| `test_random_actions.py` | Fixed | Random | Sanity check — does the env work? |
| `test_randomised_actions.py` | Randomised | Random | Sanity check — does randomisation work? |
| `train_baseline.py` | Fixed | PPO (learns) | Train the specialist agent |
| `train_randomised.py` | Randomised | PPO (learns) | Train the robust agent |
| `plot_learning_curve.py` | — | — | Plot reward over training episodes |
| `evaluate_baseline.py` | Fixed | Trained baseline | Test the specialist on its home turf |
| `evaluate_randomised.py` | Randomised | Both models | Head-to-head comparison |

## Setup

```bash
uv sync
```

## Usage

**Run the validation script** (checks the environment, prints episode stats, saves a trajectory plot):

```bash
uv run python scripts/test_random_actions.py
```

Output plot is saved to `outputs/random_actions_test.png`.

**Explore the environment interactively:**

```bash
uv run python
```

```python
from current_rider.envs.auv_simple import AUVSimpleEnv

env = AUVSimpleEnv()
obs, info = env.reset(seed=0)
print(obs)                          # 6-dim normalised observation

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(reward, info)
```

`uv run` handles the virtualenv automatically — no activation needed.

## Training

**Train the PPO baseline** (saves model to `models/baseline_fixed_physics.zip`):

```bash
uv run python scripts/train_baseline.py
```

Re-run this whenever you change `MAX_STEPS` or any physics parameter in `src/current_rider/envs/auv_simple.py` — the saved model is tied to the environment it was trained on.

**Plot the learning curve** (reads `logs/monitor/train.monitor.csv`):

```bash
uv run python scripts/plot_learning_curve.py
```

Output saved to `outputs/baseline_learning_curve.png`.

## Evaluation

**Evaluate the trained baseline** (20 deterministic episodes, saves trajectory plot):

```bash
uv run python scripts/evaluate_baseline.py
```

Output saved to `outputs/baseline_trajectories.png`.

**Run a trained model interactively:**

```bash
uv run python
```

```python
from stable_baselines3 import PPO
from current_rider.envs.auv_simple import AUVSimpleEnv

env = AUVSimpleEnv()                          # fixed physics (baseline)
# env = AUVSimpleEnv(randomise=True)          # randomised physics

model = PPO.load("models/baseline_fixed_physics")

obs, info = env.reset(seed=0)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={info['steps']}  dist={info['distance']:.2f} m  reward={reward:.3f}")
    if terminated or truncated:
        print("Goal reached!" if info["goal_reached"] else "Timed out.")
        break

env.close()
```

`deterministic=True` uses the policy mean — the agent's best guess with no exploration noise.
