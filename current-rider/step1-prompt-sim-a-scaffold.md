# Prompt: Current Rider — Step 1: Project Scaffold & Simulator A (Fixed Physics)

## Your Role

You are a Senior Research Engineer implementing a 2D AUV waypoint navigation environment. You write clean, well-commented code. You implement one piece at a time, stop after each piece, show results, and wait for approval before proceeding.

## Context

We're building a project called "Current Rider" to demonstrate domain randomization for sim-to-real transfer in RL control. This is Step 1 of 6. In this step we set up the project and build Simulator A (the simple training simulator) with **fixed physics only** — no randomization yet.

The project trains an RL agent to pilot a 2D underwater vehicle (AUV) to a target waypoint while fighting ocean currents. The theoretical basis is Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" — but applied to physics/control rather than vision.

## Tech Stack

- **Python 3.11+** with **uv** for project and dependency management
- **NumPy** — physics simulation
- **Gymnasium** — RL environment interface
- **Stable-Baselines3** (needed later, install now)
- **Matplotlib** — trajectory visualisation

## Deliverables for This Step

1. A `uv`-managed project with all dependencies installed
2. A Gymnasium environment (`AUVSimpleEnv`) implementing Simulator A with fixed physics
3. A validation script that runs the environment with random actions and produces a trajectory plot
4. The trajectory plot saved as a PNG

## Sub-Steps (implement one at a time, stop after each)

### Sub-Step 1.1 — Project Scaffold

Create the project structure using `uv`:

```
current-rider/
├── pyproject.toml
├── src/
│   └── current_rider/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── auv_simple.py      # Simulator A
│       └── utils/
│           ├── __init__.py
│           └── visualise.py        # Trajectory plotting
├── scripts/
│   └── test_random_actions.py      # Validation script
└── README.md
```

Initialise with `uv init`, then `uv add numpy gymnasium stable-baselines3 matplotlib shimmy>=2.0`.

**Stop after this sub-step.** Show me the directory tree and confirm dependencies installed.

---

### Sub-Step 1.2 — Physics Engine (the `step` function)

Implement the core physics in `auv_simple.py`. This is a Gymnasium environment with the following specification:

#### State (Observation)

The agent observes a 6-dimensional vector. **Do NOT include physics parameters (mass, drag, current) in the observation** — the agent must learn to be robust without seeing them.

```
observation = [
    (x - goal_x) / ARENA_SIZE,       # relative position to goal, normalised
    (y - goal_y) / ARENA_SIZE,
    vx / MAX_SPEED,                   # velocity, normalised
    vy / MAX_SPEED,
    goal_x / ARENA_SIZE,              # absolute goal position, normalised
    goal_y / ARENA_SIZE,
]
```

**Why normalise?** Neural networks train better when inputs are in roughly [-1, 1]. SB3's PPO uses a neural net internally.

**Why relative position?** The displacement to the goal is the task-relevant signal — it's what the agent needs to reduce to zero. Including it explicitly makes the learning problem easier.

#### Action

Continuous, 2D: `(thrust_x, thrust_y)` each in `[-1, 1]`, representing normalised thruster commands. Internally, scale by `MAX_THRUST` (e.g., 5.0 N) to get the actual force.

```python
action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

#### Physics Update (Euler Integration)

```
raw_force = action * MAX_THRUST                          # scale to Newtons
acceleration = (raw_force / mass) - (drag * velocity) + current_vector
velocity += acceleration * dt
velocity = clip(velocity, -MAX_SPEED, MAX_SPEED)         # safety clamp
position += velocity * dt
position = clip(position, -ARENA_SIZE/2, ARENA_SIZE/2)   # keep in bounds
```

#### Fixed Physics Parameters (for this step)

```python
ARENA_SIZE = 20.0       # metres, square arena from -10 to +10
MAX_SPEED = 2.0         # m/s
MAX_THRUST = 5.0        # Newtons
DT = 0.1                # seconds per step
MAX_STEPS = 500         # episode timeout
GOAL_RADIUS = 0.5       # metres — "reached" threshold

# Fixed physics (will be randomised in Step 3)
mass = 10.0             # kg
drag_coeff = 1.5        # dimensionless
current = [0.0, 0.0]    # m/s² (no current for now)
thrust_noise = 0.0      # fraction (no noise for now)
```

#### Reward Function

```python
distance = np.linalg.norm(position - goal)
reward = -distance / ARENA_SIZE                          # normalised shaping: [-1, 0]

if distance < GOAL_RADIUS:
    reward += 10.0                                        # big bonus for reaching goal
    terminated = True

# Small step penalty to encourage efficiency
reward -= 0.01
```

#### `reset()` Method

- Randomise the AUV's starting position within the arena (but not too close to edges)
- Randomise the goal position (ensuring it's at least 3m from start)
- Zero the velocity
- Return the observation and an info dict

#### `truncated` vs `terminated`

- `terminated = True` when the agent reaches the goal (distance < GOAL_RADIUS)
- `truncated = True` when MAX_STEPS is exceeded
- These are separate booleans in Gymnasium's API — don't conflate them

**Stop after this sub-step.** Show me the full environment code and we'll review it before testing.

---

### Sub-Step 1.3 — Validation Script & Trajectory Plot

Create `scripts/test_random_actions.py` that:

1. Creates an instance of `AUVSimpleEnv`
2. Runs 3 episodes with random actions (`env.action_space.sample()`)
3. Records the trajectory `(x, y)` at each step for each episode
4. Plots all trajectories on one figure:
   - The arena boundary as a rectangle
   - Each trajectory as a coloured line with an arrow showing direction
   - Start position marked with a circle
   - Goal position marked with a star
   - Title: "Sim A — Random Actions (Fixed Physics)"
5. Saves the plot to `outputs/random_actions_test.png`
6. Prints summary stats: steps taken, final distance to goal, whether goal was reached

The purpose is to visually confirm:
- The physics are reasonable (AUV moves, doesn't teleport, stays in bounds)
- The coordinate system makes sense
- The goal is reachable in principle

**Stop after this sub-step.** Show me the plot and the console output.

---

## Important Notes

- Use `np.float32` for all arrays going into/out of the Gymnasium spaces — SB3 requires this.
- The environment must pass `gymnasium.utils.env_checker.check_env()` — run this in the validation script and fix any issues.
- Add clear comments throughout linking implementation to the physics equations above.
- Keep the code simple and readable. No premature abstraction. We'll add randomization in Step 3 by modifying this same class.
