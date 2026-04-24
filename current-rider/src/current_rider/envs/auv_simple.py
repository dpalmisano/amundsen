"""
Simulator A — AUV waypoint navigation.

Supports two modes via the `randomise` flag:
  - randomise=False (default): fixed physics — used for the Step 2 baseline.
  - randomise=True:  physics parameters are resampled at each episode reset,
    implementing Tobin et al. (2017) domain randomization for control.

The observation space is IDENTICAL in both modes (6-dim, no physics params
exposed). The agent must learn a reactive strategy that works across the full
distribution of dynamics, which is what forces sim-to-real robustness.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Arena & timing constants ──────────────────────────────────────────────────
ARENA_SIZE  = 20.0   # metres; arena spans [-10, +10] on each axis
MAX_SPEED   = 2.0    # m/s — velocity safety clamp
MAX_THRUST  = 5.0    # Newtons — scales the [-1,1] action
DT          = 0.1    # seconds per simulation step
MAX_STEPS   = 1000   # episode timeout (truncation)
GOAL_RADIUS = 0.5    # metres — distance threshold for "reached goal"

# ── Fixed physics defaults ────────────────────────────────────────────────────
DEFAULT_MASS         = 10.0
DEFAULT_DRAG         = 1.5
DEFAULT_CURRENT      = (0.0, 0.0)
DEFAULT_THRUST_NOISE = 0.0
DEFAULT_GOAL_NOISE   = 0.0

# ── Randomisation ranges ──────────────────────────────────────────────────────
# Defaults sit roughly in the middle of each range, so the fixed-physics case
# is "one draw" from the randomised distribution.
RAND_MASS_RANGE         = (3.0,   5.0)
RAND_DRAG_RANGE         = (0.5,   3.0)
RAND_CURRENT_RANGE      = (-0.3,  0.3)   # per axis
RAND_THRUST_NOISE_RANGE = (0.0,   0.15)
RAND_GOAL_NOISE_RANGE   = (0.0,   0.5)   # metres std


class AUVSimpleEnv(gym.Env):
    """
    2-D AUV waypoint navigation — Simulator A.

    Observation (6-dim, normalised to roughly [-1, 1]):
        [dx/ARENA_SIZE, dy/ARENA_SIZE,   # relative displacement to *observed* goal
         vx/MAX_SPEED,  vy/MAX_SPEED,    # normalised velocity
         gx/ARENA_SIZE, gy/ARENA_SIZE]   # normalised *observed* goal position

    Action (2-dim, continuous in [-1, 1]):
        [thrust_x, thrust_y] — scaled internally by MAX_THRUST

    When randomise=True, physics params are resampled each episode. The agent
    never sees them — it must learn a strategy robust across the full
    distribution (Tobin et al. 2017).
    """

    metadata = {"render_modes": []}

    def __init__(self, randomise: bool = False):
        super().__init__()

        self.randomise = randomise

        # Physics state — defaults here; overwritten in reset() when randomised
        self.mass               = DEFAULT_MASS
        self.drag_coeff         = DEFAULT_DRAG
        self.current            = np.array(DEFAULT_CURRENT, dtype=np.float32)
        self.thrust_noise_scale = DEFAULT_THRUST_NOISE
        self.goal_noise_std     = DEFAULT_GOAL_NOISE

        # Observation and action spaces are identical regardless of randomise flag
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state (set properly in reset())
        self._pos   = np.zeros(2, dtype=np.float32)
        self._vel   = np.zeros(2, dtype=np.float32)
        self._goal  = np.zeros(2, dtype=np.float32)
        self._steps = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # ── Resample physics if in randomised mode ────────────────────────────
        if self.randomise:
            self.mass = float(self.np_random.uniform(*RAND_MASS_RANGE))
            self.drag_coeff = float(self.np_random.uniform(*RAND_DRAG_RANGE))
            self.current = self.np_random.uniform(
                RAND_CURRENT_RANGE[0], RAND_CURRENT_RANGE[1], size=2
            ).astype(np.float32)
            self.thrust_noise_scale = float(
                self.np_random.uniform(*RAND_THRUST_NOISE_RANGE)
            )
            self.goal_noise_std = float(
                self.np_random.uniform(*RAND_GOAL_NOISE_RANGE)
            )
        else:
            self.mass               = DEFAULT_MASS
            self.drag_coeff         = DEFAULT_DRAG
            self.current            = np.array(DEFAULT_CURRENT, dtype=np.float32)
            self.thrust_noise_scale = DEFAULT_THRUST_NOISE
            self.goal_noise_std     = DEFAULT_GOAL_NOISE

        # ── Spawn AUV and goal ────────────────────────────────────────────────
        half = ARENA_SIZE / 2.0
        spawn_margin = 2.0

        self._pos = self.np_random.uniform(
            low=-half + spawn_margin,
            high= half - spawn_margin,
            size=2,
        ).astype(np.float32)

        while True:
            goal = self.np_random.uniform(
                low=-half + spawn_margin,
                high= half - spawn_margin,
                size=2,
            ).astype(np.float32)
            if np.linalg.norm(goal - self._pos) >= 3.0:
                self._goal = goal
                break

        self._vel   = np.zeros(2, dtype=np.float32)
        self._steps = 0

        info = {
            "mass":               self.mass,
            "drag_coeff":         self.drag_coeff,
            "current":            self.current.tolist(),
            "thrust_noise_scale": self.thrust_noise_scale,
            "goal_noise_std":     self.goal_noise_std,
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Physics update (Euler integration) ────────────────────────────────
        # Thrust noise: multiplicative scaling models thruster efficiency variation
        if self.thrust_noise_scale > 0.0:
            noise = self.np_random.uniform(
                1.0 - self.thrust_noise_scale,
                1.0 + self.thrust_noise_scale,
                size=2,
            ).astype(np.float32)
            raw_force = action * MAX_THRUST * noise
        else:
            raw_force = action * MAX_THRUST

        # acceleration = F/m  −  drag·v  +  current
        acceleration = (
            raw_force / self.mass
            - self.drag_coeff * self._vel
            + self.current
        )

        self._vel = self._vel + acceleration * DT
        self._vel = np.clip(self._vel, -MAX_SPEED, MAX_SPEED).astype(np.float32)

        self._pos = self._pos + self._vel * DT
        self._pos = np.clip(self._pos, -ARENA_SIZE / 2, ARENA_SIZE / 2).astype(np.float32)

        self._steps += 1

        # Reward and termination use the TRUE goal — sensor noise affects only
        # what the agent observes, not the physical arrival condition.
        distance   = float(np.linalg.norm(self._pos - self._goal))
        reward     = -distance / ARENA_SIZE - 0.01
        terminated = distance < GOAL_RADIUS
        truncated  = self._steps >= MAX_STEPS

        if terminated:
            reward += 10.0

        info = {
            "distance":     distance,
            "steps":        self._steps,
            "goal_reached": terminated,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the 6-dim observation.

        Goal position is perturbed by Gaussian noise each call when
        goal_noise_std > 0, modelling a noisy acoustic positioning sensor.
        The noise std is fixed per episode (sensor quality is constant hardware)
        but each reading jitters independently. Reward and termination always
        use the true goal — only what the agent sees is noisy.
        """
        if self.goal_noise_std > 0.0:
            observed_goal = self._goal + self.np_random.normal(
                0.0, self.goal_noise_std, size=2
            ).astype(np.float32)
        else:
            observed_goal = self._goal

        dx = (self._pos[0] - observed_goal[0]) / ARENA_SIZE
        dy = (self._pos[1] - observed_goal[1]) / ARENA_SIZE
        vx = self._vel[0] / MAX_SPEED
        vy = self._vel[1] / MAX_SPEED
        gx = observed_goal[0] / ARENA_SIZE
        gy = observed_goal[1] / ARENA_SIZE
        return np.array([dx, dy, vx, vy, gx, gy], dtype=np.float32)
