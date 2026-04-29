"""
Simulator B — AUV waypoint navigation with complex "real-world" physics.

Used exclusively for evaluation (zero-shot transfer test). Agents trained in
Sim A are deployed here without retraining. The observation and action spaces
are byte-for-byte identical to AUVSimpleEnv so that SB3 policies load cleanly.

Complexity introduced vs Sim A:
  - Spatially varying current field (gyre pattern)
  - Quadratic (nonlinear) drag at higher speeds
  - Thruster response delay (2 steps)
  - Position sensor noise (AUV doesn't know its exact location)
  - Rotational inertia (heading rotates toward desired thrust direction)

None of these effects were present in Sim A training. The domain-randomised
agent is hypothesised to handle them better because it learned a reactive,
general strategy rather than memorising one specific dynamics model.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ── Shared constants (identical to Sim A) ────────────────────────────────────
ARENA_SIZE  = 20.0
MAX_SPEED   = 2.0
MAX_THRUST  = 5.0
DT          = 0.1
MAX_STEPS   = 500    # shorter budget — Sim B is harder
GOAL_RADIUS = 0.5

# ── Sim B fixed physics ───────────────────────────────────────────────────────
MASS            = 12.0    # slightly heavier than Sim A's default of 10 kg
DRAG_COEFF      = 1.5     # same base value; nonlinear model makes it behave differently
ACTION_DELAY    = 2       # thruster lag in steps
POS_NOISE_STD   = 0.2     # position sensor noise (metres)
MAX_TURN_RATE   = np.pi / 4   # max heading change per step (45°)


class AUVComplexEnv(gym.Env):
    """
    Simulator B — complex AUV environment for sim-to-real transfer evaluation.

    Observation (6-dim, identical layout to AUVSimpleEnv):
        [dx/ARENA_SIZE, dy/ARENA_SIZE,   # relative displacement (noisy position)
         vx/MAX_SPEED,  vy/MAX_SPEED,    # normalised velocity (true)
         gx/ARENA_SIZE, gy/ARENA_SIZE]   # normalised goal position (true)

    Action (2-dim, continuous in [-1, 1]):
        [thrust_x, thrust_y] — delayed by ACTION_DELAY steps before taking effect
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Spaces identical to AUVSimpleEnv — SB3 policies load without modification
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state
        self._pos     = np.zeros(2, dtype=np.float32)
        self._vel     = np.zeros(2, dtype=np.float32)
        self._goal    = np.zeros(2, dtype=np.float32)
        self._heading = 0.0   # radians; 0 = pointing right
        self._steps   = 0
        self._action_buffer = deque(
            [np.zeros(2, dtype=np.float32)] * (ACTION_DELAY + 1),
            maxlen=ACTION_DELAY + 1,
        )

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        half         = ARENA_SIZE / 2.0
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

        self._vel     = np.zeros(2, dtype=np.float32)
        self._heading = 0.0
        self._steps   = 0
        self._action_buffer = deque(
            [np.zeros(2, dtype=np.float32)] * (ACTION_DELAY + 1),
            maxlen=ACTION_DELAY + 1,
        )

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Thruster delay ────────────────────────────────────────────────────
        self._action_buffer.append(action.copy())
        delayed_action = self._action_buffer[0]   # action from ACTION_DELAY steps ago

        # ── Rotational inertia ────────────────────────────────────────────────
        thrust_mag = float(np.linalg.norm(delayed_action)) * MAX_THRUST
        if thrust_mag > 1e-6:
            desired_angle = float(np.arctan2(delayed_action[1], delayed_action[0]))
            angle_diff = desired_angle - self._heading
            # Normalise to [-π, π]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_diff) > MAX_TURN_RATE:
                self._heading += MAX_TURN_RATE * np.sign(angle_diff)
            else:
                self._heading = desired_angle

        actual_force = np.array([
            thrust_mag * np.cos(self._heading),
            thrust_mag * np.sin(self._heading),
        ], dtype=np.float32)

        # ── Spatially varying current ─────────────────────────────────────────
        current = self._get_current(self._pos)

        # ── Quadratic drag ────────────────────────────────────────────────────
        speed = float(np.linalg.norm(self._vel))
        if speed > 0.5:
            # Quadratic regime: drag force grows with speed²
            drag_force = DRAG_COEFF * self._vel * speed
        else:
            # Linear regime at low speed — avoids numerical instability near zero
            drag_force = DRAG_COEFF * self._vel

        # ── Euler integration ─────────────────────────────────────────────────
        acceleration = actual_force / MASS - drag_force + current
        self._vel = self._vel + acceleration * DT
        self._vel = np.clip(self._vel, -MAX_SPEED, MAX_SPEED).astype(np.float32)
        self._pos = self._pos + self._vel * DT
        self._pos = np.clip(self._pos, -ARENA_SIZE / 2, ARENA_SIZE / 2).astype(np.float32)

        self._steps += 1

        # Reward and termination always use the TRUE position
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

    # ── Physics helpers ────────────────────────────────────────────────────────

    def _get_current(self, position: np.ndarray) -> np.ndarray:
        """
        Spatially varying current field — a circular gyre centred at the origin.

        Current is tangential to the position vector (like a real ocean gyre),
        with strength increasing with distance from centre. A constant background
        drift is added on top. This creates currents that change as the AUV
        moves — something Sim A's constant current never prepared the agent for.
        """
        x, y = float(position[0]), float(position[1])
        dist  = np.sqrt(x**2 + y**2) + 1e-6
        scale = 0.3 * np.tanh(dist / 5.0)   # saturates at ~0.3 m/s²

        # Tangential direction: perpendicular to the radius vector
        current_x = -scale * (y / dist) + 0.1    # + constant background drift
        current_y =  scale * (x / dist) + 0.05

        return np.array([current_x, current_y], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Build the 6-dim observation.

        Position is noisy (imperfect localisation sensor). Goal and velocity
        are unaffected — the AUV knows where it's going and how fast it's
        moving, but not precisely where it is.
        """
        noisy_pos = self._pos + self.np_random.normal(
            0.0, POS_NOISE_STD, size=2
        ).astype(np.float32)

        dx = (noisy_pos[0] - self._goal[0]) / ARENA_SIZE
        dy = (noisy_pos[1] - self._goal[1]) / ARENA_SIZE
        vx = self._vel[0] / MAX_SPEED
        vy = self._vel[1] / MAX_SPEED
        gx = self._goal[0] / ARENA_SIZE
        gy = self._goal[1] / ARENA_SIZE

        return np.array([dx, dy, vx, vy, gx, gy], dtype=np.float32)
