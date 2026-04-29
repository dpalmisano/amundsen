"""
Step 5 — Side-by-side animation of baseline vs domain-randomised agent in Sim B.

Sub-Step 5.1: Seed search + data collection  (run first)
Sub-Step 5.2: Animation render               (run after 5.1)
Sub-Step 5.3: Static filmstrip               (run after 5.1)

Run the whole script — it executes all three sub-steps in sequence.
"""

import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from current_rider.envs.auv_complex import AUVComplexEnv, ARENA_SIZE, GOAL_RADIUS

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE   = "#1f77b4"   # baseline
ORANGE = "#ff7f0e"   # randomised
GOLD   = "#FFD700"
HALF   = ARENA_SIZE / 2.0

# ── Load models ───────────────────────────────────────────────────────────────
baseline   = PPO.load(os.path.join(BASE_DIR, "models", "baseline_fixed_physics"))
randomised = PPO.load(os.path.join(BASE_DIR, "models", "randomised_physics"))
print("Models loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 5.1 — Seed search & data collection
# ═══════════════════════════════════════════════════════════════════════════════

def _run_episode(model, seed: int) -> dict:
    """Run one deterministic episode and return summary + full trajectory."""
    env = AUVComplexEnv()
    obs, _ = env.reset(seed=seed)
    start  = env._pos.copy()
    goal   = env._goal.copy()

    traj = {"pos": [start.copy()], "vel": [], "heading": [],
            "action": [], "current": [], "dist": [], "cum_reward": []}
    cum_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

        traj["pos"].append(env._pos.copy())
        traj["vel"].append(env._vel.copy())
        traj["heading"].append(env._heading)
        traj["action"].append(action.copy())
        traj["current"].append(env._get_current(env._pos).copy())
        traj["dist"].append(info["distance"])
        traj["cum_reward"].append(cum_reward)

        if terminated or truncated:
            break

    env.close()
    return {
        "start":        start,
        "goal":         goal,
        "goal_reached": info["goal_reached"],
        "final_dist":   info["distance"],
        "steps":        info["steps"],
        "traj":         {k: np.array(v) for k, v in traj.items()},
    }


# ── Scan seeds 0–50 ───────────────────────────────────────────────────────────
print("Scanning seeds 0–50 to find best scenario…")
print(f"{'Seed':>4}  {'Base dist':>9}  {'Base OK':>7}  {'Rand dist':>9}  {'Rand OK':>7}  {'Gap':>6}")
print("─" * 56)

best_seed = 0
best_gap  = -np.inf
best_base = None
best_rand = None

for seed in range(51):
    rb = _run_episode(baseline,   seed)
    rr = _run_episode(randomised, seed)

    gap = rb["final_dist"] - rr["final_dist"]
    start_goal_dist = float(np.linalg.norm(rb["goal"] - rb["start"]))

    print(f"{seed:>4}  {rb['final_dist']:>9.2f}  {str(rb['goal_reached']):>7}  "
          f"{rr['final_dist']:>9.2f}  {str(rr['goal_reached']):>7}  {gap:>6.2f}")

    # Prefer: baseline fails, randomised succeeds, large gap, start/goal far apart
    score = gap + (10.0 if not rb["goal_reached"] else 0.0) \
                + (10.0 if rr["goal_reached"] else 0.0) \
                + (5.0  if start_goal_dist >= 8.0 else 0.0)

    if score > best_gap:
        best_gap  = score
        best_seed = seed
        best_base = rb
        best_rand = rr

print("─" * 56)
print(f"\nChosen seed: {best_seed}")
print(f"  Baseline  — dist={best_base['final_dist']:.2f} m  reached={best_base['goal_reached']}")
print(f"  Randomised — dist={best_rand['final_dist']:.2f} m  reached={best_rand['goal_reached']}")
print(f"  Start→goal distance: {np.linalg.norm(best_base['goal'] - best_base['start']):.2f} m")

# ── Save animation data ───────────────────────────────────────────────────────
np.savez(
    os.path.join(OUT_DIR, "animation_data.npz"),
    seed       = best_seed,
    start      = best_base["start"],
    goal       = best_base["goal"],
    # Baseline trajectories
    base_pos     = best_base["traj"]["pos"],
    base_vel     = best_base["traj"]["vel"],
    base_heading = best_base["traj"]["heading"],
    base_action  = best_base["traj"]["action"],
    base_current = best_base["traj"]["current"],
    base_dist    = best_base["traj"]["dist"],
    base_reward  = best_base["traj"]["cum_reward"],
    base_success = best_base["goal_reached"],
    # Randomised trajectories
    rand_pos     = best_rand["traj"]["pos"],
    rand_vel     = best_rand["traj"]["vel"],
    rand_heading = best_rand["traj"]["heading"],
    rand_action  = best_rand["traj"]["action"],
    rand_current = best_rand["traj"]["current"],
    rand_dist    = best_rand["traj"]["dist"],
    rand_reward  = best_rand["traj"]["cum_reward"],
    rand_success = best_rand["goal_reached"],
)
print(f"\nData saved → {OUT_DIR}/animation_data.npz")


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 5.2 — Side-by-side animation
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding animation…")

# ── Load saved data ───────────────────────────────────────────────────────────
d = np.load(os.path.join(OUT_DIR, "animation_data.npz"), allow_pickle=True)
start = d["start"];  goal = d["goal"]

base_pos     = d["base_pos"];    rand_pos     = d["rand_pos"]
base_heading = d["base_heading"]; rand_heading = d["rand_heading"]
base_dist    = d["base_dist"];   rand_dist    = d["rand_dist"]
base_reward  = d["base_reward"]; rand_reward  = d["rand_reward"]
base_success = bool(d["base_success"]); rand_success = bool(d["rand_success"])

# Align lengths — pad shorter trajectory with its final position
n_base = len(base_dist)
n_rand = len(rand_dist)
n_frames = max(n_base, n_rand)

def _pad(arr, target_len):
    if len(arr) >= target_len:
        return arr
    pad = np.tile(arr[-1:], (target_len - len(arr), 1) if arr.ndim > 1
                  else (target_len - len(arr),))
    return np.concatenate([arr, pad])

base_pos     = _pad(base_pos,     n_frames + 1)
base_heading = _pad(base_heading, n_frames)
base_dist    = _pad(base_dist,    n_frames)
base_reward  = _pad(base_reward,  n_frames)
rand_pos     = _pad(rand_pos,     n_frames + 1)
rand_heading = _pad(rand_heading, n_frames)
rand_dist    = _pad(rand_dist,    n_frames)
rand_reward  = _pad(rand_reward,  n_frames)

# ── Pre-compute gyre quiver field ─────────────────────────────────────────────
def _gyre_field(n_grid=8):
    xs = np.linspace(-HALF + 1, HALF - 1, n_grid)
    ys = np.linspace(-HALF + 1, HALF - 1, n_grid)
    X, Y = np.meshgrid(xs, ys)
    env = AUVComplexEnv()
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(n_grid):
        for j in range(n_grid):
            c = env._get_current(np.array([X[i,j], Y[i,j]], dtype=np.float32))
            U[i,j], V[i,j] = c[0], c[1]
    env.close()
    return X, Y, U, V

GX, GY, GU, GV = _gyre_field()

# ── Triangle marker for AUV heading ──────────────────────────────────────────
def _auv_triangle(cx, cy, heading, size=0.6):
    """Return (x_pts, y_pts) for a triangle pointing along `heading`."""
    tip   = np.array([cx + size * np.cos(heading),
                      cy + size * np.sin(heading)])
    left  = np.array([cx + size * 0.5 * np.cos(heading + 2.4),
                      cy + size * 0.5 * np.sin(heading + 2.4)])
    right = np.array([cx + size * 0.5 * np.cos(heading - 2.4),
                      cy + size * 0.5 * np.sin(heading - 2.4)])
    pts = np.array([tip, left, right, tip])
    return pts[:, 0], pts[:, 1]


# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("white")
fig.suptitle("Domain Randomization: Sim-to-Real Transfer",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.93,
         "Same architecture · Same training budget · Only difference: "
         "randomised physics during training",
         ha="center", fontsize=9, color="#555555")

def _init_ax(ax, title, colour):
    ax.set_facecolor("white")
    ax.add_patch(plt.Rectangle((-HALF, -HALF), ARENA_SIZE, ARENA_SIZE,
                               linewidth=1.5, edgecolor="#cccccc",
                               facecolor="#fafafa", zorder=0))
    ax.quiver(GX, GY, GU, GV, color="#aec6e8", alpha=0.4,
              scale=5.0, width=0.003, headwidth=3, headlength=4, zorder=1)
    # Goal
    goal_circle = plt.Circle(goal, GOAL_RADIUS, color=GOLD, alpha=0.25, zorder=2)
    ax.add_patch(goal_circle)
    ax.plot(*goal, "*", color=GOLD, markersize=16, zorder=3,
            markeredgecolor="#888800", markeredgewidth=0.8)
    # Start marker
    ax.plot(*start, "o", color=colour, markersize=8, zorder=3, alpha=0.5,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlim(-HALF - 0.5, HALF + 0.5)
    ax.set_ylim(-HALF - 0.5, HALF + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=colour, pad=8)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.tick_params(labelsize=8)

_init_ax(ax_b, "BASELINE AGENT  (Fixed Physics)", BLUE)
_init_ax(ax_r, "RANDOMISED AGENT  (Domain Randomised)", ORANGE)

# Dynamic elements — initialise empty
trail_b,  = ax_b.plot([], [], color=BLUE,   alpha=0.4, linewidth=1.5, zorder=4)
trail_r,  = ax_r.plot([], [], color=ORANGE, alpha=0.4, linewidth=1.5, zorder=4)
auv_b,    = ax_b.plot([], [], color=BLUE,   linewidth=1.5, zorder=5)
auv_r,    = ax_r.plot([], [], color=ORANGE, linewidth=1.5, zorder=5)

stats_b = ax_b.text(-HALF + 0.4, -HALF + 0.4, "",
                    fontsize=8, family="monospace", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor="none"), zorder=6)
stats_r = ax_r.text(-HALF + 0.4, -HALF + 0.4, "",
                    fontsize=8, family="monospace", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor="none"), zorder=6)

step_txt = fig.text(0.5, 0.01, "", ha="center", fontsize=9, color="#333333")

HOLD_FRAMES = 50   # freeze at end

def _status(reached, step, total, dist):
    if reached and step >= total - 1:
        return "✓ GOAL REACHED!"
    elif not reached and step >= total - 1:
        return "✗ TIMED OUT"
    return "NAVIGATING..."

def update(frame):
    # Clamp to valid index
    ib = min(frame, n_base - 1)
    ir = min(frame, n_rand - 1)

    # Trails
    trail_b.set_data(base_pos[:ib+2, 0], base_pos[:ib+2, 1])
    trail_r.set_data(rand_pos[:ir+2, 0], rand_pos[:ir+2, 1])

    # AUV triangles
    bx_pts, by_pts = _auv_triangle(*base_pos[ib+1], base_heading[ib])
    rx_pts, ry_pts = _auv_triangle(*rand_pos[ir+1], rand_heading[ir])
    auv_b.set_data(bx_pts, by_pts)
    auv_r.set_data(rx_pts, ry_pts)

    # Stats text
    b_stat = _status(base_success, frame, n_frames + HOLD_FRAMES - 1, base_dist[ib])
    r_stat = _status(rand_success, frame, n_frames + HOLD_FRAMES - 1, rand_dist[ir])

    stats_b.set_text(
        f"Dist:   {base_dist[ib]:.2f} m\n"
        f"Reward: {base_reward[ib]:.1f}\n"
        f"Step:   {min(frame+1, n_base)}/{n_base}\n"
        f"{b_stat}"
    )
    stats_r.set_text(
        f"Dist:   {rand_dist[ir]:.2f} m\n"
        f"Reward: {rand_reward[ir]:.1f}\n"
        f"Step:   {min(frame+1, n_rand)}/{n_rand}\n"
        f"{r_stat}"
    )

    step_txt.set_text(f"Step {min(frame+1, n_frames)} / {n_frames}  |  Sim B (Complex Physics)")

    return trail_b, trail_r, auv_b, auv_r, stats_b, stats_r, step_txt

total_frames = n_frames + HOLD_FRAMES
anim = animation.FuncAnimation(
    fig, update, frames=total_frames,
    interval=40, blit=True,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.92])

# ── Save MP4 ─────────────────────────────────────────────────────────────────
mp4_path = os.path.join(OUT_DIR, "transfer_animation.mp4")
gif_path = os.path.join(OUT_DIR, "transfer_animation.gif")

if shutil.which("ffmpeg"):
    print("Saving MP4…")
    writer_mp4 = animation.FFMpegWriter(fps=25, bitrate=1800,
                                         extra_args=["-vcodec", "libx264"])
    anim.save(mp4_path, writer=writer_mp4, dpi=120)
    print(f"MP4 saved → {mp4_path}")
else:
    print("ffmpeg not found — skipping MP4.")

print("Saving GIF…")
writer_gif = animation.PillowWriter(fps=25)
anim.save(gif_path, writer=writer_gif, dpi=80)
print(f"GIF saved → {gif_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 5.3 — Static filmstrip
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding filmstrip…")

# Key frame indices: step 0, 25%, 50%, final
key_steps = [
    0,
    n_frames // 4,
    n_frames // 2,
    n_frames - 1,
]
col_labels = ["Start", f"Step {n_frames//4}", f"Step {n_frames//2}", "Final"]

fig_fs, axes = plt.subplots(2, 4, figsize=(16, 8))
fig_fs.suptitle("Transfer Test — Filmstrip Summary", fontsize=13, fontweight="bold")
fig_fs.patch.set_facecolor("white")

ROW_LABELS = ["Baseline\n(Fixed Physics)", "Randomised\n(Domain Random.)"]
COLOURS    = [BLUE, ORANGE]
TRAJS      = [(base_pos, base_heading, base_dist, base_reward, base_success, n_base),
              (rand_pos, rand_heading, rand_dist, rand_reward, rand_success, n_rand)]

for row, (pos, heading, dist, reward, success, n_ep) in enumerate(TRAJS):
    for col, step in enumerate(key_steps):
        ax = axes[row][col]
        s  = min(step, n_ep - 1)

        # Arena
        ax.add_patch(plt.Rectangle((-HALF, -HALF), ARENA_SIZE, ARENA_SIZE,
                                   linewidth=1, edgecolor="#cccccc",
                                   facecolor="#fafafa", zorder=0))

        # Gyre (faint)
        ax.quiver(GX, GY, GU, GV, color="#aec6e8", alpha=0.3,
                  scale=6.0, width=0.003, headwidth=3, zorder=1)

        # Trail up to this step
        if s > 0:
            ax.plot(pos[:s+2, 0], pos[:s+2, 1],
                    color=COLOURS[row], alpha=0.5, linewidth=1.2, zorder=3)

        # AUV
        tx, ty = _auv_triangle(*pos[s+1], heading[s], size=0.7)
        ax.fill(tx, ty, color=COLOURS[row], zorder=4)

        # Start & goal
        ax.plot(*start, "o", color=COLOURS[row], markersize=5, alpha=0.4, zorder=3)
        ax.add_patch(plt.Circle(goal, GOAL_RADIUS, color=GOLD, alpha=0.2, zorder=2))
        ax.plot(*goal, "*", color=GOLD, markersize=12, zorder=3,
                markeredgecolor="#888800", markeredgewidth=0.6)

        ax.set_xlim(-HALF - 0.3, HALF + 0.3)
        ax.set_ylim(-HALF - 0.3, HALF + 0.3)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.2)

        # Column header
        if row == 0:
            ax.set_title(col_labels[col], fontsize=9, fontweight="bold")

        # Row label on left
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row], fontsize=8, fontweight="bold",
                          color=COLOURS[row])

        # Final column annotation
        if col == 3:
            if success:
                msg = f"SUCCESS\n{dist[s]:.2f} m  step {n_ep}"
                box_colour = "#d4edda"
            else:
                msg = f"FAILED\n{dist[s]:.2f} m from goal"
                box_colour = "#f8d7da"
            ax.text(0, -HALF + 1.2, msg,
                    ha="center", fontsize=7, fontweight="bold",
                    color="#155724" if success else "#721c24",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=box_colour, edgecolor="none"),
                    zorder=5)

fig_fs.tight_layout()
filmstrip_path = os.path.join(OUT_DIR, "transfer_filmstrip.png")
fig_fs.savefig(filmstrip_path, dpi=150, bbox_inches="tight")
print(f"Filmstrip saved → {filmstrip_path}")

plt.close("all")
print("\nDone.")
