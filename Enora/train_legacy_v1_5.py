"""
train_legacy_v1_5.py — Training / evaluation script for the v1.5 environment.

This file bundles the v1.5 SkiEnv (14D obs, 2D action, heading-based physics)
so that checkpoints trained on that version (e.g. policy_final_ft_best.pt) can
be loaded and evaluated without modifying the current codebase.

Usage:
    python train_legacy_v1_5.py --eval --checkpoint "manual saves/policy_final_ft_best.pt" --render
    python train_legacy_v1_5.py --eval --checkpoint "manual saves/policy_final_ft_best.pt" --difficulty 0.0 0.2 0.5
    python train_legacy_v1_5.py --difficulty-sweep --checkpoint "manual saves/policy_final_ft_best.pt"
    python train_legacy_v1_5.py                    # train from scratch on v1.5 env
    python train_legacy_v1_5.py --curriculum --checkpoint "manual saves/policy_final_ft_best.pt"
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import importlib
_parent = str(Path(__file__).resolve().parent)
sys.path.insert(0, _parent)
# Import from local agents/ directory, avoiding collision with any pip 'agents' package
_spec = importlib.util.spec_from_file_location(
    "agents.ppo_agent",
    str(Path(_parent) / "agents" / "ppo_agent.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ActorCritic = _mod.ActorCritic
RolloutBuffer = _mod.RolloutBuffer
ppo_update = _mod.ppo_update

import gymnasium as gym
from gymnasium import spaces


# ======================================================================
# v1.5 SkiEnv — inlined from commit 750e375
# ======================================================================

class SkiEnvLegacy(gym.Env):
    """Competition slalom skiing environment (v1.5).

    14D observation, 2D action [steer, brake].
    Heading-based physics without edge drag.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- Physics ---
    GRAVITY      = 9.81
    SLOPE_ANGLE  = 20.0
    DT           = 0.05
    MAX_STEPS    = 700

    MAX_STEER_RATE = 2.5
    BASE_DRAG    = 0.02
    BRAKE_DRAG   = 0.25
    TURN_DRAG    = 0.10
    MAX_SPEED    = 25.0

    CRASH_LATERAL_G = 8.0

    # --- Track ---
    SLOPE_LENGTH = 120.0
    TRACK_WIDTH  = 10.0
    N_GATES_DEFAULT = 8

    # --- Scoring ---
    GATE_PASS_REWARD  =  60.0
    FINISH_BONUS      = 200.0
    SPEED_BONUS_RATE  =   0.3
    FALL_PENALTY      =  80.0

    def __init__(self, render_mode=None, n_gates=N_GATES_DEFAULT, difficulty=0.0):
        super().__init__()
        self.render_mode = render_mode
        self.n_gates     = n_gates
        self.difficulty  = float(np.clip(difficulty, 0.0, 1.0))

        self.observation_space = spaces.Box(
            low=np.full(14, -5.0, dtype=np.float32),
            high=np.full(14,  5.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._slope_rad   = np.radians(self.SLOPE_ANGLE)
        self._gravity_acc = self.GRAVITY * np.sin(self._slope_rad)

        self.state        = None
        self.gates        = []
        self.gates_passed = []
        self.gates_missed = []
        self.step_count   = 0
        self.screen       = None
        self.clock        = None

    # --- Difficulty-dependent parameters ---

    @property
    def gate_offset(self):
        return 0.5 + self.difficulty * 5.5

    @property
    def gate_width(self):
        return 2.5 - self.difficulty * 1.3

    @property
    def gate_miss_penalty(self):
        return 20.0 + self.difficulty * 130.0

    @property
    def gate_y_tolerance(self):
        return 2.0 - self.difficulty * 0.8

    # ------------------------------------------------------------------

    def set_difficulty(self, d: float):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float64)
        self.step_count   = 0
        self.gates        = self._generate_gates()
        self.gates_passed = [False] * self.n_gates
        self.gates_missed = [False] * self.n_gates
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        steer = float(action[0])
        brake = float(np.clip(action[1], 0.0, 1.0))

        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)

        speed_factor = 1.0 / (1.0 + 0.03 * speed)
        d_heading = steer * self.MAX_STEER_RATE * speed_factor * self.DT
        heading = heading + d_heading

        grav_ax = 0.0
        grav_ay = self._gravity_acc

        drag_coeff = (self.BASE_DRAG
                      + brake * self.BRAKE_DRAG
                      + abs(steer) * self.TURN_DRAG)

        if speed > 0.01:
            drag_ax = -drag_coeff * vx * speed
            drag_ay = -drag_coeff * vy * speed
        else:
            drag_ax = 0.0
            drag_ay = 0.0

        target_dx = np.sin(heading)
        target_dy = np.cos(heading)

        carve_strength = 5.0 * speed
        carve_ax = carve_strength * (target_dx - (vx / max(speed, 0.1))) * self.DT
        carve_ay = carve_strength * (target_dy - (vy / max(speed, 0.1))) * self.DT

        ax = grav_ax + drag_ax + carve_ax
        ay = grav_ay + drag_ay + carve_ay
        vx += ax * self.DT
        vy += ay * self.DT

        speed_new = np.sqrt(vx**2 + vy**2)
        if speed_new > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed_new
            vx *= scale
            vy *= scale
            speed_new = self.MAX_SPEED

        vx += self.np_random.normal(0, 0.05) * self.DT
        vy += self.np_random.normal(0, 0.05) * self.DT

        x += vx * self.DT
        y += vy * self.DT

        self.state = np.array([x, y, vx, vy, heading])
        self.step_count += 1

        centripetal = speed * abs(d_heading) / self.DT if self.DT > 0 else 0
        crashed = centripetal > self.CRASH_LATERAL_G
        out_OOB = abs(x) > self.TRACK_WIDTH
        reached = y >= self.SLOPE_LENGTH
        timeout = self.step_count >= self.MAX_STEPS

        terminated = crashed or out_OOB or reached
        truncated  = timeout and not terminated

        gate_reward = self._check_gates(x, y)
        reward = self._compute_reward(
            x, y, vx, vy, heading, speed_new,
            crashed, out_OOB, reached, gate_reward, brake
        )

        n_passed = sum(self.gates_passed)
        n_missed = sum(self.gates_missed)
        info = {
            "gates_passed":  n_passed,
            "gates_missed":  n_missed,
            "n_gates":       self.n_gates,
            "perfect_run":   reached and n_missed == 0,
            "reached":       reached,
            "fell":          crashed or out_OOB,
            "timeout":       truncated,
            "steps":         self.step_count,
            "difficulty":    self.difficulty,
            "speed":         speed_new,
            "score":         n_passed * self.GATE_PASS_REWARD
                             - n_missed * self.gate_miss_penalty,
        }

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, x, y, vx, vy, heading, speed,
                        crashed, out_OOB, reached, gate_reward, brake):
        reward = 0.0
        reward += vy * self.DT * 0.8
        gx = self._next_gate_x(y)
        alignment_weight = 0.05 + 0.15 * self.difficulty
        reward -= alignment_weight * abs(x - gx) * self.DT
        reward -= 0.02 * brake
        reward += gate_reward
        if reached:
            steps_remaining = self.MAX_STEPS - self.step_count
            reward += self.FINISH_BONUS + steps_remaining * self.SPEED_BONUS_RATE
        if crashed or out_OOB:
            reward -= self.FALL_PENALTY
        return float(reward)

    def _generate_gates(self):
        ys = np.linspace(12, self.SLOPE_LENGTH - 12, self.n_gates)
        offset = self.gate_offset
        xs = np.array([
            offset * (1 if i % 2 == 0 else -1)
            for i in range(self.n_gates)
        ], dtype=float)
        noise_scale = 0.3 + self.difficulty * 0.5
        xs += self.np_random.uniform(-noise_scale, noise_scale, self.n_gates)
        return list(zip(ys.tolist(), xs.tolist()))

    def _check_gates(self, x, y):
        reward = 0.0
        tol = self.gate_y_tolerance
        for i, (gy, gx) in enumerate(self.gates):
            if self.gates_passed[i] or self.gates_missed[i]:
                continue
            if abs(y - gy) < tol:
                if abs(x - gx) <= self.gate_width:
                    self.gates_passed[i] = True
                    reward += self.GATE_PASS_REWARD
                else:
                    self.gates_missed[i] = True
                    reward -= self.gate_miss_penalty
        return reward

    def _next_gate_x(self, y):
        for i, (gy, gx) in enumerate(self.gates):
            if not self.gates_passed[i] and not self.gates_missed[i] and gy > y:
                return gx
        return 0.0

    def _get_obs(self):
        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)

        base = np.array([
            x       / self.TRACK_WIDTH,
            y       / self.SLOPE_LENGTH,
            vx      / 10.0,
            vy      / 20.0,
            heading / np.pi,
            speed   / self.MAX_SPEED,
        ], dtype=np.float32)

        upcoming = self._get_upcoming_gates(y, n=2)

        gate_width_norm = self.gate_width / 2.5
        next_gx = self._next_gate_x(y)
        lateral_offset = next_gx - x
        lateral_error = np.sign(lateral_offset) * vx / 10.0
        gates_passed_ratio = sum(self.gates_passed) / self.n_gates

        context = np.array([
            gate_width_norm,
            lateral_error,
            gates_passed_ratio,
            self.difficulty,
        ], dtype=np.float32)

        return np.clip(np.concatenate([base, upcoming, context]), -5.0, 5.0)

    def _get_upcoming_gates(self, y, n=2):
        result = []
        for i, (gy, gx) in enumerate(self.gates):
            if len(result) >= n * 2:
                break
            if gy > y and not self.gates_passed[i] and not self.gates_missed[i]:
                result.append(gx       / self.TRACK_WIDTH)
                result.append((gy - y) / self.SLOPE_LENGTH)
        while len(result) < n * 2:
            result.append(0.0)
        return np.array(result, dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        try:
            import pygame
        except ImportError:
            print("pip install pygame")
            return None

        W, H = 600, 520
        TRACK_TOP = 60
        TRACK_H   = 400
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Ski RL — v1.5 Legacy")
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()

        self.screen.fill((170, 205, 235))
        pygame.draw.rect(self.screen, (230, 243, 255), (0, TRACK_TOP, W, H))

        left_x  = int(W / 2 - W * 0.45)
        right_x = int(W / 2 + W * 0.45)
        pygame.draw.line(self.screen, (150, 180, 220),
                         (left_x, TRACK_TOP), (left_x, TRACK_TOP + TRACK_H), 2)
        pygame.draw.line(self.screen, (150, 180, 220),
                         (right_x, TRACK_TOP), (right_x, TRACK_TOP + TRACK_H), 2)

        finish_y = TRACK_TOP + TRACK_H
        pygame.draw.line(self.screen, (220, 30, 30),
                         (left_x, finish_y), (right_x, finish_y), 4)

        for i, (gy, gx) in enumerate(self.gates):
            sx = int(W / 2 + (gx / self.TRACK_WIDTH) * W * 0.45)
            sy = int(TRACK_TOP + (gy / self.SLOPE_LENGTH) * TRACK_H)
            hw = int(self.gate_width / self.TRACK_WIDTH * W * 0.45)

            if self.gates_passed[i]:
                col = (60, 200, 60)
            elif self.gates_missed[i]:
                col = (100, 100, 100)
            else:
                col = (200, 40, 40) if i % 2 == 0 else (40, 80, 200)

            pygame.draw.line(self.screen, col, (sx - hw, sy), (sx + hw, sy), 4)
            pygame.draw.circle(self.screen, col, (sx - hw, sy), 6)
            pygame.draw.circle(self.screen, col, (sx + hw, sy), 6)

            font_sm = pygame.font.SysFont(None, 18)
            lbl = font_sm.render(str(i + 1), True, col)
            self.screen.blit(lbl, (sx + hw + 4, sy - 8))

        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)
        sx = int(W / 2 + (x / self.TRACK_WIDTH) * W * 0.45)
        sy = int(TRACK_TOP + (y / self.SLOPE_LENGTH) * TRACK_H)

        L = 18
        dx_h = int(L * np.sin(heading))
        dy_h = int(L * np.cos(heading))
        pygame.draw.line(self.screen, (20, 60, 180),
                         (sx - dx_h, sy - dy_h), (sx + dx_h, sy + dy_h), 4)
        pygame.draw.circle(self.screen, (255, 200, 140),
                           (sx - dx_h, sy - dy_h), 7)

        ski_len = 12
        for side in [-1, 1]:
            ox = side * int(4 * np.cos(heading))
            oy = side * int(4 * np.sin(heading))
            pygame.draw.line(
                self.screen, (30, 30, 80),
                (sx + ox - int(ski_len * np.sin(heading)),
                 sy + oy - int(ski_len * np.cos(heading))),
                (sx + ox + int(ski_len * np.sin(heading)),
                 sy + oy + int(ski_len * np.cos(heading))),
                3
            )

        font = pygame.font.SysFont(None, 21)
        n_passed = sum(self.gates_passed)
        n_missed = sum(self.gates_missed)
        score = n_passed * self.GATE_PASS_REWARD - n_missed * self.gate_miss_penalty
        hud1 = font.render(
            f"y={y:.1f}m  speed={speed:.1f}m/s  heading={np.degrees(heading):.1f}deg",
            True, (10, 10, 10)
        )
        hud2 = font.render(
            f"passed={n_passed}  missed={n_missed}  score={score:.0f}  "
            f"diff={self.difficulty:.2f}",
            True, (10, 10, 10)
        )
        self.screen.blit(hud1, (8, 8))
        self.screen.blit(hud2, (8, 28))

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit("Window closed by user.")
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None


# ======================================================================
# Config
# ======================================================================

CFG = dict(
    total_steps    = 500_000,
    rollout_steps  = 2048,
    lr             = 3e-4,
    gamma          = 0.99,
    lam            = 0.95,
    clip_eps       = 0.2,
    vf_coef        = 0.5,
    ent_coef       = 0.01,
    n_epochs       = 4,
    log_interval   = 10,
    save_interval  = 50_000,
    save_dir       = "checkpoints",
)


# ======================================================================
# Training
# ======================================================================

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(render: bool = False, seed: int = 0, checkpoint: str | None = None):
    _set_seed(seed)

    render_mode = "human" if render else None
    env = SkiEnvLegacy(render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v1.5] Training on {device}  |  obs={obs_dim}  act={act_dim}  seed={seed}")

    policy = ActorCritic(obs_dim, act_dim).to(device)
    if checkpoint:
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"  Loaded checkpoint: {checkpoint}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=CFG["lr"])
    buffer    = RolloutBuffer()

    run_dir = Path(CFG["save_dir"]) / f"legacy_v1_5_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []

    ep_rewards = []
    recent_rewards = deque(maxlen=100)
    global_step = 0
    ep_reward = 0.0
    ep_len    = 0
    ep_count  = 0

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    while global_step < CFG["total_steps"]:
        buffer.reset()
        for _ in range(CFG["rollout_steps"]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            buffer.add(
                obs, action_np,
                log_prob.item(), reward,
                value.item(), float(terminated or truncated),
            )

            ep_reward += reward
            ep_len    += 1
            global_step += 1
            obs = next_obs

            if terminated or truncated:
                ep_rewards.append(ep_reward)
                recent_rewards.append(ep_reward)
                ep_count += 1

                if ep_count % CFG["log_interval"] == 0:
                    mean_r = np.mean(recent_rewards)
                    elapsed = time.time() - t_start
                    print(
                        f"Step {global_step:>7d} | "
                        f"Ep {ep_count:>5d} | "
                        f"MeanR(100) {mean_r:>8.1f} | "
                        f"EpLen {ep_len:>4d} | "
                        f"{elapsed:.0f}s"
                    )

                ep_reward = 0.0
                ep_len    = 0
                obs, _ = env.reset()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_value = policy.get_action(obs_t)

        ppo_update(
            policy, optimizer, buffer, last_value.item(),
            clip_eps=CFG["clip_eps"], vf_coef=CFG["vf_coef"],
            ent_coef=CFG["ent_coef"], n_epochs=CFG["n_epochs"],
            gamma=CFG["gamma"], lam=CFG["lam"], device=device,
        )

        if global_step % CFG["save_interval"] < CFG["rollout_steps"]:
            ckpt = run_dir / f"policy_{global_step}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")

            metrics = validate(policy, device)
            metrics["step"] = global_step
            metrics_log.append(metrics)
            print_validation(metrics, global_step)

    env.close()

    final_ckpt = run_dir / "policy_final.pt"
    torch.save(policy.state_dict(), final_ckpt)

    metrics = validate(policy, device)
    metrics["step"] = global_step
    metrics_log.append(metrics)
    print_validation(metrics, global_step)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": CFG, "validation": metrics_log,
                   "episode_rewards": ep_rewards}, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")
    print(f"[v1.5] Training complete (seed={seed}).")
    return metrics_log


# ======================================================================
# Validation
# ======================================================================

def validate(policy, device, n_episodes: int = 20, render_mode=None,
             difficulty: float = 0.0):
    env = SkiEnvLegacy(render_mode=render_mode, difficulty=difficulty)
    policy.eval()

    results = {
        "reward": [], "reached": [], "fell": [], "timeout": [],
        "gates_passed": [], "gates_missed": [],
        "perfect_run": [], "score": [],
    }

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_t, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            total_reward += reward
            done = terminated or truncated

        results["reward"].append(total_reward)
        results["reached"].append(float(info.get("reached", False)))
        results["fell"].append(float(info.get("fell", False)))
        results["timeout"].append(float(info.get("timeout", False)))
        results["gates_passed"].append(info.get("gates_passed", 0))
        results["gates_missed"].append(info.get("gates_missed", 0))
        results["perfect_run"].append(float(info.get("perfect_run", False)))
        results["score"].append(info.get("score", 0.0))

    env.close()
    policy.train()

    return {
        "difficulty":        difficulty,
        "finish_rate":       np.mean(results["reached"]),
        "fall_rate":         np.mean(results["fell"]),
        "timeout_rate":      np.mean(results["timeout"]),
        "avg_gates_passed":  np.mean(results["gates_passed"]),
        "avg_gates_missed":  np.mean(results["gates_missed"]),
        "perfect_run_rate":  np.mean(results["perfect_run"]),
        "avg_score":         np.mean(results["score"]),
        "avg_reward":        np.mean(results["reward"]),
    }


def print_validation(metrics, step):
    diff = metrics.get("difficulty", 0.0)
    print(
        f"\n{'='*60}\n"
        f"  VALIDATION @ step {step}  (difficulty={diff:.2f})\n"
        f"  Finish rate:      {metrics['finish_rate']:>6.1%}\n"
        f"  Fall rate:        {metrics['fall_rate']:>6.1%}\n"
        f"  Timeout rate:     {metrics['timeout_rate']:>6.1%}\n"
        f"  Avg gates passed: {metrics['avg_gates_passed']:>6.1f}\n"
        f"  Avg gates missed: {metrics['avg_gates_missed']:>6.1f}\n"
        f"  Perfect run rate: {metrics['perfect_run_rate']:>6.1%}\n"
        f"  Avg score:        {metrics['avg_score']:>8.1f}\n"
        f"  Avg reward:       {metrics['avg_reward']:>8.1f}\n"
        f"{'='*60}\n"
    )


# ======================================================================
# Evaluation (load checkpoint, run, print)
# ======================================================================

def evaluate(checkpoint: str, n_episodes: int = 20, render: bool = False,
             difficulties: list[float] | None = None):
    env_tmp = SkiEnvLegacy()
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ActorCritic(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"[v1.5] Loaded {checkpoint}  (obs={obs_dim}, act={act_dim})")

    if difficulties is None:
        difficulties = [0.0]

    render_mode = "human" if render else None
    for diff in difficulties:
        metrics = validate(policy, device, n_episodes=n_episodes,
                           render_mode=render_mode, difficulty=diff)
        print_validation(metrics, step="final")


# ======================================================================
# Difficulty sweep
# ======================================================================

def run_difficulty_sweep(checkpoint: str, n_episodes: int = 30, steps: int = 11):
    env_tmp = SkiEnvLegacy()
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ActorCritic(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))

    difficulties = np.linspace(0.0, 1.0, steps)
    sweep_results = []

    for diff in difficulties:
        print(f"  Evaluating difficulty={diff:.2f} ...", end=" ", flush=True)
        metrics = validate(policy, device, n_episodes=n_episodes, difficulty=diff)
        sweep_results.append(metrics)
        print(
            f"finish={metrics['finish_rate']:.0%}  "
            f"fall={metrics['fall_rate']:.0%}  "
            f"gates={metrics['avg_gates_passed']:.1f}  "
            f"score={metrics['avg_score']:.0f}  "
            f"reward={metrics['avg_reward']:.0f}"
        )

    report_dir = Path("baseline_results")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "legacy_v1_5_difficulty_sweep.json"
    with open(report_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "n_episodes": n_episodes,
                   "results": sweep_results}, f, indent=2)
    print(f"\nSweep saved to {report_path}")


# ======================================================================
# Curriculum training
# ======================================================================

CURRICULUM_STRATEGIES = {
    "conservative": dict(
        total_steps=1_000_000, start_difficulty=0.0, max_difficulty=1.0,
        difficulty_step=0.05, promote_threshold=0.75, promote_finish=0.80,
        promote_window=3, eval_interval=20_000, regress_threshold=0.6,
        sampling="fixed",
    ),
    "focused": dict(
        total_steps=1_500_000, start_difficulty=0.0, max_difficulty=1.0,
        difficulty_step=0.05, promote_threshold=0.70, promote_finish=0.80,
        promote_window=3, eval_interval=15_000, regress_threshold=0.5,
        sampling="range", sampling_spread=0.08,
    ),
    "fine_tune": dict(
        total_steps=1_500_000, start_difficulty=0.2, max_difficulty=1.0,
        difficulty_step=0.05, promote_threshold=0.70, promote_finish=0.80,
        promote_window=3, eval_interval=15_000, regress_threshold=0.5,
        sampling="range", sampling_spread=0.08,
    ),
}


def _sample_difficulty(cur_diff, cfg, rng):
    if cfg["sampling"] == "fixed":
        return cur_diff
    spread = cfg.get("sampling_spread", 0.15)
    lo = max(0.0, cur_diff - spread)
    hi = min(cfg["max_difficulty"], cur_diff + spread)
    return float(rng.uniform(lo, hi))


def train_curriculum(seed: int = 0, checkpoint: str | None = None,
                     strategy: str = "focused"):
    if strategy not in CURRICULUM_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(CURRICULUM_STRATEGIES.keys())}")

    cur_cfg = CURRICULUM_STRATEGIES[strategy]
    _set_seed(seed)
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SkiEnvLegacy()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, act_dim).to(device)
    if checkpoint:
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"[v1.5] Warm-started from {checkpoint}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=CFG["lr"])
    buffer = RolloutBuffer()

    warm_tag = "warm" if checkpoint else "scratch"
    run_dir = Path(CFG["save_dir"]) / f"legacy_v1_5_curriculum_{strategy}_{warm_tag}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    difficulty = cur_cfg["start_difficulty"]
    ep_difficulty = difficulty
    promote_count = 0
    env.set_difficulty(ep_difficulty)

    ep_rewards = []
    recent_rewards = deque(maxlen=100)
    global_step = 0
    ep_reward = 0.0
    ep_len = 0
    ep_count = 0
    metrics_log = []
    difficulty_history = []

    total_steps = cur_cfg["total_steps"]
    eval_interval = cur_cfg["eval_interval"]
    last_eval_step = 0

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    print(f"[v1.5] Curriculum on {device}  |  seed={seed}  strategy={strategy}")

    while global_step < total_steps:
        buffer.reset()
        for _ in range(CFG["rollout_steps"]):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            buffer.add(
                obs, action_np,
                log_prob.item(), reward,
                value.item(), float(terminated or truncated),
            )

            ep_reward += reward
            ep_len += 1
            global_step += 1
            obs = next_obs

            if terminated or truncated:
                ep_rewards.append(ep_reward)
                recent_rewards.append(ep_reward)
                ep_count += 1

                if ep_count % CFG["log_interval"] == 0:
                    mean_r = np.mean(recent_rewards)
                    elapsed = time.time() - t_start
                    print(
                        f"Step {global_step:>7d} | "
                        f"Ep {ep_count:>5d} | "
                        f"MeanR(100) {mean_r:>8.1f} | "
                        f"Diff {difficulty:.2f} "
                        f"(ep={ep_difficulty:.2f}) | "
                        f"{elapsed:.0f}s"
                    )

                ep_reward = 0.0
                ep_len = 0

                ep_difficulty = _sample_difficulty(difficulty, cur_cfg, rng)
                env.set_difficulty(ep_difficulty)
                obs, _ = env.reset()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_value = policy.get_action(obs_t)

        ppo_update(
            policy, optimizer, buffer, last_value.item(),
            clip_eps=CFG["clip_eps"], vf_coef=CFG["vf_coef"],
            ent_coef=CFG["ent_coef"], n_epochs=CFG["n_epochs"],
            gamma=CFG["gamma"], lam=CFG["lam"], device=device,
        )

        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            difficulty_history.append((global_step, difficulty))

            m_curr = validate(policy, device, n_episodes=20, difficulty=difficulty)
            m_curr["step"] = global_step
            m_curr["curriculum_difficulty"] = difficulty

            m_base = validate(policy, device, n_episodes=20, difficulty=0.0)

            gate_rate = m_curr["avg_gates_passed"] / env.n_gates
            finish_rate = m_curr["finish_rate"]
            base_gate_rate = m_base["avg_gates_passed"] / env.n_gates

            print(
                f"\n  CURRICULUM CHECK @ step {global_step}  "
                f"(target diff={difficulty:.2f})\n"
                f"    Current:  gates={m_curr['avg_gates_passed']:.1f}/{env.n_gates}  "
                f"({gate_rate:.0%})  finish={finish_rate:.0%}\n"
                f"    Base d=0: gates={m_base['avg_gates_passed']:.1f}/{env.n_gates}  "
                f"({base_gate_rate:.0%})  finish={m_base['finish_rate']:.0%}"
            )

            regressing = base_gate_rate < cur_cfg["regress_threshold"]
            gates_ok = gate_rate >= cur_cfg["promote_threshold"]
            finish_ok = finish_rate >= cur_cfg["promote_finish"]

            if regressing:
                promote_count = 0
                print(f"    >> HOLDING — regression")
            elif gates_ok and finish_ok:
                promote_count += 1
                if promote_count >= cur_cfg["promote_window"]:
                    old_diff = difficulty
                    difficulty = min(difficulty + cur_cfg["difficulty_step"],
                                    cur_cfg["max_difficulty"])
                    promote_count = 0
                    print(f"    >> PROMOTED {old_diff:.2f} -> {difficulty:.2f}")
                else:
                    print(f"    >> Streak: {promote_count}/{cur_cfg['promote_window']}")
            else:
                promote_count = 0
                print(f"    >> Not ready")

            metrics_log.append({
                "current": m_curr,
                "baseline_regression": {
                    "difficulty": 0.0,
                    "gate_rate": base_gate_rate,
                    "finish_rate": m_base["finish_rate"],
                },
                "training_difficulty": difficulty,
            })

            ckpt = run_dir / f"policy_{global_step}_d{difficulty:.2f}.pt"
            torch.save(policy.state_dict(), ckpt)

    env.close()

    final_ckpt = run_dir / "policy_final.pt"
    torch.save(policy.state_dict(), final_ckpt)
    print(f"\n[v1.5] Curriculum complete (seed={seed}, strategy={strategy}, "
          f"final difficulty={difficulty:.2f}).")
    return metrics_log


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / evaluate with the v1.5 legacy environment (14D obs, 2D action)")
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0])
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--difficulty-sweep", action="store_true")
    parser.add_argument("--sweep-steps", type=int, default=11)
    parser.add_argument("--sweep-episodes", type=int, default=30)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-strategy", default="focused",
                        choices=list(CURRICULUM_STRATEGIES.keys()))
    args = parser.parse_args()

    if args.curriculum:
        train_curriculum(seed=args.seed, checkpoint=args.checkpoint,
                         strategy=args.curriculum_strategy)
    elif args.difficulty_sweep:
        if args.checkpoint is None:
            raise ValueError("Pass --checkpoint <path> for difficulty sweep")
        run_difficulty_sweep(args.checkpoint, n_episodes=args.sweep_episodes,
                             steps=args.sweep_steps)
    elif args.eval:
        if args.checkpoint is None:
            raise ValueError("Pass --checkpoint <path> to evaluate")
        evaluate(args.checkpoint, n_episodes=args.n_episodes,
                 render=args.render, difficulties=args.difficulty)
    else:
        train(render=args.render, seed=args.seed, checkpoint=args.checkpoint)
