"""
ski_env.py — Competition slalom skiing environment with curriculum learning.

Curriculum stages (controlled by `difficulty` in [0.0, 1.0]):
  - Gates spread from narrow (easy) to wide alternating (hard)
  - Gate width shrinks as difficulty rises
  - Miss penalty grows with difficulty
  - Track width stays constant

State (10D):
    [x, y, vx, vy, angle, angular_vel,
     next_gate_rel_x, next_gate_rel_y,
     next2_gate_rel_x, next2_gate_rel_y]

Action (1D): continuous steering torque in [-1, 1]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkiEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- Physics ---
    GRAVITY     = 9.81
    SLOPE_ANGLE = 20.0
    MASS        = 70.0
    DRAG        = 0.3
    MAX_TORQUE  = 50.0
    DT          = 0.05
    MAX_STEPS   = 700

    # --- Track ---
    SLOPE_LENGTH = 100.0
    TRACK_WIDTH  = 8.0
    FALL_ANGLE   = 35.0

    # --- Scoring (fixed reference values) ---
    GATE_PASS_REWARD  =  60.0
    FINISH_BONUS      = 200.0
    SPEED_BONUS_RATE  =   0.3
    FALL_PENALTY      =  80.0

    def __init__(self, render_mode=None, n_gates=8, difficulty=0.0):
        """
        difficulty: float in [0.0, 1.0]
          0.0 = easy (gates close to centre, wide gates, small miss penalty)
          1.0 = hard (full alternating offset, narrow gates, large penalty)
        """
        super().__init__()
        self.render_mode = render_mode
        self.n_gates     = n_gates
        self.difficulty  = float(np.clip(difficulty, 0.0, 1.0))

        self.observation_space = spaces.Box(
            low=np.full(10, -5.0, dtype=np.float32),
            high=np.full(10,  5.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._slope_rad = np.radians(self.SLOPE_ANGLE)
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
        """Lateral offset of alternating gates. 0.5 (easy) → 5.5 (hard)."""
        return 0.5 + self.difficulty * 5.0

    @property
    def gate_width(self):
        """Gate half-width. 2.5 (easy) → 1.2 (hard)."""
        return 2.5 - self.difficulty * 1.3

    @property
    def gate_miss_penalty(self):
        """Miss penalty. 20 (easy) → 150 (hard)."""
        return 20.0 + self.difficulty * 130.0

    @property
    def gate_y_tolerance(self):
        """Y-window to detect gate crossing. Wider when easy."""
        return 2.0 - self.difficulty * 0.8   # 2.0 → 1.2

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def set_difficulty(self, d: float):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state        = np.zeros(6, dtype=np.float64)
        self.step_count   = 0
        self.gates        = self._generate_gates()
        self.gates_passed = [False] * self.n_gates
        self.gates_missed = [False] * self.n_gates
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        torque = float(action[0]) * self.MAX_TORQUE

        x, y, vx, vy, theta, omega = self.state

        ax    = -self.DRAG * vx / self.MASS + torque / self.MASS * 0.1
        ay    = self.GRAVITY * np.sin(self._slope_rad) - self.DRAG * vy / self.MASS
        alpha = (torque - 0.5 * self.MASS * self.GRAVITY * np.sin(theta)) / (0.3 * self.MASS)
        alpha += self.np_random.normal(0, 0.3)

        vx    += ax    * self.DT
        vy    += ay    * self.DT
        x     += vx    * self.DT
        y     += vy    * self.DT
        omega += alpha * self.DT
        theta += omega * self.DT

        self.state      = np.array([x, y, vx, vy, theta, omega])
        self.step_count += 1

        fell    = abs(np.degrees(theta)) > self.FALL_ANGLE
        out_OOB = abs(x) > self.TRACK_WIDTH
        reached = y >= self.SLOPE_LENGTH
        timeout = self.step_count >= self.MAX_STEPS

        terminated = fell or out_OOB or reached
        truncated  = timeout and not terminated

        gate_reward = self._check_gates(x, y)
        reward = self._compute_reward(
            x, y, vx, vy, theta, fell, out_OOB, reached, gate_reward
        )

        n_passed = sum(self.gates_passed)
        n_missed = sum(self.gates_missed)
        info = {
            "gates_passed":  n_passed,
            "gates_missed":  n_missed,
            "n_gates":       self.n_gates,
            "perfect_run":   reached and n_missed == 0,
            "reached":       reached,
            "fell":          fell or out_OOB,
            "steps":         self.step_count,
            "difficulty":    self.difficulty,
            "score":         n_passed * self.GATE_PASS_REWARD
                             - n_missed * self.gate_miss_penalty,
        }

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward — shaped for learning
    # ------------------------------------------------------------------

    def _compute_reward(self, x, y, vx, vy, theta, fell, out_OOB, reached, gate_reward):
        reward = 0.0

        # 1. Forward progress (dense)
        reward += vy * self.DT * 1.0

        # 2. Alignment toward next gate (dense, scales with difficulty)
        gx = self._next_gate_x(y)
        alignment_weight = 0.05 + 0.15 * self.difficulty   # stronger pull when hard
        reward -= alignment_weight * abs(x - gx) * self.DT

        # 3. Stay upright
        reward += 0.05 * (np.cos(theta) - 1.0)

        # 4. Gate outcome (dominant sparse signal)
        reward += gate_reward

        # 5. Terminal
        if reached:
            steps_remaining = self.MAX_STEPS - self.step_count
            reward += self.FINISH_BONUS + steps_remaining * self.SPEED_BONUS_RATE
        if fell or out_OOB:
            reward -= self.FALL_PENALTY

        return float(reward)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _generate_gates(self):
        ys = np.linspace(10, 90, self.n_gates)
        offset = self.gate_offset
        xs = np.array([
            offset * (1 if i % 2 == 0 else -1)
            for i in range(self.n_gates)
        ], dtype=float)
        noise_scale = 0.3 + self.difficulty * 0.5   # more noise when hard
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
                    self.state[2] *= 0.4   # stumble effect
        return reward

    def _next_gate_x(self, y):
        for i, (gy, gx) in enumerate(self.gates):
            if not self.gates_passed[i] and not self.gates_missed[i] and gy > y:
                return gx
        return 0.0

    # ------------------------------------------------------------------
    # Observation (10D)
    # ------------------------------------------------------------------

    def _get_obs(self):
        x, y, vx, vy, theta, omega = self.state

        base = np.array([
            x     / self.TRACK_WIDTH,
            y     / self.SLOPE_LENGTH,
            vx    / 10.0,
            vy    / 20.0,
            theta / np.radians(self.FALL_ANGLE),
            omega / 5.0,
        ], dtype=np.float32)

        upcoming = self._get_upcoming_gates(y, n=2)
        return np.clip(np.concatenate([base, upcoming]), -5.0, 5.0)

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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
        TRACK_H   = 400          # vertical pixels for the slope
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Ski RL — Competition Slalom")
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()

        # Background — sky + slope
        self.screen.fill((170, 205, 235))
        pygame.draw.rect(self.screen, (230, 243, 255), (0, TRACK_TOP, W, H))

        # Draw track boundary lines
        left_x  = int(W / 2 - self.TRACK_WIDTH / self.TRACK_WIDTH * W * 0.45)
        right_x = int(W / 2 + self.TRACK_WIDTH / self.TRACK_WIDTH * W * 0.45)
        pygame.draw.line(self.screen, (150, 180, 220), (left_x,  TRACK_TOP), (left_x,  TRACK_TOP + TRACK_H), 2)
        pygame.draw.line(self.screen, (150, 180, 220), (right_x, TRACK_TOP), (right_x, TRACK_TOP + TRACK_H), 2)

        # Draw finish line (red line)
        finish_y = TRACK_TOP + TRACK_H
        pygame.draw.line(self.screen, (220, 30, 30), (left_x, finish_y), (right_x, finish_y), 4)

        # Draw gates
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

        # Draw skier
        x, y, *_, theta, _ = self.state
        sx = int(W / 2 + (x / self.TRACK_WIDTH) * W * 0.45)
        sy = int(TRACK_TOP + (y / self.SLOPE_LENGTH) * TRACK_H)
        L  = 20
        dx, dy = int(L * np.sin(theta)), int(L * np.cos(theta))
        pygame.draw.line(self.screen, (20, 60, 180), (sx, sy + dy), (sx, sy - dy), 4)
        pygame.draw.circle(self.screen, (255, 200, 140), (sx, sy - dy - 7), 7)

        # Skis
        ski_len = 14
        ski_angle = theta + 0.1
        for side in [-1, 1]:
            sx2 = sx + side * int(5 * np.cos(theta))
            sy2 = sy + side * int(5 * np.sin(theta))
            pygame.draw.line(
                self.screen, (30, 30, 80),
                (sx2 - int(ski_len * np.sin(ski_angle)), sy2 - int(ski_len * np.cos(ski_angle))),
                (sx2 + int(ski_len * np.sin(ski_angle)), sy2 + int(ski_len * np.cos(ski_angle))),
                3
            )

        # HUD
        font     = pygame.font.SysFont(None, 21)
        n_passed = sum(self.gates_passed)
        n_missed = sum(self.gates_missed)
        score    = n_passed * self.GATE_PASS_REWARD - n_missed * self.gate_miss_penalty
        hud1 = font.render(
            f"y={y:.1f}m  vy={self.state[3]:.1f}m/s  tilt={np.degrees(theta):.1f}°",
            True, (10, 10, 10)
        )
        hud2 = font.render(
            f"passed={n_passed}  missed={n_missed}  score={score:.0f}  "
            f"difficulty={self.difficulty:.2f}",
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