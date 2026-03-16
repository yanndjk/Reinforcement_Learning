"""
ski_env.py — Competition slalom skiing environment (v2).

Redesigned dynamics with heading-based turning and 2D action space.

Physics model:
  - Skier moves on a slope with gravity pulling downhill (+y direction)
  - Heading angle controls direction of travel relative to the fall line
  - Steering action rotates heading; sharper turns at high speed cost more grip
  - Brake action applies edge pressure, increasing drag to trade speed for control
  - Crash if: speed too high during sharp turn (loss of edge grip), or out of bounds

State (5 internal): [x, y, vx, vy, heading]
Observation (14D): see _get_obs
Action (2D): [steer, brake] both in [-1, 1], brake clamped to [0, 1]

Curriculum stages (controlled by `difficulty` in [0.0, 1.0]):
  - Gates spread from narrow (easy) to wide alternating (hard)
  - Gate width shrinks as difficulty rises
  - Miss penalty grows with difficulty
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkiEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- Physics ---
    GRAVITY      = 9.81
    SLOPE_ANGLE  = 20.0          # degrees, steepness of hill
    DT           = 0.05
    MAX_STEPS    = 700

    # Turning
    MAX_STEER_RATE = 2.5         # rad/s — max heading change rate
    # Speed
    BASE_DRAG    = 0.02          # air/snow friction (always present)
    BRAKE_DRAG   = 0.25          # additional drag when braking fully
    TURN_DRAG    = 0.10          # additional drag from carving (proportional to |steer|)
    MAX_SPEED    = 25.0          # absolute speed cap (terminal velocity)

    # Crash conditions
    CRASH_LATERAL_G = 8.0        # max centripetal acceleration before wipeout (m/s^2)

    # --- Track ---
    SLOPE_LENGTH = 120.0         # longer slope for more gate spacing
    TRACK_WIDTH  = 10.0          # wider track for larger gate offsets
    N_GATES_DEFAULT = 4

    # --- Scoring (survival-first, gates secondary) ---
    GATE_PASS_REWARD  =  15.0    # reduced — don't tempt risky maneuvers
    FINISH_BONUS      = 500.0    # dominant goal: get down the slope alive
    SPEED_BONUS_RATE  =   0.1    # small bonus for finishing fast
    FALL_PENALTY      = 300.0    # crashing is catastrophic
    SAFE_SPEED        =  15.0    # speed above this gets penalised during turns

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
        # Action: [steer, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._slope_rad   = np.radians(self.SLOPE_ANGLE)
        self._gravity_acc = self.GRAVITY * np.sin(self._slope_rad)

        self.state        = None    # [x, y, vx, vy, heading]
        self.gates        = []
        self.gates_passed = []
        self.gates_missed = []
        self.step_count   = 0
        self.screen       = None
        self.clock        = None

    # --- Difficulty-dependent parameters ---

    @property
    def gate_offset(self):
        """Lateral offset of alternating gates. 0.5 (easy) -> 6.0 (hard)."""
        return 0.5 + self.difficulty * 5.5

    @property
    def gate_width(self):
        """Gate half-width. 2.5 (easy) -> 1.2 (hard)."""
        return 2.5 - self.difficulty * 1.3

    @property
    def gate_miss_penalty(self):
        """Miss penalty. 20 (easy) -> 150 (hard)."""
        return 20.0 + self.difficulty * 130.0

    @property
    def gate_y_tolerance(self):
        """Y-window to detect gate crossing. Loosened to isolate steering
        quality from crossing-timing sensitivity during early training.
        Tighten later once steering is reliable."""
        return 3.0 - self.difficulty * 1.0   # 3.0 -> 2.0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def set_difficulty(self, d: float):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # State: [x, y, vx, vy, heading]
        # Start at top, slight downhill velocity, heading straight down
        self.state = np.array([0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float64)
        self.step_count   = 0
        self.gates        = self._generate_gates()
        self.gates_passed = [False] * self.n_gates
        self.gates_missed = [False] * self.n_gates
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        steer = float(action[0])                     # [-1, 1]
        brake = float(np.clip(action[1], 0.0, 1.0))  # [0, 1]

        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)

        # --- Heading update ---
        # Steering rate scales with action; slight reduction at very high speed
        # (harder to turn when going fast, like real skiing)
        speed_factor = 1.0 / (1.0 + 0.03 * speed)
        d_heading = steer * self.MAX_STEER_RATE * speed_factor * self.DT
        heading = heading + d_heading

        # --- Acceleration ---
        # Gravity component along the slope (always pushes +y = downhill)
        grav_ax = 0.0
        grav_ay = self._gravity_acc

        # Total drag = base + brake contribution + turn contribution
        drag_coeff = (self.BASE_DRAG
                      + brake * self.BRAKE_DRAG
                      + abs(steer) * self.TURN_DRAG)

        # Drag opposes current velocity
        if speed > 0.01:
            drag_ax = -drag_coeff * vx * speed
            drag_ay = -drag_coeff * vy * speed
        else:
            drag_ax = 0.0
            drag_ay = 0.0

        # Heading steers the velocity: apply a lateral force that rotates
        # velocity toward the heading direction.
        # Target velocity direction from heading (heading=0 means straight down)
        target_dx = np.sin(heading)
        target_dy = np.cos(heading)

        # Carving force: redirects velocity toward heading direction
        # Strength proportional to speed (no speed = no carving)
        carve_strength = 5.0 * speed
        carve_ax = carve_strength * (target_dx - (vx / max(speed, 0.1))) * self.DT
        carve_ay = carve_strength * (target_dy - (vy / max(speed, 0.1))) * self.DT

        # Integrate
        ax = grav_ax + drag_ax + carve_ax
        ay = grav_ay + drag_ay + carve_ay
        vx += ax * self.DT
        vy += ay * self.DT

        # Speed cap
        speed_new = np.sqrt(vx**2 + vy**2)
        if speed_new > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed_new
            vx *= scale
            vy *= scale
            speed_new = self.MAX_SPEED

        # Add small noise for realism
        vx += self.np_random.normal(0, 0.05) * self.DT
        vy += self.np_random.normal(0, 0.05) * self.DT

        # Position update
        x += vx * self.DT
        y += vy * self.DT

        self.state = np.array([x, y, vx, vy, heading])
        self.step_count += 1

        # --- Crash detection ---
        # Centripetal acceleration: v^2 * d_heading / dt
        centripetal = speed * abs(d_heading) / self.DT if self.DT > 0 else 0
        crashed = centripetal > self.CRASH_LATERAL_G
        out_OOB = abs(x) > self.TRACK_WIDTH
        reached = y >= self.SLOPE_LENGTH
        timeout = self.step_count >= self.MAX_STEPS

        terminated = crashed or out_OOB or reached
        truncated  = timeout and not terminated

        # --- Reward ---
        gate_reward = self._check_gates(x, y)
        reward = self._compute_reward(
            x, y, vx, vy, heading, speed_new,
            crashed, out_OOB, reached, gate_reward,
            centripetal, steer
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

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, x, y, vx, vy, heading, speed,
                        crashed, out_OOB, reached, gate_reward,
                        centripetal, steer):
        reward = 0.0

        # --- Priority 1: Survival ---

        # 1a. Alive bonus — every step not crashed is good
        reward += 0.15

        # 1b. Crash danger shaping — penalise approaching the wipeout threshold
        #     Ramps from 0 at 50% of limit to -0.5 at the limit
        danger = centripetal / self.CRASH_LATERAL_G   # 0 → 1+
        if danger > 0.5:
            reward -= 1.0 * (danger - 0.5)

        # --- Priority 2: Finish the run ---

        # 2a. Forward progress — gentle downhill incentive
        reward += vy * self.DT * 0.5

        # --- Priority 3: Safe turning & speed management ---

        # 3a. Penalise high speed during sharp turns
        #     (teaches brake-before-turn coordination)
        if speed > self.SAFE_SPEED and abs(steer) > 0.3:
            excess = (speed - self.SAFE_SPEED) / self.MAX_SPEED
            reward -= 0.3 * excess * abs(steer)

        # --- Priority 4: Gates (reduced — survival first) ---

        # 4a. Alignment toward next gate (very light)
        gx = self._next_gate_x(y)
        alignment_weight = 0.02 + 0.03 * self.difficulty
        reward -= alignment_weight * abs(x - gx) * self.DT

        # 4b. Gate outcome (sparse, reduced magnitude)
        reward += gate_reward

        # --- Terminal ---
        if reached:
            steps_remaining = self.MAX_STEPS - self.step_count
            reward += self.FINISH_BONUS + steps_remaining * self.SPEED_BONUS_RATE
        if crashed or out_OOB:
            reward -= self.FALL_PENALTY

        return float(reward)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _generate_gates(self):
        # Wider spacing: gates from y=12 to y=108 on a 120m slope
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

    # ------------------------------------------------------------------
    # Observation (14D)
    # ------------------------------------------------------------------

    def _get_obs(self):
        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)

        base = np.array([
            x       / self.TRACK_WIDTH,
            y       / self.SLOPE_LENGTH,
            vx      / 10.0,
            vy      / 20.0,
            heading / np.pi,               # normalise heading to [-1, 1]
            speed   / self.MAX_SPEED,      # normalised speed
        ], dtype=np.float32)

        # 2-gate lookahead (4 values)
        upcoming = self._get_upcoming_gates(y, n=2)

        # Gate width normalised
        gate_width_norm = self.gate_width / 2.5

        # Lateral error: is vx moving toward or away from next gate?
        next_gx = self._next_gate_x(y)
        lateral_offset = next_gx - x
        lateral_error = np.sign(lateral_offset) * vx / 10.0

        # Course progress
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
        TRACK_H   = 400
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Ski RL — Competition Slalom v2")
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()

        # Background
        self.screen.fill((170, 205, 235))
        pygame.draw.rect(self.screen, (230, 243, 255), (0, TRACK_TOP, W, H))

        # Track boundaries
        left_x  = int(W / 2 - W * 0.45)
        right_x = int(W / 2 + W * 0.45)
        pygame.draw.line(self.screen, (150, 180, 220),
                         (left_x, TRACK_TOP), (left_x, TRACK_TOP + TRACK_H), 2)
        pygame.draw.line(self.screen, (150, 180, 220),
                         (right_x, TRACK_TOP), (right_x, TRACK_TOP + TRACK_H), 2)

        # Finish line
        finish_y = TRACK_TOP + TRACK_H
        pygame.draw.line(self.screen, (220, 30, 30),
                         (left_x, finish_y), (right_x, finish_y), 4)

        # Gates
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

        # Skier
        x, y, vx, vy, heading = self.state
        speed = np.sqrt(vx**2 + vy**2)
        sx = int(W / 2 + (x / self.TRACK_WIDTH) * W * 0.45)
        sy = int(TRACK_TOP + (y / self.SLOPE_LENGTH) * TRACK_H)

        # Body oriented along heading
        L = 18
        dx_h = int(L * np.sin(heading))
        dy_h = int(L * np.cos(heading))
        pygame.draw.line(self.screen, (20, 60, 180),
                         (sx - dx_h, sy - dy_h), (sx + dx_h, sy + dy_h), 4)
        pygame.draw.circle(self.screen, (255, 200, 140),
                           (sx - dx_h, sy - dy_h), 7)

        # Skis — aligned with heading
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

        # HUD
        font = pygame.font.SysFont(None, 21)
        n_passed = sum(self.gates_passed)
        n_missed = sum(self.gates_missed)
        score = n_passed * self.GATE_PASS_REWARD - n_missed * self.gate_miss_penalty
        hud1 = font.render(
            f"y={y:.1f}m  speed={speed:.1f}m/s  heading={np.degrees(heading):.1f}°",
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
