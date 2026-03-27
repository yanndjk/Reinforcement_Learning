"""
td3_agent.py — Twin Delayed Deep Deterministic Policy Gradient (TD3).

Drop-in replacement for ppo_agent.py in the skiing RL project.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Actor network — deterministic policy μ(s) → a ∈ [-1, 1]
# ------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),   # squash to [-1, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small final layer → near-zero initial actions
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ------------------------------------------------------------------
# Twin critic networks — Q1(s,a) and Q2(s,a)
# ------------------------------------------------------------------

class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + act_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Used for actor gradient — only need Q1 to save a forward pass."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1)


# ------------------------------------------------------------------
# Replay buffer — uniform random sampling
# ------------------------------------------------------------------

class ReplayBuffer:
    """Circular replay buffer for off-policy TD3 learning."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 200_000):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0

        self.obs     = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards = np.zeros((capacity,),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros((capacity,),         dtype=np.float32)

    def add(self, obs, action, reward: float, next_obs, done: bool):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      dtype=torch.float32).to(device),
            torch.tensor(self.actions[idx],  dtype=torch.float32).to(device),
            torch.tensor(self.rewards[idx],  dtype=torch.float32).to(device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32).to(device),
            torch.tensor(self.dones[idx],    dtype=torch.float32).to(device),
        )

    def __len__(self):
        return self.size


# ------------------------------------------------------------------
# TD3 agent — wraps actor, twin critics, target networks, and updates
# ------------------------------------------------------------------

class TD3Agent:
    """
    Self-contained TD3 agent.

    Args:
        obs_dim:        observation dimensionality
        act_dim:        action dimensionality
        hidden:         hidden layer width for all networks (default 256)
        lr_actor:       actor learning rate (default 3e-4)
        lr_critic:      critic learning rate (default 3e-4)
        gamma:          discount factor (default 0.99)
        tau:            Polyak averaging coefficient for target nets (default 5e-3)
        policy_noise:   std of smoothing noise added to target actions (default 0.2)
        noise_clip:     clipping range for target policy noise (default 0.5)
        policy_delay:   critic updates per actor update (default 2)
        expl_noise:     std of Gaussian exploration noise added at act-time (default 0.1)
        buffer_capacity:size of replay buffer (default 200_000)
        batch_size:     minibatch size for gradient updates (default 256)
        warmup_steps:   random actions taken before learning starts (default 1_000)
        device:         "cpu" or "cuda"
    """

    def __init__(
        self,
        obs_dim:          int,
        act_dim:          int,
        hidden:           int   = 256,
        lr_actor:         float = 3e-4,
        lr_critic:        float = 3e-4,
        gamma:            float = 0.99,
        tau:              float = 5e-3,
        policy_noise:     float = 0.2,
        noise_clip:       float = 0.5,
        policy_delay:     int   = 2,
        expl_noise:       float = 0.1,
        buffer_capacity:  int   = 200_000,
        batch_size:       int   = 256,
        warmup_steps:     int   = 1_000,
        device:           str   = "cpu",
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise   = expl_noise
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.device       = device

        # --- Networks ---
        self.actor  = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = TwinCritic(obs_dim, act_dim, hidden).to(device)

        # Target networks start as exact copies
        self.actor_target  = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Targets are never directly trained — only Polyak-updated
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # --- Optimisers ---
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_capacity)

        # Internal step counter for delayed actor updates
        self._update_count = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Return an action for the given observation.

        During training (explore=True) Gaussian noise is added.
        During evaluation (explore=False) the deterministic action is returned.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()

        if explore:
            noise  = np.random.normal(0.0, self.expl_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def random_action(self, act_dim: int) -> np.ndarray:
        """Pure random action used during warmup."""
        return np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        Draw one minibatch from the replay buffer and do a TD3 update.
        Returns a stats dict (empty if buffer not yet ready).
        """
        if len(self.buffer) < self.batch_size:
            return {}

        self._update_count += 1
        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # ---- Critic update ----
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            # Conservative Q-target: min of twin critics
            q1_t, q2_t = self.critic_target(next_obs, next_actions)
            q_target    = rewards + self.gamma * (1.0 - dones) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        stats = {"critic_loss": critic_loss.item(), "actor_loss": 0.0}

        # ---- Delayed actor update ----
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            stats["actor_loss"] = actor_loss.item()

            # ---- Polyak target network updates ----
            self._polyak_update(self.actor,  self.actor_target)
            self._polyak_update(self.critic, self.critic_target)

        return stats

    def _polyak_update(self, src: nn.Module, tgt: nn.Module):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p_src.data)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt":     self.actor_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "_update_count": self._update_count,
        }

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self._update_count = sd.get("_update_count", 0)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
