"""
ppo_agent.py — Minimal PPO implementation (actor-critic, continuous actions).

Designed for educational clarity over raw performance.
For production training, swap in stable-baselines3 PPO directly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ------------------------------------------------------------------
# Neural network: shared backbone + separate actor / critic heads
# ------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )

        self.actor_mean    = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic        = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, obs):
        h     = self.backbone(obs)
        mean  = self.actor_mean(h)
        std   = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        raw  = mean if deterministic else dist.sample()
        action   = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(-1)
        # Change-of-variables correction for tanh squashing
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        """action is the squashed (tanh) action stored in the buffer."""
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        # Recover pre-tanh value: atanh(action)
        raw  = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw).sum(-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ------------------------------------------------------------------
# Rollout buffer
# ------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones   = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """Generalised Advantage Estimation (GAE)."""
        advantages, returns = [], []
        gae    = 0.0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns

    def to_tensors(self, device="cpu"):
        return (
            torch.tensor(np.array(self.obs),       dtype=torch.float32).to(device),
            torch.tensor(np.array(self.actions),   dtype=torch.float32).to(device),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(device),
        )


# ------------------------------------------------------------------
# PPO update
# ------------------------------------------------------------------

def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    *,
    clip_eps: float = 0.2,
    vf_coef: float  = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int   = 4,
    gamma: float    = 0.99,
    lam: float      = 0.95,
    device: str     = "cpu",
):
    advantages, returns = buffer.compute_returns(last_value, gamma, lam)

    obs_t, act_t, old_lp_t = buffer.to_tensors(device)
    adv_t = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_t = torch.tensor(returns,    dtype=torch.float32).to(device)

    # Normalise advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

    for _ in range(n_epochs):
        new_lp, values, entropy = policy.evaluate(obs_t, act_t)

        ratio = (new_lp - old_lp_t).exp()
        surr1 = ratio * adv_t
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = 0.5 * (ret_t - values).pow(2).mean()
        ent_loss    = -entropy.mean()

        loss = policy_loss + vf_coef * value_loss + ent_coef * ent_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        stats["policy_loss"] += policy_loss.item()
        stats["value_loss"]  += value_loss.item()
        stats["entropy"]     += (-ent_loss).item()

    for k in stats:
        stats[k] /= n_epochs
    return stats