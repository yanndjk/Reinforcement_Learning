"""
train.py — Main training loop for the skiing RL agent.

Usage:
    python train.py                    # train from scratch
    python train.py --render           # train with live rendering (slow)
    python train.py --eval             # evaluate a saved checkpoint

Dependencies:
    pip install gymnasium torch numpy matplotlib
    pip install pygame          # optional, for rendering
    pip install stable-baselines3   # optional, see comment below
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from envs.ski_env_v2 import SkiEnv
from agents.ppo_agent_v2 import ActorCritic, RolloutBuffer, ppo_update

# Track layout — set via --layout CLI flag (default: "wide")
LAYOUT = None


# ------------------------------------------------------------------
# Config — tweak these freely
# ------------------------------------------------------------------

CFG = dict(
    # Training
    total_steps    = 1_200_000,
    rollout_steps  = 2048,      # larger rollouts for more stable updates
    lr             = 3e-4,
    gamma          = 0.995,     # longer horizon — finish matters more
    lam            = 0.97,
    clip_eps       = 0.2,
    vf_coef        = 0.5,
    ent_coef       = 0.010,     # less exploration noise — more exploitation
    n_epochs       = 4,         # stronger PPO updates per rollout
    # Logging
    log_interval   = 10,        # episodes between console prints
    save_interval  = 50_000,    # steps between checkpoints
    save_dir       = "checkpoints_v3",
)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(render: bool = False, seed: int = 0):
    _set_seed(seed)

    render_mode = "human" if render else None
    env = SkiEnv(render_mode=render_mode, layout=LAYOUT)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}  |  obs={obs_dim}  act={act_dim}  seed={seed}")

    policy    = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=CFG["lr"])
    buffer    = RolloutBuffer()

    # Per-seed output directories
    run_dir  = Path(CFG["save_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []  # validation metrics collected during training

    # Tracking
    ep_rewards = []
    ep_lengths = []
    recent_rewards = deque(maxlen=100)
    global_step = 0
    ep_reward = 0.0
    ep_len    = 0
    ep_count  = 0

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    while global_step < CFG["total_steps"]:
        # -- Collect rollout -----------------------------------------
        buffer.reset()
        for _ in range(CFG["rollout_steps"]):
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
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
                ep_lengths.append(ep_len)
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
                        f"{elapsed:.0f}s elapsed"
                    )

                ep_reward = 0.0
                ep_len    = 0
                obs, _ = env.reset()

        # -- PPO update ----------------------------------------------
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_value = policy.get_action(obs_t)

        stats = ppo_update(
            policy, optimizer, buffer, last_value.item(),
            clip_eps=CFG["clip_eps"], vf_coef=CFG["vf_coef"],
            ent_coef=CFG["ent_coef"], n_epochs=CFG["n_epochs"],
            gamma=CFG["gamma"], lam=CFG["lam"], device=device,
        )

        # -- Checkpoint + Validation ----------------------------------
        if global_step % CFG["save_interval"] < CFG["rollout_steps"]:
            ckpt = run_dir / f"policy_{global_step}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"  ↳ Saved checkpoint: {ckpt}")

            metrics = validate(policy, device)
            metrics["step"] = global_step
            metrics["seed"] = seed
            metrics_log.append(metrics)
            print_validation(metrics, global_step)

    env.close()

    # Final validation
    metrics = validate(policy, device)
    metrics["step"] = global_step
    metrics["seed"] = seed
    metrics_log.append(metrics)
    print_validation(metrics, global_step)

    # Save all metrics and training curve for this seed
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": CFG, "validation": metrics_log,
                   "episode_rewards": ep_rewards}, f, indent=2)
    print(f"  ↳ Metrics saved to {metrics_path}")

    _plot_training(ep_rewards, seed=seed)
    print(f"Training complete (seed={seed}).")
    return metrics_log


# ------------------------------------------------------------------
# Validation — run N deterministic episodes and aggregate task metrics
# ------------------------------------------------------------------

def validate(policy, device, n_episodes: int = 20, render_mode=None,
             difficulty: float = 0.0):
    """Run deterministic rollouts and return aggregated task metrics."""
    env = SkiEnv(render_mode=render_mode, difficulty=difficulty, layout=LAYOUT)
    policy.eval()

    results = {
        "reward": [], "reached": [], "fell": [], "timeout": [],
        "gates_passed": [], "gates_missed": [],
        "perfect_run": [], "score": [], "progress": [],
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
        results["progress"].append(info.get("progress", 0.0))

    env.close()
    policy.train()

    metrics = {
        "difficulty":        difficulty,
        "finish_rate":       np.mean(results["reached"]),
        "fall_rate":         np.mean(results["fell"]),
        "timeout_rate":      np.mean(results["timeout"]),
        "avg_gates_passed":  np.mean(results["gates_passed"]),
        "avg_gates_missed":  np.mean(results["gates_missed"]),
        "perfect_run_rate":  np.mean(results["perfect_run"]),
        "avg_score":         np.mean(results["score"]),
        "avg_progress":      np.mean(results["progress"]),
        "avg_reward":        np.mean(results["reward"]),
    }
    return metrics


def print_validation(metrics, step):
    diff = metrics.get("difficulty", 0.0)
    print(
        f"\n{'='*60}\n"
        f"  VALIDATION @ step {step}  (difficulty={diff:.2f})\n"
        f"  Avg progress:     {metrics['avg_progress']:>6.1%}\n"
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


# ------------------------------------------------------------------
# Evaluation (CLI entry point — loads checkpoint, renders, prints)
# ------------------------------------------------------------------

def evaluate(checkpoint: str, n_episodes: int = 20, render: bool = False,
             difficulties: list[float] | None = None):
    env_tmp = SkiEnv(layout=LAYOUT)
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    if difficulties is None:
        difficulties = [0.0]

    render_mode = "human" if render else None
    for diff in difficulties:
        metrics = validate(policy, device="cpu", n_episodes=n_episodes,
                           render_mode=render_mode, difficulty=diff)
        print_validation(metrics, step="final")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _plot_training(rewards, seed: int = 0):
    smoothed = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(rewards,  alpha=0.3, label="Episode reward")
    plt.plot(smoothed, linewidth=2, label="Smoothed (20-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Skiing RL — Training curve (seed={seed})")
    plt.legend()
    plt.tight_layout()

    save_dir = Path("training_curves")
    save_dir.mkdir(exist_ok=True)
    fname = (
        f"curve_seed{seed}"
        f"_steps{CFG['total_steps']}"
        f"_lr{CFG['lr']}"
        f"_gamma{CFG['gamma']}"
        f"_clip{CFG['clip_eps']}"
        f"_ent{CFG['ent_coef']}"
        f"_rollout{CFG['rollout_steps']}"
        f"_epochs{CFG['n_epochs']}"
        f".png"
    )
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Training curve saved to {save_path}")


# ------------------------------------------------------------------
# Curriculum training — progressive difficulty with strategy presets
# ------------------------------------------------------------------

CURRICULUM_STRATEGIES = {
    # Original conservative: small steps, strict promotion, single difficulty
    "conservative": dict(
        total_steps       = 1_000_000,
        start_difficulty  = 0.0,
        max_difficulty    = 1.0,
        difficulty_step   = 0.05,
        promote_threshold = 0.75,        # gate pass rate
        promote_finish    = 0.80,        # finish rate
        promote_window    = 3,
        eval_interval     = 20_000,
        regress_threshold = 0.6,
        # Sampling: train at exactly the current difficulty
        sampling          = "fixed",
    ),
    # Aggressive: bigger jumps, faster promotion, reaches hard levels sooner
    "aggressive": dict(
        total_steps       = 1_000_000,
        start_difficulty  = 0.0,
        max_difficulty    = 1.0,
        difficulty_step   = 0.10,
        promote_threshold = 0.60,        # gate pass rate
        promote_finish    = 0.70,        # finish rate
        promote_window    = 2,
        eval_interval     = 15_000,
        regress_threshold = 0.5,
        sampling          = "fixed",
    ),
    # Mixed: each episode samples difficulty from [current - spread, current + spread]
    # This gives the policy diverse gate layouts every rollout
    "mixed": dict(
        total_steps       = 1_000_000,
        start_difficulty  = 0.0,
        max_difficulty    = 1.0,
        difficulty_step   = 0.10,
        promote_threshold = 0.60,        # gate pass rate
        promote_finish    = 0.70,        # finish rate
        promote_window    = 2,
        eval_interval     = 15_000,
        regress_threshold = 0.5,
        sampling          = "range",
        sampling_spread   = 0.15,   # sample from [d - 0.15, d + 0.15]
    ),
    # Focused: tighter sampling window, smaller steps, stricter promotion.
    # Spends more time per difficulty level with less signal blur.
    # Designed for the 0.2–0.4 transition zone where gate performance collapses.
    "focused": dict(
        total_steps       = 1_500_000,   # more budget to spend longer per level
        start_difficulty  = 0.0,
        max_difficulty    = 1.0,
        difficulty_step   = 0.05,        # half the mixed jump
        promote_threshold = 0.70,        # gate pass rate — stricter than mixed
        promote_finish    = 0.80,        # finish rate
        promote_window    = 3,           # must prove consistency
        eval_interval     = 15_000,
        regress_threshold = 0.5,
        sampling          = "range",
        sampling_spread   = 0.08,        # tight window: ±0.08 around target
    ),
    #same as focused but starts at 0.2 to skip the very easy levels and get to the gate learning sooner
    "fine_tune": dict(
        total_steps       = 1_500_000,
        start_difficulty  = 0.2,
        max_difficulty    = 1.0,
        difficulty_step   = 0.05,
        promote_threshold = 0.70,
        promote_finish    = 0.80,
        promote_window    = 3,
        eval_interval     = 15_000,
        regress_threshold = 0.5,
        sampling          = "range",
        sampling_spread   = 0.08,
    )
}


def _sample_difficulty(cur_diff: float, cfg: dict, rng: np.random.Generator) -> float:
    """Pick a training difficulty for the next episode."""
    if cfg["sampling"] == "fixed":
        return cur_diff
    # "range": uniform sample around current difficulty
    spread = cfg.get("sampling_spread", 0.15)
    lo = max(0.0, cur_diff - spread)
    hi = min(cfg["max_difficulty"], cur_diff + spread)
    return float(rng.uniform(lo, hi))


def train_curriculum(seed: int = 0, checkpoint: str | None = None,
                     strategy: str = "mixed"):
    """Train with progressive difficulty.

    strategy: "conservative", "aggressive", or "mixed" (default).
    """
    if strategy not in CURRICULUM_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(CURRICULUM_STRATEGIES.keys())}")

    cur_cfg = CURRICULUM_STRATEGIES[strategy]
    _set_seed(seed)
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SkiEnv(layout=LAYOUT)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, act_dim).to(device)
    if checkpoint:
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Warm-started from {checkpoint}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=CFG["lr"])
    buffer = RolloutBuffer()

    warm_tag = "warm" if checkpoint else "scratch"
    run_dir = Path(CFG["save_dir"]) / f"curriculum_{strategy}_{warm_tag}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum state
    difficulty = cur_cfg["start_difficulty"]
    ep_difficulty = difficulty  # per-episode sampled difficulty
    promote_count = 0
    env.set_difficulty(ep_difficulty)

    # Tracking
    ep_rewards = []
    recent_rewards = deque(maxlen=100)
    global_step = 0
    ep_reward = 0.0
    ep_len = 0
    ep_count = 0
    metrics_log = []
    difficulty_history = []  # (step, difficulty) pairs

    total_steps = cur_cfg["total_steps"]
    eval_interval = cur_cfg["eval_interval"]
    last_eval_step = 0

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    print(f"Curriculum training on {device}  |  seed={seed}  "
          f"strategy={strategy}  start={warm_tag}")
    print(f"  Config: step={cur_cfg['difficulty_step']}  "
          f"promote=gates≥{cur_cfg['promote_threshold']:.0%}+finish≥{cur_cfg['promote_finish']:.0%}"
          f"x{cur_cfg['promote_window']}  "
          f"sampling={cur_cfg['sampling']}")

    while global_step < total_steps:
        # -- Collect rollout -----------------------------------------
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

                # Sample a new difficulty for the next episode
                ep_difficulty = _sample_difficulty(difficulty, cur_cfg, rng)
                env.set_difficulty(ep_difficulty)
                obs, _ = env.reset()

        # -- PPO update ----------------------------------------------
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_value = policy.get_action(obs_t)

        ppo_update(
            policy, optimizer, buffer, last_value.item(),
            clip_eps=CFG["clip_eps"], vf_coef=CFG["vf_coef"],
            ent_coef=CFG["ent_coef"], n_epochs=CFG["n_epochs"],
            gamma=CFG["gamma"], lam=CFG["lam"], device=device,
        )

        # -- Curriculum evaluation & promotion -----------------------
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            difficulty_history.append((global_step, difficulty))

            # Validate at current curriculum difficulty
            save_difficulty = difficulty   # snapshot before promotion
            m_curr = validate(policy, device, n_episodes=20,
                              difficulty=difficulty)
            m_curr["step"] = global_step
            m_curr["curriculum_difficulty"] = difficulty

            # Validate at d=0.0 (regression check)
            m_base = validate(policy, device, n_episodes=20,
                              difficulty=0.0)

            gate_rate = m_curr["avg_gates_passed"] / env.n_gates
            finish_rate = m_curr["finish_rate"]
            base_gate_rate = m_base["avg_gates_passed"] / env.n_gates

            finish_thresh = cur_cfg["promote_finish"]

            print(
                f"\n  CURRICULUM CHECK @ step {global_step}  "
                f"(target diff={difficulty:.2f})\n"
                f"    Current diff:  gates={m_curr['avg_gates_passed']:.1f}/{env.n_gates}  "
                f"({gate_rate:.0%})  finish={finish_rate:.0%}  "
                f"progress={m_curr['avg_progress']:.0%}\n"
                f"    Regression d=0: gates={m_base['avg_gates_passed']:.1f}/{env.n_gates}  "
                f"({base_gate_rate:.0%})  finish={m_base['finish_rate']:.0%}  "
                f"progress={m_base['avg_progress']:.0%}"
            )

            # Promotion logic — both gate rate AND finish rate must meet threshold
            regressing = base_gate_rate < cur_cfg["regress_threshold"]
            gates_ok = gate_rate >= cur_cfg["promote_threshold"]
            finish_ok = finish_rate >= finish_thresh

            if regressing:
                promote_count = 0
                print(f"    >> HOLDING — regression at d=0.0 "
                      f"(gate rate {base_gate_rate:.0%} < "
                      f"{cur_cfg['regress_threshold']:.0%})")
            elif gates_ok and finish_ok:
                promote_count += 1
                if promote_count >= cur_cfg["promote_window"]:
                    old_diff = difficulty
                    difficulty = min(
                        difficulty + cur_cfg["difficulty_step"],
                        cur_cfg["max_difficulty"],
                    )
                    promote_count = 0
                    print(f"    >> PROMOTED {old_diff:.2f} → {difficulty:.2f}")
                else:
                    print(f"    >> Promote streak: "
                          f"{promote_count}/{cur_cfg['promote_window']}")
            else:
                promote_count = 0
                reasons = []
                if not gates_ok:
                    reasons.append(f"gates {gate_rate:.0%} < {cur_cfg['promote_threshold']:.0%}")
                if not finish_ok:
                    reasons.append(f"finish {finish_rate:.0%} < {finish_thresh:.0%}")
                print(f"    >> Not ready ({', '.join(reasons)})")

            metrics_log.append({
                "current": m_curr,
                "baseline_regression": {
                    "difficulty": 0.0,
                    "gate_rate": base_gate_rate,
                    "finish_rate": m_base["finish_rate"],
                },
                "training_difficulty": difficulty,
            })

            # Save checkpoint at each eval (use pre-promotion difficulty)
            ckpt = run_dir / f"policy_{global_step}_d{save_difficulty:.2f}.pt"
            torch.save(policy.state_dict(), ckpt)

    env.close()

    # -- Final outputs -----------------------------------------------
    difficulty_history.append((global_step, difficulty))

    # Save metrics
    report = {
        "config": CFG,
        "curriculum_config": cur_cfg,
        "strategy": strategy,
        "warm_start": checkpoint,
        "seed": seed,
        "difficulty_history": difficulty_history,
        "evaluations": metrics_log,
        "episode_rewards": ep_rewards,
    }
    metrics_path = run_dir / "curriculum_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ↳ Curriculum metrics saved to {metrics_path}")

    # Final difficulty sweep
    print("\n  Running final difficulty sweep...")
    final_ckpt = run_dir / "policy_final.pt"
    torch.save(policy.state_dict(), final_ckpt)

    difficulties = np.linspace(0.0, 1.0, 11)
    sweep = []
    for diff in difficulties:
        m = validate(policy, device, n_episodes=30, difficulty=diff)
        sweep.append(m)
        print(f"    d={diff:.1f}  gates={m['avg_gates_passed']:.1f}  "
              f"finish={m['finish_rate']:.0%}")

    _plot_curriculum(difficulty_history, metrics_log, difficulties, sweep,
                     run_dir, seed, strategy, cur_cfg)
    _plot_episode_rewards(ep_rewards, run_dir, seed, strategy)

    print(f"\nCurriculum training complete (seed={seed}, strategy={strategy}, "
          f"final difficulty={difficulty:.2f}).")
    return metrics_log


def _plot_curriculum(diff_history, eval_log, sweep_diffs, sweep_results,
                     run_dir, seed, strategy, cur_cfg):
    """Plot curriculum progression and final sweep comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Curriculum Training — {strategy} (seed={seed})", fontsize=13)

    # 1. Difficulty over time
    ax = axes[0]
    steps = [h[0] for h in diff_history]
    diffs = [h[1] for h in diff_history]
    ax.step(steps, diffs, where="post", linewidth=2, color="blue")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Difficulty")
    ax.set_title("Difficulty progression")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. Gate rate at training difficulty over time
    ax = axes[1]
    eval_steps = [e["current"]["step"] for e in eval_log]
    gate_rates = [e["current"]["avg_gates_passed"] / 8 for e in eval_log]
    base_rates = [e["baseline_regression"]["gate_rate"] for e in eval_log]
    ax.plot(eval_steps, gate_rates, "o-", label="At training diff", color="blue")
    ax.plot(eval_steps, base_rates, "s-", label="At d=0.0 (regression)", color="green")
    ax.axhline(y=cur_cfg["promote_threshold"], color="orange",
               linestyle="--", alpha=0.7, label="Promote threshold")
    ax.axhline(y=cur_cfg["regress_threshold"], color="red",
               linestyle="--", alpha=0.7, label="Regress threshold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gate pass rate")
    ax.set_title("Gate performance over training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Final sweep — gates passed
    ax = axes[2]
    ax.plot(sweep_diffs, [r["avg_gates_passed"] for r in sweep_results],
            "o-", color="blue", label=strategy)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Avg gates passed")
    ax.set_title("Final difficulty sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = run_dir / "curriculum_progress.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ↳ Curriculum plot saved to {save_path}")


def _plot_episode_rewards(rewards, run_dir, seed, strategy):
    """Plot per-episode rewards with smoothing and save to run directory."""
    if len(rewards) < 2:
        return
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(rewards,  alpha=0.3, label="Episode reward")
    plt.plot(smoothed, linewidth=2, label=f"Smoothed ({window}-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Episode Rewards — {strategy} (seed={seed})")
    plt.legend()
    plt.tight_layout()
    save_path = run_dir / "episode_rewards.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Episode rewards plot saved to {save_path}")


# ------------------------------------------------------------------
# Baseline — train across multiple seeds, aggregate results
# ------------------------------------------------------------------

def run_baseline(seeds: list[int]):
    """Train on each seed and produce a cross-seed summary."""
    all_runs = {}
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"#  BASELINE RUN — seed {seed}")
        print(f"{'#'*60}\n")
        all_runs[seed] = train(seed=seed)

    _print_cross_seed_summary(all_runs)
    _save_baseline_report(all_runs)


def _print_cross_seed_summary(all_runs: dict):
    """Print a table comparing final metrics across seeds."""
    print(f"\n{'='*70}")
    print("  CROSS-SEED BASELINE SUMMARY (final checkpoint per seed)")
    print(f"{'='*70}")

    # Collect final metrics from each seed
    finals = {}
    for seed, logs in all_runs.items():
        finals[seed] = logs[-1]  # last validation entry

    keys = ["finish_rate", "fall_rate", "avg_gates_passed",
            "avg_gates_missed", "perfect_run_rate", "avg_score", "avg_reward"]

    # Per-seed rows
    header = f"  {'Metric':<20s}" + "".join(f"{'seed '+str(s):>12s}" for s in finals) + f"{'mean':>10s}{'std':>10s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for k in keys:
        vals = [finals[s][k] for s in finals]
        row = f"  {k:<20s}"
        for v in vals:
            if "rate" in k:
                row += f"{v:>11.1%} "
            else:
                row += f"{v:>11.1f} "
        row += f"{np.mean(vals):>9.1f}{np.std(vals):>10.2f}"
        print(row)

    print(f"{'='*70}\n")


def _save_baseline_report(all_runs: dict):
    """Save full baseline results to a JSON file."""
    report_dir = Path("baseline_results")
    report_dir.mkdir(exist_ok=True)

    report = {
        "config": CFG,
        "seeds": {},
    }
    for seed, logs in all_runs.items():
        report["seeds"][str(seed)] = logs

    report_path = report_dir / "baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Baseline report saved to {report_path}")


# ------------------------------------------------------------------
# Robustness — test checkpoints x repeated validation batches
# ------------------------------------------------------------------

def run_robustness(seed: int = 0, n_batches: int = 5, n_episodes: int = 20):
    """Load all checkpoints for a seed, run multiple validation batches each."""
    run_dir = Path(CFG["save_dir"]) / f"seed_{seed}"
    ckpts = sorted(run_dir.glob("policy_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))

    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    env_tmp = SkiEnv(layout=LAYOUT)
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    for ckpt in ckpts:
        step = int(ckpt.stem.split("_")[1])
        policy = ActorCritic(obs_dim, act_dim).to(device)
        policy.load_state_dict(torch.load(ckpt, map_location=device))

        batch_metrics = []
        for _ in range(n_batches):
            m = validate(policy, device, n_episodes=n_episodes)
            batch_metrics.append(m)

        all_results[step] = batch_metrics

        # Summarise this checkpoint
        keys = ["finish_rate", "fall_rate", "avg_gates_passed",
                "avg_gates_missed", "perfect_run_rate", "avg_score"]
        means = {k: np.mean([bm[k] for bm in batch_metrics]) for k in keys}
        stds  = {k: np.std([bm[k] for bm in batch_metrics])  for k in keys}

        print(f"\n  Checkpoint step={step}  ({n_batches} batches x {n_episodes} eps)")
        for k in keys:
            if "rate" in k:
                print(f"    {k:<20s} {means[k]:>6.1%} ± {stds[k]:.1%}")
            else:
                print(f"    {k:<20s} {means[k]:>6.1f} ± {stds[k]:.1f}")

    # Save robustness report
    report_dir = Path("baseline_results")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"robustness_seed{seed}.json"

    serialisable = {}
    for step, batches in all_results.items():
        serialisable[str(step)] = batches
    with open(report_path, "w") as f:
        json.dump({"seed": seed, "n_batches": n_batches,
                    "n_episodes": n_episodes, "results": serialisable}, f, indent=2)
    print(f"\nRobustness report saved to {report_path}")


# ------------------------------------------------------------------
# Difficulty sweep — map the task difficulty curve
# ------------------------------------------------------------------

def run_difficulty_sweep(checkpoint: str, n_episodes: int = 30,
                         steps: int = 11):
    """Evaluate a checkpoint across a range of difficulties."""
    env_tmp = SkiEnv(layout=LAYOUT)
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    device = "cpu"

    difficulties = np.linspace(0.0, 1.0, steps)
    sweep_results = []

    for diff in difficulties:
        print(f"  Evaluating difficulty={diff:.2f} ...", end=" ", flush=True)
        metrics = validate(policy, device, n_episodes=n_episodes,
                           difficulty=diff)
        sweep_results.append(metrics)
        print(
            f"finish={metrics['finish_rate']:.0%}  "
            f"fall={metrics['fall_rate']:.0%}  "
            f"gates={metrics['avg_gates_passed']:.1f}/{metrics['avg_gates_passed']+metrics['avg_gates_missed']:.1f}  "
            f"score={metrics['avg_score']:.0f}  "
            f"reward={metrics['avg_reward']:.0f}"
        )

    # Save raw data
    report_dir = Path("baseline_results")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "difficulty_sweep.json"
    with open(report_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "n_episodes": n_episodes,
                    "results": sweep_results}, f, indent=2)
    print(f"\nSweep data saved to {report_path}")

    _plot_difficulty_sweep(difficulties, sweep_results)


def _plot_difficulty_sweep(difficulties, results):
    """Generate a multi-panel plot answering the key questions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Difficulty Sweep — Baseline Policy", fontsize=14)

    d = difficulties

    # 1. Finish rate vs Fall rate
    ax = axes[0, 0]
    ax.plot(d, [r["finish_rate"] for r in results], "o-", label="Finish rate", color="green")
    ax.plot(d, [r["fall_rate"] for r in results], "s-", label="Fall rate", color="red")
    ax.set_ylabel("Rate")
    ax.set_title("Finish vs Fall rate")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. Gates passed vs missed
    ax = axes[0, 1]
    ax.plot(d, [r["avg_gates_passed"] for r in results], "o-", label="Passed", color="green")
    ax.plot(d, [r["avg_gates_missed"] for r in results], "s-", label="Missed", color="orange")
    ax.set_ylabel("Gates")
    ax.set_title("Avg gates passed / missed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Perfect run rate
    ax = axes[0, 2]
    ax.plot(d, [r["perfect_run_rate"] for r in results], "o-", color="purple")
    ax.set_ylabel("Rate")
    ax.set_title("Perfect run rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. Score
    ax = axes[1, 0]
    ax.plot(d, [r["avg_score"] for r in results], "o-", color="blue")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Score")
    ax.set_title("Avg score")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 5. Reward
    ax = axes[1, 1]
    ax.plot(d, [r["avg_reward"] for r in results], "o-", color="teal")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Reward")
    ax.set_title("Avg reward")
    ax.grid(True, alpha=0.3)

    # 6. Reward vs Score correlation
    ax = axes[1, 2]
    scores  = [r["avg_score"] for r in results]
    rewards = [r["avg_reward"] for r in results]
    ax.scatter(scores, rewards, c=d, cmap="coolwarm", s=60, edgecolors="black", zorder=3)
    ax.set_xlabel("Score")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Score (color=difficulty)")
    # Correlation coefficient
    if len(scores) > 2:
        corr = np.corrcoef(scores, rewards)[0, 1]
        ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_dir = Path("baseline_results")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "difficulty_sweep.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved to {save_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",     action="store_true", help="Render during training/eval")
    parser.add_argument("--eval",       action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", default=None,        help="Path to checkpoint .pt file")
    parser.add_argument("--seed",       type=int, default=0, help="Random seed (single run)")
    parser.add_argument("--baseline",   action="store_true",
                        help="Train across multiple seeds and produce baseline report")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds for --baseline (default: 0 1 2)")
    parser.add_argument("--robustness", action="store_true",
                        help="Test checkpoint robustness with repeated validation")
    parser.add_argument("--n-batches",  type=int, default=5,
                        help="Validation batches per checkpoint for --robustness")
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0],
                        help="Difficulty level(s) 0.0-1.0 for --eval (default: 0.0)")
    parser.add_argument("--curriculum",  action="store_true",
                        help="Train with progressive difficulty (curriculum learning)")
    parser.add_argument("--curriculum-strategy", default="mixed",
                        choices=["conservative", "aggressive", "mixed", "focused", "fine_tune"],
                        help="Curriculum strategy preset (default: mixed)")
    parser.add_argument("--difficulty-sweep", action="store_true",
                        help="Sweep difficulty 0.0→1.0 and plot degradation curve")
    parser.add_argument("--sweep-steps", type=int, default=11,
                        help="Number of difficulty levels in sweep (default: 11)")
    parser.add_argument("--sweep-episodes", type=int, default=30,
                        help="Episodes per difficulty level in sweep (default: 30)")
    parser.add_argument("--layout", default=None,
                        choices=list(SkiEnv.LAYOUTS.keys()),
                        help="Track layout (default: wide)")
    args = parser.parse_args()

    # Set global layout
    LAYOUT = args.layout

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
        evaluate(args.checkpoint, render=args.render,
                 difficulties=args.difficulty)
    elif args.baseline:
        run_baseline(seeds=args.seeds)
    elif args.robustness:
        run_robustness(seed=args.seed, n_batches=args.n_batches)
    else:
        train(render=args.render, seed=args.seed)
