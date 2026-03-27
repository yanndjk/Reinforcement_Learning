"""
TD3 training loop for the skiing RL agent.
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
sys.path.insert(0, str(Path(__file__).parent))
from ski_env_v3 import SkiEnv
from td3_agent import TD3Agent

LAYOUT = None   


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

CFG = dict(
    total_steps     = 1_200_000,
    lr_actor        = 3e-4,
    lr_critic       = 3e-4,
    gamma           = 0.995,
    tau             = 5e-3,
    policy_noise    = 0.2,
    noise_clip      = 0.5,
    policy_delay    = 2,
    expl_noise      = 0.15,
    buffer_capacity = 300_000,
    batch_size      = 256,
    warmup_steps    = 5_000,
    hidden          = 256,
    log_interval    = 10,
    save_interval   = 50_000,
    save_dir        = "checkpoints_td3",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(render: bool = False, seed: int = 0):
    _set_seed(seed)

    render_mode = "human" if render else None
    env = SkiEnv(render_mode=render_mode, layout=LAYOUT)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TD3] Training on {device}  |  obs={obs_dim}  act={act_dim}  seed={seed}")

    agent = TD3Agent(
        obs_dim         = obs_dim,
        act_dim         = act_dim,
        hidden          = CFG["hidden"],
        lr_actor        = CFG["lr_actor"],
        lr_critic       = CFG["lr_critic"],
        gamma           = CFG["gamma"],
        tau             = CFG["tau"],
        policy_noise    = CFG["policy_noise"],
        noise_clip      = CFG["noise_clip"],
        policy_delay    = CFG["policy_delay"],
        expl_noise      = CFG["expl_noise"],
        buffer_capacity = CFG["buffer_capacity"],
        batch_size      = CFG["batch_size"],
        warmup_steps    = CFG["warmup_steps"],
        device          = device,
    )

    run_dir = Path(CFG["save_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []

    ep_rewards     = []
    recent_rewards = deque(maxlen=100)
    global_step    = 0
    ep_reward      = 0.0
    ep_len         = 0
    ep_count       = 0
    stats          = {}

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    while global_step < CFG["total_steps"]:

        if global_step < CFG["warmup_steps"]:
            action = agent.random_action(act_dim)
        else:
            agent.train()
            action = agent.select_action(obs, explore=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        ep_reward   += reward
        ep_len      += 1
        global_step += 1
        obs          = next_obs

        if global_step >= CFG["warmup_steps"]:
            stats = agent.update()

        if done:
            ep_rewards.append(ep_reward)
            recent_rewards.append(ep_reward)
            ep_count += 1

            if ep_count % CFG["log_interval"] == 0:
                elapsed  = time.time() - t_start
                actor_l  = stats.get("actor_loss",  0.0)
                critic_l = stats.get("critic_loss", 0.0)
                print(
                    f"Step {global_step:>7d} | "
                    f"Ep {ep_count:>5d} | "
                    f"MeanR(100) {np.mean(recent_rewards):>8.1f} | "
                    f"EpLen {ep_len:>4d} | "
                    f"CritLoss {critic_l:>7.3f} | "
                    f"ActLoss {actor_l:>7.3f} | "
                    f"{elapsed:.0f}s"
                )

            ep_reward = 0.0
            ep_len    = 0
            obs, _ = env.reset()

        if global_step % CFG["save_interval"] < 1:
            ckpt = run_dir / f"agent_{global_step}.pt"
            torch.save(agent.state_dict(), ckpt)
            print(f"  ↳ Saved checkpoint: {ckpt}")

            metrics = validate(agent, device)
            metrics["step"] = global_step
            metrics["seed"] = seed
            metrics_log.append(metrics)
            print_validation(metrics, global_step)

    env.close()

    metrics = validate(agent, device)
    metrics["step"] = global_step
    metrics["seed"] = seed
    metrics_log.append(metrics)
    print_validation(metrics, global_step)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": CFG, "validation": metrics_log,
                   "episode_rewards": ep_rewards}, f, indent=2)
    print(f"  ↳ Metrics saved to {metrics_path}")

    _plot_training(ep_rewards, seed=seed)
    print(f"Training complete (seed={seed}).")
    return metrics_log


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate(agent: TD3Agent, device: str, n_episodes: int = 20,
             render_mode=None, difficulty: float = 0.0) -> dict:
    env = SkiEnv(render_mode=render_mode, difficulty=difficulty, layout=LAYOUT)
    agent.eval()

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
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
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
    agent.train()

    return {
        "difficulty":       difficulty,
        "finish_rate":      np.mean(results["reached"]),
        "fall_rate":        np.mean(results["fell"]),
        "timeout_rate":     np.mean(results["timeout"]),
        "avg_gates_passed": np.mean(results["gates_passed"]),
        "avg_gates_missed": np.mean(results["gates_missed"]),
        "perfect_run_rate": np.mean(results["perfect_run"]),
        "avg_score":        np.mean(results["score"]),
        "avg_progress":     np.mean(results["progress"]),
        "avg_reward":       np.mean(results["reward"]),
    }


def print_validation(metrics: dict, step):
    diff = metrics.get("difficulty", 0.0)
    print(
        f"\n{'='*60}\n"
        f"  [TD3] VALIDATION @ step {step}  (difficulty={diff:.2f})\n"
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
# Evaluation (CLI entry)
# ------------------------------------------------------------------

def evaluate(checkpoint: str, n_episodes: int = 20, render: bool = False,
             difficulties: list[float] | None = None):
    env_tmp = SkiEnv(layout=LAYOUT)
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = TD3Agent(obs_dim, act_dim, device=device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))

    if difficulties is None:
        difficulties = [0.0]

    render_mode = "human" if render else None
    for diff in difficulties:
        metrics = validate(agent, device=device, n_episodes=n_episodes,
                           render_mode=render_mode, difficulty=diff)
        print_validation(metrics, step="final")


# ------------------------------------------------------------------
# Baseline (multi-seed)
# ------------------------------------------------------------------

def run_baseline(seeds: list[int] = (0, 1, 2)):
    all_metrics = {}
    for seed in seeds:
        print(f"\n{'#'*60}\n  Seed {seed}\n{'#'*60}")
        all_metrics[seed] = train(seed=seed)

    final = [m[-1] for m in all_metrics.values()]
    keys  = ["finish_rate", "fall_rate", "avg_gates_passed",
             "avg_gates_missed", "perfect_run_rate", "avg_score", "avg_reward"]
    print(f"\n{'='*60}\n[TD3] BASELINE SUMMARY (seeds={list(seeds)})\n{'='*60}")
    for k in keys:
        vals = [f[k] for f in final]
        tag  = "%" if "rate" in k else ""
        mult = 100 if "rate" in k else 1
        print(f"  {k:<22s}  {np.mean(vals)*mult:>6.1f}{tag} ± {np.std(vals)*mult:.1f}{tag}")


# ------------------------------------------------------------------
# Curriculum training
# ------------------------------------------------------------------

CURRICULUM_STRATEGIES = {
    "conservative": dict(
        total_steps=1_000_000, start_difficulty=0.0, max_difficulty=1.0,
        difficulty_step=0.05, promote_threshold=0.75, promote_finish=0.80,
        promote_window=3, eval_interval=20_000, regress_threshold=0.6,
        sampling="fixed",
    ),
    "aggressive": dict(
        total_steps=1_000_000, start_difficulty=0.0, max_difficulty=1.0,
        difficulty_step=0.10, promote_threshold=0.60, promote_finish=0.70,
        promote_window=2, eval_interval=15_000, regress_threshold=0.5,
        sampling="fixed",
    ),
    "mixed": dict(
        total_steps=1_000_000, start_difficulty=0.0, max_difficulty=1.0,
        difficulty_step=0.10, promote_threshold=0.60, promote_finish=0.70,
        promote_window=2, eval_interval=15_000, regress_threshold=0.5,
        sampling="range", sampling_spread=0.15,
    ),
}


def _sample_difficulty(cur_diff: float, cfg: dict, rng: np.random.Generator) -> float:
    if cfg["sampling"] == "fixed":
        return cur_diff
    spread = cfg.get("sampling_spread", 0.15)
    return float(rng.uniform(max(0.0, cur_diff - spread),
                             min(cfg["max_difficulty"], cur_diff + spread)))


def train_curriculum(seed: int = 0, checkpoint: str | None = None,
                     strategy: str = "mixed"):
    cur_cfg = CURRICULUM_STRATEGIES[strategy]
    _set_seed(seed)
    rng = np.random.default_rng(seed)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    env     = SkiEnv(layout=LAYOUT)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim, device=device,
                     hidden=CFG["hidden"], lr_actor=CFG["lr_actor"],
                     lr_critic=CFG["lr_critic"], gamma=CFG["gamma"],
                     tau=CFG["tau"], policy_noise=CFG["policy_noise"],
                     noise_clip=CFG["noise_clip"], policy_delay=CFG["policy_delay"],
                     expl_noise=CFG["expl_noise"],
                     buffer_capacity=CFG["buffer_capacity"],
                     batch_size=CFG["batch_size"],
                     warmup_steps=CFG["warmup_steps"])

    if checkpoint:
        agent.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Warm-started from {checkpoint}")

    warm_tag = "warm" if checkpoint else "scratch"
    run_dir  = Path(CFG["save_dir"]) / f"curriculum_{strategy}_{warm_tag}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    difficulty    = cur_cfg["start_difficulty"]
    promote_count = 0
    env.set_difficulty(difficulty)

    ep_rewards     = []
    recent_rewards = deque(maxlen=100)
    global_step    = 0
    ep_reward      = 0.0
    ep_len         = 0
    ep_count       = 0
    metrics_log    = []
    last_eval_step = 0
    total_steps    = cur_cfg["total_steps"]

    obs, _ = env.reset(seed=seed)
    t_start = time.time()

    print(f"[TD3] Curriculum on {device}  seed={seed}  strategy={strategy}")

    while global_step < total_steps:

        if global_step < CFG["warmup_steps"]:
            action = agent.random_action(act_dim)
        else:
            agent.train()
            action = agent.select_action(obs, explore=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        ep_reward   += reward
        ep_len      += 1
        global_step += 1
        obs          = next_obs

        if global_step >= CFG["warmup_steps"]:
            agent.update()

        if done:
            ep_rewards.append(ep_reward)
            recent_rewards.append(ep_reward)
            ep_count += 1

            if ep_count % CFG["log_interval"] == 0:
                elapsed = time.time() - t_start
                print(
                    f"Step {global_step:>7d} | Ep {ep_count:>5d} | "
                    f"MeanR(100) {np.mean(recent_rewards):>8.1f} | "
                    f"Diff {difficulty:.2f} | {elapsed:.0f}s"
                )

            ep_reward = 0.0
            ep_len    = 0
            ep_diff = _sample_difficulty(difficulty, cur_cfg, rng)
            env.set_difficulty(ep_diff)
            obs, _ = env.reset()

        if (global_step - last_eval_step) >= cur_cfg["eval_interval"]:
            last_eval_step = global_step
            m = validate(agent, device, n_episodes=20, difficulty=difficulty)
            metrics_log.append({**m, "step": global_step, "difficulty": difficulty})
            print_validation(m, global_step)

            gate_rate   = m["avg_gates_passed"] / max(
                m["avg_gates_passed"] + m["avg_gates_missed"], 1)
            finish_rate = m["finish_rate"]

            if (gate_rate >= cur_cfg["promote_threshold"]
                    and finish_rate >= cur_cfg["promote_finish"]):
                promote_count += 1
                if promote_count >= cur_cfg["promote_window"]:
                    new_diff = min(difficulty + cur_cfg["difficulty_step"],
                                   cur_cfg["max_difficulty"])
                    print(f"  ↑ Promoting: {difficulty:.2f} → {new_diff:.2f}")
                    difficulty    = new_diff
                    promote_count = 0
            elif gate_rate < cur_cfg.get("regress_threshold", 0.5):
                promote_count = 0

    env.close()

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": CFG, "curriculum_config": cur_cfg,
                   "validation": metrics_log,
                   "episode_rewards": ep_rewards}, f, indent=2)
    print(f"  ↳ Metrics saved to {metrics_path}")
    _plot_training(ep_rewards, seed=seed, tag=f"curriculum_{strategy}")
    print("Curriculum training complete.")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _plot_training(rewards: list, seed: int = 0, tag: str = ""):
    smoothed = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(rewards,  alpha=0.3, label="Episode reward")
    plt.plot(smoothed, linewidth=2, label="Smoothed (20-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Skiing RL — TD3 Training curve (seed={seed})")
    plt.legend(loc="upper left")
    plt.tight_layout()

    save_dir = Path("training_curves")
    save_dir.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    save_path = save_dir / f"td3_curve_seed{seed}{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Training curve saved to {save_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 training for SkiEnv")
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--baseline",   action="store_true")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0])
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-strategy", default="mixed",
                        choices=list(CURRICULUM_STRATEGIES.keys()))
    parser.add_argument("--layout",     default=None,
                        choices=list(SkiEnv.LAYOUTS.keys()))
    args = parser.parse_args()

    LAYOUT = args.layout

    if args.curriculum:
        train_curriculum(seed=args.seed, checkpoint=args.checkpoint,
                         strategy=args.curriculum_strategy)
    elif args.eval:
        if args.checkpoint is None:
            raise ValueError("Pass --checkpoint <path> to evaluate")
        evaluate(args.checkpoint, render=args.render,
                 difficulties=args.difficulty)
    elif args.baseline:
        run_baseline(seeds=args.seeds)
    else:
        train(render=args.render, seed=args.seed)