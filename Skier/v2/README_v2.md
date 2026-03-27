# Enora — Skiing RL Agent

A PPO-based reinforcement learning agent that learns competition slalom skiing.

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### Train a single run

```bash
python train.py
```

Trains the agent from scratch with seed 0. Add `--seed 5` to pick a different seed, or `--render` to watch the agent live (much slower).

### Evaluate a saved checkpoint

```bash
python train.py --eval --checkpoint checkpoints/seed_0/policy_500000.pt
```

Loads a previously saved model and runs 20 test episodes to print all the validation metrics (finish rate, gates passed, etc.). Add `--render` to watch the episodes play out visually.

### Test on different difficulty levels

Evaluate a checkpoint on a single difficulty (0.0 = easy, 1.0 = hard):

```bash
python train.py --eval --checkpoint checkpoints/seed_0/policy_500000.pt --difficulty 0.5
```

Or sweep across multiple difficulties at once to see how the agent degrades:

```bash
python train.py --eval --checkpoint checkpoints/seed_0/policy_500000.pt --difficulty 0.0 0.25 0.5 0.75 1.0
```

This prints a separate validation block for each difficulty level, making it easy to compare performance across the full range.

### Difficulty sweep

```bash
python train.py --difficulty-sweep --checkpoint checkpoints/seed_0/policy_500000.pt
```

Evaluates the baseline policy across 11 evenly-spaced difficulty levels (0.0 to 1.0) and generates a 6-panel plot answering:

- At what difficulty does finish rate drop and fall rate spike?
- Do gates get missed gradually or does the agent collapse?
- Does the perfect run rate vanish early?
- Does score track the same pattern as reward, or do they diverge?

Use `--sweep-steps 21` for a finer grid, or `--sweep-episodes 50` for lower-variance estimates.

**What it produces:**

- `baseline_results/difficulty_sweep.json` — raw metrics at each difficulty
- `baseline_results/difficulty_sweep.png` — the 6-panel plot

### Curriculum training

Train with progressively increasing difficulty. Three strategies are available:

| Strategy | Promotion | Sampling | Best for |
|---|---|---|---|
| `conservative` | +0.05, 75% gate rate, 3 consecutive passes | Fixed (single difficulty) | Preserving easy-mode, slow and safe |
| `aggressive` | +0.10, 60% gate rate, 2 consecutive passes | Fixed (single difficulty) | Reaching harder levels faster |
| `mixed` | +0.10, 60% gate rate, 2 consecutive passes | Range (uniform ± 0.15 around current) | Broad generalization across layouts |
| `focused` | +0.05, 70% gate rate, 3 consecutive passes | Range (uniform ± 0.08 around current) | Precise learning in the 0.2–0.4 zone |

**From scratch** (test whether the agent can build gate-tracking from zero):

```bash
python train.py --curriculum --curriculum-strategy mixed --seed 0
```

**Warm-start** from baseline (preserves existing skiing ability):

```bash
python train.py --curriculum --curriculum-strategy mixed --seed 0 --checkpoint checkpoints/seed_0/policy_500000.pt
```

**Experiment: compare strategies and warm-start vs scratch**

```bash
# Warm-start comparisons
python train.py --curriculum --curriculum-strategy conservative --seed 0 --checkpoint checkpoints/seed_0/policy_500000.pt
python train.py --curriculum --curriculum-strategy aggressive   --seed 0 --checkpoint checkpoints/seed_0/policy_500000.pt
python train.py --curriculum --curriculum-strategy mixed        --seed 0 --checkpoint checkpoints/seed_0/policy_500000.pt

# From-scratch comparison (mixed only — the most likely to benefit)
python train.py --curriculum --curriculum-strategy mixed --seed 0
```

Then compare final sweep results with:

```bash
python train.py --difficulty-sweep --checkpoint checkpoints/curriculum_mixed_warm_seed_0/policy_final.pt
python train.py --difficulty-sweep --checkpoint checkpoints/curriculum_mixed_scratch_seed_0/policy_final.pt
python train.py --difficulty-sweep --checkpoint checkpoints/curriculum_focused_scratch_seed_0/policy_final.pt
```

**Note on observation changes:** The observation space was expanded from 10D to 16D to give the agent better gate-planning information. Old checkpoints (10D) are not compatible — you must retrain from scratch. See the observation section below for details.

**How promotion works (all strategies):**

- At regular intervals, the agent is validated at the current target difficulty
- If gate pass rate exceeds the promote threshold for N consecutive checks, difficulty increases
- A regression check at d=0.0 runs each time — if easy-mode performance drops, promotion is paused
- The `mixed` strategy samples a different difficulty each episode from a window around the target, so the rollout buffer contains diverse gate layouts

**What it produces:**

- `checkpoints/curriculum_{strategy}_{warm|scratch}_seed_N/` — checkpoints tagged with difficulty
- `curriculum_metrics.json` — full log of promotions, evaluations, and difficulty history
- `curriculum_progress.png` — 3-panel plot:
  - Difficulty over time (staircase)
  - Gate pass rate at training difficulty + d=0.0 regression
  - Final difficulty sweep (gates passed vs difficulty)

### Run the full baseline

```bash
python train.py --baseline
```

Trains 3 separate runs (seeds 0, 1, 2) back-to-back and compares their final results side by side. This tells you whether your results are consistent or just a lucky seed. To pick different seeds:

```bash
python train.py --baseline --seeds 0 1 2 3 4
```

**What it produces:**

- `checkpoints/seed_N/` — model files + `metrics.json` for each seed
- `training_curves/` — one plot per seed
- `baseline_results/baseline_report.json` — everything in one file
- A printed summary table in the terminal

### Test robustness

```bash
python train.py --robustness --seed 0
```

Takes a seed that has already been trained, loads every checkpoint it saved, and runs the validation 5 times on each one. This reveals:

- Whether the agent's performance is stable near convergence (last few checkpoints should agree)
- Whether 20 evaluation episodes is enough (low std = reliable)

To run more batches:

```bash
python train.py --robustness --seed 0 --n-batches 10
```

**What it produces:**

- `baseline_results/robustness_seed0.json` — all raw numbers
- A printed per-checkpoint breakdown with mean ± std

## All flags

| Flag | What it does |
|---|---|
| `--render` | Show the game window during training or eval |
| `--eval` | Evaluate mode (requires `--checkpoint`) |
| `--checkpoint PATH` | Path to a `.pt` model file |
| `--seed N` | Set random seed (default: 0) |
| `--baseline` | Train multiple seeds and compare |
| `--seeds 0 1 2 ...` | Which seeds to use with `--baseline` |
| `--robustness` | Test checkpoints with repeated validation |
| `--n-batches N` | How many validation batches per checkpoint (default: 5) |
| `--difficulty 0.0 ...` | Difficulty level(s) for `--eval`, from 0.0 (easy) to 1.0 (hard) |
| `--curriculum` | Train with progressive difficulty (curriculum learning) |
| `--curriculum-strategy` | Strategy preset: `conservative`, `aggressive`, `mixed`, or `focused` (default: mixed) |
| `--difficulty-sweep` | Sweep difficulty 0.0 to 1.0 and plot degradation curves |
| `--sweep-steps N` | Number of difficulty levels in sweep (default: 11) |
| `--sweep-episodes N` | Episodes per difficulty level in sweep (default: 30) |

## Environment (v2)

### Physics model

The skier moves on a slope with gravity pulling downhill. The dynamics are heading-based:

- **Heading** controls the direction of travel relative to the fall line (straight down = 0)
- **Steering** rotates the heading; turning gets harder at high speed (reduced steer rate)
- **Braking** applies edge pressure, increasing drag to trade speed for control
- **Carving force** redirects velocity to align with the current heading (like real ski edges)
- **Crash** if centripetal acceleration exceeds the grip limit (too fast + too sharp a turn), or out of bounds

### Action space (2D)

| Dim | Name | Range | Effect |
|---|---|---|---|
| 0 | Steer | [-1, 1] | Rotates heading left/right |
| 1 | Brake | [0, 1] | Edge pressure — increases drag, slows the skier |

### Speed regulation

Three sources of drag, all additive:
- **Base drag** (0.02) — always present, air/snow friction
- **Brake drag** (up to 0.25) — proportional to brake action
- **Turn drag** (up to 0.10) — proportional to |steer|, carving costs speed

Speed is hard-capped at 25 m/s. The agent must learn to manage speed: brake before sharp turns, carry speed through straights.

### Track

- Slope length: 120m (up from 100m, giving more room between gates)
- Track width: 10m (up from 8m, accommodating wider gate offsets at high difficulty)
- 4 gates spaced evenly from y=12 to y=108

### Observation space (14D)

| Dims | Name | Description |
|---|---|---|
| 0-5 | Physics state | x, y, vx, vy, heading, speed (all normalised) |
| 6-7 | Gate 1 | Relative x and y to the next upcoming gate |
| 8-9 | Gate 2 | Relative x and y to the gate after next |
| 10 | Gate width | Current gate width normalised to [0, 1] (shrinks with difficulty) |
| 11 | Lateral error | Whether lateral velocity is toward (+) or away from (-) the next gate |
| 12 | Course progress | Fraction of gates already passed (0.0 to 1.0) |
| 13 | Difficulty | Current difficulty level (0.0 to 1.0) |

**Note:** All previous checkpoints are incompatible with v2 (different state, action, and dynamics). Retrain from scratch.
