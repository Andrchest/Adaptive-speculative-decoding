# Handoff — Bandit Routing for Speculative Decoding (m.krylov)

> Branch: `research/m.krylov`, HEAD `3192748`
> Working tree: clean
> Single implementation file: `experiments/bandit_routing.py` (1918 lines)

---

## Environment Setup

```bash
# 1. Clone and enter the repo
cd /home/jovyan/persistent_volume/Adaptive-speculative-decoding
git checkout research/m.krylov

# 2. Install dependencies (uv manages the .venv)
uv sync

# 3. Verify
uv run python src/main.py --list --research
```

**Requirements**: Python 3.12, CUDA GPU (A100 80GB tested), `uv` package manager.

---

## What This Does

Replaces static MLP drafter-routing with **online multi-armed bandit** algorithms that learn which drafter to pick per prompt from live decoding performance — no separate training phase required.

**Three bandit families**: UCB1, Thompson Sampling, LinUCB (contextual). Each optionally combined with **per-arm online distillation** so drafters improve while the router learns.

### Reward

Computed per prompt (aggregated over all decode steps):

```
reward = total_accepted_tokens / max(total_wall_time_ms / 1000, 1e-6)    # tokens/sec
```

Balances quality (acceptance count) against speed (wall-clock time).

### Data Flow

```
prompt input_ids
    ↓
router.select_drafter(input_ids)  ← bandit algorithm
    ↓
selected drafter generates K tokens
    ↓
target verifies in 1 forward pass
    ↓
accepted tokens emitted, bonus residual sampled
    ↓
reward computed from StepResult list
    ↓
router.update(reward)
    ↓
[if distillation enabled] → PerArmBuffer → distiller.step()
```

---

## Running Experiments

All commands use `uv run` (auto-activates the project venv).

### Quick smoke test (tiny models, 5 samples, ~30s)

```bash
uv run python src/main.py --experiment bandit_ucb --tiny -n 5
```

### Serious experiments (real Qwen models)

Default models: `Qwen2.5-0.5B` + `Qwen2.5-1.5B` drafters, `Qwen2.5-7B` target, gsm8k dataset.

```bash
# UCB1 baseline (100 samples)
uv run python src/main.py --experiment bandit_ucb -n 100

# Thompson Sampling (compare with UCB)
uv run python src/main.py --experiment bandit_thompson -n 100

# Exploration parameter sweep (UCB c + LinUCB α)
uv run python src/main.py --experiment bandit_exploration_sweep -n 100

# Bandit vs MLP router (MLP trained online)
uv run python src/main.py --experiment bandit_vs_mlp_ucb -n 100
uv run python src/main.py --experiment bandit_vs_mlp_thompson -n 100

# Multi-dataset (gsm8k, mbpp, alpaca, xsum)
uv run python src/main.py --experiment bandit_multidataset_ucb -n 200

# With per-arm distillation
uv run python src/main.py --experiment bandit_ucb_distill -n 100
uv run python src/main.py --experiment bandit_contextual_distill -n 100
```

### Useful flags

| Flag | Effect |
|---|---|
| `--tiny` / `-t` | Use `opt-125m` / `opt-350m` instead of Qwen |
| `-n N` | Override sample count (default 500) |
| `--no-mlflow` | Disable MLflow tracking (default: `sqlite:///mlflow.db`) |
| `--log-level VERBOSE` | Show all logs (default: QUIET = tqdm only) |
| `--research` / `-r` | Run all research experiments |

### List all experiments

```bash
uv run python src/main.py --list --research
```

---

## Analyzing Results

### MLflow UI (browse all runs visually)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Per-experiment JSON

```bash
cat results/bandit_ucb.json | python -m json.tool | head -40
```

### Quick comparison (UCB vs Thompson)

```python
import json

for name in ["bandit_ucb", "bandit_thompson"]:
    with open(f"results/{name}.json") as f:
        m = json.load(f)["metrics"]
    print(f"\n{name}:")
    print(f"  mean_reward:      {m.get('bandit_mean_reward', 'N/A')}")
    print(f"  acceptance_rate:  {m.get('acceptance_rate', 0)*100:.1f}%")
    print(f"  tokens_per_sec:   {m.get('tokens_per_sec', 0):.1f}")
    print(f"  router stats:     {json.dumps(m.get('bandit_router', {}), indent=2)}")
```

### Exploration sweep convergence analysis

```python
import json

with open("results/bandit_exploration_sweep.json") as f:
    sweep = json.load(f)["metrics"]["exploration_sweep"]

for key, v in sweep.items():
    print(f"{key:20s} reward={v['mean_reward']:8.1f}  conv@{v['convergence_sample']}")
```

### Bandit vs MLP agreement

```python
import json

with open("results/bandit_vs_mlp_ucb.json") as f:
    comp = json.load(f)["metrics"]["bandit_vs_mlp_comparison"]
print(f"Agreements:    {comp['agreements']}")
print(f"Disagreements: {comp['disagreements']}")
print(f"MLP trained:   {comp['mlp_online_train_count']} rounds")
```

### Multi-dataset breakdown

```python
import json

with open("results/bandit_multidataset_ucb.json") as f:
    ds = json.load(f)["metrics"]["per_dataset_metrics"]
for name, m in ds.items():
    print(f"{name:10s} reward={m['mean_reward']:8.1f}  acc={m['acceptance_rate']*100:5.1f}%")
```

---

## Experiment Catalog (11 experiments)

| Name | Algorithm | Distillation | Notes |
|---|---|---|---|
| `bandit_ucb` | UCB1 | no | Baseline bandit |
| `bandit_thompson` | Thompson | no | Bayesian baseline |
| `bandit_ucb_distill` | UCB1 | yes | Per-arm distillation |
| `bandit_thompson_distill` | Thompson | yes | Per-arm distillation |
| `bandit_contextual` | LinUCB | no | 8 prompt features, vocab_size-aware |
| `bandit_contextual_distill` | LinUCB | yes | Contextual + distillation |
| `bandit_vs_mlp_ucb` | UCB1 | no | Head-to-head vs MLP (MLP trained online) |
| `bandit_vs_mlp_thompson` | Thompson | no | Head-to-head vs MLP |
| `bandit_multidataset_ucb` | UCB1 | no | gsm8k, mbpp, alpaca, xsum |
| `bandit_multidataset_thompson` | Thompson | no | Cross-dataset Thompson |
| `bandit_exploration_sweep` | UCB1 + LinUCB | no | Sweeps exploration params (c: 0.1–5.0, α: 0.1–5.0) |

### Router Algorithms

**UCB1** — `score(arm) = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)`. One hyperparameter `c` (default 2.0).

**Thompson Sampling** — Normal-Gamma conjugate posterior per arm. Prior: μ₀=0, κ₀=1, α₀=1, β₀=1. Fixed RNG seed 42.

**LinUCB (contextual)** — Ridge-regression weight vector θ ∈ ℝᵈ (d=8). Score = θᵀx + α·√(xᵀA⁻¹x). Features: log(prompt_length), vocab diversity, mean/std token ID (normalised by vocab_size), token frequency bins.

### Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `exploration` | 2.0 | UCB `c` or LinUCB `α` |
| `enable_distillation` | False | Per-arm online distillation |
| `buffer_capacity` | 4096 | PerArmBuffer max entries |
| `replay_every` | 32 | Distillation replay frequency (prompts) |
| `replay_batch` | 8 | Distillation batch size |
| `reward_window` | 0 | Sliding window size (0 = all history) |
| `per_step_update` | False | Per-step vs per-prompt bandit updates |
| `convergence_window` | 3 | Consecutive same-arm to count as converged (sweep only) |

---

## Remaining Tasks

### Phase 2 — [ ] Compare UCB vs Thompson
Run both on same dataset with same seed, compare convergence speed, final reward, arm distribution.

### Phase 3 — [ ] Monitor distillation effects on bandit behaviour
Use `reward_window > 0` to track adaptation when drafters improve mid-run.

### Phase 3 — [ ] Tune distillation parameters
Optimal `replay_every` and `replay_batch` depend on dataset and drafter sizes. Current defaults (32, 8) are placeholders.

### Phase 4 — [ ] Analyse exploration vs exploitation trade-off
Larger sweeps (`n ≥ 50`) needed for convergence analysis. Smoke test (`n=5`) shows UCB `c=0.1` is best at small sample sizes.

### Phase 4 — [ ] Add humaneval to multi-dataset
Currently covers gsm8k, mbpp, alpaca, xsum. humaneval not yet in `DATASETS`.

### Phase 2b — [ ] Non-stationary reward analysis
With `reward_window > 0`, study how policies behave when drafters improve mid-run. Standard bandit regret bounds don't apply.

---

## Smoke Test Results (for reference)

**Command**: `uv run python src/main.py --experiment bandit_exploration_sweep --tiny -n 5`
**Models**: `opt-125m` vs `opt-350m` as drafters, `opt-350m` as target, gsm8k, 5 samples.

### UCB sweep

| c value | mean_reward | acceptance_rate | arm distribution | convergence |
|---|---|---|---|---|
| 0.1 | 362.6 | 70.8% | opt-350m: 4/5 | sample 2 |
| 0.5 | 315.9 | 64.2% | opt-350m: 4/5 | sample 2 |
| 1.0 | 310.2 | 63.4% | opt-350m: 4/5 | sample 2 |
| 2.0 | 250.2 | 55.0% | opt-350m: 4/5 | sample 2 |
| 5.0 | 256.5 | 55.9% | opt-125m: 3/5 | None |

### LinUCB sweep

| α value | mean_reward | acceptance_rate | arm distribution | convergence |
|---|---|---|---|---|
| 0.1 | 345.7 | 68.4% | opt-350m: 4/5 | sample 2 |
| 1.0 | 293.8 | 61.1% | opt-350m: 4/5 | sample 2 |
| 5.0 | 273.7 | 58.3% | opt-350m: 4/5 | sample 2 |

**Best overall**: `ucb_0.1` with mean_reward=362.6, acceptance_rate=70.8%.

---

## Files

```
research/m.krylov/
├── README.md                        # Research hypothesis, phased plan, task checklist
├── HANDOFF.md                       # This file
└── experiments/
    ├── __init__.py                  # Docstring only (discovery uses __all__ from bandit_routing.py)
    └── bandit_routing.py            # Everything: routers, buffer, 11 experiment classes (1918 lines)
```

MLP routing components (`DynamicRouter`, `RouterModel`, `DrafterSpec`) are imported from `src/core/extensions/routing/router.py`.
