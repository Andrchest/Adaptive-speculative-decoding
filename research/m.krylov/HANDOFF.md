# Handoff — Bandit Routing Experiment (m.krylov)

> Commit `1d4ecd9` — 2026-06-24
> Last updated: 2026-06-29 (AI assistant session)

---

## 1. Project Overview

### What the project does

**Adaptive Speculative Decoding** is a research framework for speeding up LLM inference. The core idea: instead of a large "target" model generating tokens one at a time (slow), a small "drafter" model proposes multiple tokens at once, and the target verifies them all in a single forward pass. Accepted tokens are kept; rejected ones trigger a residual sample. The theoretical guarantee is **zero quality loss** — output distribution matches the target exactly.

### Core pipeline

```
Prompt → Drafter drafts k tokens → Cross-vocab translation → Target verifies (1 pass)
       → Accept/reject each token → Residual bonus sample → Update cache + distill
```

### Key modules

| Module | Purpose |
|---|---|
| `core/decoder/speculative.py` | Main decode loop — draft, verify, accept/reject, residual sample |
| `core/models/draft_model.py` | Small fast model wrapper (autoregressive draft generation) |
| `core/models/target_model.py` | Large slow model wrapper (batch verification, optional 4-bit NF4) |
| `core/translation/` | Cross-vocabulary mapping when drafter/target use different tokenizers (Rule1 = exact match, Rule2 = approximate via Aho-Corasick, or TokenizerLattice = exact DP) |
| `core/cache/ngram.py` | N-gram cache with LRU/LFU/acc/hybrid eviction |
| `core/distillation/online.py` | Online distillation — fine-tunes drafter during inference using KL + NLL loss |
| `core/extensions/routing/` | MLP-based dynamic router (selects drafter per prompt) |
| `core/extensions/adaptive/` | SpeedupPredictor — adapts draft length k per context |
| `core/extensions/contrastive/` | InfoNCE loss using rejected tokens as hard negatives |
| `core/extensions/replay/` | Replay buffer with FIFO or prioritized sampling |
| `core/extensions/multitarget/` | UniversalDrafter — single drafter for multiple target families via per-target embeddings |

### Experiment framework

Uses a **Strategy pattern**. Every experiment is a `BaseExperiment` subclass that overrides `build_*` methods (translator, cache, distiller, router, etc.) and `on_*` hooks (before/after decode, per-step). The `ExperimentRunner` orchestrates model loading, dataset loading, GPU memory management, and result persistence.

```bash
python src/main.py --smoke                     # Quick test
python src/main.py --suite ablation             # All 11 built-in experiments
python src/main.py --research                   # All research experiments
python src/main.py --experiment bandit_ucb      # Single experiment
python src/main.py --list                       # List all experiments
```

### Ablation suite (11 built-in experiments)

1. `01_baseline` — plain speculative decoding
2. `02_+lattice` — exact tokenizer lattice instead of Rule2 heuristic
3. `03_+translator` — learned translator blended with Rule1+Rule2
4. `04_+online_distil` — online distillation
5. `05_+replay_fifo` — replay with FIFO sampling
6. `06_+replay_prioritized` — replay with prioritized sampling
7. `07_+contrastive` — contrastive loss on rejected tokens
8. `08_+speedup_adapt` — adaptive draft length
9. `09_+routing` — MLP-based dynamic multi-drafter router
10. `10_+universal` — universal multi-target drafter
11. `11_full_system` — everything combined

---

## 2. Commit 1d4ecd9 — Bandit Routing Experiment

### What was added

Two files:

| File | Lines | Purpose |
|---|---|---|
| `research/m.krylov/experiments/bandit_routing.py` | +641 | Full implementation |
| `research/m.krylov/README.md` | +91/-7 | Research hypothesis, phased plan, references |

### The research hypothesis

The existing routing approach (`09_+routing`) uses a trained MLP classifier that maps prompt embeddings to drafter indices. This requires:
- A training phase to collect (prompt, best-drafter) pairs
- A separate router model to maintain
- Static assignment — doesn't adapt during inference

**Proposal**: Replace the MLP router with a **multi-armed bandit** algorithm that learns routing policy **online** from actual decoding performance, while drafters are simultaneously improved through online distillation.

### Data flow

```
Prompt arrives
    ↓
Bandit router selects drafter αᵢ (UCB score or Thompson sample)
    ↓
Selected drafter generates K tokens
    ↓
Target verifies draft tokens (1 forward pass)
    ↓
Some tokens accepted, rest rejected + residual bonus
    ↓
Reward computed: rᵢ = accepted_count / (T_draft + T_target)
    ↓
Router updates bandit policy with reward
    ↓
Accepted tokens + target logits stored in per-arm buffer
    ↓
Periodically: replay buffer → distill selected drafter
```

### Reward definition

```
rᵢ = A / (T_draft + T_target)
```

Where:
- **A** = number of accepted draft tokens
- **T_draft** = wall-clock time for draft generation
- **T_target** = wall-clock time for target verification

This reward balances quality against speed — a fast but inaccurate drafter and an accurate but slow drafter are both penalized.

### Two algorithms implemented

#### UCB1 (Upper Confidence Bound)

```
score(arm) = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)
```

- Simple, deterministic, one hyperparameter (`c`, default 2.0)
- Exploration naturally decays as arms are pulled more
- Round-robin until every arm has been pulled once

#### Thompson Sampling (Normal-Gamma posterior)

- Bayesian: maintains full posterior over each arm's mean reward
- Prior: `mu ~ N(mu_0, sigma²/kappa_0)`, `sigma² ~ InvGamma(alpha_0, beta_0)`
- After N observations: updates kappa, mu, alpha, beta analytically
- Selection: sample from each arm's posterior, pick highest
- Naturally balances exploration/exploitation without manual tuning

### Per-arm distillation

Each drafter has its own `OnlineDistiller` and optimizer. The `PerArmBuffer` tags every training entry with the arm index, so during replay each drafter is trained **only on data it generated**. This prevents distribution mismatch — drafter A shouldn't learn from drafter B's proposals.

### Key classes

| Class | Role |
|---|---|
| `UCBBanditRouter` | UCB1 selection, `select_drafter()` / `update(reward)` |
| `ThompsonSamplingRouter` | Thompson Sampling with Normal-Gamma posterior |
| `_GaussianArm` | Internal: tracks posterior parameters for one arm |
| `DrafterEntry` | One bandit arm: name, model, pulls, total_reward |
| `PerArmBuffer` | FIFO buffer tagged by arm index; `sample_for_arm()` |
| `BanditRoutingExperiment` | Main experiment — builds drafters, router, distillers |
| `BanditUCBExperiment` | Convenience: UCB without distillation (phases 1-2) |
| `BanditThompsonExperiment` | Convenience: Thompson without distillation (phases 1-2) |
| `BanditUCBDistillExperiment` | Convenience: UCB + distillation (phase 3+) |
| `BanditThompsonDistillExperiment` | Convenience: Thompson + distillation (phase 4) |

### Phased development plan

| Phase | Status | Description |
|---|---|---|
| **1** | ✅ Done | UCB1 router, reward signal, no distillation |
| **2** | ✅ Done | Thompson Sampling, arm switching during decoding |
| **3** | ✅ Done | Per-arm distillation with tagged buffer |
| **4** | ⏳ Remaining | Compare against MLP routing, multiple datasets, contextual bandit |

### How to run

```bash
# UCB without distillation
python src/main.py --research --experiment bandit_ucb

# Thompson Sampling without distillation
python src/main.py --research --experiment bandit_thompson

# UCB with distillation
python src/main.py --research --experiment bandit_ucb_distill

# Thompson with distillation
python src/main.py --research --experiment bandit_thompson_distill

# All research experiments
python src/main.py --research

# Quick test with tiny models
python src/main.py --research --tiny -n 5
```

### Metrics produced

In addition to standard metrics (acceptance_rate, tokens_per_sec, etc.), the experiment reports:

- `bandit_router` — algorithm, exploration params, per-arm pulls and mean rewards
- `bandit_mean_reward` / `bandit_std_reward` — reward statistics across all steps
- `mean_wall_time_ms` — average decode step timing
- `buffer_stats` — total entries and per-arm breakdown (when distillation enabled)

### Novelty vs prior work

The closest prior work is **Online Speculative Decoding**, which combines online distillation with model routing. Their routing uses a BERT-based classifier that assigns draft models by prompt domain (static, requires training data). The bandit approach learns directly from decoding performance (acceptance rate + wall-clock time) with no separate training phase — the routing policy improves continuously during inference.

### Open questions for Phase 4

1. Does bandit routing outperform MLP routing (`09_+routing`) on acceptance rate and throughput?
2. How does exploration vs exploitation trade-off evolve over a long decoding session?
3. Can a contextual bandit (using prompt features like length, domain, token distribution) do even better?
4. How do bandit policies behave when drafters are being simultaneously improved by distillation (non-stationary rewards)?

---

## 3. Bug fixes applied (2026-06-29, original)

Four bugs were found and fixed in the bandit routing experiment:

### Bug 1: `on_decode_step` received a `dict` instead of `StepResult` (CRITICAL)

The runner's `BaseExperiment.run()` called `self.on_decode_step(decode_ctx, decoder.stats(), i)` **after** clearing `decoder._step_results`. Since `decoder.stats()` reads from `_step_results`, the hook got an empty dict. The bandit's `on_decode_step` tried `step_result.accepted_count` which fell through to `hasattr → else 0`, so **reward was always 0** and the bandit never learned.

**Fix:** Changed the runner to capture `_step_results = list(decoder._step_results)` **before** clearing, and pass the list to the hook. Updated `on_decode_step` signature to accept `list[StepResult]`. The bandit experiment now aggregates `total_accepted`, `total_wall_ms`, `total_draft` across all steps in the prompt.

### Bug 2: Duplicate experiment names

`__all__` exported 5 classes but 3 shared the name `bandit_routing_ucb` (base class defaulting to ucb + `BanditUCBExperiment` + `BanditUCBDistillExperiment`) and 2 shared `bandit_routing_thompson`. Running `--experiment bandit_routing_ucb` executed all 3.

**Fix:** Removed base `BanditRoutingExperiment` from `__all__`. Gave each convenience class a unique `meta.name`: `bandit_ucb`, `bandit_thompson`, `bandit_ucb_distill`, `bandit_thompson_distill`.

### Bug 3: `on_decode_step` granularity mismatch

The docstring said "per decode step" but the runner calls it once per completed prompt. The bandit's `step_count` was labeled as steps but was actually counting prompts.

**Fix:** Updated docstrings to reflect that the hook is called once per prompt, receiving a list of `StepResult` objects (one per decode step within that prompt).

### Bug 4: Missing `__init__.py`

`research/m.krylov/experiments/` had no `__init__.py`.

**Fix:** Created `__init__.py`.

### Files changed

| File | Change |
|---|---|
| `src/experiments/base.py` | Pass `list[StepResult]` to `on_decode_step` before clearing |
| `research/m.krylov/experiments/bandit_routing.py` | Fixed `on_decode_step`, unique names, removed base from `__all__` |
| `research/m.krylov/experiments/__init__.py` | Created |

---

## 4. Additional fixes (2026-06-29, AI assistant session)

### Bug 5: `--tiny` didn't override `drafter_model_paths`

The `--tiny` CLI flag only overrode `drafter_model_path` (singular) and `target_model_path`, but `build_router` reads `drafter_model_paths` (plural) which was hardcoded in the config to Qwen models. This caused the bandit experiment to try loading **Qwen/Qwen2.5-0.5B + Qwen/Qwen2.5-1.5B as drafters** even with `--tiny`, leading to OOM.

**Fix (`src/main.py`):** `_apply_overrides` now also sets `drafter_model_paths` when `--tiny` is used:
```python
exp.set_config_override("drafter_model_paths", [
    "facebook/opt-125m",
    "facebook/opt-350m",
])
```

### Bug 6: `--research -e <name>` ran ALL research experiments instead of filtering

The CLI selection logic checked `--research` before `-e`, so `--research -e bandit_ucb` discovered and ran all 4 research experiments instead of just the named one.

**Fix (`src/main.py`):** Moved the `experiment` check to be evaluated first. When both `--research` and `-e` are given, it searches only research experiments for the matching name.

### Bug 7: MLflow run left active after experiment failure

When an experiment failed after `_setup_mlflow()` started an MLflow run, the run was never ended. The next experiment then failed with "Run with UUID ... is already active."

**Fix (`src/experiments/runner.py`):** `_setup_mlflow` now checks for and ends any active MLflow run before starting a new one.

### Bug 8: Thompson Sampling used non-existent `torch.gamma()`

`_GaussianArm.sample()` called `torch.gamma(...)` which doesn't exist in PyTorch. The correct API is `torch.distributions.Gamma`.

**Fix (`bandit_routing.py`):** Replaced with:
```python
import torch.distributions as D
gamma_dist = D.Gamma(concentration=torch.tensor(self.alpha_N), rate=torch.tensor(self.beta_N))
tau = gamma_dist.sample().item()
```

### Bug 9: `build_router` loaded duplicate drafter on GPU

The runner loads `ctx.drafter` (e.g., opt-125m). Then `build_router` loaded a **second copy** of the same model from `drafter_model_paths[0]` (also opt-125m). Two copies of the same model on GPU = wasted VRAM.

**Fix (`bandit_routing.py`):** `build_router` now reuses `ctx.drafter` when the path matches `cfg.drafter_model_path` instead of loading a new `DraftModel`.

### Bug 10: CUDA allocator OOM on MIG partitions

The PyTorch CUDA caching allocator fails with `NVML_SUCCESS == r INTERNAL ASSERT FAILED at "CUDACachingAllocator.cpp":1015` on NVIDIA A100 MIG partitions. This affects **all experiments** (including built-in baseline), not just the bandit experiment. The error triggers during `scatter_add_` / `index_add_` in the vocabulary translation step.

**Partial fix (`src/main.py`):** Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the top of `main.py` before any torch imports. This helps on some systems but **does not fully resolve** the issue on this particular MIG partition (A100-SXM4-80GB MIG 3g.40gb, PyTorch 2.6.0+cu124, driver 555.42.06).

**Also changed (`src/core/translation/rules.py`):** Replaced `index_add_` with `scatter_add_` as an alternative kernel, but this also triggers the same allocator bug on this hardware.

**⚠️ BLOCKER:** The experiments **cannot run on this machine** due to the CUDA allocator bug. This is a known PyTorch issue with MIG partitions. On a non-MIG GPU or full GPU, the experiments should work. Workarounds to try:
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already done in code)
- Use a non-MIG GPU
- Upgrade PyTorch (this may be fixed in later versions)
- Run on CPU with `--device cpu` (slow but should work for correctness testing)

### Files changed in this session

| File | Changes |
|---|---|
| `src/main.py` | Added `PYTORCH_CUDA_ALLOC_CONF` env var; fixed `--tiny` to override `drafter_model_paths`; fixed `--research -e` selection order |
| `src/experiments/runner.py` | Fixed MLflow run leak (end active run before starting new one) |
| `src/core/translation/rules.py` | Replaced `index_add_` with `scatter_add_` in Rule1 `map_logits` |
| `research/m.krylov/experiments/bandit_routing.py` | Fixed `torch.gamma` → `torch.distributions.Gamma`; `build_router` reuses `ctx.drafter` to avoid duplicate GPU loads |

---

## 5. Current state of the branch

The branch `research/m.krylov` is the **active working branch**. It is based on `main` with the full experiment framework, all built-in experiments, and the bandit routing experiment on top.

Most recent commits (newest first):

| Commit | Author | Description |
|---|---|---|
| `1d4ecd9` | S1norin | Bandit routing experiment (UCB + Thompson + per-arm distillation) |
| `49922b5` | Andrchest | Split `drafter.py` into `draft_model.py` + `target_model.py` |
| `50c3538` | Andrchest | Sub-optimization fixes (P0-P2) + profiler |
| `0acf039` | Andrchest | Fix 5 memory leaks + critical fixes test suite |
| `f64b7ab` | Andrchest | Add experiment descriptions to research/README.md |

**Uncommitted changes** from this session are in the files listed above. They should be reviewed and committed.

---

## 6. Quick reference

### Hardware note

The current dev machine has an **NVIDIA A100-SXM4-80GB in MIG mode (3g.40gb partition = 40 GB VRAM)**. PyTorch 2.6.0+cu124 has a CUDA allocator bug on this setup that causes OOM during tensor operations. The code changes are correct but **cannot be verified on this machine**. Test on a non-MIG GPU.

### Project layout (relevant parts)

```
src/
├── core/
│   ├── decoder/speculative.py       # Main decode loop
│   ├── models/draft_model.py        # DraftModel
│   ├── models/target_model.py       # TargetModel
│   ├── translation/                 # Cross-vocab translation
│   ├── cache/ngram.py               # N-gram cache
│   ├── distillation/online.py       # OnlineDistiller
│   └── extensions/
│       ├── routing/router.py        # MLP-based DynamicRouter
│       ├── adaptive/                # SpeedupPredictor
│       ├── contrastive/             # InfoNCE loss
│       ├── replay/buffer.py         # ReplayBuffer
│       └── multitarget/             # UniversalDrafter
├── experiments/
│   ├── base.py                      # BaseExperiment ABC
│   ├── runner.py                    # ExperimentRunner + ExperimentConfig
│   ├── suites.py                    # ABLATION_SUITE, discovery
│   ├── built_in/                    # 10 built-in experiments
│   └── templates/minimal_template.py # Copy-paste starting point
└── main.py                          # CLI entry point

research/m.krylov/
├── README.md                        # Research hypothesis + plan
├── HANDOFF.md                       # This file
└── experiments/
    ├── __init__.py                  # Package init
    └── bandit_routing.py            # Bandit routing implementation
```

### Configuration defaults

```python
# Original (full models):
drafter_model_paths = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
target_model_path = "Qwen/Qwen2.5-7B-Instruct"

# --tiny override:
drafter_model_paths = ["facebook/opt-125m", "facebook/opt-350m"]
target_model_path = "facebook/opt-350m"
```

### Running the bandit experiments

```bash
# List all research experiments
python src/main.py --list --research

# Run a single experiment (use --research flag so -e searches research experiments)
python src/main.py --research --experiment bandit_ucb --tiny -n 10
python src/main.py --research --experiment bandit_thompson --tiny -n 10
python src/main.py --research --experiment bandit_ucb_distill --tiny -n 10
python src/main.py --research --experiment bandit_thompson_distill --tiny -n 10

# All research experiments
python src/main.py --research --tiny -n 5

# On CPU (slow but avoids CUDA allocator bug)
python src/main.py --research --experiment bandit_ucb --tiny -n 2 --device cpu --no-mlflow
```
