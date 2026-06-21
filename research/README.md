# Research Area — Shared

Each team member has their own folder here for experiments, results, and notes.

## Folder Structure

```
research/<username>/
├── README.md              — project description, hypotheses, tasks
├── experiments/           — experiment classes (auto-discovered)
│   └── my_experiment.py
├── configs/               — experiment configurations
│   └── *.yaml
├── results/               — results
│   ├── *.csv
│   └── *.json
├── notebooks/             — Jupyter notebooks
│   └── *.ipynb
└── plots/                 — plots
    └── *.png
```

## Rules

- Each research branch is created from `main` and lives independently.
- Results are written to `research/<username>/results/`.
- Coding experiments go in `src/`, research analysis goes in notebooks.
- Before merging to `main` — at least one passing test.

---

## Built-in Experiments (Ablation Suite)

The project ships with **11 built-in experiments** that reproduce the original
flag-based ablation suite.  They form a dependency chain — each experiment
adds one or more components on top of its predecessor.

| # | ID | Name | Description | Key Components |
|---|----|------|-------------|----------------|
| 1 | `01_baseline` | Baseline | Core speculative decoding: Rule 1 (exact match) + Rule 2 (heuristic redistribution) + LRU N-gram cache. No distillation, no lattice. | `use_rule1`, `use_rule2`, `NgramCache(LRU)` |
| 2 | `02_+lattice` | Lattice | Replaces approximate Rule 2 with exact dynamic-programming lattice mapping. More accurate token translation at higher compute cost. | `TokenizerLattice` (DP), `use_rule1` |
| 3 | `03_+translator` | Translator | Adds a learned Transformer-based translator (lightweight encoder) alongside the lattice, operating in hybrid mode. | `TranslatorModel`, `lattice`, `translator_weight=0.3` |
| 4 | `04_+online_distil` | Online Distillation | Enables online distillation during decoding — the drafter is trained on-the-fly via KL divergence + N-gram NLL loss using accepted/rejected draft tokens as signal. Optionally uses LoRA adapters. | `OnlineDistiller`, `KL + N-gram NLL`, optional LoRA |
| 5 | `05_+replay_fifo` | Replay (FIFO) | Wraps online distillation with a FIFO replay buffer. Periodically replays old traces for more stable training. | `ReplayBuffer(FIFO)`, `OnlineDistiller` |
| 6 | `06_+replay_prio` | Replay (Prioritized) | Same as FIFO replay but samples by `(1 − acceptance_rate)` to focus on hard-to-draft tokens. | `ReplayBuffer(prioritized)`, `OnlineDistiller` |
| 7 | `07_+contrastive` | Contrastive | Adds contrastive rejection learning: rejected draft tokens become hard negatives in an InfoNCE loss, pushing the drafter away from tokens the target model rejects. | `ContrastiveLoss (InfoNCE)`, `OnlineDistiller` |
| 8 | `08_+speedup_adapt` | Adaptive Drafting | Dynamically selects draft length `k` per step. A small MLP predictor estimates tokens/sec speedup for each candidate `k`; the controller picks `argmax(speedup)`. | `SpeedupPredictor`, `AdaptiveDraftController` |
| 9 | `09_+routing` | Dynamic Router | Multi-drafter routing: a lightweight MLP maps prompt embeddings to drafter indices from a pool of drafters (0.5B / 1.5B / 7B), selecting the most efficient drafter per prompt. | `DynamicRouter`, `RouterModel`, 3 drafter specs |
| 10 | `10_+universal` | Universal Drafter | A single drafter model trained to draft for multiple target LLM families. Learnable target-specific embeddings are injected at each transformer layer via forward hooks. | `UniversalDrafter`, target embeddings, adapter |
| 11 | `11_full_system` | Full System | All components enabled simultaneously: learned translator + online distillation + prioritized replay + contrastive loss + adaptive drafting + dynamic routing + universal drafter. | Everything above |

### Dependency Chain

```
01_baseline
  ├── 02_+lattice
  │     └── 03_+translator
  ├── 04_+online_distil
  │     ├── 05_+replay_fifo
  │     ├── 06_+replay_prio
  │     └── 07_+contrastive
  ├── 08_+speedup_adapt
  ├── 09_+routing
  ├── 10_+universal
  └── 11_full_system  (depends on: translator, replay_prio, contrastive, speedup_adapt, routing, universal)
```

### Component Matrix

Every experiment composes the same set of core modules.  The table below shows
which component each experiment enables (`✓`) and where it remains disabled (`—`).

| Component | Module | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|-----------|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Rule 1 (exact match) | `core.translation.rules` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rule 2 (heuristic) | `core.translation.rules` | ✓ | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| N-gram Cache (LRU) | `core.cache.ngram` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TokenizerLattice (DP) | `core.extensions.lattice.tokenizer_lattice` | — | ✓ | ✓ | — | — | — | — | — | — | — | — |
| Learned Translator | `core.extensions.translator.model` | — | — | ✓ | — | — | — | — | — | — | — | ✓ |
| Online Distiller | `core.distillation.online` | — | — | — | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ |
| Replay Buffer | `core.extensions.replay.buffer` | — | — | — | — | ✓ | ✓ | — | — | — | — | ✓ |
| Contrastive Loss | `core.extensions.contrastive.loss` | — | — | — | — | — | — | ✓ | — | — | — | ✓ |
| Adaptive Drafting | `core.extensions.adaptive.speedup_predictor` | — | — | — | — | — | — | — | ✓ | — | — | ✓ |
| Dynamic Router | `core.extensions.routing.router` | — | — | — | — | — | — | — | — | ✓ | — | ✓ |
| Universal Drafter | `core.extensions.multitarget.universal_drafter` | — | — | — | — | — | — | — | — | — | ✓ | ✓ |

**Legend:** `✓` = enabled, `—` = disabled

### Running Built-in Experiments

```bash
# List all
python src/main.py --list

# Run a specific experiment
python src/main.py --experiment 01_baseline

# Run the full ablation suite
python src/main.py --suite ablation

# Quick smoke test (1 sample, tiny models)
python src/main.py --smoke
```

---

## Creating a New Experiment

The experiment framework uses a **Strategy pattern**: each experiment is a
self-contained class that inherits from `BaseExperiment`.  You create a file
with your class and register it — **no changes to shared code required**.

### Step-by-step

1. **Create the experiments directory** (if it doesn't exist):

   ```bash
   mkdir -p research/ivan/experiments
   ```

2. **Copy the template**:

   ```bash
   cp src/experiments/templates/minimal_template.py \
      research/ivan/experiments/phonetic_translation.py
   ```

3. **Edit the file**:

   - Change `ExperimentMeta` (name, description, tags).
   - Override `get_config()` to set the right flags.
   - Override any `build_*` methods you need (translator, distiller, etc.).
   - Override any `on_*` hooks if you need custom decode behaviour.
   - Make sure your class name is listed in `__all__` at the bottom.

   Minimal example:

   ```python
   from experiments.base import BaseExperiment, ExperimentMeta
   from experiments.runner import ExperimentConfig

   class PhoneticTranslationExperiment(BaseExperiment):
       def __init__(self):
           super().__init__(ExperimentMeta(
               name="phonetic_translation",
               description="Phonetic mapping instead of Rule 2",
               tags=["translation", "ivan"],
           ))

       def get_config(self) -> ExperimentConfig:
           return ExperimentConfig(
               name=self.meta.name,
               use_rule1=True,
               use_lattice=True,
               # ... other flags
           )

       def build_translator(self, ctx):
           translator = super().build_translator(ctx)
           # Add your phonetic mapping component
           translator.phonetic = PhoneticMapper(...)
           return translator

   __all__ = ["PhoneticTranslationExperiment"]
   ```

4. **Run your experiment**:

   ```bash
   # Run all research experiments
   python src/main.py --research

   # Run a specific experiment by name
   python src/main.py --experiment phonetic_translation

   # Fast iteration with tiny models
   python src/main.py --research --tiny -n 5
   ```

### Inheriting from Built-in Experiments

Instead of starting from scratch, extend an existing built-in experiment:

```python
from experiments.built_in import BaselineExperiment
from experiments.base import ExperimentMeta

class MyExtendedBaseline(BaselineExperiment):
    """Everything from baseline, but with my custom distiller."""

    def __init__(self):
        super().__init__()
        self.meta = ExperimentMeta(
            name="my_extended_baseline",
            description="Baseline + my custom distiller",
            tags=["distillation", "ivan"],
        )

    def build_distiller(self, ctx):
        # Build your custom distiller here
        ...
```

### Available Build Methods

Override these methods in `BaseExperiment` to customize components:

| Method | Returns | Purpose |
|--------|---------|---------|
| `build_translator(ctx)` | `CrossVocabTranslator` | Cross-vocab translation (Rule1/2, lattice, learned) |
| `build_cache(ctx)` | `NgramCache` | N-gram cache with eviction strategy |
| `build_distiller(ctx)` | `OnlineDistiller \| None` | Online distillation (KL, replay, contrastive) |
| `build_adaptive_controller(ctx)` | `Any \| None` | Adaptive draft-length controller |
| `build_router(ctx)` | `DynamicRouter \| None` | Dynamic drafter routing |
| `build_universal_drafter(ctx)` | `Any \| None` | Universal drafter adapter |

### Available Hooks

Override these methods to customize decode behaviour:

| Hook | When | Purpose |
|------|------|---------|
| `on_before_decode(ctx)` | Once, before decode loop | Initialize state, warm-up |
| `on_decode_step(ctx, stats, prompt_index)` | After each prompt | Log, adapt, collect stats |
| `on_after_decode(ctx)` | Once, after all prompts | Finalize, flush buffers |
| `on_extra_metrics(summary)` | At the end | Add custom metrics to results |

### Adding Custom Metrics

```python
def on_extra_metrics(self, summary: dict) -> dict:
    # Add your custom metrics
    summary["my_custom_metric"] = self._compute_metric()
    return summary
```

### Listing Experiments

```bash
# List all experiments (built-in + research)
python src/main.py --list

# List research experiments only
python src/main.py --list --research
```

### Useful Patterns

**Parameterized experiments** (like `ReplayExperiment(strategy="fifo")`):

```python
class MyParameterizedExperiment(BaseExperiment):
    def __init__(self, temperature: float = 1.0):
        super().__init__(ExperimentMeta(
            name=f"my_exp_t{temperature}",
        ))
        self.temperature = temperature

__all__ = ["MyParameterizedExperiment"]
```

**Accessing components from build methods**:

```python
def build_distiller(self, ctx):
    # ctx.drafter — DraftModel instance
    # ctx.target — TargetModel instance
    # ctx.device — "cuda" or "cpu"
    # ctx.config — ExperimentConfig
    # ctx.components["translator"] — CrossVocabTranslator
    # ctx.components["cache"] — NgramCache
    ...
```

## References

- **Template**: `src/experiments/templates/minimal_template.py`
- **Built-in examples**: `src/experiments/built_in/`
  - `with_lattice.py` — simplest extension (override `build_translator`)
  - `with_online_distil.py` — override `build_distiller`
  - `full_system.py` — override everything
- **Plan**: `docs/plans/experiment-refactor-option-b.md`
