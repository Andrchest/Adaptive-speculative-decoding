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
