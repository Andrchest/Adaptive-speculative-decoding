# AI Agent Guidelines

## 🌍 Language Policy

All files in this repository — code, comments, documentation, configs, commit messages, PR descriptions — **must be in English**. No Russian, Chinese, or other non-English text. Exception: user-provided data (e.g. prompts, test inputs) may contain any language.

## 📁 Project Layout

```
src/
├── core/                # Core library
│   ├── models/          # DraftModel, TargetModel, UniversalDrafter
│   ├── decoder/         # SpeculativeDecoder
│   ├── translation/     # Cross-vocab translation (Rule1, Rule2)
│   ├── cache/           # N-gram cache with eviction strategies
│   ├── distillation/    # Online distillation
│   ├── profiling/       # Substep timer, torch profiler
│   └── extensions/      # Experimental modules
│       ├── adaptive/    # Acceptance & speedup predictors
│       ├── contrastive/ # Contrastive loss
│       ├── lattice/     # Tokenizer lattice
│       ├── multitarget/ # Universal drafter
│       ├── replay/      # Replay buffer
│       ├── routing/     # Dynamic router
│       └── translator/  # Learned translator
├── experiments/         # Experiment runner & ablation suite
│   ├── built_in/        # 12 built-in experiments
│   └── templates/       # Copy-paste template for researchers
├── benchmarks/          # Metrics collection
└── config/              # Configuration

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── extension_tests/     # Extension tests

research/                # Per-researcher work area
scripts/                 # Standalone scripts (profiler, etc.)
```

## 🔧 Key Conventions

1. **All imports are absolute from `src/` root** — no `sys.path.insert`
2. **Configuration** lives in `src/config/` — never hardcode params
3. **Type hints** are mandatory for public APIs (enforced by mypy)
4. **Docstrings** use Google style
5. **Extensions** go in `src/core/extensions/` — clearly experimental

## 🧪 Testing Rules

- New code → new tests
- Run `pytest` before committing
- GPU tests marked with `@pytest.mark.gpu` — skipped in CI unless GPU available

## 🔒 Branch Rules

- **`main`** — always stable, CI must pass
- **`research/<name>`** — experimental work, can be messy
- PR to `main` requires: lint ✅ + test ✅ + type ✅ + review ✅

## 📝 Code Style Summary

| Rule | Tool | Config |
|------|------|--------|
| Format | `ruff format` | 100 chars, 4-space indent |
| Lint | `ruff check` | see `ruff.toml` |
| Types | `mypy` | strict mode (see `.mypy.ini`) |
| Imports | `ruff` | isort + no unused |
| Pre-commit | `pre-commit` | all hooks auto-run |

## 🧪 Experiment Architecture

Experiments use a **Strategy pattern**: each experiment is a `BaseExperiment` subclass.

### Creating a New Experiment

1. Create `research/<name>/experiments/<file>.py`
2. Subclass `BaseExperiment` and override `get_config()` plus any `build_*` methods
3. Register the class in `__all__`
4. Run: `python src/main.py --research` or `--experiment <name>`

**Never** modify `src/experiments/` shared code for new experiments. Copy
`src/experiments/templates/minimal_template.py` as a starting point.

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `BaseExperiment` | `src/experiments/base.py` | ABC for all experiments |
| `ExperimentRunner` | `src/experiments/runner.py` | Orchestrator (models, datasets, persistence) |
| `ExperimentConfig` | `src/experiments/runner.py` | Configuration dataclass |
| `ABLATION_SUITE` | `src/experiments/suites.py` | Standard 12-experiment ablation |
| `discover_experiments()` | `src/experiments/suites.py` | Auto-discover built-in + research |

### Built-in Experiments

All live in `src/experiments/built_in/` and correspond to the original
flag-based `ABLATION_SUITE`.  New experiments should extend these rather
than duplicate their logic.

### CLI

```bash
python src/main.py --suite ablation       # Run all 12 ablation experiments
python src/main.py --experiment 01_baseline  # Run one experiment
python src/main.py --research              # Run all research experiments
python src/main.py --list                  # List all experiments
python src/main.py --list --research       # List research experiments only
python src/main.py --smoke                 # Quick smoke test
python src/main.py --research --tiny -n 5  # Fast iteration
```

## 🗣 Communication

- Ask questions in PR comments
- Document decisions in `docs/`
- Research results → `research/<username>/`
