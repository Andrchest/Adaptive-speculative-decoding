# Adaptive Speculative Decoding

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Adaptive Speculative Decoding** for LLM inference — research framework for optimizing generation speed through intelligent draft selection, cross-vocabulary translation, and online distillation.

## 🚀 Quick Start

```bash
# 1. Clone
git clone git@github.com:Andrchest/Adaptive-speculative-decoding.git
cd Adaptive-speculative-decoding

# 2. Install (requires Python 3.12)
uv sync --all-extras

# 3. Run smoke test
uv run python src/main.py --smoke

# 4. Run ablation suite
uv run python src/main.py --suite ablation
```

## 📐 Project Structure

```
Adaptive-speculative-decoding/
├── src/                      # Main source code
│   ├── core/                 # Core library
│   │   ├── models/           # DraftModel, TargetModel, UniversalDrafter
│   │   ├── decoder/          # SpeculativeDecoder
│   │   ├── translation/      # Cross-vocab translation (Rule1, Rule2)
│   │   ├── cache/            # N-gram cache with eviction strategies
│   │   ├── distillation/     # Online distillation
│   │   └── extensions/       # Experimental modules
│   ├── experiments/          # Experiment runner & ablation suite
│   │   ├── base.py           # BaseExperiment (Strategy pattern)
│   │   ├── runner.py         # ExperimentRunner (orchestrator)
│   │   ├── suites.py         # ABLATION_SUITE, discovery
│   │   ├── built_in/         # 10 built-in experiments
│   │   └── templates/        # Copy-paste template for researchers
│   ├── benchmarks/           # Metrics collection
│   ├── config/               # Configuration
│   ├── utils/                # Shared utilities
│   ├── inference/            # Production inference (future)
│   └── main.py               # CLI entry point
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── extension_tests/      # Extension-specific tests
├── research/                 # Research areas per team member
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks for analysis
└── pyproject.toml            # Project configuration (uv + hatchling)
```

## 🔬 Research Tracks

Each team member works on their own research track (separate branch).
Experiments use a **Strategy pattern** — each experiment is a self-contained
`BaseExperiment` subclass in `research/<name>/experiments/`.

### Creating a New Experiment (for researchers)

```bash
# 1. Create your research directory
mkdir -p research/<your_name>/experiments

# 2. Copy the template
cp src/experiments/templates/minimal_template.py \
   research/<your_name>/experiments/my_idea.py

# 3. Edit: change meta, get_config(), optional build_* / on_* overrides

# 4. Run
eu run python src/main.py --research          # all research experiments
eu run python src/main.py --experiment my_idea  # single experiment
```

See `src/experiments/templates/minimal_template.py` and `research/README.md` for details.

### Research Team Members

| Member | Branch | Direction |
|--------|--------|-----------|
| m.krylov | `research/m.krylov` | — |
| v.poponnikov | `research/v.poponnikov` | — |
| a.polevoi | `research/a.polevoi` | — |
| al.khadeeva | `research/al.khadeeva` | — |
| da.popov | `research/da.popov` | — |
| e.pestrovskii | `research/e.pestrovskii` | — |

## 🛠 Tools

- **Python 3.12** — base language
- **uv** — package management & dependency resolution
- **ruff** — linting & formatting
- **mypy** — static type checking
- **pytest** — testing
- **Docker** — reproducible environments (CUDA 12.4)
- **MLflow** — experiment tracking

## 📋 Experiment Runner

The project uses a Strategy-pattern experiment framework: each experiment is an
independent class that overrides `build_*` methods and `on_*` hooks.

### Built-in Experiments (Ablation Suite)

11 experiments reproducing the original flag-based suite:

```bash
uv run python src/main.py --list              # list all 11 experiments
uv run python src/main.py --smoke             # run smoke test (1 sample, tiny models)
uv run python src/main.py --suite ablation    # run all 11 ablation experiments
uv run python src/main.py --experiment 01_baseline   # run one experiment
uv run python src/main.py --research           # run all research experiments
uv run python src/main.py --list --research    # list research experiments only
```

### Architecture

| File | Purpose |
|------|---------|
| `src/experiments/base.py` | `BaseExperiment` ABC, `BuildContext`, `DecodeContext` |
| `src/experiments/runner.py` | `ExperimentRunner` orchestrator, `ExperimentConfig` |
| `src/experiments/suites.py` | `ABLATION_SUITE`, `discover_experiments()` |
| `src/experiments/built_in/` | 10 built-in experiment classes |
| `src/experiments/templates/` | `minimal_template.py` for researchers |

## 🐳 Docker

```bash
# Build
docker compose build

# Run (with GPU)
docker compose run --rm app python src/main.py --smoke

# Start MLflow tracking
docker compose up -d mlflow
# → http://localhost:5000
```

## 🔗 References

- **Inspired by OmniDraft**: [odspd](odspd/) — legacy codebase for reference
- **Paper**: _Adaptive Speculative Decoding_ (TBD)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
