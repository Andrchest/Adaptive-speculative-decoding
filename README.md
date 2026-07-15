# Adaptive Speculative Decoding

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Adaptive Speculative Decoding** for LLM inference — research framework for optimizing generation speed through intelligent draft selection, cross-vocabulary translation, and online distillation.

## 🚀 Quick Start

```bash
# 0. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# → installs to ~/.local/bin/uv (no sudo required)
# → add to PATH: export PATH="$HOME/.local/bin:$PATH"

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

## 🧩 Available Models

### Drafter Models (small, fast)

| Model | Params | Speed | Notes |
|-------|--------|-------|-------|
| `EleutherAI/pythia-70m` | 70M | ⚡⚡⚡ | Very fast, good for smoke tests |
| `JackFram/llama-68m` | 68M | ⚡⚡⚡ | Very fast, needs `sentencepiece` |
| `facebook/opt-125m` | 125M | ⚡⚡ | Used in LoRA distillation experiments |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | ⚡ | Default drafter |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ⚡⚡ | Used in ExperimentConfig defaults |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ⚡ | Multi-drafter routing |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | — | Larger drafter option |

### Target Models (large, accurate)

| Model | Params | Notes |
|-------|--------|-------|
| `facebook/opt-350m` | 350M | Used in LoRA distillation experiments |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Default target in ExperimentConfig |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Larger target option |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | Largest target option |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Default in `default.yaml` |

### CLI Flags for Model Selection

```bash
# Override drafter model
uv run python src/main.py --drafter-model mistralai/Mistral-7B-v0.1

# Override target model
uv run python src/main.py --target-model meta-llama/Llama-3-8B

# Use tiny models (opt-125m / opt-350m) for fast testing
uv run python src/main.py --tiny

# Combine with other flags
uv run python src/main.py --tiny -n 5 --log-level VERBOSE
```

### Quick Examples

```bash
# Fast smoke test with smallest models
uv run python src/main.py --smoke

# Run with specific drafter/target
uv run python src/main.py --drafter-model Qwen/Qwen2.5-0.5B-Instruct \
                          --target-model Qwen/Qwen2.5-7B-Instruct

# Run ablation suite with 10 samples
uv run python src/main.py --suite ablation -n 10
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
│   │   ├── profiling/        # Substep timer, torch profiler
│   │   └── extensions/       # Experimental modules
│   ├── experiments/          # Experiment runner & ablation suite
│   │   ├── base.py           # BaseExperiment (Strategy pattern)
│   │   ├── runner.py         # ExperimentRunner (orchestrator)
│   │   ├── suites.py         # ABLATION_SUITE, discovery
│   │   ├── built_in/         # 12 built-in experiments
│   │   └── templates/        # Copy-paste template for researchers
│   ├── benchmarks/           # Metrics collection
│   └── config/               # Configuration
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── extension_tests/      # Extension-specific tests
├── research/                 # Research areas per team member
├── scripts/                  # Standalone scripts (profiler, etc.)
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
uv run python src/main.py --research          # all research experiments
uv run python src/main.py --experiment my_idea  # single experiment
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

### Results and Metrics

Every experiment always produces:
- **`acceptance_rate`** — average acceptance rate across all prompts
- **`tokens_per_sec`** — overall throughput (total tokens / total time)
- **`gpu_mem_peak_gb`** — peak GPU memory (GB) during the run
- **`gpu_mem_mean_gb`** — average GPU memory (GB) during the run
- **`wall_time_total_s`** / **`wall_time_mean_s`** — total and average decode time

Conditional metrics (only when features are enabled):
- `wall_clock_speedup` — when a baseline is set
- `mean_kl_divergence`, `training_loss_mean/std` — when distillation is on
- `router` — when dynamic routing is used

Results are saved as JSON (`results/<name>.json`) and CSV (`results/comparison_table.csv`).

### Built-in Experiments (Ablation Suite)

12 experiments reproducing the original flag-based suite:

```bash
uv run python src/main.py --list              # list all 12 experiments
uv run python src/main.py --smoke             # run smoke test (1 sample, tiny models)
uv run python src/main.py --suite ablation    # run all 12 ablation experiments
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
| `src/experiments/built_in/` | 12 built-in experiment classes |
| `src/experiments/templates/` | `minimal_template.py` for researchers |

## 🔗 References

- **Inspired by OmniDraft**: [odspd](odspd/) — legacy codebase for reference
- **Paper**: _Adaptive Speculative Decoding_ (TBD)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
