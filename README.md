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
│   │   ├── core/             # Speculative decoder, drafters
│   │   ├── translation/      # Cross-vocab translation (Rule1, Rule2)
│   │   ├── cache/            # N-gram cache with eviction strategies
│   │   ├── distillation/     # Online distillation
│   │   └── extensions/       # Experimental modules
│   ├── experiments/          # Experiment runner & ablation suite
│   ├── benchmarks/           # Metrics collection
│   ├── config/               # Configuration management
│   └── inference/            # Production inference (future)
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── extension_tests/      # Extension-specific tests
├── research/                 # Research areas per team member
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks for analysis
├── scripts/                  # Utility scripts
└── pyproject.toml            # Project configuration (uv + hatchling)
```

## 🔬 Research Tracks

Each team member works on their own research track (separate branch):

| Member | Branch | Direction |
|--------|--------|-----------|
| m.krylov (Михаил Крылов) | `research/m.krylov` | — |
| v.poponnikov (Вадим Попонников) | `research/v.poponnikov` | — |
| a.polevoi (Андрей Полевой) | `research/a.polevoi` | — |
| al.khadeeva (Алия Хадеева) | `research/al.khadeeva` | — |
| da.popov (Данил Попов) | `research/da.popov` | — |
| e.pestrovskii (Евгений Пестровский) | `research/e.pestrovskii` | — |

## 🛠 Tools

- **Python 3.12** — base language
- **uv** — package management & dependency resolution
- **ruff** — linting & formatting
- **mypy** — static type checking
- **pytest** — testing
- **Docker** — reproducible environments (CUDA 12.4)
- **MLflow** — experiment tracking

## 📋 Ablation Suite

The project includes a configurable ablation suite for comparing system components:

```bash
uv run python src/main.py --list          # list all experiments
uv run python src/main.py --smoke         # run smoke test
uv run python src/main.py --suite ablation  # run full suite
```

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
