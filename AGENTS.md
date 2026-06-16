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
│   └── extensions/      # Experimental modules
├── experiments/         # Experiment runner & ablation suite
├── benchmarks/          # Metrics collection
├── config/              # Configuration
├── utils/               # Shared utilities (logging, model loading)
├── inference/           # API (future)
└── main.py              # Entry point

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── extension_tests/     # Extension tests

research/                # Per-researcher work area
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

## 🗣 Communication

- Ask questions in PR comments
- Document decisions in `docs/`
- Research results → `research/<username>/`
