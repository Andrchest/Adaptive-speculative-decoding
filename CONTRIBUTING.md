# Contributing to Adaptive Speculative Decoding

## 📋 For Human Contributors

### Development Workflow

```bash
# 1. Fork → Clone → Install
git clone git@github.com:<your-username>/Adaptive-speculative-decoding.git
cd Adaptive-speculative-decoding
uv sync --all-extras

# 2. Pre-commit hooks (auto-installed)
pre-commit install

# 3. Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature

# 4. Work... then commit with conventional messages
git add .
git commit -m "feat: add speculative decoding optimization"

# 5. Push & open PR
git push -u origin feature/your-feature
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

types:
  feat:     new feature
  fix:      bug fix
  docs:     documentation
  refactor: code restructuring (no feature change)
  test:     add/modify tests
  chore:    tooling, config, maintenance
  research: research-specific changes (experiments, notebooks)
```

### Code Style

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` — all errors must be fixed
- **Types**: `mypy` — all public APIs must be typed
- **Tests**: `pytest` — all new features need tests

### Creating Research Experiments

The experiment framework uses a **Strategy pattern** — each experiment is a
`BaseExperiment` subclass in `research/<username>/experiments/`.

**Steps:**

1. `mkdir -p research/<your_name>/experiments`
2. `cp src/experiments/templates/minimal_template.py research/<your_name>/experiments/my_idea.py`
3. Edit: override `get_config()`, optional `build_*`/`on_*` methods
4. Register your class in `__all__` at the bottom of the file
5. Run: `python src/main.py --research`

See `src/experiments/templates/minimal_template.py` and `research/README.md` for full details.

### Pull Requests

1. Branch from `main`
2. CI must pass (lint + test + type-check)
3. At least one review from team member
4. Squash merge to `main`

## 🤖 For AI Agents

If you're an AI coding agent working on this project:

1. **Read these files first**:
   - `AGENTS.md` — project-specific AI instructions
   - `pyproject.toml` — dependencies and config
   - `ruff.toml` — linting rules
   - `src/` — code structure

2. **Before making changes**:
   - Understand the current code
   - Check if a similar pattern exists
   - Follow existing conventions

3. **After making changes**:
   - Run `ruff check . && ruff format .`
   - Run `mypy src/`
   - Run `pytest` (at least `--co -q` to verify tests exist)
   - Update tests if needed

4. **Never**:
   - Remove type annotations
   - Break existing tests
   - Hard-code configuration values
   - Modify `pyproject.toml` without consensus
