# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Project scaffolding with `pyproject.toml` (uv + hatchling)
- Pre-commit hooks (ruff, mypy, pre-commit-hooks)
- Research workspace for 6 team members
- Documentation skeleton (architecture, modules, research)
- GitHub templates (PR, issues, CODEOWNERS)
- CI pipeline (lint, type-check, test)
- AI agent guidelines (AGENTS.md)

### Changed
- **Experiment refactoring (Phase 0):** Introduced Strategy-pattern architecture
  for experiments (`BaseExperiment` ABC, `BuildContext`, `DecodeContext`,
  `ExperimentMeta`, `ExperimentResult`).  `ExperimentRunner` now supports
  both legacy ``ExperimentConfig`` lists and new ``BaseExperiment`` instances.
  Added `src/experiments/built_in/baseline.py` as proof-of-concept.
  Migrated ``_load_dataset`` to instance method with tokenizer variant.
  Added unit tests for all base classes (20 tests, all passing).
  Full plan: ``docs/plans/experiment-refactor-option-b.md``.
- **Experiment refactoring (Phase 1):** Migrated all 12 ablation experiments
  to individual ``BaseExperiment`` subclasses:
  ``LatticeExperiment``, ``TranslatorExperiment``, ``OnlineDistillExperiment``,
  ``ReplayExperiment`` (parameterized fifo/prioritized), ``ContrastiveExperiment``,
  ``SpeedupAdaptiveExperiment``, ``RoutingExperiment``, ``UniversalDrafterExperiment``,
  and ``FullSystemExperiment``.  Added ``experiments/suites.py`` with
  ``ABLATION_SUITE``, ``CACHE_SUITE``, ``DATASET_SUITE``, and
  ``discover_experiments()``.  Updated public API exports.
- **Experiment refactoring (Phase 2):** Removed legacy code from ``ExperimentRunner``:
  deleted ``_run_one()`` (~350 lines), ``_apply_lora()``, and hardcoded ``ABLATION_SUITE``.
  Runner reduced from ~946 to ~390 lines. ``__init__`` accepts only ``experiments``
  (no ``configs``). Moved ``_apply_lora()`` into ``OnlineDistillExperiment`` and
  ``FullSystemExperiment`` as static methods. Rewrote ``main.py`` to use the new
  architecture: ``--suite`` uses ``BaseExperiment`` instances, ``--list`` shows
  tags/descriptions, ``--smoke`` uses ``_SmokeTestExperiment``, CLI overrides
  (``--tiny``, ``--max-samples``) applied via ``set_config_override()``.
  Updated tests: ``test_experiment_base.py`` (new runner API + override tests),
  ``test_fixes.py`` (``_run_one`` → ``BaseExperiment.run()``).
- **Experiment refactoring (Phase 3):** Added researcher infrastructure:
  ``src/experiments/templates/minimal_template.py`` (copy-paste starting point),
  ``--research`` CLI flag for running/listing research experiments,
  ``discover_research_experiments()`` for scanning ``research/*/experiments/*.py``,
  robust ``_load_experiment_from_file()`` using ``importlib.util.SourceFileLoader``
  (no Python package required for research directories). Updated ``research/README.md``
  with comprehensive guidelines: creating experiments, inheriting built-ins,
  build methods, hooks, custom metrics, and CLI usage.
- **Experiment refactoring (Phase 4):** Added ``tests/unit/test_experiment_suites.py``
  with 56 tests covering all built-in experiment classes (import, instantiation,
  config validation, metadata), inheritance hierarchy, ``ABLATION_SUITE`` composition,
  discovery functions, and template package.
- **Experiment refactoring (Phase 5):** Updated ``AGENTS.md`` with experiment
  architecture section (Strategy pattern, CLI reference, key classes). Verified
  ``ruff check`` and ``ruff format`` clean across all experiment modules.
- **Cleanup:** Removed Dockerfile, docker-compose.yml, docs/, reports/, and stale
  plan files. Moved ``src/profiler.py`` to ``scripts/profiler.py``. Moved
  ``src/tests/unit/test_sub_optimal_fixes.py`` to ``tests/unit/``. Formatted all
  files with ``ruff format``. Updated AGENTS.md, README.md, CHANGELOG.md to match
  actual codebase state.
