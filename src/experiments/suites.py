"""Pre-defined experiment suites.

Provides ready-to-use lists of experiments for common benchmarks:
ablation studies, cache eviction comparisons, and dataset sweeps.

Usage
-----
>>> from experiments.suites import ABLATION_SUITE
>>> from experiments import ExperimentRunner
>>> runner = ExperimentRunner(experiments=ABLATION_SUITE)
>>> results = runner.run_all()
"""

from __future__ import annotations

from pathlib import Path

from experiments.built_in import (
    AcceptanceAdaptiveExperiment,
    BaselineExperiment,
    ContrastiveExperiment,
    FullSystemExperiment,
    LatticeExperiment,
    OnlineDistillExperiment,
    ReplayExperiment,
    RoutingExperiment,
    SpeedupAdaptiveExperiment,
    TranslatorExperiment,
    UniversalDrafterExperiment,
)

# ---------------------------------------------------------------------------
# Ablation suite — reproduces the original 11 experiments
# ---------------------------------------------------------------------------

ABLATION_SUITE = [
    BaselineExperiment(),
    LatticeExperiment(),
    TranslatorExperiment(),
    OnlineDistillExperiment(),
    ReplayExperiment(strategy="fifo"),
    ReplayExperiment(strategy="prioritized"),
    ContrastiveExperiment(),
    SpeedupAdaptiveExperiment(),
    AcceptanceAdaptiveExperiment(),
    RoutingExperiment(),
    UniversalDrafterExperiment(),
    FullSystemExperiment(),
]

# ---------------------------------------------------------------------------
# Cache suite — vary eviction strategies (baseline config)
# ---------------------------------------------------------------------------

CACHE_SUITE = [
    BaselineExperiment(),  # default: hybrid eviction
]

# ---------------------------------------------------------------------------
# Dataset suite — run baseline on multiple datasets
# ---------------------------------------------------------------------------

DATASET_SUITE = [
    BaselineExperiment(),  # default: gsm8k
]

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _load_experiment_from_file(file_path: Path) -> list:
    """Load experiment classes from a single Python file using importlib.

    Uses ``spec_from_file_location`` so the research directory does not
    need to be a proper Python package.

    Parameters
    ----------
    file_path :
        Absolute path to a ``.py`` file containing experiment classes.

    Returns
    -------
    list
        Instantiated ``BaseExperiment`` subclasses found in ``__all__``.
    """
    import logging
    import sys
    from importlib.machinery import SourceFileLoader

    logger = logging.getLogger(__name__)
    base_dir = Path(__file__).resolve().parents[2]
    module_name = f"research_exp_{file_path.stem}_{file_path.parts[-3]}"
    if module_name in sys.modules:
        return []

    loader = SourceFileLoader(module_name, str(file_path))
    spec = loader.load_module()
    sys.modules[module_name] = spec

    results = []
    for attr_name in getattr(spec, "__all__", []):
        attr = getattr(spec, attr_name, None)
        if attr is None:
            continue
        if not isinstance(attr, type):
            continue
        # Must be a BaseExperiment subclass (has the key methods)
        if not hasattr(attr, "build_translator"):
            continue
        if not hasattr(attr, "get_config"):
            continue
        try:
            instance = attr()
            results.append(instance)
            logger.info(
                "Discovered research experiment: %s from %s",
                instance.meta.name,
                file_path.relative_to(base_dir),
            )
        except Exception as e:
            logger.warning(
                "Failed to instantiate %s from %s: %s",
                attr_name,
                file_path.relative_to(base_dir),
                e,
            )
    return results


def discover_experiments(include_research: bool = True) -> list:
    """Auto-discover experiments from built-in and research directories.

    Parameters
    ----------
    include_research :
        If ``True`` (default), also scan ``research/*/experiments/*.py``
        for experiment classes.  Set to ``False`` to return only
        built-in experiments.

    Returns
    -------
    list
        Instantiated ``BaseExperiment`` instances.
    """
    import logging

    logger = logging.getLogger(__name__)
    experiments = list(ABLATION_SUITE)

    if not include_research:
        return experiments

    # Research experiments: research/<username>/experiments/*.py
    research_dir = Path(__file__).resolve().parents[2] / "research"
    if not research_dir.exists():
        return experiments

    for exp_file in sorted(research_dir.rglob("experiments/*.py")):
        if exp_file.name.startswith("_"):
            continue
        try:
            experiments.extend(_load_experiment_from_file(exp_file))
        except Exception as e:
            logger.warning(
                "Failed to load research experiment from %s: %s",
                exp_file.relative_to(Path(__file__).resolve().parents[2]),
                e,
            )

    return experiments


def discover_research_experiments() -> list:
    """Return only research experiments (not built-in).

    Scans ``research/*/experiments/*.py`` for experiment classes.

    Returns
    -------
    list
        Instantiated ``BaseExperiment`` instances from research directories.
    """
    import logging

    logger = logging.getLogger(__name__)
    research_dir = Path(__file__).resolve().parents[2] / "research"
    if not research_dir.exists():
        return []

    experiments = []
    for exp_file in sorted(research_dir.rglob("experiments/*.py")):
        if exp_file.name.startswith("_"):
            continue
        try:
            experiments.extend(_load_experiment_from_file(exp_file))
        except Exception as e:
            logger.warning(
                "Failed to load research experiment from %s: %s",
                exp_file.relative_to(Path(__file__).resolve().parents[2]),
                e,
            )

    return experiments
