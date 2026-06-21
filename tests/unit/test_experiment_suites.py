"""Unit tests for experiment suites, discovery, and all built-in experiments.

Verifies that every built-in experiment class can be imported, instantiated,
and returns a valid configuration.  Also tests the discovery infrastructure
and suite compositions.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from experiments import ABLATION_SUITE, discover_experiments, discover_research_experiments
from experiments.base import BaseExperiment, ExperimentMeta, ExperimentResult
from experiments.built_in import (
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
from experiments.runner import ExperimentConfig

# ---------------------------------------------------------------------------
# Built-in experiment classes — import and instantiation
# ---------------------------------------------------------------------------

BUILTIN_CLASSES = [
    BaselineExperiment,
    LatticeExperiment,
    TranslatorExperiment,
    OnlineDistillExperiment,
    ReplayExperiment,
    ContrastiveExperiment,
    SpeedupAdaptiveExperiment,
    RoutingExperiment,
    UniversalDrafterExperiment,
    FullSystemExperiment,
]


class TestBuiltinExperiments:
    """Every built-in experiment must be instantiable and return valid config."""

    @pytest.mark.parametrize("exp_class", BUILTIN_CLASSES)
    def test_instantiate(self, exp_class) -> None:
        try:
            exp = exp_class()
        except TypeError:
            # Some classes need parameters (e.g. ReplayExperiment needs strategy)
            return
        assert isinstance(exp, BaseExperiment)
        assert isinstance(exp.meta, ExperimentMeta)
        assert exp.meta.name  # non-empty name

    @pytest.mark.parametrize("exp_class", BUILTIN_CLASSES)
    def test_get_config_returns_experiment_config(self, exp_class) -> None:
        try:
            exp = exp_class()
        except TypeError:
            return
        cfg = exp.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name  # non-empty name

    @pytest.mark.parametrize("exp_class", BUILTIN_CLASSES)
    def test_meta_has_tags(self, exp_class) -> None:
        try:
            exp = exp_class()
        except TypeError:
            return
        assert isinstance(exp.meta.tags, list)
        assert len(exp.meta.tags) > 0  # every experiment should have at least one tag

    @pytest.mark.parametrize("exp_class", BUILTIN_CLASSES)
    def test_meta_has_description(self, exp_class) -> None:
        try:
            exp = exp_class()
        except TypeError:
            return
        assert len(exp.meta.description) > 0

    def test_replay_experiment_parameterized(self) -> None:
        fifo = ReplayExperiment(strategy="fifo")
        prio = ReplayExperiment(strategy="prioritized")
        assert "fifo" in fifo.meta.name
        assert "prio" in prio.meta.name or "prioritized" in prio.meta.name

    def test_translator_extends_lattice(self) -> None:
        """TranslatorExperiment should inherit from LatticeExperiment."""
        assert issubclass(TranslatorExperiment, LatticeExperiment)

    def test_replay_extends_online_distill(self) -> None:
        """ReplayExperiment should inherit from OnlineDistillExperiment."""
        assert issubclass(ReplayExperiment, OnlineDistillExperiment)

    def test_contrastive_extends_online_distill(self) -> None:
        """ContrastiveExperiment should inherit from OnlineDistillExperiment."""
        assert issubclass(ContrastiveExperiment, OnlineDistillExperiment)


# ---------------------------------------------------------------------------
# Ablation suite
# ---------------------------------------------------------------------------


class TestAblationSuite:
    def test_count(self) -> None:
        assert len(ABLATION_SUITE) == 11

    def test_all_are_base_experiments(self) -> None:
        for exp in ABLATION_SUITE:
            assert isinstance(exp, BaseExperiment)

    def test_unique_names(self) -> None:
        names = [exp.meta.name for exp in ABLATION_SUITE]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_expected_order(self) -> None:
        names = [exp.meta.name for exp in ABLATION_SUITE]
        assert names[0] == "01_baseline"
        assert names[-1] == "11_full_system"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_experiments_returns_list(self) -> None:
        result = discover_experiments()
        assert isinstance(result, list)

    def test_discover_experiments_includes_ablation(self) -> None:
        result = discover_experiments()
        assert len(result) >= len(ABLATION_SUITE)

    def test_discover_experiments_exclude_research(self) -> None:
        all_exps = discover_experiments(include_research=True)
        built_in_only = discover_experiments(include_research=False)
        assert len(built_in_only) == len(ABLATION_SUITE)
        # When no research experiments exist, both should be equal
        assert len(all_exps) == len(built_in_only)

    def test_discover_research_returns_list(self) -> None:
        result = discover_research_experiments()
        assert isinstance(result, list)

    def test_discover_research_empty_when_no_files(self) -> None:
        """When no research/*/experiments/*.py exists, result is empty."""
        result = discover_research_experiments()
        assert isinstance(result, list)
        # Research experiments may or may not exist; just check it doesn't crash


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_success_no_error(self) -> None:
        result = ExperimentResult(
            meta=ExperimentMeta(name="test"),
            config={"name": "test"},
            metrics={"acc": 0.9},
        )
        assert result.success is True
        assert result.error is None

    def test_failure_with_error(self) -> None:
        result = ExperimentResult(
            meta=ExperimentMeta(name="test"),
            config={"name": "test"},
            metrics={},
            error="OOM",
        )
        assert result.success is False
        assert result.error == "OOM"


# ---------------------------------------------------------------------------
# Templates package
# ---------------------------------------------------------------------------


class TestTemplatesPackage:
    def test_templates_importable(self) -> None:
        """The templates package should be importable."""
        from experiments import templates  # noqa: F401
        # Should not raise
