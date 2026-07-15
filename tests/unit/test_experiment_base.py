"""Unit tests for the new experiment base classes.

These tests verify the Strategy-pattern architecture without requiring
GPU models.  Full equivalence tests (same metrics as legacy runner)
require a GPU and are marked with ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is on the path (mirrors existing test pattern)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from experiments.base import (
    BaseExperiment,
    BuildContext,
    DecodeContext,
    ExperimentMeta,
    ExperimentResult,
)
from experiments.runner import ExperimentConfig, ExperimentRunner

# ---------------------------------------------------------------------------
# ExperimentMeta
# ---------------------------------------------------------------------------


class TestExperimentMeta:
    def test_defaults(self) -> None:
        meta = ExperimentMeta(name="test")
        assert meta.name == "test"
        assert meta.description == ""
        assert meta.tags == []
        assert meta.dimensions == []
        assert meta.depends_on == []

    def test_full(self) -> None:
        meta = ExperimentMeta(
            name="my_exp",
            description="A test experiment",
            tags=["test", "unit"],
            dimensions=["translation"],
            depends_on=["baseline"],
        )
        assert meta.name == "my_exp"
        assert meta.tags == ["test", "unit"]


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_success(self) -> None:
        result = ExperimentResult(
            meta=ExperimentMeta(name="ok"),
            config={"name": "ok"},
            metrics={"acceptance_rate": 0.8},
        )
        assert result.success is True
        assert result.error is None

    def test_failure(self) -> None:
        result = ExperimentResult(
            meta=ExperimentMeta(name="fail"),
            config={"name": "fail"},
            metrics={},
            error="OOM",
        )
        assert result.success is False
        assert result.error == "OOM"


# ---------------------------------------------------------------------------
# BaseExperiment ABC
# ---------------------------------------------------------------------------


class TestBaseExperimentABC:
    def test_abstract_get_config(self) -> None:
        """Subclass without get_config() cannot be instantiated (ABC enforcement)."""

        class IncompleteExperiment(BaseExperiment):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteExperiment()

    def test_default_meta_from_class_name(self) -> None:
        """When meta is None, use class name."""

        class NamedExperiment(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="named")

        exp = NamedExperiment()
        # meta defaults to class name only when __init__ is called without meta
        # But our NamedExperiment calls super().__init__() which auto-generates
        assert exp.meta.name == "NamedExperiment"

    def test_explicit_meta(self) -> None:
        """Explicit meta overrides class name."""

        class MyExperiment(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="my")

        exp = MyExperiment(ExperimentMeta(name="custom_name", tags=["test"]))
        assert exp.meta.name == "custom_name"
        assert exp.meta.tags == ["test"]

    def test_default_build_methods_return_sensible_values(self) -> None:
        """Default build_* methods should not raise."""

        class MinimalExperiment(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="minimal")

        exp = MinimalExperiment()
        # These should not raise (they return defaults)
        assert exp.build_distiller(None) is None  # type: ignore[arg-type]
        assert exp.build_adaptive_controller(None) is None  # type: ignore[arg-type]
        assert exp.build_router(None) is None  # type: ignore[arg-type]
        assert exp.build_universal_drafter(None) is None  # type: ignore[arg-type]

    def test_default_hooks_no_op(self) -> None:
        """Default on_* hooks should be no-ops."""

        class MinimalExperiment(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="minimal")

        exp = MinimalExperiment()
        # Should not raise
        exp.on_before_decode(None)  # type: ignore[arg-type]
        exp.on_decode_step(None, None, 0)  # type: ignore[arg-type]
        exp.on_after_decode(None)  # type: ignore[arg-type]

    def test_on_extra_metrics_passthrough(self) -> None:
        """Default on_extra_metrics returns summary unchanged."""

        class MinimalExperiment(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="minimal")

        exp = MinimalExperiment()
        summary = {"acceptance_rate": 0.8, "tps": 50.0}
        result = exp.on_extra_metrics(summary)
        assert result == summary


# ---------------------------------------------------------------------------
# BuildContext / DecodeContext
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_defaults(self) -> None:
        ctx = BuildContext(
            device="cpu",
            drafter=None,  # type: ignore[arg-type]
            target=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
        )
        assert ctx.device == "cpu"
        assert ctx.components == {}

    def test_components_storage(self) -> None:
        ctx = BuildContext(
            device="cpu",
            drafter=None,  # type: ignore[arg-type]
            target=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
        )
        ctx.components["translator"] = "mock_translator"
        assert ctx.components["translator"] == "mock_translator"


class TestDecodeContext:
    def test_defaults(self) -> None:
        ctx = DecodeContext(
            decoder=None,  # type: ignore[arg-type]
            collector=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
        )
        assert ctx.distiller is None
        assert ctx.router is None
        assert ctx.adaptive_fn is None
        assert ctx.extra_state == {}

    def test_extra_state(self) -> None:
        ctx = DecodeContext(
            decoder=None,  # type: ignore[arg-type]
            collector=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
        )
        ctx.extra_state["counter"] = 42
        assert ctx.extra_state["counter"] == 42


# ---------------------------------------------------------------------------
# ExperimentRunner initialization
# ---------------------------------------------------------------------------


class TestExperimentRunnerInit:
    def test_experiments(self) -> None:
        class DummyExp(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="dummy")

        runner = ExperimentRunner(
            experiments=[DummyExp()],
            output_dir="/tmp/test_exp",
        )
        assert len(runner.experiments) == 1

    def test_empty_experiments(self) -> None:
        runner = ExperimentRunner(experiments=[], output_dir="/tmp/test_exp")
        assert len(runner.experiments) == 0

    def test_defaults(self) -> None:
        runner = ExperimentRunner(output_dir="/tmp/test_exp")
        assert runner.experiments == []
        assert runner.device == "cuda"
        assert runner.output_dir == "/tmp/test_exp"


class TestConfigOverrides:
    def test_set_config_override(self) -> None:
        class DummyExp(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="dummy", max_samples=500)

        exp = DummyExp()
        exp.set_config_override("max_samples", 10)
        # Overrides are stored but applied only during run()
        assert exp._overrides["max_samples"] == 10

    def test_multiple_overrides(self) -> None:
        class DummyExp(BaseExperiment):
            def get_config(self) -> ExperimentConfig:
                return ExperimentConfig(name="dummy")

        exp = DummyExp()
        exp.set_config_override("max_samples", 5)
        exp.set_config_override("max_new_tokens", 32)
        exp.set_config_override("drafter_model_path", "tiny")
        assert len(exp._overrides) == 3


# ---------------------------------------------------------------------------
# BaselineExperiment (built-in)
# ---------------------------------------------------------------------------


class TestBaselineExperiment:
    def test_import_and_instantiate(self) -> None:
        from experiments.built_in import BaselineExperiment

        exp = BaselineExperiment()
        assert exp.meta.name == "01_baseline"
        assert "baseline" in exp.meta.tags

    def test_get_config(self) -> None:
        from experiments.built_in import BaselineExperiment

        exp = BaselineExperiment()
        cfg = exp.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "01_baseline"
        assert cfg.use_rule1 is True
        assert cfg.use_rule2 is True
        assert cfg.use_lattice is False
        assert cfg.use_online_distil is False


# ---------------------------------------------------------------------------
# GPU memory and time metrics (always present in summary)
# ---------------------------------------------------------------------------


class TestAlwaysPresentMetrics:
    """Verify that GPU memory and time metrics are always in summary."""

    def test_gpu_mem_peak_and_mean(self) -> None:
        from benchmarks.metrics.collector import BenchmarkCollector

        c = BenchmarkCollector("metrics_test")
        c._gpu_mem_samples = [1.0, 2.5, 3.0, 2.0]

        with c.record_sequence(prompt_len=10) as rec:
            rec.add_step(draft_len=5, accepted=3)

        summary = c.summary()
        assert summary["gpu_mem_peak_gb"] == 3.0
        assert summary["gpu_mem_mean_gb"] == pytest.approx(2.125)

    def test_gpu_mem_empty(self) -> None:
        from benchmarks.metrics.collector import BenchmarkCollector

        c = BenchmarkCollector("empty_test")
        with c.record_sequence(prompt_len=10) as rec:
            rec.add_step(draft_len=5, accepted=3)

        summary = c.summary()
        assert summary["gpu_mem_peak_gb"] == 0.0
        assert summary["gpu_mem_mean_gb"] == 0.0

    def test_time_metrics_always_present(self) -> None:
        from benchmarks.metrics.collector import BenchmarkCollector

        c = BenchmarkCollector("time_test")
        with c.record_sequence(prompt_len=10) as rec:
            rec.add_step(draft_len=5, accepted=3)

        summary = c.summary()
        assert "wall_time_total_s" in summary
        assert "wall_time_mean_s" in summary
        assert summary["wall_time_total_s"] > 0
        assert summary["wall_time_mean_s"] > 0

    def test_all_core_metrics_present(self) -> None:
        """Every experiment run must always produce these metrics."""
        from benchmarks.metrics.collector import BenchmarkCollector

        c = BenchmarkCollector("core_test")
        c._gpu_mem_samples = [0.5, 1.5]
        with c.record_sequence(prompt_len=10) as rec:
            rec.add_step(draft_len=5, accepted=3)

        summary = c.summary()
        required = [
            "acceptance_rate",
            "gpu_mem_peak_gb",
            "gpu_mem_mean_gb",
            "wall_time_total_s",
            "wall_time_mean_s",
            "tokens_per_sec",
        ]
        for key in required:
            assert key in summary, f"Missing required metric: {key}"

    def test_get_config(self) -> None:
        from experiments.built_in import BaselineExperiment

        exp = BaselineExperiment()
        cfg = exp.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "01_baseline"
        assert cfg.use_rule1 is True
        assert cfg.use_rule2 is True
        assert cfg.use_lattice is False
        assert cfg.use_online_distil is False
