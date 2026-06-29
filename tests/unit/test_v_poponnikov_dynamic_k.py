"""Tests for v.poponnikov stochastic dynamic-k research controllers."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, "src")

from core.decoder.speculative import SpeculativeDecoder, StepResult
from experiments.runner import ExperimentConfig


def _load_research_module():
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "research" / "v.poponnikov" / "experiments" / "stochastic_dynamic_k.py"
    spec = importlib.util.spec_from_file_location("v_poponnikov_stochastic_dynamic_k", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    def __init__(self, vocab_size: int = 8) -> None:
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": 4})()
        self.vocab_size = vocab_size

    def __call__(self, context: torch.Tensor, **kwargs) -> _FakeOutput:
        del kwargs
        logits = torch.zeros(1, context.shape[1], self.vocab_size)
        last_id = int(context.reshape(-1)[-1].item())
        logits[0, -1, (last_id + 1) % self.vocab_size] = 2.0
        return _FakeOutput(logits)


class _ScriptedDrafter:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.calls = 0
        self.rows = [
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 1, 3],
            [1, 2, 2, 3],
        ]

    def draft(
        self,
        context: torch.Tensor,
        k: int,
        distill: bool = False,
        temperature: float = 1.0,
    ) -> tuple[list[int], torch.Tensor]:
        del context, distill, temperature
        row = self.rows[self.calls % len(self.rows)][:k]
        self.calls += 1
        logits = torch.zeros(k, self.model.vocab_size)
        for pos, token in enumerate(row):
            logits[pos, token] = 3.0
            logits[pos, (token + 1) % self.model.vocab_size] = 1.0
        return row, logits


def test_epistemic_consensus_selects_bounded_k_and_tracks_consensus() -> None:
    module = _load_research_module()
    controller = module.EpistemicConsensusK(
        _ScriptedDrafter(),
        k_min=1,
        k_max=4,
        n_trajectories=3,
        theta=0.05,
        seed=1,
    )

    selected = controller(torch.tensor([[0, 1, 2]], dtype=torch.long))

    assert 1 <= selected <= 4
    assert controller.selections
    assert controller.selections[-1].consensus[:2] == pytest.approx([1.0, 1.0])


def test_epistemic_consensus_threshold_reacts_to_acceptance() -> None:
    module = _load_research_module()
    controller = module.EpistemicConsensusK(
        _ScriptedDrafter(),
        k_min=1,
        k_max=4,
        n_trajectories=3,
        target_acceptance=0.8,
        adaptation_rate=0.1,
        theta=0.5,
        seed=2,
    )
    initial_theta = controller.theta

    controller.observe_step(StepResult(draft_length=4, accepted_count=1, rejected_at=1))

    assert controller.theta > initial_theta
    assert controller.summary()["consensus_k_mean_step_acceptance"] == pytest.approx(0.25)


def test_latent_regime_updates_posterior_and_lambdas() -> None:
    module = _load_research_module()
    controller = module.LatentRegimeK(
        _ScriptedDrafter(),
        k_min=1,
        k_max=4,
        lambdas=(4.0, 3.0, 2.0, 1.0),
        seed=3,
    )

    selected = controller(torch.tensor([[0, 1, 2]], dtype=torch.long))
    controller.observe_step(StepResult(draft_length=selected, accepted_count=0, rejected_at=0))

    assert 1 <= selected <= 4
    assert float(controller.posterior.sum().item()) == pytest.approx(1.0)
    assert controller.posterior_entropy_history
    assert controller.summary()["regime_k_mean_selected_k"] >= 1


def test_decoder_reports_step_result_to_adaptive_observer() -> None:
    class Observer:
        def __init__(self) -> None:
            self.accepted: list[int] = []

        def observe_step(self, result: StepResult) -> None:
            self.accepted.append(result.accepted_count)

    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    observer = Observer()

    decoder._notify_adaptive_result(
        observer,
        StepResult(draft_length=3, accepted_count=2, rejected_at=2),
    )

    assert observer.accepted == [2]


def test_research_experiment_configs_are_valid() -> None:
    module = _load_research_module()
    experiments = [
        module.StochasticConsensusKExperiment(),
        module.LatentRegimeKExperiment(),
    ]

    for experiment in experiments:
        cfg = experiment.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.use_speedup_adaptive is True
        assert "v.poponnikov" in experiment.meta.tags
