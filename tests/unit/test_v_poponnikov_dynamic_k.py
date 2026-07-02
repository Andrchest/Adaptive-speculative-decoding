"""Tests for v.poponnikov stochastic dynamic-k research controller."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

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


def _load_comparison_module():
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "research" / "v.poponnikov" / "notebooks" / "dynamic_k_comparison.py"
    spec = importlib.util.spec_from_file_location("v_poponnikov_dynamic_k_comparison", path)
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
        self.tokenizer = type("Tokenizer", (), {"decode": lambda _self, ids: "\n"})()
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


def test_latent_regime_uses_tokenizer_for_token_class() -> None:
    module = _load_research_module()
    controller = module.LatentRegimeK(_ScriptedDrafter(), k_min=1, k_max=4)

    token_class = controller._token_class(torch.tensor([[3]], dtype=torch.long))

    assert token_class == pytest.approx(1.0)


def test_invalid_dynamic_k_parameters_raise() -> None:
    module = _load_research_module()

    with pytest.raises(ValueError, match="k_max"):
        module.LatentRegimeK(_ScriptedDrafter(), k_min=4, k_max=1)
    with pytest.raises(ValueError, match="lambdas"):
        module.LatentRegimeK(_ScriptedDrafter(), lambdas=(1.0, 2.0))
    with pytest.raises(ValueError, match="transition_stay_prob"):
        module.LatentRegimeK(_ScriptedDrafter(), transition_stay_prob=1.1)


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
    experiments = [module.LatentRegimeKExperiment()]

    for experiment in experiments:
        cfg = experiment.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.use_speedup_adaptive is True
        assert "v.poponnikov" in experiment.meta.tags


def test_dynamic_k_comparison_csv_keeps_research_metrics(tmp_path) -> None:
    module = _load_comparison_module()
    results = [
        {
            "config": {"name": "01_baseline"},
            "metrics": {"tokens_per_sec": 10.0, "acceptance_rate": 0.5},
        },
        {
            "config": {"name": "latent_regime_k"},
            "metrics": {
                "tokens_per_sec": 12.0,
                "regime_k_mean_selected_k": 3.5,
                "regime_k_k_distribution": {1: 2, 3: 4},
            },
        },
    ]
    path = tmp_path / "dynamic_k_comparison.csv"

    module.write_union_csv(results, path)

    text = path.read_text(encoding="utf-8")
    assert "tokens_per_sec" in text
    assert "regime_k_mean_selected_k" in text
    assert '"{""1"": 2, ""3"": 4}"' in text
