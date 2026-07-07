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
        self.calls = 0

    def __call__(self, context: torch.Tensor, **kwargs) -> _FakeOutput:
        del kwargs
        self.calls += 1
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
    controller = module.LatentRegimeK(_ScriptedDrafter(), k_min=1, k_max=4, use_token_class=True)

    token_class = controller._token_class(torch.tensor([[3]], dtype=torch.long))

    assert token_class == pytest.approx(1.0)


def test_latent_regime_avoids_drafter_forward_for_out_of_vocab_context() -> None:
    module = _load_research_module()
    drafter = _ScriptedDrafter()
    controller = module.LatentRegimeK(drafter, k_min=1, k_max=4)

    selected = controller(torch.tensor([[0, 99]], dtype=torch.long))

    assert 1 <= selected <= 4
    assert drafter.model.calls == 0
    assert controller._pending_entropy == pytest.approx(0.45)
    assert controller._pending_token_class == pytest.approx(0.1)


def test_latent_regime_entropy_probe_checks_vocab_when_enabled() -> None:
    module = _load_research_module()
    drafter = _ScriptedDrafter()
    controller = module.LatentRegimeK(drafter, k_min=1, k_max=4, use_drafter_entropy=True)

    entropy = controller._draft_entropy(torch.tensor([[0, 99]], dtype=torch.long))

    assert entropy == pytest.approx(0.45)
    assert drafter.model.calls == 0


def test_latent_regime_sampler_does_not_clamp_overflow_to_k_max() -> None:
    module = _load_research_module()
    controller = module.LatentRegimeK(_ScriptedDrafter(), k_min=1, k_max=10, seed=123)

    samples = [controller._sample_truncated_poisson(10.0) for _ in range(500)]
    max_share = samples.count(10) / len(samples)

    assert min(samples) >= 1
    assert max(samples) <= 10
    assert max_share < 0.4


def test_categorical_latent_regime_samples_from_discrete_support() -> None:
    module = _load_research_module()
    controller = module.CategoricalLatentRegimeK(
        _ScriptedDrafter(),
        k_min=1,
        k_max=10,
        seed=321,
    )

    samples = [controller(torch.tensor([[0, 1, 2]], dtype=torch.long)) for _ in range(200)]
    summary = controller.summary()

    assert min(samples) >= 1
    assert max(samples) <= 10
    assert len(set(samples)) > 2
    assert summary["regime_k_selector"] == "categorical"
    assert summary["regime_k_categorical_temperature"] == pytest.approx(0.9)


def test_latent_regime_lambda_update_is_less_conservative() -> None:
    module = _load_research_module()
    controller = module.LatentRegimeK(
        _ScriptedDrafter(),
        k_min=1,
        k_max=8,
        lambdas=(5.0, 4.0, 3.0, 2.0),
        lambda_lr=0.1,
        shrink_lr_scale=0.25,
        target_acceptance=0.55,
    )
    controller._last_regime_idx = 1

    controller.observe_step(StepResult(draft_length=4, accepted_count=4, rejected_at=-1))

    grown_lambda = float(controller.lambdas[1].item())
    assert grown_lambda > 4.0

    controller.observe_step(StepResult(draft_length=8, accepted_count=0, rejected_at=0))

    shrunk_lambda = float(controller.lambdas[1].item())
    assert shrunk_lambda < grown_lambda
    assert shrunk_lambda > 2.0


def test_invalid_dynamic_k_parameters_raise() -> None:
    module = _load_research_module()

    with pytest.raises(ValueError, match="k_max"):
        module.LatentRegimeK(_ScriptedDrafter(), k_min=4, k_max=1)
    with pytest.raises(ValueError, match="lambdas"):
        module.LatentRegimeK(_ScriptedDrafter(), lambdas=(1.0, 2.0))
    with pytest.raises(ValueError, match="regime_lambda_floor"):
        module.LatentRegimeK(_ScriptedDrafter(), regime_lambda_floor=(1.0, 2.0))
    with pytest.raises(ValueError, match="initial_posterior"):
        module.LatentRegimeK(_ScriptedDrafter(), initial_posterior=(1.0, 0.0))
    with pytest.raises(ValueError, match="transition_stay_prob"):
        module.LatentRegimeK(_ScriptedDrafter(), transition_stay_prob=1.1)
    with pytest.raises(ValueError, match="categorical_temperature"):
        module.CategoricalLatentRegimeK(_ScriptedDrafter(), categorical_temperature=0.0)


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
        module.LatentRegimeKExperiment(),
        module.LatentRegimeCategoricalKExperiment(),
    ]

    for experiment in experiments:
        cfg = experiment.get_config()
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.use_speedup_adaptive is True
        assert cfg.k_max == 10
        assert "v.poponnikov" in experiment.meta.tags


def test_dynamic_k_comparison_csv_keeps_research_metrics(tmp_path) -> None:
    module = _load_comparison_module()
    assert "stochastic_consensus_k" not in module.DEFAULT_EXPERIMENTS
    assert "latent_regime_categorical_k" in module.DEFAULT_EXPERIMENTS
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
        {
            "config": {"name": "latent_regime_categorical_k"},
            "metrics": {
                "tokens_per_sec": 11.0,
                "regime_k_selector": "categorical",
                "regime_k_k_distribution": {4: 1, 6: 3},
            },
        },
    ]
    path = tmp_path / "dynamic_k_comparison.csv"

    module.write_union_csv(results, path)

    text = path.read_text(encoding="utf-8")
    assert "tokens_per_sec" in text
    assert "regime_k_mean_selected_k" in text
    assert "latent_regime_categorical_k" in text
    assert '"{""1"": 2, ""3"": 4}"' in text


def test_model_matrix_helpers_build_pair_outputs(tmp_path) -> None:
    module = _load_comparison_module()
    args = type(
        "Args",
        (),
        {
            "draft_sizes": ["70m"],
            "target_sizes": ["1.5b"],
            "include_large_targets": False,
        },
    )()

    pairs = module.build_model_pairs(args)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.slug == "70m-1_5b"
    assert pair.drafter.path == "EleutherAI/pythia-70m"
    assert pair.target.path == "Qwen/Qwen2.5-1.5B-Instruct"

    results = [
        {
            "config": {"name": "latent_regime_k"},
            "metrics": {"tokens_per_sec": 12.0, "regime_k_mean_selected_k": 3.5},
        }
    ]
    path = tmp_path / "model_matrix_metrics.csv"

    module.write_matrix_csv([(pair, results)], path)

    text = path.read_text(encoding="utf-8")
    assert "70m-1_5b" in text
    assert "EleutherAI/pythia-70m" in text
    assert "Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "regime_k_mean_selected_k" in text


def test_comparison_select_experiments_reuses_cached_prototypes() -> None:
    module = _load_comparison_module()

    class _FakeExperiment:
        def __init__(self) -> None:
            self.meta = type("Meta", (), {"name": "latent_regime_k"})()

    previous = module._EXPERIMENT_PROTOTYPES
    module._EXPERIMENT_PROTOTYPES = {"latent_regime_k": _FakeExperiment()}
    try:
        first = module.select_experiments(["latent_regime_k"])
        second = module.select_experiments(["latent_regime_k"])
    finally:
        module._EXPERIMENT_PROTOTYPES = previous

    assert first[0] is not second[0]
    assert first[0].meta.name == "latent_regime_k"
    assert second[0].meta.name == "latent_regime_k"
