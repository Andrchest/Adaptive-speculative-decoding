"""Research experiment for stochastic dynamic draft-length selection.

This module keeps the v.poponnikov research code local to the research area.
It defines the LatentRegimeK adaptive controller, a hidden-regime controller
with change-point resets.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig

if TYPE_CHECKING:
    from core.decoder.speculative import StepResult


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _validate_k_bounds(k_min: int, k_max: int) -> None:
    if k_min < 1:
        raise ValueError("k_min must be at least 1")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min")


@dataclass
class RegimeSelection:
    """Debug information for one LatentRegimeK decision."""

    selected_k: int
    regime: str
    change_point_prob: float
    lambda_t: float
    posterior: list[float]


@dataclass
class DynamicKStats:
    """Shared metric history for stochastic k controllers."""

    selected_k: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    acceptance_rate: list[float] = field(default_factory=list)
    rejected_at: list[int] = field(default_factory=list)

    def observe(self, result: StepResult) -> None:
        self.selected_k.append(result.draft_length)
        self.accepted.append(result.accepted_count)
        rate = result.accepted_count / max(result.draft_length, 1)
        self.acceptance_rate.append(rate)
        self.rejected_at.append(result.rejected_at)

    def summary(self, prefix: str) -> dict[str, object]:
        counts = Counter(self.selected_k)
        return {
            f"{prefix}_mean_selected_k": _mean([float(k) for k in self.selected_k]),
            f"{prefix}_min_selected_k": min(self.selected_k, default=0),
            f"{prefix}_max_selected_k": max(self.selected_k, default=0),
            f"{prefix}_mean_step_acceptance": _mean(self.acceptance_rate),
            f"{prefix}_k_distribution": dict(sorted(counts.items())),
        }


class LatentRegimeK:
    """Choose draft length from an online hidden-regime model."""

    regime_names = ("easy", "normal", "hard", "transition")

    def __init__(
        self,
        drafter,
        *,
        k_min: int = 1,
        k_max: int = 10,
        lambdas: tuple[float, float, float, float] = (8.0, 6.0, 3.5, 2.0),
        lambda_min: float = 2.5,
        regime_lambda_floor: tuple[float, float, float, float] = (3.0, 2.0, 1.0, 1.0),
        initial_posterior: tuple[float, float, float, float] = (0.45, 0.4, 0.12, 0.03),
        transition_stay_prob: float = 0.9,
        reward_penalty: float = 0.5,
        lambda_lr: float = 0.05,
        shrink_lr_scale: float = 0.5,
        target_acceptance: float = 0.5,
        full_accept_bonus: float = 0.2,
        default_entropy: float = 0.45,
        unknown_token_class: float = 0.1,
        change_point_scale: float = 0.8,
        use_drafter_entropy: bool = False,
        use_token_class: bool = False,
        seed: int = 42,
    ) -> None:
        _validate_k_bounds(k_min, k_max)
        if len(lambdas) != len(self.regime_names):
            raise ValueError("lambdas must contain one value per regime")
        if len(regime_lambda_floor) != len(self.regime_names):
            raise ValueError("regime_lambda_floor must contain one value per regime")
        if len(initial_posterior) != len(self.regime_names):
            raise ValueError("initial_posterior must contain one value per regime")
        if any(value <= 0 for value in lambdas):
            raise ValueError("all regime lambdas must be positive")
        if lambda_min <= 0:
            raise ValueError("lambda_min must be positive")
        if any(value <= 0 for value in regime_lambda_floor):
            raise ValueError("all regime lambda floors must be positive")
        if any(value < 0 for value in initial_posterior):
            raise ValueError("initial_posterior values must be non-negative")
        if sum(initial_posterior) <= 0:
            raise ValueError("initial_posterior must have positive mass")
        if not 0.0 <= transition_stay_prob <= 1.0:
            raise ValueError("transition_stay_prob must be in [0, 1]")
        if not 0.0 <= target_acceptance <= 1.0:
            raise ValueError("target_acceptance must be in [0, 1]")
        if shrink_lr_scale <= 0:
            raise ValueError("shrink_lr_scale must be positive")
        if change_point_scale < 0:
            raise ValueError("change_point_scale must be non-negative")
        if not 0.0 <= default_entropy <= 1.0:
            raise ValueError("default_entropy must be in [0, 1]")
        if not 0.0 <= unknown_token_class <= 1.0:
            raise ValueError("unknown_token_class must be in [0, 1]")

        self.drafter = drafter
        self.k_min = k_min
        self.k_max = k_max
        self.lambdas = torch.tensor(lambdas, dtype=torch.float32)
        self.lambda_min = lambda_min
        self.regime_lambda_floor = torch.tensor(
            [min(value, k_max) for value in regime_lambda_floor],
            dtype=torch.float32,
        )
        self.transition = self._build_transition(transition_stay_prob)
        self.reward_penalty = reward_penalty
        self.lambda_lr = lambda_lr
        self.shrink_lr_scale = shrink_lr_scale
        self.target_acceptance = target_acceptance
        self.full_accept_bonus = full_accept_bonus
        self.default_entropy = default_entropy
        self.unknown_token_class = unknown_token_class
        self.change_point_scale = change_point_scale
        self.use_drafter_entropy = use_drafter_entropy
        self.use_token_class = use_token_class
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        self.posterior = torch.tensor(initial_posterior, dtype=torch.float32)
        self.posterior = self.posterior / self.posterior.sum()
        self.feature_mean = torch.tensor(
            [
                [0.9, 0.25, 0.1, 1.0, 0.0],
                [0.7, 0.45, 0.3, 0.6, 0.2],
                [0.45, 0.7, 0.55, 0.35, 0.5],
                [0.25, 0.8, 0.75, 0.1, 1.0],
            ],
            dtype=torch.float32,
        )
        self.feature_std = torch.tensor([0.22, 0.25, 0.25, 0.3, 0.4], dtype=torch.float32)

        self.stats = DynamicKStats()
        self.selections: list[RegimeSelection] = []
        self.posterior_entropy_history: list[float] = []
        self.change_point_history: list[float] = []
        self._pending_entropy = 0.0
        self._pending_token_class = 0.0
        self._last_regime_idx = 0
        self._last_k = k_min

    def __call__(self, context: torch.Tensor) -> int:
        self._pending_entropy = self._draft_entropy(context)
        self._pending_token_class = self._token_class(context)

        change_point_prob = (1.0 - float(self.posterior.max().item())) * self.change_point_scale
        regime_idx = int(self.posterior.argmax().item())
        lambda_regime = float(self.lambdas[regime_idx].item())
        lambda_t = (1.0 - change_point_prob) * lambda_regime
        lambda_t += change_point_prob * self.lambda_min
        selected_k = self._sample_truncated_poisson(lambda_t)

        self._last_regime_idx = regime_idx
        self._last_k = selected_k
        self.change_point_history.append(change_point_prob)
        self.selections.append(
            RegimeSelection(
                selected_k=selected_k,
                regime=self.regime_names[regime_idx],
                change_point_prob=change_point_prob,
                lambda_t=lambda_t,
                posterior=[float(v) for v in self.posterior.tolist()],
            )
        )
        return selected_k

    def observe_step(self, result: StepResult) -> None:
        """Update regime posterior and per-regime lambda after verification."""
        self.stats.observe(result)
        feature = self._build_feature(result)
        prior = self.transition.T @ self.posterior
        likelihood = self._gaussian_likelihood(feature)
        posterior = prior * likelihood
        self.posterior = posterior / posterior.sum().clamp(min=1e-8)
        self.posterior_entropy_history.append(self._posterior_entropy())

        old_lambda = float(self.lambdas[self._last_regime_idx].item())
        new_lambda = self._update_lambda_from_acceptance(old_lambda, result)
        floor = max(self.k_min, float(self.regime_lambda_floor[self._last_regime_idx].item()))
        self.lambdas[self._last_regime_idx] = _clamp(new_lambda, floor, self.k_max)

    def record_result(self, accepted_count: int) -> None:
        """Compatibility hook for controllers that only receive accepted count."""
        result = _StepProxy(self._last_k, accepted_count)
        self.observe_step(result)  # type: ignore[arg-type]

    def summary(self, prefix: str = "regime_k") -> dict[str, object]:
        metrics = self.stats.summary(prefix)
        counts = Counter(selection.regime for selection in self.selections)
        metrics.update(
            {
                f"{prefix}_posterior_entropy_mean": _mean(self.posterior_entropy_history),
                f"{prefix}_change_point_mean": _mean(self.change_point_history),
                f"{prefix}_lambda_easy": float(self.lambdas[0].item()),
                f"{prefix}_lambda_normal": float(self.lambdas[1].item()),
                f"{prefix}_lambda_hard": float(self.lambdas[2].item()),
                f"{prefix}_lambda_transition": float(self.lambdas[3].item()),
                f"{prefix}_regime_distribution": dict(sorted(counts.items())),
                f"{prefix}_selector": "bounded_poisson",
                f"{prefix}_target_acceptance": self.target_acceptance,
                f"{prefix}_default_entropy": self.default_entropy,
                f"{prefix}_change_point_scale": self.change_point_scale,
            }
        )
        return metrics

    def _build_feature(self, result: StepResult) -> torch.Tensor:
        acceptance = result.accepted_count / max(result.draft_length, 1)
        disagreement = 1.0 - acceptance
        if result.rejected_at < 0:
            reject_position = 1.0
        else:
            reject_position = result.rejected_at / max(result.draft_length, 1)
        return torch.tensor(
            [
                acceptance,
                self._pending_entropy,
                disagreement,
                reject_position,
                self._pending_token_class,
            ],
            dtype=torch.float32,
        )

    def _gaussian_likelihood(self, feature: torch.Tensor) -> torch.Tensor:
        z = (feature.unsqueeze(0) - self.feature_mean) / self.feature_std.clamp(min=1e-6)
        log_likelihood = -0.5 * (z**2).sum(dim=-1)
        log_likelihood = log_likelihood - log_likelihood.max()
        return torch.exp(log_likelihood).clamp(min=1e-8)

    def _sample_truncated_poisson(self, lambda_t: float) -> int:
        lam = max(lambda_t, 1e-3)
        support = torch.arange(self.k_min, self.k_max + 1, dtype=torch.float32)
        log_probs = support * math.log(lam) - torch.lgamma(support + 1.0)
        probs = torch.exp(log_probs - log_probs.max())
        probs = probs / probs.sum().clamp(min=1e-8)
        index = int(torch.multinomial(probs, 1, generator=self.rng).item())
        return int(support[index].item())

    def _update_lambda_from_acceptance(self, old_lambda: float, result: StepResult) -> float:
        acceptance = result.accepted_count / max(result.draft_length, 1)
        signal = acceptance - self.target_acceptance
        if result.rejected_at < 0 or result.accepted_count >= result.draft_length:
            signal += self.full_accept_bonus

        if signal >= 0.0:
            lr = self.lambda_lr
        else:
            lr = self.lambda_lr * self.shrink_lr_scale
        return old_lambda + lr * signal * self.k_max

    def _draft_entropy(self, context: torch.Tensor) -> float:
        if not self.use_drafter_entropy:
            return self.default_entropy
        if not self._context_in_drafter_vocab(context):
            return self.default_entropy
        with torch.no_grad():
            out = self.drafter.model(context)
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :].float()
            probs = functional.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum()
            return float((entropy / math.log(max(logits.shape[-1], 2))).item())

    def _token_class(self, context: torch.Tensor) -> float:
        if not self.use_token_class:
            return self.unknown_token_class
        token_id = int(context.reshape(-1)[-1].item())
        if not self._token_in_drafter_vocab(token_id):
            return self.unknown_token_class

        tokenizer = getattr(self.drafter, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                text = str(tokenizer.decode([token_id]))
                return self._classify_token_text(text)
            except Exception:
                pass

        if token_id <= 2:
            return 1.0
        if token_id <= 10:
            return 0.5
        return 0.0

    @staticmethod
    def _classify_token_text(text: str) -> float:
        stripped = text.strip()
        if not stripped or "\n" in text:
            return 1.0
        if any(ch in stripped for ch in "{}[]();:=<>"):
            return 0.65
        if any(ch in stripped for ch in "+-*/=^"):
            return 0.6
        if any(ch in stripped for ch in "#*_`|>"):
            return 0.5
        if any(ch.isdigit() for ch in stripped):
            return 0.45
        return 0.0

    def _context_in_drafter_vocab(self, context: torch.Tensor) -> bool:
        vocab_size = self._drafter_vocab_size()
        if vocab_size is None or context.numel() == 0:
            return True
        min_token = int(context.min().detach().cpu().item())
        max_token = int(context.max().detach().cpu().item())
        return min_token >= 0 and max_token < vocab_size

    def _token_in_drafter_vocab(self, token_id: int) -> bool:
        vocab_size = self._drafter_vocab_size()
        return vocab_size is None or 0 <= token_id < vocab_size

    def _drafter_vocab_size(self) -> int | None:
        model = getattr(self.drafter, "model", None)
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size
        vocab_size = getattr(model, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size
        return None

    def _posterior_entropy(self) -> float:
        entropy = -(self.posterior * self.posterior.clamp(min=1e-8).log()).sum()
        return float((entropy / math.log(len(self.regime_names))).item())

    @staticmethod
    def _build_transition(stay_prob: float) -> torch.Tensor:
        n_regimes = len(LatentRegimeK.regime_names)
        off_diag = (1.0 - stay_prob) / max(n_regimes - 1, 1)
        transition = torch.full((n_regimes, n_regimes), off_diag, dtype=torch.float32)
        transition.fill_diagonal_(stay_prob)
        return transition


class CategoricalLatentRegimeK(LatentRegimeK):
    """Choose draft length from regime-conditioned categorical distributions."""

    def __init__(
        self,
        drafter,
        *,
        categorical_widths: tuple[float, float, float, float] = (2.3, 2.0, 1.45, 0.9),
        categorical_temperature: float = 0.9,
        max_k_penalty: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(drafter, **kwargs)
        if len(categorical_widths) != len(self.regime_names):
            raise ValueError("categorical_widths must contain one value per regime")
        if any(value <= 0 for value in categorical_widths):
            raise ValueError("all categorical widths must be positive")
        if categorical_temperature <= 0:
            raise ValueError("categorical_temperature must be positive")
        if max_k_penalty < 0:
            raise ValueError("max_k_penalty must be non-negative")

        self.categorical_widths = torch.tensor(categorical_widths, dtype=torch.float32)
        self.categorical_temperature = categorical_temperature
        self.max_k_penalty = max_k_penalty

    def __call__(self, context: torch.Tensor) -> int:
        self._pending_entropy = self._draft_entropy(context)
        self._pending_token_class = self._token_class(context)

        change_point_prob = (1.0 - float(self.posterior.max().item())) * self.change_point_scale
        regime_idx = int(self.posterior.argmax().item())
        support, probs = self._categorical_distribution(change_point_prob)
        index = int(torch.multinomial(probs, 1, generator=self.rng).item())
        selected_k = int(support[index].item())
        expected_k = float((support.float() * probs).sum().item())

        self._last_regime_idx = regime_idx
        self._last_k = selected_k
        self.change_point_history.append(change_point_prob)
        self.selections.append(
            RegimeSelection(
                selected_k=selected_k,
                regime=self.regime_names[regime_idx],
                change_point_prob=change_point_prob,
                lambda_t=expected_k,
                posterior=[float(v) for v in self.posterior.tolist()],
            )
        )
        return selected_k

    def summary(self, prefix: str = "regime_k") -> dict[str, object]:
        metrics = super().summary(prefix)
        metrics.update(
            {
                f"{prefix}_selector": "categorical",
                f"{prefix}_categorical_temperature": self.categorical_temperature,
                f"{prefix}_max_k_penalty": self.max_k_penalty,
            }
        )
        return metrics

    def _categorical_distribution(
        self,
        change_point_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support = torch.arange(self.k_min, self.k_max + 1, dtype=torch.float32)
        regime_probs = torch.stack(
            [
                self._regime_categorical_distribution(index, support)
                for index in range(len(self.regime_names))
            ]
        )
        posterior = self.posterior.to(dtype=regime_probs.dtype)
        mixed_probs = posterior @ regime_probs
        transition_probs = regime_probs[self.regime_names.index("transition")]
        change_point_prob = _clamp(change_point_prob, 0.0, 1.0)
        probs = (1.0 - change_point_prob) * mixed_probs
        probs += change_point_prob * transition_probs
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum().clamp(min=1e-8)
        return support, probs

    def _regime_categorical_distribution(
        self,
        regime_idx: int,
        support: torch.Tensor,
    ) -> torch.Tensor:
        center = float(self.lambdas[regime_idx].item())
        center = _clamp(center, float(self.k_min), float(self.k_max))
        width = float(self.categorical_widths[regime_idx].item())
        logits = -0.5 * ((support - center) / width) ** 2
        logits = logits / self.categorical_temperature
        if self.max_k_penalty > 0:
            logits = logits - self.max_k_penalty * (support == self.k_max).float()
        return torch.softmax(logits, dim=0)


@dataclass
class _StepProxy:
    """Small StepResult-compatible object for record_result fallbacks."""

    draft_length: int
    accepted_count: int
    rejected_at: int = -1


class LatentRegimeKExperiment(BaseExperiment):
    """Research experiment for LatentRegimeK."""

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="latent_regime_k",
                description="Stochastic dynamic k via latent regimes and change points",
                tags=["research", "v.poponnikov", "adaptive", "stochastic", "regime"],
                dimensions=["draft_length_strategy"],
                depends_on=["01_baseline", "08_+speedup_adapt"],
            )
        )
        self._controller: LatentRegimeK | None = None

    def get_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.meta.name,
            use_rule1=True,
            use_rule2=True,
            use_lattice=False,
            use_translator=False,
            use_online_distil=False,
            use_replay=False,
            use_contrastive=False,
            use_speedup_adaptive=True,
            use_dynamic_routing=False,
            use_universal_drafter=False,
            draft_length=6,
            k_min=1,
            k_max=10,
        )

    def build_adaptive_controller(self, ctx: BuildContext) -> LatentRegimeK:
        cfg = ctx.config
        controller = LatentRegimeK(
            ctx.drafter,
            k_min=getattr(cfg, "k_min", 1),
            k_max=getattr(cfg, "k_max", 8),
            seed=getattr(cfg, "seed", 42) + 2001,
        )
        self._controller = controller
        return controller

    def on_extra_metrics(self, summary: dict) -> dict:
        if self._controller is not None:
            summary.update(self._controller.summary("regime_k"))
            summary["dynamic_k_method"] = "latent_regime"
        return summary


class LatentRegimeCategoricalKExperiment(LatentRegimeKExperiment):
    """Research experiment for the categorical LatentRegimeK variant."""

    def __init__(self) -> None:
        BaseExperiment.__init__(
            self,
            ExperimentMeta(
                name="latent_regime_categorical_k",
                description="Stochastic dynamic k via latent regimes and categorical sampling",
                tags=[
                    "research",
                    "v.poponnikov",
                    "adaptive",
                    "stochastic",
                    "regime",
                    "categorical",
                ],
                dimensions=["draft_length_strategy"],
                depends_on=["01_baseline", "08_+speedup_adapt", "latent_regime_k"],
            ),
        )
        self._controller: CategoricalLatentRegimeK | None = None

    def build_adaptive_controller(self, ctx: BuildContext) -> CategoricalLatentRegimeK:
        cfg = ctx.config
        controller = CategoricalLatentRegimeK(
            ctx.drafter,
            k_min=getattr(cfg, "k_min", 1),
            k_max=getattr(cfg, "k_max", 8),
            seed=getattr(cfg, "seed", 42) + 3001,
        )
        self._controller = controller
        return controller

    def on_extra_metrics(self, summary: dict) -> dict:
        if self._controller is not None:
            summary.update(self._controller.summary("regime_k"))
            summary["dynamic_k_method"] = "latent_regime_categorical"
        return summary


__all__ = ["LatentRegimeKExperiment", "LatentRegimeCategoricalKExperiment"]
