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
        k_max: int = 8,
        lambdas: tuple[float, float, float, float] = (8.0, 5.0, 2.0, 1.0),
        lambda_min: float = 1.0,
        transition_stay_prob: float = 0.9,
        reward_penalty: float = 0.5,
        lambda_lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        _validate_k_bounds(k_min, k_max)
        if len(lambdas) != len(self.regime_names):
            raise ValueError("lambdas must contain one value per regime")
        if any(value <= 0 for value in lambdas):
            raise ValueError("all regime lambdas must be positive")
        if not 0.0 <= transition_stay_prob <= 1.0:
            raise ValueError("transition_stay_prob must be in [0, 1]")

        self.drafter = drafter
        self.k_min = k_min
        self.k_max = k_max
        self.lambdas = torch.tensor(lambdas, dtype=torch.float32)
        self.lambda_min = lambda_min
        self.transition = self._build_transition(transition_stay_prob)
        self.reward_penalty = reward_penalty
        self.lambda_lr = lambda_lr
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        self.posterior = torch.full((len(self.regime_names),), 1.0 / len(self.regime_names))
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

        change_point_prob = 1.0 - float(self.posterior.max().item())
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

        accepted = float(result.accepted_count)
        rejected = float(max(result.draft_length - result.accepted_count, 0))
        reward = (accepted - self.reward_penalty * rejected) / max(result.draft_length, 1)
        old_lambda = float(self.lambdas[self._last_regime_idx].item())
        new_lambda = old_lambda + self.lambda_lr * reward * (self._last_k - old_lambda)
        self.lambdas[self._last_regime_idx] = _clamp(new_lambda, self.k_min, self.k_max)

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
        lam = torch.tensor(max(lambda_t, 1e-3), dtype=torch.float32)
        for _ in range(20):
            sample = int(torch.poisson(lam, generator=self.rng).item())
            if self.k_min <= sample <= self.k_max:
                return sample
        return max(self.k_min, min(round(lambda_t), self.k_max))

    def _draft_entropy(self, context: torch.Tensor) -> float:
        with torch.no_grad():
            out = self.drafter.model(context)
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :].float()
            probs = functional.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum()
            return float((entropy / math.log(max(logits.shape[-1], 2))).item())

    def _token_class(self, context: torch.Tensor) -> float:
        token_id = int(context.reshape(-1)[-1].item())

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
            draft_length=5,
            k_min=1,
            k_max=8,
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


__all__ = ["LatentRegimeKExperiment"]
