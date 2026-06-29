"""Research experiments for stochastic dynamic draft-length selection.

This module keeps the v.poponnikov research code local to the research area.
It defines two adaptive controllers:

- EpistemicConsensusK: stochastic self-consensus over several drafter runs.
- LatentRegimeK: hidden-regime controller with change-point resets.
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


@dataclass
class ConsensusSelection:
    """Debug information for one EpistemicConsensusK decision."""

    selected_k: int
    theta: float
    consensus: list[float]
    logprob_variance: list[float]
    margin: list[float]
    score: list[float]
    continue_prob: list[float]


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


class EpistemicConsensusK:
    """Choose draft length from stochastic agreement between drafter runs."""

    def __init__(
        self,
        drafter,
        *,
        k_min: int = 1,
        k_max: int = 8,
        n_trajectories: int = 4,
        target_acceptance: float = 0.78,
        adaptation_rate: float = 0.03,
        tau_k: float = 0.6,
        tau_margin: float = 1.0,
        lambda_consensus: float = 1.0,
        lambda_margin: float = 0.35,
        lambda_uncertainty: float = 0.15,
        theta: float = 0.75,
        theta_min: float = 0.05,
        theta_max: float = 2.5,
        consensus_temperature: float = 1.0,
        logit_noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.drafter = drafter
        self.k_min = k_min
        self.k_max = k_max
        self.n_trajectories = n_trajectories
        self.target_acceptance = target_acceptance
        self.adaptation_rate = adaptation_rate
        self.tau_k = tau_k
        self.tau_margin = tau_margin
        self.lambda_consensus = lambda_consensus
        self.lambda_margin = lambda_margin
        self.lambda_uncertainty = lambda_uncertainty
        self.theta = theta
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.consensus_temperature = consensus_temperature
        self.logit_noise_std = logit_noise_std
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        self.stats = DynamicKStats()
        self.selections: list[ConsensusSelection] = []
        self.theta_history: list[float] = [theta]

    def __call__(self, context: torch.Tensor) -> int:
        tokens, logits = self._sample_trajectories(context)
        selection = self._select_from_trajectories(tokens, logits)
        self.selections.append(selection)
        return selection.selected_k

    def observe_step(self, result: StepResult) -> None:
        """Update the caution threshold after target verification."""
        self.stats.observe(result)
        acceptance = result.accepted_count / max(result.draft_length, 1)
        self.theta = _clamp(
            self.theta + self.adaptation_rate * (self.target_acceptance - acceptance),
            self.theta_min,
            self.theta_max,
        )
        self.theta_history.append(self.theta)

    def record_result(self, accepted_count: int) -> None:
        """Compatibility hook for controllers that only receive accepted count."""
        if not self.selections:
            return
        result = _StepProxy(self.selections[-1].selected_k, accepted_count)
        self.observe_step(result)  # type: ignore[arg-type]

    def summary(self, prefix: str = "consensus_k") -> dict[str, object]:
        metrics = self.stats.summary(prefix)
        latest_scores = [value for s in self.selections for value in s.score]
        consensus = [value for s in self.selections for value in s.consensus]
        variance = [value for s in self.selections for value in s.logprob_variance]
        continuation = [value for s in self.selections for value in s.continue_prob]
        metrics.update(
            {
                f"{prefix}_theta_final": self.theta,
                f"{prefix}_theta_mean": _mean(self.theta_history),
                f"{prefix}_score_mean": _mean(latest_scores),
                f"{prefix}_consensus_mean": _mean(consensus),
                f"{prefix}_logprob_variance_mean": _mean(variance),
                f"{prefix}_continue_prob_mean": _mean(continuation),
                f"{prefix}_n_trajectories": self.n_trajectories,
            }
        )
        return metrics

    def _sample_trajectories(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_rows: list[list[int]] = []
        logits_rows: list[torch.Tensor] = []

        for _ in range(self.n_trajectories):
            tokens, logits = self.drafter.draft(
                context,
                self.k_max,
                temperature=self.consensus_temperature,
            )
            logits = self._normalize_logits(logits)
            if self.logit_noise_std > 0:
                logits = logits + torch.randn_like(logits) * self.logit_noise_std
            token_rows.append(tokens[: self.k_max])
            logits_rows.append(logits[: self.k_max])

        device = logits_rows[0].device
        token_tensor = torch.tensor(token_rows, dtype=torch.long, device=device)
        logits_tensor = torch.stack(logits_rows, dim=0)
        return token_tensor, logits_tensor

    def _select_from_trajectories(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> ConsensusSelection:
        log_probs = functional.log_softmax(logits.float(), dim=-1)
        majority_tokens: list[int] = []
        consensus: list[float] = []

        for pos in range(tokens.shape[1]):
            counts = Counter(int(tok) for tok in tokens[:, pos].detach().cpu().tolist())
            token, count = counts.most_common(1)[0]
            majority_tokens.append(token)
            consensus.append(count / max(self.n_trajectories, 1))

        majority = torch.tensor(majority_tokens, dtype=torch.long, device=logits.device)
        position = torch.arange(tokens.shape[1], device=logits.device)
        majority_log_probs = log_probs[:, position, majority]
        logprob_variance = majority_log_probs.var(dim=0, unbiased=False)

        if log_probs.shape[-1] >= 2:
            mean_log_probs = log_probs.mean(dim=0)
            top2 = torch.topk(mean_log_probs, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.zeros(tokens.shape[1], device=logits.device)

        consensus_t = torch.tensor(consensus, dtype=torch.float32, device=logits.device)
        score = (
            self.lambda_consensus * consensus_t
            + self.lambda_margin * torch.sigmoid(margin / max(self.tau_margin, 1e-6))
            - self.lambda_uncertainty * logprob_variance
        )
        continue_prob = torch.sigmoid((score - self.theta) / max(self.tau_k, 1e-6))
        selected_k = self._sample_k(continue_prob.detach().cpu())

        return ConsensusSelection(
            selected_k=selected_k,
            theta=self.theta,
            consensus=[float(v) for v in consensus_t.detach().cpu().tolist()],
            logprob_variance=[float(v) for v in logprob_variance.detach().cpu().tolist()],
            margin=[float(v) for v in margin.detach().cpu().tolist()],
            score=[float(v) for v in score.detach().cpu().tolist()],
            continue_prob=[float(v) for v in continue_prob.detach().cpu().tolist()],
        )

    def _sample_k(self, continue_prob: torch.Tensor) -> int:
        selected = 0
        for pos, prob in enumerate(continue_prob.tolist(), start=1):
            draw = torch.rand((), generator=self.rng).item()
            if draw < prob:
                selected = pos
            else:
                break
        return max(self.k_min, min(selected, self.k_max))

    @staticmethod
    def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3 and logits.shape[1] == 1:
            return logits.squeeze(1)
        return logits


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

    @staticmethod
    def _token_class(context: torch.Tensor) -> float:
        token_id = int(context.reshape(-1)[-1].item())
        if token_id <= 2:
            return 1.0
        if token_id <= 10:
            return 0.5
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


class StochasticConsensusKExperiment(BaseExperiment):
    """Research experiment for EpistemicConsensusK."""

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="stochastic_consensus_k",
                description="Stochastic dynamic k via drafter self-consensus",
                tags=["research", "v.poponnikov", "adaptive", "stochastic"],
                dimensions=["draft_length_strategy"],
                depends_on=["01_baseline", "08_+speedup_adapt"],
            )
        )
        self._controller: EpistemicConsensusK | None = None

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

    def build_adaptive_controller(self, ctx: BuildContext) -> EpistemicConsensusK:
        cfg = ctx.config
        controller = EpistemicConsensusK(
            ctx.drafter,
            k_min=getattr(cfg, "k_min", 1),
            k_max=getattr(cfg, "k_max", 8),
            n_trajectories=4,
            target_acceptance=0.78,
            adaptation_rate=0.03,
            seed=getattr(cfg, "seed", 42) + 1001,
        )
        self._controller = controller
        return controller

    def on_extra_metrics(self, summary: dict) -> dict:
        if self._controller is not None:
            summary.update(self._controller.summary("consensus_k"))
            summary["dynamic_k_method"] = "epistemic_consensus"
        return summary


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


__all__ = ["LatentRegimeKExperiment", "StochasticConsensusKExperiment"]
