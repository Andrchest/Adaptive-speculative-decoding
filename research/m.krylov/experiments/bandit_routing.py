"""Bandit-based routing with online distillation.

Combines multi-armed bandit routing (UCB / Thompson Sampling) with online
distillation.  The router learns which drafter to pick for each prompt by
observing a reward computed from acceptance rate and wall-clock time, while
the drafters are simultaneously improved through online distillation.

Phased development
------------------
Phase 1 — UCB only, reward signal, no distillation
Phase 2 — Enable arm switching, verify exploration → convergence
Phase 3 — Add per-arm distillation with tagged buffer
Phase 4 — Thompson Sampling, compare against MLP routing (09_+routing)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import torch
import torch.optim as optim

from experiments.base import BaseExperiment, BuildContext, DecodeContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drafter arm
# ---------------------------------------------------------------------------


@dataclass
class DrafterEntry:
    """One arm of the bandit."""

    name: str
    model: object | None  # DraftModel instance
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.pulls, 1)


# ---------------------------------------------------------------------------
# UCB1 router
# ---------------------------------------------------------------------------


class UCBBanditRouter:
    """Upper Confidence Bound (UCB1) multi-armed bandit router.

    Score = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)
    """

    def __init__(
        self,
        drafters: list[DrafterEntry],
        exploration: float = 2.0,
    ) -> None:
        self.arms = list(drafters)
        self.c = exploration
        self.total_pulls: int = len(drafters)
        self._last_selected_idx: int = 0

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self.total_pulls < n_arms:
            idx = self.total_pulls % n_arms
            self._last_selected_idx = idx
            return self.arms[idx].model, idx

        scores: list[float] = []
        for arm in self.arms:
            exploitation = arm.mean_reward
            exploration = self.c * math.sqrt(
                math.log(self.total_pulls) / max(arm.pulls, 1)
            )
            scores.append(exploitation + exploration)

        idx = int(max(range(n_arms), key=lambda i: scores[i]))
        self._last_selected_idx = idx
        logger.debug(
            "UCB select: arm=%d (%s) score=%.4f mean=%.4f pulls=%d",
            idx,
            self.arms[idx].name,
            scores[idx],
            self.arms[idx].mean_reward,
            self.arms[idx].pulls,
        )
        return self.arms[idx].model, idx

    def update(self, reward: float) -> None:
        arm = self.arms[self._last_selected_idx]
        arm.pulls += 1
        arm.total_reward += reward
        self.total_pulls += 1

    def stats(self) -> dict:
        return {
            "algorithm": "ucb1",
            "exploration_c": self.c,
            "total_pulls": self.total_pulls,
            "arms": [
                {"name": a.name, "pulls": a.pulls, "mean_reward": round(a.mean_reward, 6)}
                for a in self.arms
            ],
        }


# ---------------------------------------------------------------------------
# Gaussian Thompson Sampling router
# ---------------------------------------------------------------------------


@dataclass
class _GaussianArm:
    """Normal-Gamma posterior for one arm.

    Prior: mu ~ N(mu_0, sigma^2 / kappa_0), sigma^2 ~ InvGamma(alpha_0, beta_0)

    After N observations with sum S and sum of squares S2:
        kappa_N = kappa_0 + N
        mu_N    = (kappa_0 * mu_0 + S) / kappa_N
        alpha_N = alpha_0 + N / 2
        beta_N  = beta_0 + 0.5 * (S2 - S^2 / N
                    + kappa_0 * N * (S/N - mu_0)^2 / (kappa_0 + N))

    Sampling: draw precision tau ~ Gamma(alpha_N, beta_N),
              then draw mu ~ N(mu_N, 1 / (kappa_N * tau))
    """

    name: str
    N: int = 0
    S: float = 0.0       # sum of rewards
    S2: float = 0.0      # sum of squared rewards
    mu_0: float = 0.0    # prior mean
    kappa_0: float = 1.0 # prior "pseudo-count" for mean
    alpha_0: float = 1.0 # prior shape for precision
    beta_0: float = 1.0  # prior rate for precision

    @property
    def kappa_N(self) -> float:
        return self.kappa_0 + self.N

    @property
    def mu_N(self) -> float:
        return (self.kappa_0 * self.mu_0 + self.S) / self.kappa_N

    @property
    def alpha_N(self) -> float:
        return self.alpha_0 + self.N / 2

    @property
    def beta_N(self) -> float:
        if self.N == 0:
            return self.beta_0
        sample_mean = self.S / self.N
        return (
            self.beta_0
            + 0.5
            * (
                self.S2
                - self.S * self.S / self.N
                + self.kappa_0 * self.N * (sample_mean - self.mu_0) ** 2 / self.kappa_N
            )
        )

    def sample(self, rng: torch.Generator | None = None) -> float:
        """Draw a sample from the posterior over mu."""
        # Draw precision tau ~ Gamma(alpha_N, beta_N)
        # PyTorch Gamma uses (shape, rate) parameterisation
        tau = torch.gamma(
            torch.tensor(self.alpha_N),
            torch.tensor(self.beta_N),
            generator=rng,
        ).item()
        # Draw mu ~ N(mu_N, 1 / (kappa_N * tau))
        std = 1.0 / math.sqrt(max(self.kappa_N * tau, 1e-10))
        mu = self.mu_N + std * torch.randn(1, generator=rng).item()
        return mu

    def update(self, reward: float) -> None:
        self.N += 1
        self.S += reward
        self.S2 += reward * reward

    def posterior_mean(self) -> float:
        """Point estimate (posterior mean of mu)."""
        return self.mu_N


class ThompsonSamplingRouter:
    """Thompson Sampling with Normal-Gamma posterior for continuous rewards."""

    def __init__(
        self,
        drafters: list[DrafterEntry],
        prior_mean: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.arms = [
            _GaussianArm(
                name=d.name,
                mu_0=prior_mean,
                kappa_0=prior_kappa,
                alpha_0=prior_alpha,
                beta_0=prior_beta,
            )
            for d in drafters
        ]
        self._drafters = drafters
        self._last_selected_idx: int = 0
        self._total_pulls: int = len(drafters)
        self._rng = torch.Generator()
        self._rng.manual_seed(42)

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self._total_pulls < n_arms:
            idx = self._total_pulls % n_arms
            self._last_selected_idx = idx
            return self._drafters[idx].model, idx

        # Sample from each arm's posterior and pick the best
        samples = [arm.sample(self._rng) for arm in self.arms]
        idx = int(max(range(n_arms), key=lambda i: samples[i]))
        self._last_selected_idx = idx
        logger.debug(
            "TS select: arm=%d (%s) sample=%.4f posterior_mean=%.4f N=%d",
            idx,
            self.arms[idx].name,
            samples[idx],
            self.arms[idx].posterior_mean(),
            self.arms[idx].N,
        )
        return self._drafters[idx].model, idx

    def update(self, reward: float) -> None:
        self.arms[self._last_selected_idx].update(reward)
        self._total_pulls += 1

    def stats(self) -> dict:
        return {
            "algorithm": "thompson_sampling",
            "total_pulls": self._total_pulls,
            "arms": [
                {
                    "name": a.name,
                    "N": a.N,
                    "posterior_mean": round(a.posterior_mean(), 6),
                    "alpha_N": round(a.alpha_N, 2),
                    "beta_N": round(a.beta_N, 4),
                }
                for a in self.arms
            ],
        }


# ---------------------------------------------------------------------------
# Per-arm distillation buffer
# ---------------------------------------------------------------------------


@dataclass
class BufferEntry:
    """One training example in the distillation buffer."""

    arm_index: int
    draft_logits: torch.Tensor  # (k, drafter_vocab)
    target_logits: torch.Tensor  # (k, target_vocab)
    draft_tokens: list[int]
    accepted_mask: list[bool]


class PerArmBuffer:
    """FIFO buffer that tags entries with the drafter arm index.

    When replaying, entries are filtered by arm so each drafter is only
    trained on data it generated.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._entries: list[BufferEntry] = []

    def push(
        self,
        arm_index: int,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
        accepted_mask: list[bool],
    ) -> None:
        entry = BufferEntry(
            arm_index=arm_index,
            draft_logits=draft_logits.detach().cpu(),
            target_logits=target_logits.detach().cpu(),
            draft_tokens=list(draft_tokens),
            accepted_mask=list(accepted_mask),
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            self._entries.pop(0)

    def sample_for_arm(
        self, arm_index: int, batch_size: int = 8
    ) -> list[BufferEntry]:
        """Return up to batch_size entries for the given arm."""
        arm_entries = [e for e in self._entries if e.arm_index == arm_index]
        if not arm_entries:
            return []
        # Simple random sample
        indices = torch.randperm(len(arm_entries))[:batch_size].tolist()
        return [arm_entries[i] for i in indices]

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        arm_counts: dict[int, int] = {}
        for e in self._entries:
            arm_counts[e.arm_index] = arm_counts.get(e.arm_index, 0) + 1
        return {"total": len(self._entries), "per_arm": arm_counts}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class BanditRoutingExperiment(BaseExperiment):
    """Bandit-based routing with online distillation.

    Uses UCB1 or Thompson Sampling to select among multiple drafters per
    prompt.  After each decode step, the router is updated with a reward
    computed from acceptance count and wall-clock time.  Accepted tokens
    are distilled back into the selected drafter through a per-arm buffer.
    """

    def __init__(
        self,
        *,
        algorithm: str = "ucb",
        exploration: float = 2.0,
        enable_distillation: bool = False,
        buffer_capacity: int = 4096,
        replay_every: int = 32,
        replay_batch: int = 8,
    ) -> None:
        super().__init__(
            ExperimentMeta(
                name=f"bandit_routing_{algorithm}",
                description=(
                    f"Bandit routing ({algorithm})"
                    + (" + online distillation" if enable_distillation else "")
                    + ". Reward = accepted / (T_draft + T_target)"
                ),
                tags=["research", "m.krylov", "routing", "bandit"],
                dimensions=["drafter_selection"],
                depends_on=["01_baseline"],
            )
        )
        self.algorithm = algorithm
        self.exploration = exploration
        self.enable_distillation = enable_distillation
        self.buffer_capacity = buffer_capacity
        self.replay_every = replay_every
        self.replay_batch = replay_batch

        # Populated during build
        self._drafters: list[DrafterEntry] = []
        self._buffer: PerArmBuffer | None = None
        self._distillers: list[object] = []  # one per arm

    def get_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.meta.name,
            drafter_model_path="Qwen/Qwen2.5-0.5B-Instruct",
            drafter_model_paths=[
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ],
            target_model_path="Qwen/Qwen2.5-7B-Instruct",
            use_rule1=True,
            use_rule2=True,
            use_lattice=False,
            use_translator=False,
            use_online_distil=self.enable_distillation,
            use_replay=False,
            use_contrastive=False,
            use_speedup_adaptive=False,
            use_dynamic_routing=False,
            use_universal_drafter=False,
        )

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def build_router(self, ctx: BuildContext) -> UCBBanditRouter | ThompsonSamplingRouter:
        """Build the bandit router with multiple drafter arms."""
        from core.models.drafter import DraftModel

        cfg = ctx.config
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]

        self._drafters = []
        for path in drafter_paths:
            model = DraftModel(path, device=ctx.device)
            self._drafters.append(DrafterEntry(name=path, model=model))

        # Add default drafter if not already present
        default_name = cfg.drafter_model_path
        if default_name not in drafter_paths:
            self._drafters.append(DrafterEntry(name=default_name, model=ctx.drafter))

        logger.info(
            "Building %s router with %d drafters: %s",
            self.algorithm,
            len(self._drafters),
            [d.name for d in self._drafters],
        )

        if self.algorithm == "ucb":
            return UCBBanditRouter(self._drafters, exploration=self.exploration)
        elif self.algorithm == "thompson":
            return ThompsonSamplingRouter(self._drafters)
        else:
            raise ValueError(f"Unknown bandit algorithm: {self.algorithm!r}")

    def build_distiller(self, ctx: BuildContext):
        """Build per-arm distillers and a shared buffer.

        Returns None if distillation is disabled (phase 1-2).
        For phase 3+, returns a placeholder so the decoder knows
        distillation is active (gradients enabled).
        """
        if not self.enable_distillation:
            return None

        translator = ctx.components.get("translator")
        if translator is None:
            logger.warning("No translator found; skipping distillation setup")
            return None

        self._buffer = PerArmBuffer(capacity=self.buffer_capacity)
        self._distillers = []

        for i, entry in enumerate(self._drafters):
            drafter = entry.model
            drafter.prepare_for_training(torch.float32)
            drafter.model.train()

            from core.distillation.online import OnlineDistiller

            opt = optim.Adam(
                drafter.model.parameters(),
                lr=getattr(ctx.config, "distil_lr", 1e-5),
            )
            distiller = OnlineDistiller(
                drafter_model=drafter,
                translator=translator,
                optimizer=opt,
                lambda_ngram=getattr(ctx.config, "lambda_ngram", 0.5),
            )
            self._distillers.append(distiller)
            logger.info("Distiller %d ready for %s", i, entry.name)

        # Return the first distiller as the "active" one.  The actual
        # distillation target is swapped in on_decode_step based on the
        # router's selection.
        return self._distillers[0] if self._distillers else None

    # ------------------------------------------------------------------
    # Decode hooks
    # ------------------------------------------------------------------

    def on_before_decode(self, ctx: DecodeContext) -> None:
        ctx.extra_state["step_count"] = 0
        ctx.extra_state["reward_history"] = []
        ctx.extra_state["timing_history"] = []

    def on_decode_step(
        self,
        ctx: DecodeContext,
        step_result,
        prompt_index: int,
    ) -> None:
        """Compute reward, update bandit, and optionally distill."""
        router = ctx.router
        if router is None:
            return

        step_count = ctx.extra_state["step_count"]
        ctx.extra_state["step_count"] = step_count + 1
        arm_idx = getattr(router, "_last_selected_idx", 0)

        # --- Reward computation ---
        accepted = step_result.accepted_count if hasattr(step_result, "accepted_count") else 0
        wall_time_ms = getattr(step_result, "wall_time_ms", 0.0)

        # If the decoder doesn't report timing, use a placeholder based on
        # draft length (each token ~ constant cost).  Replace with real
        # timing once _decode_step() is instrumented.
        if wall_time_ms <= 0:
            draft_len = getattr(step_result, "draft_length", 5)
            # Rough estimate: ~1ms per draft token + ~2ms target verify
            wall_time_ms = draft_len * 1.0 + 2.0

        reward = accepted / max(wall_time_ms / 1000.0, 1e-6)  # tokens per second

        ctx.extra_state["reward_history"].append({
            "step": step_count,
            "prompt": prompt_index,
            "arm": arm_idx,
            "accepted": accepted,
            "wall_time_ms": wall_time_ms,
            "reward": reward,
        })
        ctx.extra_state["timing_history"].append(wall_time_ms)

        # --- Update bandit ---
        router.update(reward)

        # --- Periodic distillation (phase 3+) ---
        if (
            self.enable_distillation
            and self._buffer is not None
            and step_count % self.replay_every == 0
        ):
            self._replay_for_arm(ctx, arm_idx)

    def _replay_for_arm(self, ctx: DecodeContext, arm_idx: int) -> None:
        """Run a distillation step using buffered data for the given arm."""
        if self._buffer is None or not self._distillers:
            return
        if arm_idx >= len(self._distillers):
            return

        batch = self._buffer.sample_for_arm(arm_idx, self.replay_batch)
        if not batch:
            return

        distiller = self._distillers[arm_idx]
        for entry in batch:
            try:
                distiller.step(
                    draft_logits=entry.draft_logits.to(ctx.decoder.drafter.device),
                    target_logits=entry.target_logits.to(ctx.decoder.target.device),
                    draft_tokens=entry.draft_tokens,
                    accepted_mask=entry.accepted_mask,
                )
            except Exception as e:
                logger.warning("Distillation replay error: %s", e)

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Save references for on_extra_metrics."""
        self._last_router = ctx.router
        self._last_rewards = ctx.extra_state.get("reward_history", [])
        self._last_timings = ctx.extra_state.get("timing_history", [])

    def on_extra_metrics(self, summary: dict) -> dict:
        """Augment summary with bandit statistics."""
        router = getattr(self, "_last_router", None)
        if router is not None and hasattr(router, "stats"):
            summary["bandit_router"] = router.stats()

        rewards = self._last_rewards
        if rewards:
            reward_values = [r["reward"] for r in rewards]
            summary["bandit_mean_reward"] = sum(reward_values) / len(reward_values)
            mean_r = summary["bandit_mean_reward"]
            summary["bandit_std_reward"] = (
                sum((r - mean_r) ** 2 for r in reward_values) / len(reward_values)
            ) ** 0.5

        timings = self._last_timings
        if timings:
            summary["mean_wall_time_ms"] = sum(timings) / len(timings)

        if self._buffer is not None:
            summary["buffer_stats"] = self._buffer.stats()

        return summary


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


class BanditUCBExperiment(BanditRoutingExperiment):
    """UCB1 bandit routing (no distillation — phases 1-2)."""

    def __init__(self) -> None:
        super().__init__(algorithm="ucb", exploration=2.0, enable_distillation=False)


class BanditThompsonExperiment(BanditRoutingExperiment):
    """Thompson Sampling bandit routing (no distillation — phases 1-2)."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson", enable_distillation=False)


class BanditUCBDistillExperiment(BanditRoutingExperiment):
    """UCB1 bandit routing + online distillation (phase 3+)."""

    def __init__(self) -> None:
        super().__init__(algorithm="ucb", exploration=2.0, enable_distillation=True)


class BanditThompsonDistillExperiment(BanditRoutingExperiment):
    """Thompson Sampling + online distillation (phase 4)."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson", enable_distillation=True)


__all__ = [
    "BanditRoutingExperiment",
    "BanditUCBExperiment",
    "BanditThompsonExperiment",
    "BanditUCBDistillExperiment",
    "BanditThompsonDistillExperiment",
]
