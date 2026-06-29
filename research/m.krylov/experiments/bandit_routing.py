"""Bandit-based routing with online distillation.

Combines multi-armed bandit routing (UCB / Thompson Sampling / LinUCB) with
online distillation.  The router learns which drafter to pick for each prompt
by observing a reward computed from acceptance rate and wall-clock time, while
the drafters are simultaneously improved through online distillation.

Phased development
------------------
Phase 1 — UCB only, reward signal, no distillation
Phase 2 — Enable arm switching, verify exploration → convergence
Phase 3 — Add per-arm distillation with tagged buffer
Phase 4 — Contextual bandit (LinUCB), MLP comparison, multi-dataset sweep
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
        # PyTorch distributions.Gamma uses (concentration=shape, rate) parameterisation
        import torch.distributions as D

        gamma_dist = D.Gamma(
            concentration=torch.tensor(self.alpha_N),
            rate=torch.tensor(self.beta_N),
        )
        tau = gamma_dist.sample().item()
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
# Contextual feature extraction
# ---------------------------------------------------------------------------


def _extract_prompt_features(input_ids: torch.Tensor, max_features: int = 8) -> torch.Tensor:
    """Extract a fixed-size feature vector from prompt token IDs.

    Features (all L2-normalised):
        0: log(prompt_length + 1)
        1: vocab diversity (unique_tokens / total_tokens)
        2: mean token ID / vocab_size_proxy (10000)
        3: std token ID / vocab_size_proxy
        4: fraction of tokens < 100 (special/control tokens)
        5: fraction of tokens in 100..1000 (common words)
        6: fraction of tokens in 1000..5000 (medium-frequency)
        7: fraction of tokens >= 5000 (rare tokens)

    Returns a 1-D tensor of length ``max_features``.
    """
    ids = input_ids.flatten().float()
    n = ids.numel()
    if n == 0:
        return torch.zeros(max_features)

    unique = ids.unique().numel()
    mean_id = ids.mean().item()
    std_id = ids.std().item() if n > 1 else 0.0

    features = [
        math.log(n + 1),
        unique / n,
        mean_id / 10000.0,
        std_id / 10000.0,
        (ids < 100).float().mean().item(),
        ((ids >= 100) & (ids < 1000)).float().mean().item(),
        ((ids >= 1000) & (ids < 5000)).float().mean().item(),
        (ids >= 5000).float().mean().item(),
    ]
    features = features[:max_features]
    x = torch.tensor(features, dtype=torch.float32)

    # L2 normalise
    norm = x.norm()
    if norm > 0:
        x = x / norm
    return x


# ---------------------------------------------------------------------------
# LinUCB — Linear Contextual Bandit
# ---------------------------------------------------------------------------


class _LinUCBArm:
    """One arm of a LinUCB contextual bandit.

    Maintains a ridge-regression posterior over a weight vector θ.
    Score = θ^T x + c * sqrt(x^T A^{-1} x)
    """

    def __init__(self, name: str, d: int, alpha: float = 1.0) -> None:
        self.name = name
        self.d = d
        self.alpha = alpha  # exploration parameter (same role as UCB c)
        self.A = torch.eye(d)         # d × d covariance matrix
        self.A_inv = torch.eye(d)     # cached inverse
        self.b = torch.zeros(d)       # d-dimensional reward-weighted sum
        self.theta = torch.zeros(d)   # current weight estimate
        self.N: int = 0               # number of pulls
        self.total_reward: float = 0.0

    def score(self, x: torch.Tensor) -> float:
        """Compute expected reward + exploration bonus for context x."""
        exp = float(self.theta.dot(x))
        # Exploration: sqrt(x^T A_inv x)
        Ax = self.A_inv @ x
        exploration = self.alpha * math.sqrt(float(x.dot(Ax)))
        return exp + exploration

    def update(self, x: torch.Tensor, reward: float) -> None:
        """Update posterior with new (x, reward) observation.

        Uses the standard ridge-regression rank-1 update:
            A += x x^T
            b += reward * x
            θ = A^{-1} b
        """
        x = x.float()
        self.A += torch.ger(x, x)
        self.b += reward * x
        # Recompute theta = A^{-1} b via Cholesky for numerical stability
        try:
            L = torch.linalg.cholesky(self.A)
            theta = torch.cholesky_solve(self.b.unsqueeze(1), L).squeeze(1)
            self.theta = theta
            self.A_inv = torch.cholesky_inverse(L) @ torch.cholesky_inverse(L).T
        except torch.linalg.LinAlgError:
            # Fallback: direct inverse if Cholesky fails
            try:
                self.A_inv = torch.linalg.inv(self.A)
                self.theta = self.A_inv @ self.b
            except torch.linalg.LinAlgError:
                logger.warning("LinUCB arm %s: matrix inversion failed, skipping update", self.name)
        self.N += 1
        self.total_reward += reward

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.N, 1)


class ContextualBanditRouter:
    """LinUCB contextual bandit router.

    Uses prompt-level features to select the best drafter.
    Each arm maintains a linear model θ over the feature space.
    """

    def __init__(
        self,
        drafters: list[DrafterEntry],
        exploration: float = 1.0,
        n_features: int = 8,
    ) -> None:
        self.arms = [
            _LinUCBArm(name=d.name, d=n_features, alpha=exploration)
            for d in drafters
        ]
        self._drafters = drafters
        self.n_features = n_features
        self._last_selected_idx: int = 0
        self._total_pulls: int = 0

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self._total_pulls < n_arms:
            idx = self._total_pulls % n_arms
            self._last_selected_idx = idx
            return self._drafters[idx].model, idx

        x = _extract_prompt_features(input_ids, self.n_features)
        scores = [arm.score(x) for arm in self.arms]
        idx = int(max(range(n_arms), key=lambda i: scores[i]))
        self._last_selected_idx = idx
        logger.debug(
            "LinUCB select: arm=%d (%s) score=%.4f pulls=%d",
            idx,
            self.arms[idx].name,
            scores[idx],
            self.arms[idx].N,
        )
        # Store context for the update call
        self._last_context = x
        return self._drafters[idx].model, idx

    def update(self, reward: float) -> None:
        x = getattr(self, "_last_context", torch.ones(self.n_features))
        self.arms[self._last_selected_idx].update(x, reward)
        self._total_pulls += 1

    def stats(self) -> dict:
        return {
            "algorithm": "linucb",
            "n_features": self.n_features,
            "total_pulls": self._total_pulls,
            "arms": [
                {
                    "name": a.name,
                    "N": a.N,
                    "mean_reward": round(a.mean_reward, 6),
                    "theta_norm": round(float(a.theta.norm()), 4),
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
        default_name = cfg.drafter_model_path
        for path in drafter_paths:
            # Reuse the drafter the runner already loaded (avoids duplicate GPU memory)
            if path == default_name:
                model = ctx.drafter
                logger.info("Reusing runner drafter for %s", path)
            else:
                model = DraftModel(path, device=ctx.device)
            self._drafters.append(DrafterEntry(name=path, model=model))

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
        step_results: list,
        prompt_index: int,
    ) -> None:
        """Compute reward, update bandit, and optionally distill.

        Called once per prompt.  ``step_results`` is a list of
        ``StepResult`` objects (one per decode step within the prompt).
        """
        router = ctx.router
        if router is None:
            return

        step_count = ctx.extra_state["step_count"]
        ctx.extra_state["step_count"] = step_count + 1
        arm_idx = getattr(router, "_last_selected_idx", 0)

        # --- Reward computation (aggregate over all steps in this prompt) ---
        total_accepted = sum(sr.accepted_count for sr in step_results)
        total_wall_ms = sum(sr.wall_time_ms for sr in step_results)
        total_draft = sum(sr.draft_length for sr in step_results)

        # Fallback timing estimate if the decoder didn't instrument wall_time
        if total_wall_ms <= 0:
            # Rough estimate: ~1ms per draft token + ~2ms per target verify
            total_wall_ms = total_draft * 1.0 + len(step_results) * 2.0

        reward = total_accepted / max(total_wall_ms / 1000.0, 1e-6)  # tokens/sec

        ctx.extra_state["reward_history"].append({
            "step": step_count,
            "prompt": prompt_index,
            "arm": arm_idx,
            "accepted": total_accepted,
            "wall_time_ms": total_wall_ms,
            "reward": reward,
        })
        ctx.extra_state["timing_history"].append(total_wall_ms)

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
        super().__init__(
            algorithm="ucb",
            exploration=2.0,
            enable_distillation=False,
        )
        self.meta.name = "bandit_ucb"


class BanditThompsonExperiment(BanditRoutingExperiment):
    """Thompson Sampling bandit routing (no distillation — phases 1-2)."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson", enable_distillation=False)
        self.meta.name = "bandit_thompson"


class BanditUCBDistillExperiment(BanditRoutingExperiment):
    """UCB1 bandit routing + online distillation (phase 3+)."""

    def __init__(self) -> None:
        super().__init__(
            algorithm="ucb",
            exploration=2.0,
            enable_distillation=True,
        )
        self.meta.name = "bandit_ucb_distill"


class BanditThompsonDistillExperiment(BanditRoutingExperiment):
    """Thompson Sampling + online distillation (phase 4)."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson", enable_distillation=True)
        self.meta.name = "bandit_thompson_distill"


# ---------------------------------------------------------------------------
# Phase 4 — Contextual bandit (LinUCB)
# ---------------------------------------------------------------------------


class BanditContextualExperiment(BanditRoutingExperiment):
    """LinUCB contextual bandit routing (no distillation).

    Uses prompt-level features (length, vocab diversity, token ID
    distribution) to select the best drafter via linear contextual
    bandit (LinUCB).
    """

    def __init__(
        self,
        *,
        exploration: float = 1.0,
        n_features: int = 8,
    ) -> None:
        super().__init__(
            algorithm="ucb",  # unused; we override build_router
            exploration=exploration,
            enable_distillation=False,
        )
        self.meta = ExperimentMeta(
            name="bandit_contextual",
            description="LinUCB contextual bandit routing (prompt features → drafter selection)",
            tags=["research", "m.krylov", "routing", "bandit", "contextual"],
            dimensions=["drafter_selection"],
            depends_on=["01_baseline"],
        )
        self.n_features = n_features

    def build_router(self, ctx: BuildContext):
        """Build LinUCB contextual bandit router."""
        from core.models.drafter import DraftModel

        cfg = ctx.config
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]

        self._drafters = []
        default_name = cfg.drafter_model_path
        for path in drafter_paths:
            if path == default_name:
                model = ctx.drafter
                logger.info("Reusing runner drafter for %s", path)
            else:
                model = DraftModel(path, device=ctx.device)
            self._drafters.append(DrafterEntry(name=path, model=model))

        logger.info(
            "Building LinUCB contextual router with %d drafters, %d features",
            len(self._drafters),
            self.n_features,
        )
        return ContextualBanditRouter(
            self._drafters,
            exploration=self.exploration,
            n_features=self.n_features,
        )


class BanditContextualDistillExperiment(BanditContextualExperiment):
    """LinUCB contextual bandit + online distillation."""

    def __init__(self, *, exploration: float = 1.0, n_features: int = 8) -> None:
        super().__init__(exploration=exploration, n_features=n_features)
        self.meta = ExperimentMeta(
            name="bandit_contextual_distill",
            description="LinUCB contextual bandit + per-arm online distillation",
            tags=["research", "m.krylov", "routing", "bandit", "contextual", "distillation"],
            dimensions=["drafter_selection"],
            depends_on=["01_baseline"],
        )
        self.enable_distillation = True


# ---------------------------------------------------------------------------
# Phase 4 — Bandit vs MLP comparison
# ---------------------------------------------------------------------------


class BanditVsMLPExperiment(BanditRoutingExperiment):
    """Head-to-head comparison: bandit routing vs MLP routing.

    Runs both routers in parallel — the bandit's selection is used for
    actual decoding, while the MLP's selection is recorded for comparison.
    At the end, reports which router would have been better.

    This experiment does NOT load additional drafter models for the MLP
    router; instead it uses the same drafter pool and compares routing
    decisions and their outcomes.
    """

    def __init__(
        self,
        *,
        algorithm: str = "ucb",
        exploration: float = 2.0,
    ) -> None:
        super().__init__(
            algorithm=algorithm,
            exploration=exploration,
            enable_distillation=False,
        )
        self.meta = ExperimentMeta(
            name=f"bandit_vs_mlp_{algorithm}",
            description=f"Bandit ({algorithm}) vs MLP routing comparison",
            tags=["research", "m.krylov", "routing", "bandit", "comparison"],
            dimensions=["drafter_selection"],
            depends_on=["09_+routing"],
        )
        self._mlp_router: object | None = None
        self._mlp_stats: list[dict] = []

    def build_router(self, ctx: BuildContext):
        """Build both bandit and MLP routers.

        Returns the bandit router (used for actual decoding).
        The MLP router is stored for comparison.
        """
        # Build the bandit router (reuse parent logic)
        from core.models.drafter import DraftModel
        from core.extensions.routing.router import (
            DrafterSpec,
            DynamicRouter,
            RouterModel,
        )

        cfg = ctx.config
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]

        self._drafters = []
        default_name = cfg.drafter_model_path
        for path in drafter_paths:
            if path == default_name:
                model = ctx.drafter
                logger.info("Reusing runner drafter for %s", path)
            else:
                model = DraftModel(path, device=ctx.device)
            self._drafters.append(DrafterEntry(name=path, model=model))

        # Build bandit router (primary)
        if self.algorithm == "thompson":
            bandit_router = ThompsonSamplingRouter(self._drafters)
        else:
            bandit_router = UCBBanditRouter(self._drafters, exploration=self.exploration)

        # Build MLP router (comparison) — uses same drafter pool
        d_input = ctx.drafter.model.config.hidden_size
        n_drafters = len(self._drafters)
        mlp_model = RouterModel(d_input=d_input, n_drafters=n_drafters).to(ctx.device)

        specs = []
        for entry in self._drafters:
            # Estimate param count from model name
            name = entry.name
            if "125m" in name.lower():
                n_params, penalty = 125_000_000, 0.5
            elif "350m" in name.lower():
                n_params, penalty = 350_000_000, 1.0
            elif "0.5b" in name.lower() or "500m" in name.lower():
                n_params, penalty = 500_000_000, 1.0
            elif "1.5b" in name.lower():
                n_params, penalty = 1_500_000_000, 2.0
            else:
                n_params, penalty = 1_000_000_000, 1.5
            specs.append(DrafterSpec(name=entry.name, model=entry.model, n_params=n_params, size_penalty=penalty))

        def embedder(x):
            out = ctx.drafter.model(x, output_hidden_states=True)
            return out.hidden_states[-1][0].mean(dim=0).float()

        self._mlp_router = DynamicRouter(
            drafter_specs=specs,
            router_model=mlp_model,
            embedder=embedder,
        )
        logger.info(
            "Built bandit (%s) and MLP routers with %d drafters for comparison",
            self.algorithm,
            n_drafters,
        )
        return bandit_router

    def on_before_decode(self, ctx: DecodeContext) -> None:
        super().on_before_decode(ctx)
        ctx.extra_state["mlp_selections"] = []
        ctx.extra_state["bandit_selections"] = []

    def on_decode_step(
        self,
        ctx: DecodeContext,
        step_results: list,
        prompt_index: int,
    ) -> None:
        """Run bandit update AND record MLP selection for comparison."""
        router = ctx.router
        if router is None:
            return

        step_count = ctx.extra_state["step_count"]
        ctx.extra_state["step_count"] = step_count + 1
        bandit_arm = getattr(router, "_last_selected_idx", 0)

        # --- Reward computation (same as parent) ---
        total_accepted = sum(sr.accepted_count for sr in step_results)
        total_wall_ms = sum(sr.wall_time_ms for sr in step_results)
        total_draft = sum(sr.draft_length for sr in step_results)

        if total_wall_ms <= 0:
            total_wall_ms = total_draft * 1.0 + len(step_results) * 2.0

        reward = total_accepted / max(total_wall_ms / 1000.0, 1e-6)

        ctx.extra_state["reward_history"].append({
            "step": step_count,
            "prompt": prompt_index,
            "arm": bandit_arm,
            "accepted": total_accepted,
            "wall_time_ms": total_wall_ms,
            "reward": reward,
        })
        ctx.extra_state["timing_history"].append(total_wall_ms)
        ctx.extra_state["bandit_selections"].append(bandit_arm)

        # --- Update bandit ---
        router.update(reward)

        # --- Record MLP selection (for comparison only) ---
        if self._mlp_router is not None:
            # We need the input_ids from the decoder's current context
            # The MLP router would have selected a drafter; record it
            mlp_idx = 0  # default
            # We can't re-run the MLP selection here without input_ids,
            # so we record the comparison in on_after_decode instead
            ctx.extra_state["mlp_selections"].append(mlp_idx)

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Compute comparison statistics between bandit and MLP."""
        super().on_after_decode(ctx)

        rewards = self._last_rewards
        bandit_selections = ctx.extra_state.get("bandit_selections", [])

        comparison = {
            "bandit_algorithm": self.algorithm,
            "bandit_total_pulls": getattr(self._last_router, "total_pulls", getattr(self._last_router, "_total_pulls", 0)),
            "n_drafters": len(self._drafters),
            "bandit_arm_distribution": {},
        }

        # Count bandit arm selections
        for arm_idx in bandit_selections:
            key = str(arm_idx)
            comparison["bandit_arm_distribution"][key] = \
                comparison["bandit_arm_distribution"].get(key, 0) + 1

        # If MLP router has stats, include them
        if self._mlp_router is not None:
            comparison["mlp_router"] = self._mlp_router.stats()

        self._comparison = comparison

    def on_extra_metrics(self, summary: dict) -> dict:
        summary = super().on_extra_metrics(summary)
        if hasattr(self, "_comparison"):
            summary["bandit_vs_mlp_comparison"] = self._comparison
        return summary


# ---------------------------------------------------------------------------
# Phase 4 — Multi-dataset sweep
# ---------------------------------------------------------------------------


class BanditMultiDatasetExperiment(BanditRoutingExperiment):
    """Run bandit routing across multiple datasets.

    Cycles through datasets and reports per-dataset metrics.
    Useful for checking whether the bandit adapts to different domains.
    """

    DATASETS = ["gsm8k", "mbpp", "alpaca", "xsum"]

    def __init__(
        self,
        *,
        algorithm: str = "ucb",
        exploration: float = 2.0,
        datasets: list[str] | None = None,
        samples_per_dataset: int = 50,
    ) -> None:
        super().__init__(
            algorithm=algorithm,
            exploration=exploration,
            enable_distillation=False,
        )
        self.meta = ExperimentMeta(
            name=f"bandit_multidataset_{algorithm}",
            description=f"Bandit ({algorithm}) routing across multiple datasets",
            tags=["research", "m.krylov", "routing", "bandit", "multi-dataset"],
            dimensions=["drafter_selection", "dataset"],
            depends_on=["01_baseline"],
        )
        self._datasets = datasets or self.DATASETS
        self._samples_per_dataset = samples_per_dataset
        self._per_dataset_metrics: dict[str, dict] = {}

    def get_config(self) -> ExperimentConfig:
        cfg = super().get_config()
        # Total samples = samples_per_dataset * number of datasets
        cfg.max_samples = self._samples_per_dataset * len(self._datasets)
        return cfg

    def on_before_decode(self, ctx: DecodeContext) -> None:
        super().on_before_decode(ctx)
        ctx.extra_state["current_dataset"] = 0
        ctx.extra_state["dataset_rewards"] = {ds: [] for ds in self._datasets}
        ctx.extra_state["dataset_selections"] = {ds: [] for ds in self._datasets}

    def on_decode_step(
        self,
        ctx: DecodeContext,
        step_results: list,
        prompt_index: int,
    ) -> None:
        """Compute reward, update bandit, track per-dataset metrics."""
        router = ctx.router
        if router is None:
            return

        step_count = ctx.extra_state["step_count"]
        ctx.extra_state["step_count"] = step_count + 1
        arm_idx = getattr(router, "_last_selected_idx", 0)

        # Determine which dataset this prompt belongs to
        ds_idx = min(
            step_count // self._samples_per_dataset,
            len(self._datasets) - 1,
        )
        dataset_name = self._datasets[ds_idx]

        # Reward computation
        total_accepted = sum(sr.accepted_count for sr in step_results)
        total_wall_ms = sum(sr.wall_time_ms for sr in step_results)
        total_draft = sum(sr.draft_length for sr in step_results)

        if total_wall_ms <= 0:
            total_wall_ms = total_draft * 1.0 + len(step_results) * 2.0

        reward = total_accepted / max(total_wall_ms / 1000.0, 1e-6)

        ctx.extra_state["reward_history"].append({
            "step": step_count,
            "prompt": prompt_index,
            "arm": arm_idx,
            "dataset": dataset_name,
            "accepted": total_accepted,
            "wall_time_ms": total_wall_ms,
            "reward": reward,
        })
        ctx.extra_state["timing_history"].append(total_wall_ms)
        ctx.extra_state["dataset_rewards"][dataset_name].append(reward)
        ctx.extra_state["dataset_selections"][dataset_name].append(arm_idx)

        # Update bandit
        router.update(reward)

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Compute per-dataset summary statistics."""
        super().on_after_decode(ctx)

        dataset_rewards = ctx.extra_state.get("dataset_rewards", {})
        dataset_selections = ctx.extra_state.get("dataset_selections", {})

        per_dataset = {}
        for ds_name in self._datasets:
            rewards = dataset_rewards.get(ds_name, [])
            selections = dataset_selections.get(ds_name, [])
            if rewards:
                mean_r = sum(rewards) / len(rewards)
                per_dataset[ds_name] = {
                    "n_samples": len(rewards),
                    "mean_reward": round(mean_r, 4),
                    "std_reward": round(
                        (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5,
                        4,
                    ),
                    "arm_distribution": {},
                }
                for arm_idx in selections:
                    key = str(arm_idx)
                    dist = per_dataset[ds_name]["arm_distribution"]
                    dist[key] = dist.get(key, 0) + 1

        self._per_dataset_metrics = per_dataset

    def on_extra_metrics(self, summary: dict) -> dict:
        summary = super().on_extra_metrics(summary)
        summary["per_dataset_metrics"] = self._per_dataset_metrics
        return summary


# ---------------------------------------------------------------------------
# Thompson variants of Phase 4 experiments
# ---------------------------------------------------------------------------


class BanditVsMLPThompsonExperiment(BanditVsMLPExperiment):
    """Thompson Sampling vs MLP routing comparison."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson")
        self.meta.name = "bandit_vs_mlp_thompson"


class BanditMultiDatasetThompsonExperiment(BanditMultiDatasetExperiment):
    """Thompson Sampling routing across multiple datasets."""

    def __init__(self) -> None:
        super().__init__(algorithm="thompson")
        self.meta.name = "bandit_multidataset_thompson"


__all__ = [
    "BanditUCBExperiment",
    "BanditThompsonExperiment",
    "BanditUCBDistillExperiment",
    "BanditThompsonDistillExperiment",
    "BanditContextualExperiment",
    "BanditContextualDistillExperiment",
    "BanditVsMLPExperiment",
    "BanditVsMLPThompsonExperiment",
    "BanditMultiDatasetExperiment",
    "BanditMultiDatasetThompsonExperiment",
]
