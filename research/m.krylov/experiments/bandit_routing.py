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
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.distributions as D
import torch.optim as optim

from experiments.base import BaseExperiment, BuildContext, DecodeContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drafter arm
# ---------------------------------------------------------------------------


@dataclass
class DrafterEntry:
    """One arm of the bandit.

    When ``reward_window`` > 0 only the most recent rewards are used
    for ``mean_reward`` so the bandit can adapt when drafter quality
    changes over time (e.g. during online distillation).

    Uses an O(1) running-sum sliding window to avoid recomputing
    ``sum(history)`` on every ``mean_reward`` access.
    """

    name: str
    model: object | None  # DraftModel instance
    pulls: int = 0
    total_reward: float = 0.0
    reward_window: int = 0  # 0 = no window (use all history)

    # Internal buffer for sliding window (only used when reward_window > 0)
    _reward_history: deque = field(default_factory=deque, repr=False)
    _window_sum: float = 0.0

    def record_reward(self, reward: float) -> None:
        """Record a reward (call this instead of mutating fields directly)."""
        self.pulls += 1
        self.total_reward += reward
        if self.reward_window > 0:
            self._reward_history.append(reward)
            self._window_sum += reward
            if len(self._reward_history) > self.reward_window:
                evicted = self._reward_history.popleft()
                self._window_sum -= evicted

    @property
    def mean_reward(self) -> float:
        if self.reward_window > 0 and self._reward_history:
            return self._window_sum / len(self._reward_history)
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
        self.total_pulls: int = 0
        self._last_selected_idx: int = 0
        self._round_robin_count: int = 0  # tracks initial exploration phase

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self._round_robin_count < n_arms:
            idx = self._round_robin_count % n_arms
            self._round_robin_count += 1
            self._last_selected_idx = idx
            return self.arms[idx].model, idx

        log_total = math.log(max(self.total_pulls, 1))
        best_idx = 0
        best_score = -1e30
        for i, arm in enumerate(self.arms):
            exploitation = arm.mean_reward
            exploration = self.c * math.sqrt(log_total / max(arm.pulls, 1))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_idx = i

        self._last_selected_idx = best_idx
        logger.debug(
            "UCB select: arm=%d (%s) score=%.4f mean=%.4f pulls=%d",
            best_idx,
            self.arms[best_idx].name,
            best_score,
            self.arms[best_idx].mean_reward,
            self.arms[best_idx].pulls,
        )
        return self.arms[best_idx].model, best_idx

    def update(self, reward: float) -> None:
        arm = self.arms[self._last_selected_idx]
        arm.record_reward(reward)
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
    S: float = 0.0  # sum of rewards
    S2: float = 0.0  # sum of squared rewards
    mu_0: float = 0.0  # prior mean
    kappa_0: float = 1.0  # prior "pseudo-count" for mean
    alpha_0: float = 2.0  # prior shape for precision (alpha=2 gives finite variance)
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
        return self.beta_0 + 0.5 * (
            self.S2
            - self.S * self.S / self.N
            + self.kappa_0 * self.N * (sample_mean - self.mu_0) ** 2 / self.kappa_N
        )

    def sample(self, rng: torch.Generator | None = None) -> float:
        """Draw a sample from the posterior over mu."""
        gamma_dist = D.Gamma(
            concentration=torch.tensor(self.alpha_N),
            rate=torch.tensor(self.beta_N),
        )
        tau = gamma_dist.sample().item()
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
        prior_alpha: float = 2.0,
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
        self._total_pulls: int = 0
        self._round_robin_count: int = 0  # tracks initial exploration phase
        self._rng = torch.Generator()
        self._rng.manual_seed(42)
        self._samples: list[float] = [0.0] * len(drafters)

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self._round_robin_count < n_arms:
            idx = self._round_robin_count % n_arms
            self._round_robin_count += 1
            self._last_selected_idx = idx
            return self._drafters[idx].model, idx

        # Sample from each arm's posterior and pick the best
        samples = self._samples
        for i, arm in enumerate(self.arms):
            samples[i] = arm.sample(self._rng)
        best_idx = 0
        best_val = -1e30
        for i, v in enumerate(samples):
            if v > best_val:
                best_val = v
                best_idx = i
        idx = best_idx
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
        # Update the Gaussian posterior
        self.arms[self._last_selected_idx].update(reward)
        # Also update the DrafterEntry for sliding-window support
        self._drafters[self._last_selected_idx].record_reward(reward)
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


def _extract_prompt_features(
    input_ids: torch.Tensor,
    max_features: int = 8,
    vocab_size: int = 50257,  # default OPT vocab; override with actual model vocab_size
) -> torch.Tensor:
    """Extract a fixed-size feature vector from prompt token IDs.

    Features (all L2-normalised):
        0: log(prompt_length + 1)
        1: vocab diversity (unique_tokens / total_tokens)
        2: mean token ID / vocab_size
        3: std token ID / vocab_size
        4: fraction of tokens < 100 (special/control tokens)
        5: fraction of tokens in 100..1000 (common words)
        6: fraction of tokens in 1000..5000 (medium-frequency)
        7: fraction of tokens >= 5000 (rare tokens)

    ``vocab_size`` should be set to the actual model vocabulary size
    (e.g. ~50k for OPT, ~151k for Qwen2.5) so features 2–3 stay in
    a sensible [0, 1] range.

    All features are computed as a single GPU tensor to avoid
    multiple ``.item()`` syncs that stall the GPU pipeline.

    Returns a 1-D tensor of length ``max_features`` on CPU.
    """
    ids = input_ids.flatten().float()
    n = ids.size(0)
    if n == 0:
        return torch.zeros(max_features)

    vs = float(max(vocab_size, 1))
    features = torch.stack([
        torch.tensor(math.log(n + 1), device=ids.device),
        torch.tensor(ids.unique().size(0), device=ids.device) / n,
        ids.mean() / vs,
        (ids.std() if n > 1 else torch.tensor(0.0, device=ids.device)) / vs,
        (ids < 100).float().mean(),
        ((ids >= 100) & (ids < 1000)).float().mean(),
        ((ids >= 1000) & (ids < 5000)).float().mean(),
        (ids >= 5000).float().mean(),
    ])
    features = features[:max_features].cpu()
    norm = features.norm()
    if norm > 0:
        features = features / norm
    return features


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
        self.A = torch.eye(d)  # d × d covariance matrix
        self.A_inv = torch.eye(d)  # cached inverse
        self.b = torch.zeros(d)  # d-dimensional reward-weighted sum
        self.theta = torch.zeros(d)  # current weight estimate
        self.N: int = 0  # number of pulls
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

        Uses Sherman-Morrison for O(d²) rank-1 update of A_inv with
        full Cholesky recompute as fallback when the denominator is
        near zero.
        """
        x = x.float()
        self.b += reward * x

        # Sherman-Morrison: (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        x_col = x.unsqueeze(1)  # (d, 1)
        Ainv_x = self.A_inv @ x_col  # (d, 1)
        denom = 1.0 + float(x.dot(Ainv_x.squeeze(1)))

        if abs(denom) > 1e-8:
            self.A.addmm_(x_col, x_col.T, beta=1.0, alpha=1.0)
            self.A_inv.addmm_(x_col, Ainv_x.T, beta=1.0, alpha=-1.0 / denom)
            self.theta = self.A_inv @ self.b
        else:
            # Fallback: full Cholesky recompute
            self.A += torch.ger(x, x)
            try:
                L = torch.linalg.cholesky(self.A)
                self.A_inv = torch.cholesky_inverse(L)
                self.theta = torch.cholesky_solve(self.b.unsqueeze(1), L).squeeze(1)
            except torch.linalg.LinAlgError:
                try:
                    self.A_inv = torch.linalg.inv(self.A)
                    self.theta = self.A_inv @ self.b
                except torch.linalg.LinAlgError:
                    logger.warning(
                        "LinUCB arm %s: matrix inversion failed, skipping update", self.name
                    )
                    self.N += 1
                    self.total_reward += reward
                    return
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
        vocab_size: int = 50257,
    ) -> None:
        self.arms = [_LinUCBArm(name=d.name, d=n_features, alpha=exploration) for d in drafters]
        self._drafters = drafters
        self.n_features = n_features
        self.vocab_size = vocab_size
        self._last_selected_idx: int = 0
        self._total_pulls: int = 0
        self._round_robin_count: int = 0  # tracks initial exploration phase

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        n_arms = len(self.arms)

        # Round-robin until every arm has been pulled at least once
        if self._round_robin_count < n_arms:
            idx = self._round_robin_count % n_arms
            self._round_robin_count += 1
            self._last_selected_idx = idx
            return self._drafters[idx].model, idx

        x = _extract_prompt_features(input_ids, self.n_features, vocab_size=self.vocab_size)
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
        arm = self.arms[self._last_selected_idx]
        arm.update(x, reward)
        # Also update the DrafterEntry for consistency
        if self._last_selected_idx < len(self._drafters):
            self._drafters[self._last_selected_idx].record_reward(reward)
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

    draft_logits: torch.Tensor  # (k, drafter_vocab)
    target_logits: torch.Tensor  # (k, target_vocab)
    draft_tokens: list[int]
    accepted_mask: list[bool]


class PerArmBuffer:
    """Per-arm FIFO buffers for distillation replay.

    Each arm gets its own ``deque(maxlen=capacity_per_arm)`` so push is
    O(1), sampling is O(k), and there are no cross-arm index bugs.

    Uses a seeded RNG for reproducible sampling across runs.
    """

    def __init__(
        self,
        num_arms: int,
        capacity_per_arm: int = 2048,
        seed: int = 42,
    ) -> None:
        self.num_arms = num_arms
        self.capacity_per_arm = capacity_per_arm
        self._queues: dict[int, deque] = {
            i: deque(maxlen=capacity_per_arm) for i in range(num_arms)
        }
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def push(
        self,
        arm_index: int,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
        accepted_mask: list[bool],
    ) -> None:
        if arm_index not in self._queues:
            return
        entry = BufferEntry(
            draft_logits=draft_logits.detach().cpu(),
            target_logits=target_logits.detach().cpu(),
            draft_tokens=list(draft_tokens),
            accepted_mask=list(accepted_mask),
        )
        self._queues[arm_index].append(entry)

    def sample_for_arm(self, arm_index: int, batch_size: int = 8) -> list[BufferEntry]:
        """Return up to batch_size entries for the given arm."""
        q = self._queues.get(arm_index)
        if not q:
            return []
        entries = list(q)
        if len(entries) <= batch_size:
            return entries
        indices = torch.randperm(len(entries), generator=self._rng)[:batch_size].tolist()
        return [entries[i] for i in indices]

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def stats(self) -> dict:
        per_arm = {str(i): len(q) for i, q in self._queues.items()}
        return {"total": len(self), "per_arm": per_arm}


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
        reward_window: int = 0,  # 0 = all history; >0 = sliding window (non-stationary)
        per_step_update: bool = False,  # True = update bandit per decode step, not per prompt
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
        self.reward_window = reward_window
        self.per_step_update = per_step_update
        self.reward_clip_min = 0.0
        self.reward_clip_max = 100.0

        # Populated during build
        self._drafters: list[DrafterEntry] = []
        self._buffer: PerArmBuffer | None = None
        self._distillers: list[object] = []  # one per arm
        self._vocab_size: int = 50257  # default; updated in build_router

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

    def _load_drafters(self, ctx: BuildContext) -> list[DrafterEntry]:
        """Load drafter models from config and return a list of DrafterEntry."""
        from core.models.drafter import DraftModel

        cfg = ctx.config
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]

        # Grab actual vocab size from the drafter model config
        self._vocab_size = getattr(getattr(ctx.drafter.model, "config", None), "vocab_size", 50257)

        entries = []
        default_name = cfg.drafter_model_path
        for path in drafter_paths:
            # Reuse the drafter the runner already loaded (avoids duplicate GPU memory)
            if path == default_name:
                model = ctx.drafter
                logger.info("Reusing runner drafter for %s", path)
            else:
                model = DraftModel(path, device=ctx.device)
            entries.append(
                DrafterEntry(
                    name=path,
                    model=model,
                    reward_window=self.reward_window,
                )
            )
        self._drafters = entries
        return entries

    def build_router(self, ctx: BuildContext) -> UCBBanditRouter | ThompsonSamplingRouter:
        """Build the bandit router with multiple drafter arms."""
        self._load_drafters(ctx)

        logger.info(
            "Building %s router with %d drafters: %s (vocab_size=%d, reward_window=%d)",
            self.algorithm,
            len(self._drafters),
            [d.name for d in self._drafters],
            self._vocab_size,
            self.reward_window,
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

        self._buffer = PerArmBuffer(
            num_arms=len(self._drafters),
            capacity_per_arm=max(self.buffer_capacity // len(self._drafters), 256),
            seed=getattr(ctx.config, "seed", 42),
        )
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
    # Reward helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, accepted: int, wall_ms: float) -> float:
        """Compute reward from accepted tokens and wall-clock time.

        Reward = accepted / wall_seconds, clipped to [clip_min, clip_max]
        to prevent outliers from destabilising bandit learning.
        """
        wall_seconds = max(wall_ms / 1000.0, 1e-6)
        reward = accepted / wall_seconds
        return float(
            torch.clamp(
                torch.tensor(reward),
                min=self.reward_clip_min,
                max=self.reward_clip_max,
            )
        )

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

        When ``per_step_update`` is True the bandit is updated once per
        decode step (finer-grained learning).  Otherwise (default) the
        reward is aggregated over all steps and the bandit is updated
        once per prompt.
        """
        router = ctx.router
        if router is None:
            return

        step_count = ctx.extra_state["step_count"]
        ctx.extra_state["step_count"] = step_count + 1
        arm_idx = getattr(router, "_last_selected_idx", 0)

        if self.per_step_update:
            self._update_per_step(ctx, router, step_results, step_count, prompt_index, arm_idx)
        else:
            self._update_per_prompt(ctx, router, step_results, step_count, prompt_index, arm_idx)

    def _update_per_prompt(
        self,
        ctx: DecodeContext,
        router: object,
        step_results: list,
        step_count: int,
        prompt_index: int,
        arm_idx: int,
    ) -> None:
        """Aggregate reward over all steps, update bandit once."""
        total_accepted = sum(sr.accepted_count for sr in step_results)
        total_wall_ms = sum(sr.wall_time_ms for sr in step_results)
        total_draft = sum(sr.draft_length for sr in step_results)

        if total_wall_ms <= 0:
            total_wall_ms = total_draft * 1.0 + len(step_results) * 2.0

        reward = self._compute_reward(total_accepted, total_wall_ms)

        ctx.extra_state["reward_history"].append(
            {
                "step": step_count,
                "prompt": prompt_index,
                "arm": arm_idx,
                "accepted": total_accepted,
                "wall_time_ms": total_wall_ms,
                "reward": reward,
            }
        )
        ctx.extra_state["timing_history"].append(total_wall_ms)

        router.update(reward)

        # --- Periodic distillation (phase 3+) ---
        if (
            self.enable_distillation
            and self._buffer is not None
            and step_count % self.replay_every == 0
        ):
            self._replay_for_arm(ctx, arm_idx)

    def _update_per_step(
        self,
        ctx: DecodeContext,
        router: object,
        step_results: list,
        step_count: int,
        prompt_index: int,
        arm_idx: int,
    ) -> None:
        """Update bandit once per decode step (finer-grained)."""
        total_wall_ms = 0.0
        for si, sr in enumerate(step_results):
            accepted = sr.accepted_count
            wall_ms = sr.wall_time_ms
            if wall_ms <= 0:
                wall_ms = sr.draft_length * 1.0 + 2.0

            reward = self._compute_reward(accepted, wall_ms)
            total_wall_ms += wall_ms

            ctx.extra_state["reward_history"].append(
                {
                    "step": step_count,
                    "prompt": prompt_index,
                    "sub_step": si,
                    "arm": arm_idx,
                    "accepted": accepted,
                    "wall_time_ms": wall_ms,
                    "reward": reward,
                }
            )

            router.update(reward)

        ctx.extra_state["timing_history"].append(total_wall_ms)

        # --- Periodic distillation (phase 3+) ---
        if (
            self.enable_distillation
            and self._buffer is not None
            and step_count % self.replay_every == 0
        ):
            self._replay_for_arm(ctx, arm_idx)

    def _replay_for_arm(self, ctx: DecodeContext, arm_idx: int) -> None:
        """Run a distillation step using buffered data for the given arm.

        Processes each entry sequentially through the distiller's
        gradient accumulation, then calls optimizer.step() once the
        accumulator has enough steps.  This avoids async threading
        issues and gives correct gradient behaviour.
        """
        if self._buffer is None or not self._distillers:
            return
        if arm_idx >= len(self._distillers):
            return

        batch = self._buffer.sample_for_arm(arm_idx, self.replay_batch)
        if not batch:
            return

        distiller = self._distillers[arm_idx]
        device_d = ctx.decoder.drafter.device
        device_t = ctx.decoder.target.device
        for entry in batch:
            try:
                distiller.step(
                    draft_logits=entry.draft_logits.to(device_d),
                    target_logits=entry.target_logits.to(device_t),
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

    def cleanup(self) -> None:
        """Release GPU memory held by drafters, distillers, and buffers."""
        import gc

        import torch

        # Clear per-arm distillers (hold optimizer + model references)
        for distiller in self._distillers:
            if hasattr(distiller, "optimizer"):
                distiller.optimizer = None
        self._distillers.clear()

        # Clear buffer entries (hold detached CPU tensors from logits)
        if self._buffer is not None:
            for q in self._buffer._queues.values():
                q.clear()
        self._buffer = None

        # Clear drafter entries — break references to DraftModel instances
        # that were loaded in build_router().  The runner-loaded drafter
        # (ctx.drafter) is managed by the runner's cleanup path.
        for entry in self._drafters:
            entry.model = None
        self._drafters.clear()

        # Clear other GPU-holding references
        self._last_router = None
        self._last_rewards = []
        self._last_timings = []
        if hasattr(self, "_comparison"):
            self._comparison = {}

        super().cleanup()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        reward_window: int = 0,
    ) -> None:
        super().__init__(
            algorithm="ucb",  # unused; we override build_router
            exploration=exploration,
            enable_distillation=False,
            reward_window=reward_window,
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
        self._drafters = self._load_drafters(ctx)

        logger.info(
            "Building LinUCB contextual router with %d drafters, %d features (vocab_size=%d)",
            len(self._drafters),
            self.n_features,
            self._vocab_size,
        )
        return ContextualBanditRouter(
            self._drafters,
            exploration=self.exploration,
            n_features=self.n_features,
            vocab_size=self._vocab_size,
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
# Dual-router wrapper — runs bandit + MLP in parallel
# ---------------------------------------------------------------------------


class DualRouter:
    """Wraps a bandit router and an MLP router.

    The runner calls ``select_drafter(input_ids)`` once per prompt.
    This wrapper queries **both** routers, records the MLP selection
    for later comparison, and returns the bandit's selection for
    actual decoding.

    ``update(reward)`` is forwarded to the bandit router only.
    The MLP router is trained online every ``mlp_train_every`` updates
    so it starts from learned weights rather than random initialisation.
    """

    def __init__(
        self,
        bandit_router: UCBBanditRouter | ThompsonSamplingRouter | ContextualBanditRouter,
        mlp_router: object,  # DynamicRouter
        mlp_train_every: int = 32,
        mlp_train_epochs: int = 10,
        mlp_train_lr: float = 1e-3,
    ) -> None:
        self.bandit = bandit_router
        self.mlp = mlp_router
        self._last_selected_idx: int = 0
        # Populated by select_drafter, consumed by on_decode_step
        self._mlp_selection: int = 0
        # Store last input_ids so MLP can record observations in on_decode_step
        self._last_input_ids: torch.Tensor | None = None
        # Online MLP training config
        self._mlp_train_every = mlp_train_every
        self._mlp_train_epochs = mlp_train_epochs
        self._mlp_train_lr = mlp_train_lr
        self._mlp_update_count: int = 0

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object | None, int]:
        # Cache input_ids for MLP online training
        self._last_input_ids = input_ids

        # Ask the MLP router and record its choice
        try:
            _, mlp_idx = self.mlp.select_drafter(input_ids)
            self._mlp_selection = mlp_idx
        except Exception as e:
            logger.warning("MLP router select failed: %s — defaulting to arm 0", e)
            self._mlp_selection = 0

        # Ask the bandit router (primary — its drafter is used for decoding)
        model, bandit_idx = self.bandit.select_drafter(input_ids)
        self._last_selected_idx = bandit_idx
        return model, bandit_idx

    def update(self, reward: float) -> None:
        self.bandit.update(reward)
        self._mlp_update_count += 1

        # Periodically train the MLP router on accumulated observations
        if (
            self._mlp_update_count % self._mlp_train_every == 0
            and hasattr(self.mlp, "_train_X")
            and len(self.mlp._train_X) >= 4  # need at least a few samples
        ):
            try:
                self.mlp.train_router(
                    n_epochs=self._mlp_train_epochs,
                    lr=self._mlp_train_lr,
                )
            except Exception as e:
                logger.warning("MLP online training failed: %s", e)

    def stats(self) -> dict:
        return {
            "bandit": self.bandit.stats() if hasattr(self.bandit, "stats") else {},
            "mlp": self.mlp.stats() if hasattr(self.mlp, "stats") else {},
            "mlp_online_train_count": self._mlp_update_count // self._mlp_train_every,
        }


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
        reward_window: int = 0,
    ) -> None:
        super().__init__(
            algorithm=algorithm,
            exploration=exploration,
            enable_distillation=False,
            reward_window=reward_window,
        )
        self.meta = ExperimentMeta(
            name=f"bandit_vs_mlp_{algorithm}",
            description=f"Bandit ({algorithm}) vs MLP routing comparison",
            tags=["research", "m.krylov", "routing", "bandit", "comparison"],
            dimensions=["drafter_selection"],
            depends_on=["09_+routing"],
        )
        self._mlp_router: object | None = None

    def build_router(self, ctx: BuildContext):
        """Build both bandit and MLP routers, wrapped in a DualRouter."""
        from core.extensions.routing.router import (
            DrafterSpec,
            DynamicRouter,
            RouterModel,
        )

        self._drafters = self._load_drafters(ctx)

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
            specs.append(
                DrafterSpec(
                    name=entry.name, model=entry.model, n_params=n_params, size_penalty=penalty
                )
            )

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
        return DualRouter(bandit_router, self._mlp_router)

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
        """Run bandit update AND record MLP selection for comparison.

        The DualRouter wrapper already queried both routers during
        ``select_drafter(input_ids)`` and stored the MLP selection in
        ``dual._mlp_selection``.  We read it here.
        """
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

        reward = self._compute_reward(total_accepted, total_wall_ms)

        ctx.extra_state["reward_history"].append(
            {
                "step": step_count,
                "prompt": prompt_index,
                "arm": bandit_arm,
                "accepted": total_accepted,
                "wall_time_ms": total_wall_ms,
                "reward": reward,
            }
        )
        ctx.extra_state["timing_history"].append(total_wall_ms)
        ctx.extra_state["bandit_selections"].append(bandit_arm)

        # --- Update bandit (DualRouter forwards to bandit internally) ---
        router.update(reward)

        # --- Record MLP selection from DualRouter ---
        if isinstance(router, DualRouter):
            mlp_idx = router._mlp_selection
        else:
            mlp_idx = 0
        ctx.extra_state["mlp_selections"].append(mlp_idx)

        # --- Feed observation to MLP router for online training ---
        if isinstance(router, DualRouter) and hasattr(router.mlp, "record"):
            acceptance_rate = total_accepted / max(total_draft, 1)
            last_ids = router._last_input_ids
            if last_ids is not None:
                router.mlp.record(
                    input_ids=last_ids,
                    drafter_idx=bandit_arm,
                    acceptance_rate=acceptance_rate,
                )

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Compute comparison statistics between bandit and MLP."""
        super().on_after_decode(ctx)

        bandit_selections = ctx.extra_state.get("bandit_selections", [])
        mlp_selections = ctx.extra_state.get("mlp_selections", [])

        comparison: dict = {
            "bandit_algorithm": self.algorithm,
            "n_drafters": len(self._drafters),
            "n_prompts": len(bandit_selections),
            "bandit_arm_distribution": {},
            "mlp_arm_distribution": {},
            "agreements": 0,
            "disagreements": 0,
        }

        # Count arm selections for both routers
        for arm_idx in bandit_selections:
            key = str(arm_idx)
            comparison["bandit_arm_distribution"][key] = (
                comparison["bandit_arm_distribution"].get(key, 0) + 1
            )

        for arm_idx in mlp_selections:
            key = str(arm_idx)
            comparison["mlp_arm_distribution"][key] = (
                comparison["mlp_arm_distribution"].get(key, 0) + 1
            )

        # Agreement / disagreement count
        for b, m in zip(bandit_selections, mlp_selections, strict=False):
            if b == m:
                comparison["agreements"] += 1
            else:
                comparison["disagreements"] += 1

        # Include per-router stats
        dual = getattr(self, "_last_router", None)
        if isinstance(dual, DualRouter):
            comparison["bandit_router"] = dual.bandit.stats()
            comparison["mlp_router"] = dual.mlp.stats()

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
        reward_window: int = 0,
    ) -> None:
        super().__init__(
            algorithm=algorithm,
            exploration=exploration,
            enable_distillation=False,
            reward_window=reward_window,
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

        reward = self._compute_reward(total_accepted, total_wall_ms)

        ctx.extra_state["reward_history"].append(
            {
                "step": step_count,
                "prompt": prompt_index,
                "arm": arm_idx,
                "dataset": dataset_name,
                "accepted": total_accepted,
                "wall_time_ms": total_wall_ms,
                "reward": reward,
            }
        )
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

    def __init__(self, *, reward_window: int = 0) -> None:
        super().__init__(algorithm="thompson", reward_window=reward_window)
        self.meta.name = "bandit_vs_mlp_thompson"


class BanditMultiDatasetThompsonExperiment(BanditMultiDatasetExperiment):
    """Thompson Sampling routing across multiple datasets."""

    def __init__(self, *, reward_window: int = 0) -> None:
        super().__init__(algorithm="thompson", reward_window=reward_window)
        self.meta.name = "bandit_multidataset_thompson"


# ---------------------------------------------------------------------------
# Phase 5 — Exploration parameter sweep
# ---------------------------------------------------------------------------


class BanditExplorationSweepExperiment(BanditRoutingExperiment):
    """Systematic sweep of exploration parameters across UCB and LinUCB.

    Runs the full decode loop multiple times, once per exploration value,
    and reports comparative metrics so you can pick the best setting.

    Sweep dimensions
    ----------------
    - UCB1: sweep `c` (exploration coefficient)
    - LinUCB: sweep `α` (exploration coefficient)
    - Both algorithms can be swept in the same experiment

    Metrics produced (in `exploration_sweep` dict)
    ----------------------------------------------
    - Per-`c`/`α`: mean_reward, std_reward, acceptance_rate, tokens_per_sec,
      arm_distribution, convergence_sample (first sample where policy
      stabilises on one arm for `convergence_window` consecutive steps)
    - Best value per algorithm (highest mean_reward)
    """

    # Default sweep ranges — customise via constructor
    DEFAULT_UCB_C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
    DEFAULT_LINUCB_ALPHA_VALUES = [0.1, 1.0, 5.0]

    def __init__(
        self,
        *,
        ucb_c_values: list[float] | None = None,
        linucb_alpha_values: list[float] | None = None,
        convergence_window: int = 3,  # consecutive same-arm to count as converged
        reward_window: int = 0,
    ) -> None:
        super().__init__(
            algorithm="ucb",
            exploration=2.0,  # placeholder; overridden per-sweep
            enable_distillation=False,
            reward_window=reward_window,
        )
        self.meta = ExperimentMeta(
            name="bandit_exploration_sweep",
            description=(
                "Exploration parameter sweep: UCB c-values + LinUCB α-values. "
                "Compares mean_reward, arm distribution, convergence per setting."
            ),
            tags=["research", "m.krylov", "routing", "bandit", "sweep"],
            dimensions=["drafter_selection", "exploration"],
            depends_on=["01_baseline"],
        )
        self.ucb_c_values = ucb_c_values or self.DEFAULT_UCB_C_VALUES
        self.linucb_alpha_values = linucb_alpha_values or self.DEFAULT_LINUCB_ALPHA_VALUES
        self.convergence_window = convergence_window

        # Sweep results
        self._sweep_results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Override build_router to support both UCB and LinUCB
    # ------------------------------------------------------------------

    def build_router(self, ctx: BuildContext) -> UCBBanditRouter | ContextualBanditRouter:
        """Build router for the current sweep value.

        Called once per sweep iteration by run().
        The algorithm is set via `_current_algorithm` and `_current_exploration`.
        Drafter models are pre-loaded in `run()` and cached in `_preloaded_drafters`.
        """
        cfg = ctx.config
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]

        self._vocab_size = getattr(getattr(ctx.drafter.model, "config", None), "vocab_size", 50257)

        self._drafters = []
        default_name = cfg.drafter_model_path
        for path in drafter_paths:
            # Use pre-loaded model (cached across sweep iterations)
            if path == default_name:
                model = ctx.drafter  # runner-loaded primary drafter
            else:
                model = self._preloaded_drafters.get(path)
                if model is None:
                    from core.models.drafter import DraftModel

                    model = DraftModel(path, device=ctx.device)
                    self._preloaded_drafters[path] = model
            self._drafters.append(
                DrafterEntry(
                    name=path,
                    model=model,
                    reward_window=self.reward_window,
                )
            )

        algo = getattr(self, "_current_algorithm", "ucb")
        exploration = getattr(self, "_current_exploration", 2.0)

        if algo == "linucb":
            return ContextualBanditRouter(
                self._drafters,
                exploration=exploration,
                n_features=8,
                vocab_size=self._vocab_size,
            )
        else:
            return UCBBanditRouter(self._drafters, exploration=exploration)

    # ------------------------------------------------------------------
    # Override run() to loop over exploration values
    # ------------------------------------------------------------------

    def run(self, runner: object) -> object:  # ExperimentRunner -> ExperimentResult
        """Run the full decode loop for each exploration value.

        For each (algorithm, exploration) pair:
        1. Build router with that exploration value
        2. Run all prompts
        3. Collect metrics
        4. Compare across values
        """
        import gc
        import random as _random

        import numpy as np

        from experiments.base import ExperimentResult

        logger = logging.getLogger(__name__)

        cfg = self.get_config()
        # Apply CLI overrides
        for key, value in self._overrides.items():
            setattr(cfg, key, value)
        cfg_dict = runner._asdict_config(cfg)

        # Deterministic seeding
        seed = getattr(cfg, "seed", 42)
        torch.manual_seed(seed)
        _random.seed(seed)
        np.random.seed(seed)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(seed)

        # Load models once (shared across sweep iterations)
        drafter, target = runner._build_models(cfg)

        # Pre-load extra drafter models (for multi-drafter sweeps)
        drafter_paths = getattr(cfg, "drafter_model_paths", [])
        if not drafter_paths:
            drafter_paths = [cfg.drafter_model_path]
        self._preloaded_drafters: dict[str, object] = {
            cfg.drafter_model_path: drafter,
        }
        from core.models.drafter import DraftModel

        for path in drafter_paths:
            if path not in self._preloaded_drafters:
                logger.info("Pre-loading extra drafter: %s", path)
                self._preloaded_drafters[path] = DraftModel(path, device=runner.device)

        # Build translator + cache (shared)
        build_ctx = BuildContext(
            device=runner.device,
            drafter=drafter,
            target=target,
            config=cfg,
            components={},
        )
        translator = self.build_translator(build_ctx)
        build_ctx.components["translator"] = translator
        cache = self.build_cache(build_ctx)
        build_ctx.components["cache"] = cache

        # Load dataset once
        prompts = runner._load_dataset(cfg)
        max_new_tokens = getattr(cfg, "max_new_tokens", 128)

        # Build decoder (drafter/target can be swapped via router)
        from core.decoder.speculative import SpeculativeDecoder

        draft_length = getattr(cfg, "draft_length", 5)
        decoder = SpeculativeDecoder(
            drafter=drafter,
            target=target,
            translator=translator,
            cache=cache,
            draft_length=draft_length,
        )

        # Benchmark collector (shared)
        from benchmarks.metrics.collector import BenchmarkCollector

        # MLflow setup
        runner._setup_mlflow(cfg)

        # Baseline TPS not measured in sweep (no autoregressive baseline method)
        baseline_tps = 0.0

        # ----------------------------------------------------------------
        # Sweep loop
        # ----------------------------------------------------------------
        sweep_configs = []
        for c_val in self.ucb_c_values:
            sweep_configs.append(("ucb", c_val))
        for alpha_val in self.linucb_alpha_values:
            sweep_configs.append(("linucb", alpha_val))

        total_sweeps = len(sweep_configs)
        logger.info("Starting exploration sweep: %d configurations", total_sweeps)

        for sweep_idx, (algo, exploration) in enumerate(sweep_configs, 1):
            sweep_name = f"{algo}_{exploration}"
            logger.info(
                "Sweep %d/%d: algorithm=%s exploration=%.2f",
                sweep_idx,
                total_sweeps,
                algo,
                exploration,
            )

            # Set current sweep params for build_router
            self._current_algorithm = algo
            self._current_exploration = exploration

            # Build router for this sweep value
            router = self.build_router(build_ctx)

            # Build decode context
            collector = BenchmarkCollector(name=f"{self.meta.name}_{sweep_name}")
            collector.set_baseline_tps(baseline_tps)

            decode_ctx = DecodeContext(
                decoder=decoder,
                collector=collector,
                config=cfg,
                distiller=None,
                router=router,
                adaptive_fn=None,
            )

            # Reset per-run state
            self._last_rewards = []
            self._last_timings = []

            # Before decode hook
            self.on_before_decode(decode_ctx)

            # Decode loop
            for i, (input_ids, prompt_len) in enumerate(prompts):
                input_ids = input_ids.to(runner.device)

                # GPU memory sampling (for gpu_mem_peak_gb metric)
                if torch.cuda.is_available():
                    collector.sample_gpu_memory(runner.device)

                # Router selection
                selected_drafter, _selected_idx = router.select_drafter(input_ids)
                if selected_drafter is not None:
                    decoder.drafter = selected_drafter

                with collector.record_sequence(prompt_len=prompt_len) as seq_rec:
                    decoder.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        adaptive_length_fn=None,
                        distiller=None,
                        rng=torch_rng,
                    )
                    for sr in decoder._step_results[-max_new_tokens:]:
                        seq_rec.add_step(
                            draft_len=sr.draft_length,
                            accepted=len(sr.accepted_tokens),
                            cache_hit=sr.cache_hit,
                        )

                _step_results = list(decoder._step_results)
                decoder._step_results.clear()

                # Hook: after each prompt
                self.on_decode_step(decode_ctx, _step_results, i)

            # After decode
            self.on_after_decode(decode_ctx)

            # Collect metrics
            summary = collector.summary()
            summary = self.on_extra_metrics(summary)
            collector.clear()

            # Compute convergence point (skip round-robin phase)
            rewards = self._last_rewards
            n_arms = len(router.arms)
            convergence_sample = self._find_convergence(
                rewards, self.convergence_window, skip=n_arms
            )

            # Build per-sweep result
            self._sweep_results[sweep_name] = {
                "algorithm": algo,
                "exploration": exploration,
                "mean_reward": summary.get("bandit_mean_reward", 0.0),
                "std_reward": summary.get("bandit_std_reward", 0.0),
                "acceptance_rate": summary.get("acceptance_rate", 0.0),
                "avg_accepted_tokens": summary.get("avg_accepted_tokens", 0.0),
                "avg_draft_length": summary.get("avg_draft_length", 0.0),
                "tokens_per_sec": summary.get("tokens_per_sec", 0.0),
                "wall_time_total_s": summary.get("wall_time_total_s", 0.0),
                "wall_clock_speedup": summary.get("wall_clock_speedup", None),
                "gpu_mem_peak_gb": summary.get("gpu_mem_peak_gb", 0.0),
                "gpu_mem_mean_gb": summary.get("gpu_mem_mean_gb", 0.0),
                "convergence_sample": convergence_sample,
                "n_samples": len(rewards),
                "router_stats": summary.get("bandit_router", {}),
            }

            # Free GPU memory between sweeps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ----------------------------------------------------------------
        # Compile sweep summary
        # ----------------------------------------------------------------
        self._compile_sweep_summary()

        # Final MLflow logging
        runner._log_mlflow_final(cfg, self._final_summary)

        return ExperimentResult(
            meta=self.meta,
            config=cfg_dict,
            metrics=self._final_summary,
        )

    def _find_convergence(self, rewards: list[dict], window: int, skip: int = 0) -> int | None:
        """Find the first sample index where the policy stabilises.

        Stabilised = same arm selected for `window` consecutive samples.
        `skip` samples are ignored at the start (e.g. round-robin phase).
        Returns the index of the first sample in the stable run, or None.
        """
        start = skip
        if len(rewards) - start < window:
            return None

        for i in range(start, len(rewards) - window + 1):
            arms = [rewards[j]["arm"] for j in range(i, i + window)]
            if len(set(arms)) == 1:
                return i
        return None

    def _compile_sweep_summary(self) -> None:
        """Build the final summary dict from per-sweep results.

        Also flattens the best-overall metrics to the top level so the CLI
        summary table (which reads top-level keys) can display them correctly.
        """
        self._final_summary = {
            "exploration_sweep": self._sweep_results,
            "n_sweep_configs": len(self._sweep_results),
        }

        # Best per algorithm
        for algo in ("ucb", "linucb"):
            algo_results = {k: v for k, v in self._sweep_results.items() if v["algorithm"] == algo}
            if algo_results:
                best_key = max(algo_results, key=lambda k: algo_results[k]["mean_reward"])
                best = algo_results[best_key]
                self._final_summary[f"best_{algo}"] = {
                    "exploration": best["exploration"],
                    "mean_reward": best["mean_reward"],
                    "acceptance_rate": best["acceptance_rate"],
                    "tokens_per_sec": best["tokens_per_sec"],
                    "convergence_sample": best["convergence_sample"],
                }

        # Cross-algorithm best
        if self._sweep_results:
            best_key = max(
                self._sweep_results,
                key=lambda k: self._sweep_results[k]["mean_reward"],
            )
            best = self._sweep_results[best_key]
            self._final_summary["best_overall"] = {
                "config": best_key,
                "mean_reward": best["mean_reward"],
            }

            # Flatten best-overall metrics to top level for CLI summary table
            self._final_summary["acceptance_rate"] = best["acceptance_rate"]
            self._final_summary["avg_accepted_tokens"] = best.get("avg_accepted_tokens", 0.0)
            self._final_summary["avg_draft_length"] = best.get("avg_draft_length", 0.0)
            self._final_summary["tokens_per_sec"] = best["tokens_per_sec"]
            self._final_summary["wall_time_total_s"] = best["wall_time_total_s"]
            self._final_summary["bandit_mean_reward"] = best["mean_reward"]
            self._final_summary["bandit_std_reward"] = best["std_reward"]
            # GPU memory: aggregate across all sweep configs (peak of peaks)
            all_gpu_peaks = [
                v.get("gpu_mem_peak_gb", 0.0)
                for v in self._sweep_results.values()
                if v.get("gpu_mem_peak_gb", 0.0) > 0
            ]
            self._final_summary["gpu_mem_peak_gb"] = max(all_gpu_peaks, default=0.0)
            self._final_summary["gpu_mem_mean_gb"] = sum(
                v.get("gpu_mem_mean_gb", 0.0) for v in self._sweep_results.values()
            ) / max(len(self._sweep_results), 1)
            # Wall-clock speedup (from best config)
            speedup = best.get("wall_clock_speedup")
            if speedup is not None:
                self._final_summary["wall_clock_speedup"] = speedup

    def on_extra_metrics(self, summary: dict) -> dict:
        """Augment with bandit stats (reuse parent logic)."""
        return super().on_extra_metrics(summary)


__all__ = [
    "BanditContextualDistillExperiment",
    "BanditContextualExperiment",
    "BanditExplorationSweepExperiment",
    "BanditMultiDatasetExperiment",
    "BanditMultiDatasetThompsonExperiment",
    "BanditThompsonDistillExperiment",
    "BanditThompsonExperiment",
    "BanditUCBDistillExperiment",
    "BanditUCBExperiment",
    "BanditVsMLPExperiment",
    "BanditVsMLPThompsonExperiment",
]
