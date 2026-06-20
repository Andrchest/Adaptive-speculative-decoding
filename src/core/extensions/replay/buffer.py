"""
extensions/replay/buffer.py

Replay buffer for continual online distillation.

Stores speculative decoding traces and replays them during training.

Priority:  p = 1 - acceptance_rate  (harder examples → higher priority)

Strategies:
  - fifo        : uniform random from chronological FIFO buffer
  - prioritized : weighted sampling by (1 - acceptance_rate)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    prompt_ids: list[int]
    draft_logits: torch.Tensor  # (k, drafter_vocab)
    target_logits: torch.Tensor  # (k, target_vocab)
    draft_tokens: list[int]
    accepted_tokens: list[int]
    accepted_mask: list[bool]  # position-level, length == len(draft_tokens)
    rejected_tokens: list[int]
    acceptance_rate: float

    @property
    def priority(self) -> float:
        """Higher priority for harder (low-acceptance) examples."""
        return 1.0 - self.acceptance_rate + 1e-6


class ReplayBuffer:
    """
    Parameters
    ----------
    capacity    : maximum number of traces to store
    strategy    : 'fifo' | 'prioritized'
    alpha       : exponent for priority (0 = uniform, 1 = full priority)
    beta        : importance-sampling correction exponent (for prioritized)
    """

    def __init__(
        self,
        capacity: int = 4096,
        strategy: str = "prioritized",
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.capacity = capacity
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta

        self._buffer: list[Trace] = []
        self._ptr: int = 0
        logger.info("ReplayBuffer initialized: capacity=%d strategy=%s", capacity, strategy)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, trace: Trace) -> None:
        """Add a trace (overwrites oldest if full)."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(trace)
        else:
            self._buffer[self._ptr] = trace
        self._ptr = (self._ptr + 1) % self.capacity
        logger.debug(
            "Buffer add: size=%d/%d strategy=%s", len(self._buffer), self.capacity, self.strategy
        )

    def sample(self, batch_size: int) -> list[Trace]:
        """Sample a batch of traces."""
        if len(self._buffer) == 0:
            return []
        batch_size = min(batch_size, len(self._buffer))

        if self.strategy == "fifo":
            return random.sample(self._buffer, batch_size)
        elif self.strategy == "prioritized":
            return self._priority_sample(batch_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def __len__(self) -> int:
        return len(self._buffer)

    def mean_acceptance_rate(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(t.acceptance_rate for t in self._buffer) / len(self._buffer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _priority_sample(self, batch_size: int) -> list[Trace]:
        priorities = np.array([t.priority**self.alpha for t in self._buffer], dtype=np.float32)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False, p=probs)
        return [self._buffer[i] for i in indices]


# ------------------------------------------------------------------
# Replay distiller — orchestrates sampling + distillation steps
# ------------------------------------------------------------------


class ReplayDistiller:
    """
    Wraps OnlineDistiller with a replay buffer.

    After every *replay_every* live steps, samples *replay_batch* traces
    from the buffer and replays them through the distiller.

    Parameters
    ----------
    distiller     : OnlineDistiller
    buffer        : ReplayBuffer
    replay_every  : how many live steps between replay phases
    replay_batch  : how many traces to replay per phase
    """

    def __init__(
        self,
        distiller,
        buffer: ReplayBuffer,
        replay_every: int = 32,
        replay_batch: int = 8,
    ) -> None:
        self.distiller = distiller
        self.buffer = buffer
        self.replay_every = replay_every
        self.replay_batch = replay_batch
        self._live_steps = 0
        logger.info(
            "ReplayDistiller initialized: replay_every=%d replay_batch=%d",
            replay_every,
            replay_batch,
        )

    def step(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_tokens: list,
        accepted_mask: list,
        prompt_ids: list | None = None,
    ) -> float | None:
        accepted_tokens = [t for t, a in zip(draft_tokens, accepted_mask, strict=False) if a]
        rejected_tokens = [t for t, a in zip(draft_tokens, accepted_mask, strict=False) if not a]
        return self.live_step(
            draft_logits=draft_logits,
            target_logits=target_logits,
            draft_tokens=draft_tokens,
            accepted_mask=accepted_mask,
            accepted_tokens=accepted_tokens,
            rejected_tokens=rejected_tokens,
            prompt_ids=prompt_ids or [],
        )

    def live_step(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_tokens: list,
        accepted_mask: list,
        accepted_tokens: list,
        rejected_tokens: list,
        prompt_ids: list,
    ) -> float | None:
        """Process one live decoding step with full trace info (legacy interface)."""
        acc_rate = sum(accepted_mask) / max(len(accepted_mask), 1)
        trace = Trace(
            prompt_ids=prompt_ids,
            draft_logits=draft_logits.detach().cpu(),
            target_logits=target_logits.detach().cpu(),
            draft_tokens=draft_tokens,
            accepted_tokens=accepted_tokens,
            accepted_mask=accepted_mask,
            rejected_tokens=rejected_tokens,
            acceptance_rate=acc_rate,
        )
        self.buffer.add(trace)

        # Live distillation step
        loss = self.distiller.step(
            draft_logits=draft_logits,
            target_logits=target_logits,
            draft_tokens=draft_tokens,
            accepted_mask=accepted_mask,
        )

        self._live_steps += 1
        if self._live_steps % self.replay_every == 0:
            self._replay()

        return loss

    def training_stats(self) -> dict:
        return self.distiller.training_stats()

    def _replay(self) -> None:
        """Replay sampled traces through the distiller."""
        traces = self.buffer.sample(self.replay_batch)
        logger.info("Replay phase: %d traces from buffer (size=%d)", len(traces), len(self.buffer))
        device = next(self.distiller.drafter.model.parameters()).device
        for t in traces:
            loss = self.distiller._compute_loss(
                draft_logits=t.draft_logits.to(device),
                target_logits=t.target_logits.to(device),
                draft_tokens=t.draft_tokens,
                accepted_mask=t.accepted_mask,
            )
            if loss is not None:
                # Keep on GPU: .item() in _update_weights handles the scalar transfer
                self.distiller._accum_loss = self.distiller._accum_loss + loss.detach()
                self.distiller._accum_count += 1
                if self.distiller._accum_count >= self.distiller.accum_steps:
                    self.distiller._update_weights()
