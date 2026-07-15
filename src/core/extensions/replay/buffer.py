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
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    """A single speculative decoding trajectory.

    Stores only token IDs (not logits) to keep memory footprint low.
    Logits are recomputed during replay via a forward pass.
    """

    prompt_ids: list[int]
    prompt_len: int  # original prompt length (needed to slice logits correctly)
    draft_tokens: list[int]
    accepted_tokens: list[int]
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

        self._buffer: deque[Trace] = deque(maxlen=capacity)
        self._ptr: int = 0
        logger.info("ReplayBuffer initialized: capacity=%d strategy=%s", capacity, strategy)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, trace: Trace) -> None:
        """Add a trace (deque auto-evicts oldest when at capacity)."""
        self._buffer.append(trace)
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

    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

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
    target_model  : optional TargetModel for recomputing target_logits
                    during replay (stored logits are no longer kept)
    """

    def __init__(
        self,
        distiller,
        buffer: ReplayBuffer,
        replay_every: int = 32,
        replay_batch: int = 8,
        target_model=None,
    ) -> None:
        self.distiller = distiller
        self.buffer = buffer
        self.replay_every = replay_every
        self.replay_batch = replay_batch
        self._target_model = target_model
        self._live_steps = 0
        logger.info(
            "ReplayDistiller initialized: replay_every=%d replay_batch=%d",
            replay_every,
            replay_batch,
        )

    def set_contrastive_loss(self, loss_module) -> None:
        """
        Attach a ContrastiveLoss module to be added to subsequent steps.
        """
        self.distiller.set_contrastive_loss(loss_module)
        logger.info("ReplayDistiller contrastive loss attachment")

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
        # CRITICAL: Copy all lists to prevent mutation after storage.
        # The caller (speculative decoder) mutates ctx_list after this returns,
        # which would corrupt prompt_ids if we stored a reference.
        trace = Trace(
            prompt_ids=list(prompt_ids),
            prompt_len=len(prompt_ids),
            draft_tokens=list(draft_tokens),
            accepted_tokens=list(accepted_tokens),
            rejected_tokens=list(rejected_tokens),
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
        """
        Replay sampled traces through the distiller.

        Since Trace no longer stores logits (memory fix), both draft
        and target logits are recomputed via forward passes over the
        stored prompt_ids + draft_tokens.

        The ``accepted_mask`` is reconstructed POSITIONALLY from
        ``len(t.accepted_tokens)`` rather than via set membership:
        speculative decoding accepts a contiguous prefix of the draft,
        so ``accepted_mask[i] = (i < len(t.accepted_tokens))``.
        """
        traces = self.buffer.sample(self.replay_batch)
        logger.info(
            "Replay phase: %d traces from buffer (size=%d)",
            len(traces),
            len(self.buffer),
        )
        if not traces:
            return
        device = next(self.distiller.drafter.model.parameters()).device

        for t in traces:
            k = len(t.draft_tokens)
            if k == 0:
                continue

            # Reconstruct input: (prompt + draft_tokens[:-1])
            prompt_ids = torch.tensor(t.prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            draft_tokens_tensor = torch.tensor(
                t.draft_tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
            if k > 1:
                input_for_drafter = torch.cat([prompt_ids, draft_tokens_tensor[:, :-1]], dim=1)
            else:
                input_for_drafter = prompt_ids

            # 1) Re-run drafter forward to get fresh grad-enabled logits.
            try:
                out = self.distiller.drafter.model(input_for_drafter)
            except Exception as e:
                logger.warning("Replay: skipping trace due to drafter forward error: %s", e)
                continue

            # logits at positions [prompt_len-1, prompt_len, ..., prompt_len+k-2]
            fresh_logits = out.logits[0, t.prompt_len - 1 : t.prompt_len - 1 + k, :]
            # (k, drafter_vocab) — has grad_fn

            # 2) Recompute target_logits if target_model is available.
            if self._target_model is not None:
                try:
                    with torch.no_grad():
                        target_logits, _ = self._target_model.verify(
                            input_for_drafter, t.draft_tokens
                        )
                        target_logits = target_logits[:k]  # only the k draft positions
                except Exception as e:
                    logger.warning("Replay: target forward failed, skipping trace: %s", e)
                    continue
            else:
                # Fallback: skip target-specific distillation signal.
                # Draft prediction learning still works via KL on drafter logits.
                logger.debug("Replay: no target_model provided, skipping target_logits for trace")
                continue

            # Reconstruct accepted_mask positionally (C6 fix).
            n_accepted = len(t.accepted_tokens)
            accepted_mask = [i < n_accepted for i in range(k)]

            # Route through the standard step() so backward and
            # accumulation are handled consistently with live steps.
            self.distiller.step(
                draft_logits=fresh_logits,
                target_logits=target_logits,
                draft_tokens=t.draft_tokens,
                accepted_mask=accepted_mask,
            )
