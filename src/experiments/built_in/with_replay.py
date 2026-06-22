"""Built-in: Replay Buffer experiment.

Wraps the online distiller with a replay buffer for experience replay.
Supports both FIFO and prioritized sampling strategies.
Corresponds to ``05_+replay_fifo`` and ``06_+replay_prio`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from experiments.built_in.with_online_distil import OnlineDistillExperiment

if TYPE_CHECKING:
    from experiments.base import BuildContext
    from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class ReplayExperiment(OnlineDistillExperiment):
    """Online distillation with experience replay buffer.

    Stores speculative decoding traces and replays them periodically
    for more stable training.

    Parameters
    ----------
    strategy :
        Sampling strategy: ``"fifo"`` for uniform random from
        chronological buffer, or ``"prioritized"`` for weighted
        sampling by (1 - acceptance_rate).
    """

    def __init__(self, strategy: Literal["fifo", "prioritized"] = "fifo") -> None:
        super().__init__()
        self._strategy = strategy
        self.meta.name = f"05_+replay_{strategy[:4]}" if strategy == "fifo" else "06_+replay_prio"
        self.meta.description = f"Online distillation + ReplayBuffer({strategy})"
        self.meta.tags = ["distillation", "replay", strategy]
        self.meta.dimensions = ["distillation_strategy", "replay_strategy"]
        self.meta.depends_on = ["04_+online_distil"]

    def get_config(self) -> ExperimentConfig:
        cfg = super().get_config()
        cfg.name = self.meta.name
        cfg.use_replay = True
        cfg.replay_strategy = self._strategy
        return cfg

    def build_distiller(self, ctx: BuildContext):
        """Build OnlineDistiller wrapped with ReplayBuffer."""
        distiller = super().build_distiller(ctx)
        if distiller is None:
            return None

        from core.extensions.replay.buffer import ReplayBuffer, ReplayDistiller

        cfg = ctx.config
        buf = ReplayBuffer(
            capacity=getattr(cfg, "replay_capacity", 4096),
            strategy=self._strategy,
        )
        replay_distiller = ReplayDistiller(
            distiller=distiller,
            buffer=buf,
            replay_every=getattr(cfg, "replay_every", 32),
            replay_batch=getattr(cfg, "replay_batch", 8),
            target_model=ctx.target,  # for recomputing target_logits during replay
        )
        logger.info(
            "ReplayDistiller ready: strategy=%s capacity=%d",
            self._strategy,
            getattr(cfg, "replay_capacity", 4096),
        )
        return replay_distiller
