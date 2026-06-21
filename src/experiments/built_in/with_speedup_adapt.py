"""Built-in: Speedup-Aware Adaptive Drafting experiment.

Dynamically selects draft length k based on a learned speedup predictor
instead of using a fixed draft length.
Corresponds to ``08_+speedup_adapt`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class SpeedupAdaptiveExperiment(BaseExperiment):
    """Adaptive draft length via SpeedupPredictor.

    A small MLP predicts expected tokens/sec speedup for each candidate
    draft length k, and the controller selects argmax_k(predicted_speedup).
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="08_+speedup_adapt",
                description="Speedup-Aware Adaptive Drafting (learned k selection)",
                tags=["adaptive"],
                dimensions=["draft_length_strategy"],
                depends_on=["01_baseline"],
            )
        )

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
        )

    def build_adaptive_controller(self, ctx: BuildContext):
        """Build SpeedupPredictor + AdaptiveDraftController."""
        from core.extensions.adaptive.speedup_predictor import (
            AdaptiveDraftController,
            SpeedupPredictor,
        )

        cfg = ctx.config
        pred = SpeedupPredictor(
            d_hidden=ctx.drafter.model.config.hidden_size,
            k_max=getattr(cfg, "k_max", 8),
        ).to(ctx.device)
        controller = AdaptiveDraftController(
            pred, ctx.drafter, getattr(cfg, "k_min", 1), getattr(cfg, "k_max", 8)
        )
        logger.info(
            "AdaptiveDraftController ready: k_min=%d k_max=%d",
            getattr(cfg, "k_min", 1),
            getattr(cfg, "k_max", 8),
        )
        return controller
