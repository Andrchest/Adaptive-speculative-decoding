"""Built-in: Contrastive Loss experiment.

Adds contrastive rejection learning on top of online distillation.
Rejected draft tokens serve as hard negative examples via InfoNCE loss.
Corresponds to ``07_+contrastive`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from experiments.built_in.with_online_distil import OnlineDistillExperiment

if TYPE_CHECKING:
    from experiments.base import BuildContext
    from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class ContrastiveExperiment(OnlineDistillExperiment):
    """Online distillation with contrastive rejection learning.

    In addition to KL divergence and N-gram NLL, rejected draft tokens
    are treated as hard negatives in an InfoNCE contrastive loss,
    pushing the drafter away from tokens the target model rejects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.meta.name = "07_+contrastive"
        self.meta.description = "Online distillation + Contrastive Rejection Learning (InfoNCE)"
        self.meta.tags = ["distillation", "contrastive"]
        self.meta.dimensions = ["distillation_strategy"]
        self.meta.depends_on = ["04_+online_distil"]

    def get_config(self) -> ExperimentConfig:
        cfg = super().get_config()
        cfg.name = self.meta.name
        cfg.use_contrastive = True
        return cfg

    def build_distiller(self, ctx: BuildContext):
        """Build OnlineDistiller with ContrastiveLoss attached."""
        distiller = super().build_distiller(ctx)
        if distiller is None:
            return None

        from core.extensions.contrastive.loss import ContrastiveLoss

        cfg = ctx.config
        distiller.set_contrastive_loss(
            ContrastiveLoss(
                lambda_nll=getattr(cfg, "lambda_ngram", 0.5),
                lambda_contrastive=getattr(cfg, "lambda_contrastive", 0.1),
                temperature=0.07,
            )
        )
        logger.info(
            "ContrastiveLoss attached: lambda_contrastive=%.2f temperature=%.2f",
            getattr(cfg, "lambda_contrastive", 0.1),
            0.07,
        )
        return distiller
