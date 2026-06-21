"""Built-in: Online Distillation experiment.

Adds online distillation during speculative decoding to improve
the drafter model by matching target model distributions.
Corresponds to ``04_+online_distil`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class OnlineDistillExperiment(BaseExperiment):
    """Baseline with online distillation (no replay, no contrastive).

    During decoding, accepted/rejected draft tokens provide training
    signal to update the drafter via KL divergence + N-gram NLL loss.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="04_+online_distil",
                description="Online distillation (KL + N-gram NLL)",
                tags=["distillation"],
                dimensions=["distillation_strategy"],
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
            use_online_distil=True,
            use_replay=False,
            use_contrastive=False,
            use_speedup_adaptive=False,
            use_dynamic_routing=False,
            use_universal_drafter=False,
        )

    @staticmethod
    def _apply_lora(drafter, cfg) -> None:
        """Wrap the drafter model with PEFT LoRA adapters."""
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(cfg, "lora_rank", 8),
            lora_alpha=getattr(cfg, "lora_alpha", 16.0),
            lora_dropout=getattr(cfg, "lora_dropout", 0.05),
            target_modules=getattr(
                cfg, "lora_target_modules", ["q_proj", "v_proj"]
            ),
        )
        drafter.model = get_peft_model(drafter.model, config)
        trainable = sum(p.numel() for p in drafter.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in drafter.model.parameters())
        logger.info(
            "LoRA applied: rank=%d alpha=%.1f trainable=%d/%d (%.2f%%)",
            getattr(cfg, "lora_rank", 8),
            getattr(cfg, "lora_alpha", 16.0),
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )

    def build_distiller(self, ctx: BuildContext):
        """Build OnlineDistiller with the drafter and translator."""
        import torch.optim as optim

        from core.distillation.online import OnlineDistiller

        translator = ctx.components.get("translator")
        drafter = ctx.drafter

        # Prepare drafter for training
        cfg = ctx.config
        if getattr(cfg, "use_lora", False):
            self._apply_lora(drafter, cfg)

        drafter.model.train()
        if not getattr(cfg, "use_lora", False):
            for p in drafter.model.parameters():
                p.requires_grad_(True)

        opt = optim.Adam(drafter.model.parameters(), lr=getattr(cfg, "distil_lr", 1e-5))
        distiller = OnlineDistiller(
            drafter_model=drafter,
            translator=translator,
            optimizer=opt,
            lambda_ngram=getattr(cfg, "lambda_ngram", 0.5),
            use_lora=getattr(cfg, "use_lora", False),
        )
        logger.info(
            "OnlineDistiller ready: lr=%s lambda_ngram=%s",
            getattr(cfg, "distil_lr", 1e-5),
            getattr(cfg, "lambda_ngram", 0.5),
        )
        return distiller
