"""Built-in: Universal Drafter experiment.

A single drafter model trained to draft for multiple target LLM families
via learnable target-specific embeddings injected at each transformer layer.
Corresponds to ``10_+universal`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging

import torch

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class _UniversalDrafterAdapter:
    """Thin adapter to make UniversalDrafter match the DraftModel interface."""

    def __init__(
        self,
        base,
        universal: torch.nn.Module,
        target_model_path: str,
    ) -> None:
        self.base = base
        self.universal = universal
        self.tokenizer = base.tokenizer
        self.model = base.model
        self._target_model_path = target_model_path

    def draft(
        self,
        context: torch.Tensor,
        k: int,
        distill: bool = False,
        temperature: float = 1.0,
    ) -> tuple[list[int], torch.Tensor]:
        """Forward to base (distillation) or universal (inference)."""
        if distill:
            return self.base.draft(context, k, distill=True, temperature=temperature)
        with torch.no_grad():
            return self.universal.draft(context, k, target_name=self._target_model_path)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.universal.base_model(input_ids).logits.squeeze(0)


class UniversalDrafterExperiment(BaseExperiment):
    """Universal drafter with target-conditioned adapter layers.

    A single drafter serves multiple target model families by injecting
    learnable target embeddings at every transformer layer via forward hooks.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="10_+universal",
                description="UniversalDrafter (multi-target with target embeddings)",
                tags=["universal", "multi-target"],
                dimensions=["drafter_architecture"],
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
            use_speedup_adaptive=False,
            use_dynamic_routing=False,
            use_universal_drafter=True,
        )

    def build_universal_drafter(self, ctx: BuildContext):
        """Build UniversalDrafter wrapped in an adapter."""
        from core.extensions.multitarget.universal_drafter import UniversalDrafter

        cfg = ctx.config
        # Use the config's target_model_path as the target family name.
        # This allows --tiny override (e.g. facebook/opt-350m) to work.
        target_names = [cfg.target_model_path]
        universal = UniversalDrafter(
            base_model_name=cfg.drafter_model_path,
            target_names=target_names,
            d_model=ctx.drafter.model.config.hidden_size,
            trainable_base=False,
            device=ctx.device,
            dtype=ctx.drafter.model.dtype,
        )

        adapter = _UniversalDrafterAdapter(
            base=ctx.drafter,
            universal=universal,
            target_model_path=cfg.target_model_path,
        )
        logger.info("UniversalDrafter ready with targets: %s", target_names)
        return adapter
