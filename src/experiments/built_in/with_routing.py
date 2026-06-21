"""Built-in: Dynamic Router experiment.

Uses a lightweight router to select the most appropriate drafter model
per prompt from a pool of drafters of different sizes.
Corresponds to ``09_+routing`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

import logging

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class RoutingExperiment(BaseExperiment):
    """Multi-drafter routing via DynamicRouter.

    A small MLP router maps prompt embeddings to drafter indices,
    selecting the most efficient drafter per prompt based on predicted
    acceptance rate and model size penalty.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="09_+routing",
                description="Dynamic Multi-Drafter Router (MLP-based selection)",
                tags=["routing"],
                dimensions=["drafter_selection"],
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
            use_dynamic_routing=True,
            use_universal_drafter=False,
        )

    def build_router(self, ctx: BuildContext):
        """Build DynamicRouter with multiple drafter specs."""
        from core.extensions.routing.router import (
            DrafterSpec,
            DynamicRouter,
            RouterModel,
        )

        d_input = ctx.drafter.model.config.hidden_size
        router_model = RouterModel(d_input=d_input, n_drafters=3)

        specs = [
            DrafterSpec(
                name="Qwen/Qwen2.5-0.5B-Instruct",
                model=ctx.drafter,
                n_params=500_000_000,
                size_penalty=1.0,
            ),
            DrafterSpec(
                name="Qwen/Qwen2.5-1.5B-Instruct",
                model=None,
                n_params=1_500_000_000,
                size_penalty=2.0,
            ),
            DrafterSpec(
                name="Qwen/Qwen2.5-7B-Instruct",
                model=None,
                n_params=7_000_000_000,
                size_penalty=4.0,
            ),
        ]

        def embedder(x):
            out = ctx.drafter.model(x, output_hidden_states=True)
            return out.hidden_states[-1][0].mean(dim=0)

        router = DynamicRouter(
            drafter_specs=specs,
            router_model=router_model,
            embedder=embedder,
        )
        logger.info("DynamicRouter ready with %d drafter specs", len(specs))
        return router
