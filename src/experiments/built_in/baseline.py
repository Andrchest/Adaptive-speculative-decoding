"""Built-in: Baseline experiment (Rule1 + Rule2 + NgramCache, no distillation).

This reproduces the original ``01_baseline`` from ``ABLATION_SUITE``.
"""

from __future__ import annotations

from experiments.base import BaseExperiment, ExperimentMeta
from experiments.runner import ExperimentConfig


class BaselineExperiment(BaseExperiment):
    """Baseline speculative decoding: Rule1 + Rule2 translation, N-gram cache.

    No distillation, no lattice, no learned translator, no replay,
    no contrastive loss, no adaptive drafting, no routing, no universal drafter.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="01_baseline",
                description="Rule1 + Rule2 + NgramCache(LRU) + no distillation",
                tags=["baseline"],
                dimensions=[],
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
            use_universal_drafter=False,
        )
