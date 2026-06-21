"""Built-in: Lattice experiment (Rule1 + TokenizerLattice, no distillation).

Replaces the approximate Rule 2 heuristic with exact DP-based lattice mapping.
Corresponds to ``02_+lattice`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

from experiments.base import BaseExperiment, BuildContext, ExperimentMeta
from experiments.runner import ExperimentConfig


class LatticeExperiment(BaseExperiment):
    """Baseline with TokenizerLattice replacing Rule 2.

    Uses Rule 1 (direct character match) for exact matches and
    TokenizerLattice for approximate mapping, instead of the
    approximate Rule 2 redistribution heuristic.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="02_+lattice",
                description="Rule1 + TokenizerLattice (exact DP mapping)",
                tags=["translation", "lattice"],
                dimensions=["translation_strategy"],
                depends_on=["01_baseline"],
            )
        )

    def get_config(self) -> ExperimentConfig:
        cfg = ExperimentConfig(
            name=self.meta.name,
            use_rule1=True,
            use_rule2=False,
            use_lattice=True,
            use_translator=False,
            use_online_distil=False,
            use_replay=False,
            use_contrastive=False,
            use_speedup_adaptive=False,
            use_dynamic_routing=False,
            use_universal_drafter=False,
        )
        return cfg

    def build_translator(self, ctx: BuildContext):
        """Build translator with TokenizerLattice replacing Rule 2."""
        translator = super().build_translator(ctx)

        from core.extensions.lattice.tokenizer_lattice import TokenizerLattice

        lattice = TokenizerLattice(
            ctx.drafter.tokenizer,
            ctx.target.tokenizer,
            drafter_vocab_size=ctx.drafter.model.config.vocab_size,
            target_vocab_size=ctx.target.model.config.vocab_size,
        )
        translator.lattice = lattice
        return translator
