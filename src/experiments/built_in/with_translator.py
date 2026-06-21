"""Built-in: Translator experiment (Rule1 + TokenizerLattice + learned TranslatorModel).

Adds a learned Transformer-based translator in hybrid mode alongside
the lattice mapping.
Corresponds to ``03_+translator`` in the original ABLATION_SUITE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.built_in.with_lattice import LatticeExperiment

if TYPE_CHECKING:
    from experiments.base import BuildContext
    from experiments.runner import ExperimentConfig


class TranslatorExperiment(LatticeExperiment):
    """Lattice experiment with an additional learned TranslatorModel.

    The learned model is a lightweight Transformer encoder that maps
    drafter subtoken sequences to target vocabulary probabilities.
    It operates in hybrid mode with the lattice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.meta.name = "03_+translator"
        self.meta.description = "Rule1 + TokenizerLattice + learned TranslatorModel (hybrid mode)"
        self.meta.tags = ["translation", "lattice", "learned-translator"]
        self.meta.dimensions = ["translation_strategy"]
        self.meta.depends_on = ["02_+lattice"]

    def get_config(self) -> ExperimentConfig:
        cfg = super().get_config()
        cfg.name = self.meta.name
        cfg.use_translator = True
        cfg.translator_weight = 0.3
        return cfg

    def build_translator(self, ctx: BuildContext):
        """Build translator with lattice + learned TranslatorModel."""
        translator = super().build_translator(ctx)

        from core.extensions.translator.model import TranslatorModel

        learned = TranslatorModel(
            drafter_vocab_size=ctx.drafter.model.config.vocab_size,
            target_vocab_size=ctx.target.model.config.vocab_size,
        ).to(ctx.device)
        translator.learned_model = learned
        translator.learned_weight = getattr(ctx.config, "translator_weight", 0.3)
        return translator
