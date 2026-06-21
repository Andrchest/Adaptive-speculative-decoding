"""Minimal experiment template for researchers.

Copy this file to ``research/<your_name>/experiments/<your_experiment>.py``
and customize the class below.

Quick start
-----------
1. Create directory: ``mkdir -p research/ivan/experiments``
2. Copy template:
   ``cp src/experiments/templates/minimal_template.py \\``
   ``   research/ivan/experiments/phonetic_translation.py``
3. Edit the file: change ``ExperimentMeta``, ``get_config()``, and override
   any ``build_*`` or ``on_*`` methods you need.
4. Run: ``python src/main.py --research`` (auto-discovers your experiment)

Or run a specific experiment by name:
   ``python src/main.py --experiment "my_phonetic_exp"``

Overriding build methods
-------------------------
- ``build_translator(ctx)``  — customize cross-vocab translation
- ``build_cache(ctx)``       — customize N-gram cache
- ``build_distiller(ctx)``   — add online distillation
- ``build_adaptive_controller(ctx)`` — adaptive draft-length control
- ``build_router(ctx)``      — dynamic drafter routing
- ``build_universal_drafter(ctx)``   — universal drafter adapter

Overriding hooks
----------------
- ``on_before_decode(ctx)``  — called once before the decode loop
- ``on_decode_step(ctx, stats, prompt_index)`` — called after each prompt
- ``on_after_decode(ctx)``   — called once after all prompts
- ``on_extra_metrics(summary)`` — augment the final metrics dict

Example
-------
See ``src/experiments/built_in/`` for production examples:

- ``with_lattice.py`` — override ``build_translator()`` (simplest extension)
- ``with_online_distil.py`` — override ``build_distiller()``
- ``full_system.py`` — override everything (most complex)
"""

from __future__ import annotations

from experiments.base import BaseExperiment, ExperimentMeta
from experiments.runner import ExperimentConfig


class MyResearchExperiment(BaseExperiment):
    """Replace this docstring with a description of your experiment.

    What does it test?  Which built-in does it extend?  What hooks
    or build methods does it override?
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                # Unique name — used as run name in MLflow and filename stem
                name="my_research_exp",
                # One-line summary shown in --list output
                description="My novel experiment: <short description>",
                # Free-form tags for filtering (e.g. ["translation", "ivan"])
                tags=["research", "my_name"],
                # Ablation dimensions this experiment touches
                dimensions=["translation_strategy"],
                # Experiments that should run before this one (ordering hint)
                depends_on=["01_baseline"],
            )
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_config(self) -> ExperimentConfig:
        """Return the configuration for this experiment.

        Start from defaults and override only what you need.
        """
        return ExperimentConfig(
            name=self.meta.name,
            # --- Model paths (use tiny models for fast iteration) ---
            # drafter_model_path="facebook/opt-125m",
            # target_model_path="facebook/opt-350m",
            # --- Translation flags ---
            use_rule1=True,
            use_rule2=True,
            use_lattice=False,
            use_translator=False,
            # --- Distillation flags ---
            use_online_distil=False,
            use_replay=False,
            use_contrastive=False,
            # --- Other flags ---
            use_speedup_adaptive=False,
            use_dynamic_routing=False,
            use_universal_drafter=False,
        )

    # ------------------------------------------------------------------
    # Build methods — override only what you need
    # ------------------------------------------------------------------

    # def build_translator(self, ctx: BuildContext):
    #     """Customize cross-vocabulary translation.
    #
    #     Start from the default (Rule1 + Rule2) and add your component:
    #
    #         translator = super().build_translator(ctx)
    #         translator.my_component = MyComponent(...)
    #         return translator
    #     """
    #     return super().build_translator(ctx)

    # def build_distiller(self, ctx: BuildContext):
    #     """Add online distillation.
    #
    #     Access the drafter via ctx.drafter and the translator via
    #     ctx.components["translator"].
    #
    #         import torch.optim as optim
    #         from core.distillation.online import OnlineDistiller
    #
    #         drafter = ctx.drafter
    #         drafter.model.train()
    #         opt = optim.Adam(drafter.model.parameters(), lr=1e-5)
    #         return OnlineDistiller(
    #             drafter_model=drafter,
    #             translator=ctx.components["translator"],
    #             optimizer=opt,
    #         )
    #     """
    #     return super().build_distiller(ctx)

    # ------------------------------------------------------------------
    # Hooks — override to customize decode behaviour
    # ------------------------------------------------------------------

    # def on_before_decode(self, ctx):
    #     """Called once before the decode loop starts."""
    #     pass

    # def on_decode_step(self, ctx, stats, prompt_index):
    #     """Called after each prompt is decoded.
    #
    #     Parameters
    #     ----------
    #     ctx : DecodeContext
    #         Mutable context with decoder, collector, etc.
    #     stats : dict
    #         Current decoder statistics.
    #     prompt_index : int
    #         Zero-based index of the prompt just decoded.
    #     """
    #     pass

    # def on_after_decode(self, ctx):
    #     """Called once after all prompts have been decoded."""
    #     pass

    # def on_extra_metrics(self, summary: dict) -> dict:
    #     """Add experiment-specific metrics to the final summary.
    #
    #         summary["my_custom_metric"] = compute_my_metric(...)
    #         return summary
    #     """
    #     return summary


# Register this experiment for auto-discovery.
# The discover_experiments() function looks for classes listed in __all__.
__all__ = ["MyResearchExperiment"]
