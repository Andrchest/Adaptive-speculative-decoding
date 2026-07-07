"""Built-in: Acceptance-Rate Adaptive Drafting experiment.

Sibling to SpeedupAdaptiveExperiment (with_speedup_adapt.py), differing
only in the training signal fed to the k-selection predictor: this one
predicts per-k acceptance rate (then chooses k to maximize expected
accepted tokens = rate(k) * k) instead of predicting wall-clock speedup
directly. See core/extensions/adaptive/acceptance_predictor.py for the
rationale.

Numbered 08b rather than reusing 08 so both can appear side-by-side in
comparison_table.csv without colliding on the result JSON filename.
"""

from __future__ import annotations

import logging

from experiments.base import BaseExperiment, BuildContext, DecodeContext, ExperimentMeta
from experiments.runner import ExperimentConfig

logger = logging.getLogger(__name__)


class AcceptanceAdaptiveExperiment(BaseExperiment):
    """Adaptive draft length via AcceptanceRatePredictor.

    A small MLP predicts expected acceptance rate for each candidate
    draft length k; the controller selects argmax_k(rate(k) * k), i.e.
    the k maximizing expected number of accepted tokens per step.
    """

    def __init__(self) -> None:
        super().__init__(
            ExperimentMeta(
                name="08b_+acceptance_adapt",
                description="Acceptance-Rate-Aware Adaptive Drafting (learned k selection)",
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
            use_speedup_adaptive=False,  # this is the acceptance-rate variant
            use_dynamic_routing=False,
            use_universal_drafter=False,
        )

    def build_adaptive_controller(self, ctx: BuildContext):
        """Build AcceptanceRatePredictor + AcceptanceAdaptiveController."""
        from core.extensions.adaptive.acceptance_predictor import (
            AcceptanceAdaptiveController,
            AcceptanceRatePredictor,
        )

        cfg = ctx.config
        pred = AcceptanceRatePredictor(
            d_hidden=ctx.drafter.model.config.hidden_size,
            k_max=getattr(cfg, "k_max", 8),
        ).to(ctx.device)
        controller = AcceptanceAdaptiveController(
            pred, ctx.drafter, getattr(cfg, "k_min", 1), getattr(cfg, "k_max", 8)
        )
        logger.info(
            "AcceptanceAdaptiveController ready: k_min=%d k_max=%d",
            getattr(cfg, "k_min", 1),
            getattr(cfg, "k_max", 8),
        )
        return controller

    def on_decode_step(self, ctx: DecodeContext, step_result, prompt_index: int) -> None:
        """Periodically train the predictor on its replay buffer.

        Identical cadence/hyperparameter knobs to SpeedupAdaptiveExperiment
        so the two are comparable under the same training budget — if
        one trained more often than the other, a difference in final
        acceptance rate wouldn't tell you which OBJECTIVE is better, only
        which got more gradient steps.
        """
        ctrl = ctx.adaptive_fn
        if ctrl is None or not hasattr(ctrl, "predictor"):
            return
        train_every = getattr(ctx.config, "adaptive_train_every", 16)
        if (prompt_index + 1) % train_every != 0:
            return
        mean_loss = ctrl.predictor.train_on_buffer(
            n_steps=getattr(ctx.config, "adaptive_train_steps", 32),
            batch_size=getattr(ctx.config, "adaptive_batch_size", 32),
            lr=getattr(ctx.config, "adaptive_lr", 1e-3),
        )
        logger.debug(
            "Acceptance predictor trained: prompt=%d mean_loss=%.4f", prompt_index, mean_loss
        )

    def on_extra_metrics(self, summary: dict) -> dict:
        """Tag the summary so comparison_table.csv clearly separates this
        from the speedup-prediction variant when both rows are present.
        """
        summary["adaptive_objective"] = "acceptance_rate"
        return summary
