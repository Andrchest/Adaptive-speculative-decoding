"""
experiments/base.py

Base classes and context types for the Strategy-pattern experiment architecture.

Every experiment inherits from BaseExperiment and overrides the build/hook
methods it needs.  The runner (ExperimentRunner) is responsible for loading
models and datasets; the experiment is responsible for assembling components
and customizing the decode loop.

Example
-------
>>> from experiments import BaseExperiment, ExperimentMeta, ExperimentConfig
>>>
>>> class MyExperiment(BaseExperiment):
...     def __init__(self):
...         super().__init__(ExperimentMeta(
...             name="my_exp",
...             description="My novel experiment",
...             tags=["translation"],
...         ))
...
...     def get_config(self) -> ExperimentConfig:
...         cfg = ExperimentConfig(name=self.meta.name)
...         cfg.use_lattice = True
...         return cfg

See ``templates/minimal_template.py`` for a copy-paste starting point.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING



if TYPE_CHECKING:
    # Avoid circular imports at runtime; these are only used in type hints.
    from benchmarks.metrics.collector import BenchmarkCollector
    from core.cache.ngram import NgramCache
    from core.decoder.speculative import SpeculativeDecoder, StepResult
    from core.distillation.online import OnlineDistiller
    from core.models.drafter import DraftModel, TargetModel
    from core.translation.vocabulary import CrossVocabTranslator
    from experiments.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMeta:
    """Human-readable metadata attached to every experiment.

    Attributes
    ----------
    name :
        Unique identifier, used as the run name in MLflow and as the
        filename stem for result JSON (spaces replaced with underscores).
    description :
        One-line summary shown in ``--list`` output.
    tags :
        Free-form labels for filtering (e.g. ``["translation", "ivan"]``).
    dimensions :
        Ablation dimensions this experiment touches
        (e.g. ``["translation_strategy", "cache_eviction"]``).
    depends_on :
        Names of experiments that should run before this one
        (for ordering, not hard enforcement).
    """

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Immutable result of a single experiment run.

    Attributes
    ----------
    meta :
        Experiment metadata (name, tags, etc.).
    config :
        Serialised configuration dictionary (from ``asdict(cfg)``).
    metrics :
        Aggregated benchmark metrics (from ``BenchmarkCollector.summary()``
        plus any experiment-specific additions).
    error :
        Non-None if the experiment failed; contains the error message.
    """

    meta: ExperimentMeta
    config: dict
    metrics: dict
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Build Context
# ---------------------------------------------------------------------------


@dataclass
class BuildContext:
    """Shared context passed to every ``build_*`` method.

    Components built earlier in the lifecycle are available here so that
    later components can depend on them (e.g. the distiller needs the
    translator).

    Attributes
    ----------
    device :
        Torch device string (``"cuda"``, ``"cpu"``, etc.).
    drafter :
        Loaded ``DraftModel`` instance.
    target :
        Loaded ``TargetModel`` instance.
    config :
        The ``ExperimentConfig`` returned by ``get_config()``.
    components :
        Key-value store for components built so far.  Preferred keys:
        ``"translator"``, ``"cache"``, ``"distiller"``, ``"router"``,
        ``"adaptive_fn"``.
    """

    device: str
    drafter: DraftModel
    target: TargetModel
    config: object  # ExperimentConfig — avoided to prevent circular import
    components: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decode Context
# ---------------------------------------------------------------------------


@dataclass
class DecodeContext:
    """Mutable context available during the decode loop.

    Experiment hooks (``on_before_decode``, ``on_decode_step``,
    ``on_after_decode``) can read and modify this object to influence
    decoding behaviour.

    Attributes
    ----------
    decoder :
        The ``SpeculativeDecoder`` instance for this experiment.
    collector :
        The ``BenchmarkCollector`` accumulating metrics.
    config :
        The ``ExperimentConfig`` for this run.
    distiller :
        Online distiller, if built (otherwise ``None``).
    router :
        Dynamic router, if built (otherwise ``None``).
    adaptive_fn :
        Adaptive draft-length callable, if built (otherwise ``None``).
    extra_state :
        Free-form dictionary for experiment-specific mutable state.
    """

    decoder: SpeculativeDecoder
    collector: BenchmarkCollector
    config: object  # ExperimentConfig
    distiller: OnlineDistiller | None = None
    router: object | None = None  # DynamicRouter
    adaptive_fn: object | None = None  # callable | None
    extra_state: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Experiment (ABC)
# ---------------------------------------------------------------------------


class BaseExperiment(abc.ABC):
    """Abstract base class for all experiments.

    Subclasses override the ``build_*`` methods to customize which
    components are constructed and the ``on_*`` hooks to customize
    behaviour during the decode loop.

    The default implementation of every ``build_*`` method returns the
    simplest possible component (or ``None``), so subclasses only need
    to override what differs from baseline.

    Lifecycle
    ---------
    1. Runner calls ``get_config()`` → ``ExperimentConfig``
    2. Runner loads models and dataset from config
    3. Runner creates ``BuildContext`` and calls ``build_*`` methods
       in order: translator → cache → distiller → adaptive → router → universal
    4. Runner creates ``DecodeContext`` and calls ``on_before_decode()``
    5. For each prompt:
       a. Router selects drafter (if applicable)
       b. ``decoder.generate()`` runs
       c. ``on_decode_step()`` called after each step
    6. ``on_after_decode()`` called
    7. ``on_extra_metrics()`` augments the summary dict
    8. Result saved to JSON + CSV
    """

    def __init__(self, meta: ExperimentMeta | None = None) -> None:
        if meta is None:
            meta = ExperimentMeta(name=self.__class__.__name__)
        self.meta = meta
        self._components: dict[str, object] = {}
        self._overrides: dict[str, object] = {}

    def set_config_override(self, key: str, value: object) -> None:
        """Set a configuration override applied before each run.

        This is primarily used by the CLI to apply ``--tiny``,
        ``--max-samples``, and similar flags without modifying
        experiment classes.

        Parameters
        ----------
        key :
            Attribute name on ``ExperimentConfig`` (e.g. ``"max_samples"``).
        value :
            Value to set.
        """
        self._overrides[key] = value

    # ------------------------------------------------------------------
    # Configuration (must override)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_config(self) -> object:
        """Return the base configuration for this experiment.

        Returns
        -------
        ExperimentConfig
            The configuration dataclass with model paths, dataset,
            hyperparameters, etc.  Subclasses should call
            ``super().get_config()`` and override specific fields,
            or construct from scratch.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_config(). "
            "Return an ExperimentConfig with at minimum `name` set."
        )

    # ------------------------------------------------------------------
    # Build methods (override selectively)
    # ------------------------------------------------------------------

    def build_translator(self, ctx: BuildContext) -> CrossVocabTranslator:
        """Build the cross-vocabulary translator.

        Default: Rule1 + Rule2 via ``CrossVocabTranslator.from_tokenizers()``.

        Override this to add lattice, learned translator, or custom
        translation logic.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context with drafter, target, and config.

        Returns
        -------
        CrossVocabTranslator
            Configured translator instance.
        """
        from core.translation.vocabulary import CrossVocabTranslator

        translator = CrossVocabTranslator.from_tokenizers(
            ctx.drafter.tokenizer,
            ctx.target.tokenizer,
            device=ctx.device,
            drafter_vocab_size=ctx.drafter.model.config.vocab_size,
            target_vocab_size=ctx.target.model.config.vocab_size,
        )
        return translator

    def build_cache(self, ctx: BuildContext) -> NgramCache:
        """Build the N-gram cache.

        Default: ``NgramCache(max_size=65536, eviction="hybrid")``.

        Override this to change eviction strategy or capacity.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context.

        Returns
        -------
        NgramCache
            Configured cache instance.
        """
        from core.cache.ngram import NgramCache

        # Import ExperimentConfig to read fields without circular import
        max_size = getattr(ctx.config, "cache_max_size", 65536)
        eviction = getattr(ctx.config, "cache_eviction", "hybrid")
        return NgramCache(max_size=max_size, eviction=eviction)

    def build_distiller(self, ctx: BuildContext) -> OnlineDistiller | None:
        """Build the online distiller.

        Default: ``None`` (no distillation).

        Override this to enable distillation, replay buffers,
        contrastive loss, etc.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context.  The translator is available at
            ``ctx.components["translator"]``.

        Returns
        -------
        OnlineDistiller or None
            Configured distiller, or ``None`` to skip distillation.
        """
        return None

    def build_adaptive_controller(self, ctx: BuildContext) -> object | None:
        """Build the adaptive draft-length controller.

        Default: ``None`` (fixed draft length from config).

        Override this to enable SpeedupPredictor or custom adaptive logic.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context.

        Returns
        -------
        callable or None
            A callable ``context_tensor -> k`` or ``None``.
        """
        return None

    def build_router(self, ctx: BuildContext) -> object | None:
        """Build the dynamic multi-drafter router.

        Default: ``None`` (single drafter).

        Override this to enable dynamic routing between drafters.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context.

        Returns
        -------
        DynamicRouter or None
            Configured router, or ``None``.
        """
        return None

    def build_universal_drafter(self, ctx: BuildContext) -> object | None:
        """Build the universal drafter adapter.

        Default: ``None`` (single-target drafter).

        Override this to enable multi-target drafting via UniversalDrafter.

        Parameters
        ----------
        ctx : BuildContext
            Shared build context.

        Returns
        -------
        object or None
            A drafter adapter compatible with the DraftModel interface,
            or ``None``.
        """
        return None

    # ------------------------------------------------------------------
    # Decode hooks (override selectively)
    # ------------------------------------------------------------------

    def on_before_decode(self, ctx: DecodeContext) -> None:
        """Called once before the decode loop starts.

        Override to set up state, start timers, or initialize
        per-experiment counters.

        Parameters
        ----------
        ctx : DecodeContext
            Mutable decode context.
        """
        del ctx  # unused in base

    def on_decode_step(
        self,
        ctx: DecodeContext,
        step_result: StepResult,
        prompt_index: int,
    ) -> None:
        """Called after each decode step (one prompt completion).

        Override to implement per-step logic: distillation triggers,
        routing decisions, custom metric collection, etc.

        Parameters
        ----------
        ctx : DecodeContext
            Mutable decode context.
        step_result : StepResult
            Statistics for the completed step.
        prompt_index : int
            Zero-based index of the current prompt.
        """
        del ctx, step_result, prompt_index  # unused in base

    def on_after_decode(self, ctx: DecodeContext) -> None:
        """Called once after all prompts have been decoded.

        Override to finalize state, flush buffers, or compute
        post-decode statistics.

        Parameters
        ----------
        ctx : DecodeContext
            Mutable decode context.
        """
        del ctx  # unused in base

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def on_extra_metrics(self, summary: dict) -> dict:
        """Augment the collector summary with experiment-specific metrics.

        Default: returns the summary unchanged.

        Parameters
        ----------
        summary : dict
            Aggregated metrics from ``BenchmarkCollector.summary()``.

        Returns
        -------
        dict
            The (possibly modified) summary dictionary.
        """
        return summary

    # ------------------------------------------------------------------
    # Execution (usually don't override)
    # ------------------------------------------------------------------

    def run(self, runner: ExperimentRunner) -> ExperimentResult:
        """Execute the full experiment lifecycle.

        This method orchestrates: build components → load dataset →
        decode loop → collect metrics → save result.

        Subclasses generally should NOT override this.  Instead, override
        the individual ``build_*`` and ``on_*`` methods.

        Parameters
        ----------
        runner : ExperimentRunner
            The runner providing shared utilities (model loading,
            dataset loading, persistence).

        Returns
        -------
        ExperimentResult
            Named tuple with metadata, config, metrics, and optional error.
        """
        import logging
        import random as _random

        import numpy as np
        import torch

        logger = logging.getLogger(__name__)

        cfg = self.get_config()
        # Apply CLI overrides (e.g. --tiny, --max-samples)
        for key, value in self._overrides.items():
            setattr(cfg, key, value)
        cfg_dict = runner._asdict_config(cfg)
        logger.info("Running experiment: %s", self.meta.name)

        # Deterministic seeding
        seed = getattr(cfg, "seed", 42)
        torch.manual_seed(seed)
        _random.seed(seed)
        np.random.seed(seed)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(seed)

        # Load models
        drafter, target = runner._build_models(cfg)

        # GPU memory tracking: collect samples during setup
        mem_samples: list[float] = []
        if torch.cuda.is_available():
            mem_samples.append(
                torch.cuda.memory_allocated(runner.device) / 1024**3
            )
            logger.info("GPU memory after models: %.2f GB", mem_samples[-1])

        # Build components
        build_ctx = BuildContext(
            device=runner.device,
            drafter=drafter,
            target=target,
            config=cfg,
            components={},
        )

        translator = self.build_translator(build_ctx)
        build_ctx.components["translator"] = translator

        cache = self.build_cache(build_ctx)
        build_ctx.components["cache"] = cache

        distiller = self.build_distiller(build_ctx)
        build_ctx.components["distiller"] = distiller

        adaptive_fn = self.build_adaptive_controller(build_ctx)
        build_ctx.components["adaptive_fn"] = adaptive_fn

        router = self.build_router(build_ctx)
        build_ctx.components["router"] = router

        universal_adapter = self.build_universal_drafter(build_ctx)
        if universal_adapter is not None:
            drafter = universal_adapter
            build_ctx.components["drafter"] = drafter

        # Build decoder
        draft_length = getattr(cfg, "draft_length", 5)
        from core.decoder.speculative import SpeculativeDecoder

        decoder = SpeculativeDecoder(
            drafter=drafter,
            target=target,
            translator=translator,
            cache=cache,
            draft_length=draft_length,
        )

        # Load dataset
        prompts = runner._load_dataset(cfg)

        # GPU memory: after all components built
        if torch.cuda.is_available():
            mem_samples.append(
                torch.cuda.memory_allocated(runner.device) / 1024**3
            )
            logger.info("GPU memory after setup: %.2f GB", mem_samples[-1])

        # Benchmark collector
        from benchmarks.metrics.collector import BenchmarkCollector

        name = getattr(cfg, "name", self.meta.name)
        collector = BenchmarkCollector(name=name)
        collector._gpu_mem_samples = mem_samples

        # MLflow setup
        runner._setup_mlflow(cfg)

        # Decode context
        decode_ctx = DecodeContext(
            decoder=decoder,
            collector=collector,
            config=cfg,
            distiller=distiller,
            router=router,
            adaptive_fn=adaptive_fn,
        )

        # Hooks: before decode
        self.on_before_decode(decode_ctx)

        # Decode loop
        import experiments.runner as _rl_mod  # type: ignore
        _ll = getattr(_rl_mod, "_log_level", "QUIET")
        # Use tqdm for QUIET/NORMAL mode, simple loop otherwise
        _use_tqdm = _ll in ("QUIET", "NORMAL")
        _verbose_mode = (_ll == "VERBOSE")
        log_every = getattr(cfg, "log_every", 50)

        # Import tqdm if needed
        if _use_tqdm:
            try:
                from tqdm import tqdm as _tqdm
                _tqdm_available = True
            except ImportError:
                _tqdm_available = False
        else:
            _tqdm_available = False

        # Wrap prompts with tqdm or plain enumerate
        if _tqdm_available:
            _prompt_iter = _tqdm(
                enumerate(prompts),
                total=len(prompts),
                desc=f"{self.meta.name[:20]}",
                leave=False,
                ncols=70,
            )
        else:
            _prompt_iter = enumerate(prompts)

        for i, (input_ids, prompt_len) in _prompt_iter:
            input_ids = input_ids.to(runner.device)
            max_new_tokens = getattr(cfg, "max_new_tokens", 128)

            # GPU memory: at each prompt (captures per-sequence peaks)
            if torch.cuda.is_available():
                mem_samples.append(
                    torch.cuda.memory_allocated(runner.device) / 1024**3
                )

            # Router selection (if applicable)
            if router is not None:
                selected_drafter, _selected_idx = router.select_drafter(input_ids)
                if selected_drafter is not None:
                    decoder.drafter = selected_drafter
                else:
                    logger.warning(
                        "Router selected a None drafter (index %d); keeping current drafter",
                        _selected_idx,
                    )

            with collector.record_sequence(prompt_len=prompt_len) as seq_rec:
                decoder.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    adaptive_length_fn=adaptive_fn,
                    distiller=distiller,
                    rng=torch_rng,
                )
                for sr in decoder._step_results[-max_new_tokens:]:
                    seq_rec.add_step(
                        draft_len=sr.draft_length,
                        accepted=sr.accepted_count,
                        cache_hit=sr.cache_hit,
                    )
            decoder._step_results.clear()

            # Hook: after each step
            self.on_decode_step(decode_ctx, decoder.stats(), i)

            # Progress logging
            if _verbose_mode and i % log_every == 0:
                partial = collector.summary()
                logger.info(
                    "Progress [%d/%d] acc=%.3f tps=%.1f",
                    i + 1,
                    len(prompts),
                    partial.get("acceptance_rate", 0),
                    partial.get("tokens_per_sec", 0),
                )
            # Update tqdm bar in QUIET/NORMAL mode
            if _use_tqdm and _tqdm_available:
                partial = collector.summary()
                _prompt_iter.set_postfix({
                    "acc": f"{partial.get('acceptance_rate', 0):.3f}",
                    "tps": f"{partial.get('tokens_per_sec', 0):.1f}",
                })

        # Hooks: after decode
        self.on_after_decode(decode_ctx)

        # Cleanup: remove forward hooks from UniversalDrafter to break
        # reference cycles and free CUDA allocator pressure.
        if hasattr(drafter, "cleanup"):  # _UniversalDrafterAdapter has cleanup()
            drafter.cleanup()
        elif hasattr(drafter, "remove_hooks"):  # bare UniversalDrafter
            drafter.remove_hooks()

        # GPU memory: after all decoding complete
        if torch.cuda.is_available():
            mem_samples.append(
                torch.cuda.memory_allocated(runner.device) / 1024**3
            )

        # Collect metrics (final summary — collector is cleared after this)
        summary = collector.summary()
        collector.clear()  # free memory from DecodeRecord/StepRecord objects

        # Add distiller stats
        if distiller is not None:
            distil_stats = distiller.training_stats()
            summary.update(distil_stats)

        # Add router stats
        if router is not None:
            summary["router"] = decode_ctx.extra_state.get("router_stats", {})

        # Extra metrics hook
        summary = self.on_extra_metrics(summary)

        # Print end summary to console
        _cl = getattr(_rl_mod, "_log_level", "QUIET")
        if _cl != "QUIET":
            collector.print_end_summary(summary)

        # MLflow final logging
        runner._log_mlflow_final(cfg, summary)

        return ExperimentResult(
            meta=self.meta,
            config=cfg_dict,
            metrics=summary,
        )
