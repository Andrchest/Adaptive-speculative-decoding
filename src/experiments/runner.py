"""
experiments/runner.py

Experiment runner with full ablation support.

Defines all experiment configurations and runs them sequentially (or in
parallel via multiprocessing).  Results are logged to MLflow
and written to a local CSV.

Ablation matrix:
  - baseline          : Rule1 + Rule2 + NgramCache(LRU) + no distillation
  - + lattice         : Replace Rule2 with TokenizerLattice
  - + translator      : Add TranslatorModel in hybrid mode
  - + online_distil   : Add OnlineDistiller (no replay)
  - + replay_fifo     : Add ReplayBuffer(fifo)
  - + replay_prio     : Replace with ReplayBuffer(prioritized)
  - + contrastive     : Add ContrastiveLoss
  - + speedup_adapt   : Add SpeedupPredictor adaptive drafting
  - + routing         : Add DynamicRouter (multi-drafter)
  - + universal       : UniversalDrafter (multi-target)
  - full_system       : All components enabled
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    name: str

    # Model paths
    drafter_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model_path: str = "Qwen/Qwen2.5-7B-Instruct"

    # Dataset
    dataset: str = "gsm8k"  # gsm8k | mbpp | humaneval | alpaca | xsum
    max_samples: int = 500
    max_new_tokens: int = 128

    # Draft length
    draft_length: int = 5
    k_min: int = 1
    k_max: int = 8

    # Translation
    use_rule1: bool = True
    use_rule2: bool = True
    use_lattice: bool = False  # replace Rule2 with exact lattice
    use_translator: bool = False  # add learned translator in hybrid mode
    translator_weight: float = 0.3

    # Cache
    cache_max_size: int = 65536
    cache_eviction: str = "hybrid"  # lru | lfu | acc | hybrid

    # Distillation
    use_online_distil: bool = False
    lambda_ngram: float = 0.5
    distil_lr: float = 1e-5
    use_lora: bool = False
    lora_rank: int = 8

    # Replay
    use_replay: bool = False
    replay_strategy: str = "fifo"  # fifo | prioritized
    replay_capacity: int = 4096
    replay_every: int = 32
    replay_batch: int = 8

    # Contrastive
    use_contrastive: bool = False
    lambda_contrastive: float = 0.1

    # Adaptive drafting
    use_speedup_adaptive: bool = False

    # Routing
    use_dynamic_routing: bool = False
    drafter_model_paths: list[str] = field(default_factory=list)

    # Multi-target
    use_universal_drafter: bool = False
    target_model_paths: list[str] = field(default_factory=list)

    # Logging
    mlflow_experiment: str = "adaptive_speculative"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"  # http://10.93.27.4:5000/ :(
    log_every: int = 50
    seed: int = 42


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """
    Runs a list of ExperimentConfig objects and collects results.
    """

    def __init__(
        self,
        configs: list[ExperimentConfig],
        output_dir: str = "results",
        device: str = "cuda",
    ) -> None:
        self.configs = configs
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "ExperimentRunner initialized: output_dir=%s device=%s configs=%d",
            output_dir,
            device,
            len(configs),
        )

    @staticmethod
    def _clear_gpu_memory() -> None:
        """Aggressively free GPU memory between experiments."""
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                "GPU memory: %.1f MB used / %.1f MB total",
                torch.cuda.memory_allocated() / 1e6,
                torch.cuda.memory_reserved() / 1e6,
            )

    def run_all(self) -> list[dict]:
        results = []
        logger.info("Starting run_all for %d experiment(s)", len(self.configs))
        for index, cfg in enumerate(self.configs, start=1):
            logger.info("Starting experiment %d/%d: %s", index, len(self.configs), cfg.name)
            print(f"\n{'=' * 60}")
            print(f"  Running: {cfg.name}")
            print(f"{'=' * 60}")

            # Free GPU memory from previous experiment
            if index > 1:
                logger.info("Clearing GPU memory from previous experiment")
                self._clear_gpu_memory()

            result = self._run_one(cfg)
            results.append(result)
            self._save_result(result)
            logger.info("Finished experiment %d/%d: %s", index, len(self.configs), cfg.name)

            # Free GPU memory before next experiment
            logger.info("Releasing GPU memory after experiment %d", index)
            self._clear_gpu_memory()

        self._write_csv(results)
        logger.info("Finished run_all with %d result(s)", len(results))
        return results

    def _run_one(self, cfg: ExperimentConfig) -> dict:
        """
        Build pipeline from config and run benchmark.
        Import lazily to avoid loading GPU models for dry runs.
        """
        logger.info("Running experiment config: %s", cfg.name)
        import torch

        torch.manual_seed(cfg.seed)
        logger.info("Set random seed: %s", cfg.seed)

        # --- Build components ---
        logger.info("Importing core components")
        from benchmarks.metrics.collector import BenchmarkCollector
        from core.cache.ngram import NgramCache
        from core.decoder.speculative import SpeculativeDecoder
        from core.models.drafter import DraftModel, TargetModel
        from core.translation.vocabulary import CrossVocabTranslator

        logger.info("Loading drafter model: %s", cfg.drafter_model_path)
        drafter = DraftModel(cfg.drafter_model_path, device=self.device)
        logger.info("Loading target model: %s", cfg.target_model_path)
        target = TargetModel(cfg.target_model_path, device=self.device)

        # Translation
        # NOTE: model.config.vocab_size is the lm_head output dimension and
        # often differs from len(tokenizer.get_vocab()) (e.g. OPT pads
        # 50265 → 50272). Passing the model size keeps every probability
        # tensor downstream aligned with raw logits shapes.
        logger.info("Building cross-vocabulary translator")
        translator = CrossVocabTranslator.from_tokenizers(
            drafter.tokenizer,
            target.tokenizer,
            device=self.device,
            drafter_vocab_size=drafter.model.config.vocab_size,
            target_vocab_size=target.model.config.vocab_size,
        )

        if cfg.use_lattice:
            logger.info("Building tokenizer lattice")
            from core.extensions.lattice.tokenizer_lattice import TokenizerLattice

            lattice = TokenizerLattice(
                drafter.tokenizer,
                target.tokenizer,
                drafter_vocab_size=drafter.model.config.vocab_size,
                target_vocab_size=target.model.config.vocab_size,
            )
            # Monkey-patch translator rule2 with lattice
            translator.lattice = lattice
            logger.info("Attached tokenizer lattice to translator")

        if cfg.use_translator:
            logger.info("Building learned translator model")
            from core.extensions.translator.model import TranslatorModel

            learned = TranslatorModel(
                drafter_vocab_size=drafter.model.config.vocab_size,
                target_vocab_size=target.model.config.vocab_size,
            ).to(self.device)
            translator.learned_model = learned
            translator.learned_weight = cfg.translator_weight
            logger.info("Attached learned translator with weight %.4f", cfg.translator_weight)

        # Cache
        logger.info(
            "Building n-gram cache: max_size=%d eviction=%s", cfg.cache_max_size, cfg.cache_eviction
        )
        cache = NgramCache(
            max_size=cfg.cache_max_size,
            eviction=cfg.cache_eviction,
        )

        # Distiller
        distiller = None
        if cfg.use_online_distil:
            logger.info("Building online distiller")
            import torch.optim as optim

            from core.distillation.online import OnlineDistiller

            # Enable gradients for fine-tuning
            drafter.model.train()
            for p in drafter.model.parameters():
                p.requires_grad_(True)

            opt = optim.Adam(drafter.model.parameters(), lr=cfg.distil_lr)
            distiller = OnlineDistiller(
                drafter_model=drafter,
                translator=translator,
                optimizer=opt,
                lambda_ngram=cfg.lambda_ngram,
                use_lora=cfg.use_lora,
            )
            logger.info(
                "Online distiller ready: lr=%s lambda_ngram=%s", cfg.distil_lr, cfg.lambda_ngram
            )

        if cfg.use_replay and distiller is not None:
            logger.info(
                "Wrapping distiller with replay buffer: strategy=%s capacity=%d",
                cfg.replay_strategy,
                cfg.replay_capacity,
            )
            from core.extensions.replay.buffer import ReplayBuffer, ReplayDistiller

            buf = ReplayBuffer(
                capacity=cfg.replay_capacity,
                strategy=cfg.replay_strategy,
            )
            distiller = ReplayDistiller(
                distiller=distiller,
                buffer=buf,
                replay_every=cfg.replay_every,
                replay_batch=cfg.replay_batch,
            )
            logger.info("Replay distiller ready")

        # Contrastive loss wrapper (adds InfoNCE on top of online distillation)
        if cfg.use_contrastive:
            if distiller is None:
                logger.warning(
                    "use_contrastive=True but use_online_distil=False; "
                    "creating distiller implicitly for contrastive loss"
                )
                import torch.optim as optim

                from core.distillation.online import OnlineDistiller

                opt = optim.Adam(drafter.model.parameters(), lr=cfg.distil_lr)
                distiller = OnlineDistiller(
                    drafter_model=drafter,
                    translator=translator,
                    optimizer=opt,
                    lambda_ngram=cfg.lambda_ngram,
                    use_lora=cfg.use_lora,
                )
            logger.info(
                "Wrapping distiller with contrastive loss: lambda=%.2f temp=%.2f",
                cfg.lambda_contrastive,
                0.07,
            )
            from core.extensions.contrastive.loss import ContrastiveLoss

            # Use the public setter — a previous version assigned to
            # ``distiller.contrastive_loss`` (no underscore), which never
            # reached the underscore-prefixed attribute read by
            # ``OnlineDistiller._compute_loss``. The contrastive-loss
            # ablation was therefore a silent no-op.
            distiller.set_contrastive_loss(
                ContrastiveLoss(
                    lambda_nll=cfg.lambda_ngram,
                    lambda_contrastive=cfg.lambda_contrastive,
                    temperature=0.07,
                )
            )
            # Expose translator for the contrastive module (needs Rule1 mapping)
            distiller.translator = translator
            logger.info("Contrastive distiller ready")

        # Dynamic multi-drafter routing
        router = None
        # ``specs`` is populated by the routing block when routing is
        # enabled. We initialise it here (BEFORE the routing block) so
        # the universal-drafter block below can safely read it without
        # re-initialising and discarding the routing block's work
        # (a previous version had ``specs: list = []`` AFTER the routing
        # block, which unconditionally wiped the populated list and
        # broke the universal-drafter adapter insertion).
        specs: list = []
        if cfg.use_dynamic_routing:
            logger.info(
                "Building dynamic router with %d drafter(s)",
                len(cfg.drafter_model_paths) if cfg.drafter_model_paths else 1,
            )
            from core.extensions.routing.router import DrafterSpec, DynamicRouter, RouterModel

            # Build drafter specs
            if cfg.drafter_model_paths:
                drafters = {}
                for model_path in cfg.drafter_model_paths:
                    logger.info("Loading routing drafter: %s", model_path)
                    drafters[model_path] = DraftModel(
                        model_path,
                        device=self.device,
                        dtype=getattr(drafter.model, "dtype", torch.float16),
                    )
                    logger.info("Routing drafter ready: %s", model_path)
            else:
                # Default: load the main drafter + a smaller variant
                drafters = {
                    cfg.drafter_model_path: drafter,
                    "facebook/opt-125m": DraftModel(
                        "facebook/opt-125m",
                        device=self.device,
                    ),
                }

            specs = [
                DrafterSpec(
                    name=path,
                    model=d,
                    n_params=sum(p.numel() for p in d.model.parameters()),
                    size_penalty=1.0,
                )
                for path, d in drafters.items()
            ]
            n_drafters = len(specs)
            d_hidden = drafter.model.config.hidden_size
            router_model = RouterModel(d_input=d_hidden, n_drafters=n_drafters).to(self.device)
            router = DynamicRouter(
                drafter_specs=specs,
                router_model=router_model, # idk why ruff fails to recognize drafter (it is initialized after imports)
                embedder=lambda ids: drafter.model(ids, output_hidden_states=True)  # noqa: F821
                .hidden_states[-1]
                .mean(dim=1)
                .to(dtype=router_model.net[0].weight.dtype),
            )
            logger.info("Dynamic router ready: %d drafters", n_drafters)

        # Universal drafter (multi-target)
        # NOTE: ``specs`` was already initialised above (before the
        # routing block). Do NOT re-initialise it here — doing so would
        # discard the routing block's work and break the universal-drafter
        # adapter insertion loop below.
        if cfg.use_universal_drafter:
            logger.info(
                "Building universal drafter: base=%s targets=%s",
                cfg.drafter_model_path,
                cfg.target_model_paths,
            )
            from core.extensions.multitarget.universal_drafter import UniversalDrafter

            d_model = drafter.model.config.hidden_size
            # Use explicit target list, or fall back to the single target model
            target_names = cfg.target_model_paths or [cfg.target_model_path]
            universal = UniversalDrafter(
                base_model_name=cfg.drafter_model_path,
                target_names=target_names,
                d_model=d_model,
                device=self.device,
                dtype=getattr(drafter.model, "dtype", torch.float16),
            )

            # Replace the drafter reference with the universal drafter wrapper
            # We monkey-patch the expected interface: universal.draft() and universal.model
            class _UniversalDrafterAdapter:
                """Thin adapter to make UniversalDrafter match the DraftModel interface."""

                def __init__(self, base: DraftModel, universal: UniversalDrafter) -> None:
                    self.base = base
                    self.universal = universal
                    self.tokenizer = base.tokenizer
                    self.model = base.model

                def draft(
                    self,
                    context: torch.Tensor,
                    k: int,
                    distill: bool = False,
                    temperature: float = 1.0,
                ) -> tuple[list[int], torch.Tensor]:
                    """
                    Forward to either the base drafter (when distillation
                    needs gradients) or the universal drafter.

                    The base path forwards the temperature so the
                    drafter samples from the same ``q`` used by the
                    decoder's acceptance test (C1 fix). The universal
                    path is greedy (UniversalDrafter.draft uses argmax
                    under @torch.no_grad); it therefore does NOT support
                    stochastic decoding and is only correct under
                    greedy target decoding. Distillation with the
                    universal adapter is also unsupported (the universal
                    drafter cannot produce grad-enabled logits), so the
                    adapter falls back to the base drafter in that case.
                    """
                    if distill:
                        return self.base.draft(
                            context, k, distill=True, temperature=temperature
                        )
                    with torch.no_grad():
                        return self.universal.draft(
                            context, k, target_name=cfg.target_model_path
                        )

                def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
                    return self.universal.base_model(input_ids).logits.squeeze(0)

            # Find or replace the drafter in specs with the adapter
            for spec in specs if specs else []:
                if spec.name == cfg.drafter_model_path:
                    spec.model = _UniversalDrafterAdapter(drafter, universal)

            drafter = _UniversalDrafterAdapter(drafter, universal)
            logger.info("Universal drafter ready")

        # Adaptive drafting
        adaptive_fn = None
        if cfg.use_speedup_adaptive:
            logger.info("Building adaptive draft controller")
            from core.extensions.adaptive.speedup_predictor import (
                AdaptiveDraftController,
                SpeedupPredictor,
            )

            pred = SpeedupPredictor(
                d_hidden=drafter.model.config.hidden_size,
                k_max=cfg.k_max,
            ).to(self.device)
            controller = AdaptiveDraftController(pred, drafter, cfg.k_min, cfg.k_max)
            adaptive_fn = controller
            logger.info("Adaptive draft controller ready: k_min=%d k_max=%d", cfg.k_min, cfg.k_max)

        # Decoder
        logger.info(
            "Building speculative decoder: draft_length=%d temperature=%s", cfg.draft_length, 1.0
        )
        decoder = SpeculativeDecoder(
            drafter=drafter,
            target=target,
            translator=translator,
            cache=cache,
            draft_length=cfg.draft_length,
        )
        logger.info("Speculative decoder ready")

        # --- Load dataset ---
        logger.info("Loading dataset: %s max_samples=%d", cfg.dataset, cfg.max_samples)
        prompts = self._load_dataset(cfg.dataset, cfg.max_samples, target.tokenizer)
        logger.info("Loaded %d prompt(s)", len(prompts))

        # --- Run ---
        logger.info("Starting benchmark collection")
        collector = BenchmarkCollector(name=cfg.name)
        if _HAS_MLFLOW and cfg.mlflow_experiment:
            logger.info("Initializing MLflow experiment=%s run=%s", cfg.mlflow_experiment, cfg.name)
            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow.start_run(run_name=cfg.name)
            mlflow.log_params(asdict(cfg))

        # --- Routing stats ---
        n_routing_stats: dict = {"selected_drafter": 0, "drafter_counts": {}}

        for i, (input_ids, prompt_len) in enumerate(prompts):
            logger.info("Decoding prompt %d/%d prompt_len=%d", i + 1, len(prompts), prompt_len)
            input_ids = input_ids.to(self.device)

            # Select best drafter via router (if enabled)
            if router is not None:
                selected_drafter, selected_idx = router.select_drafter(input_ids)
                decoder.drafter = selected_drafter  # monkey-patch
                router_key = specs[selected_idx].name if specs else str(selected_idx)
                n_routing_stats["drafter_counts"][router_key] = (
                    n_routing_stats["drafter_counts"].get(router_key, 0) + 1
                )
                n_routing_stats["selected_drafter"] = selected_idx
                logger.info(
                    "Router selected drafter %s (%d/%d) for prompt %d",
                    router_key,
                    selected_idx,
                    len(specs) if specs else 1,
                    i,
                )

            with collector.record_sequence(prompt_len=prompt_len) as seq_rec:
                out = decoder.generate(
                    input_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    adaptive_length_fn=adaptive_fn,
                    distiller=distiller,
                )
                # Reconstruct step records from decoder stats
                for sr in decoder._step_results[-cfg.max_new_tokens :]:
                    seq_rec.add_step(
                        draft_len=sr.draft_length,
                        accepted=sr.accepted_count,
                        cache_hit=sr.cache_hit,
                    )
            # Reset step results for next sequence
            decoder._step_results.clear()
            logger.info("Finished prompt %d/%d output_len=%d", i + 1, len(prompts), out.shape[1])

            if i % cfg.log_every == 0:
                partial = collector.summary()
                logger.info(
                    "Progress [%d/%d] acc=%.3f tps=%.1f cache_hit=%.3f",
                    i + 1,
                    len(prompts),
                    partial.get("acceptance_rate", 0),
                    partial.get("tokens_per_sec", 0),
                    partial.get("cache_hit_rate", 0),
                )
                print(
                    f"  [{i}/{len(prompts)}] acc={partial.get('acceptance_rate', 0):.3f}  "
                    f"tps={partial.get('tokens_per_sec', 0):.1f}"
                )
                if _HAS_MLFLOW and cfg.mlflow_experiment:
                    metrics = {k: v for k, v in partial.items() if isinstance(v, (int, float))}
                    tags = {k: v for k, v in partial.items() if not isinstance(v, (int, float))}
                    mlflow.log_metrics(metrics, step=i)
                    for k, v in tags.items():
                        mlflow.set_tag(k, str(v))

        summary = collector.summary()
        # Add routing stats to summary
        if router is not None:
            summary["router"] = n_routing_stats
            logger.info("Router stats: %s", n_routing_stats)
        # Add distillation / contrastive stats
        if distiller is not None:
            distil_stats = distiller.training_stats()
            summary.update(distil_stats)
            logger.info("Distillation stats: %s", distil_stats)
        logger.info("Benchmark summary for %s: %s", cfg.name, summary)
        if _HAS_MLFLOW and cfg.mlflow_experiment:
            logger.info("Logging final metrics to MLflow")
            metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            tags = {k: v for k, v in summary.items() if not isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))
            mlflow.end_run()

        # Free GPU memory before returning
        logger.info("Freeing GPU memory before return")
        del drafter, target, translator, decoder, cache, distiller, router, collector
        import gc

        # Guard: torch.cuda.empty_cache() raises on CPU-only hosts.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory freed")
        return {"config": asdict(cfg), "metrics": summary}

    # ------------------------------------------------------------------
    # Dataset loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(name: str, max_samples: int, tokenizer) -> list:
        """Returns list of (input_ids_tensor, prompt_len) tuples."""

        logger.info("Loading dataset %s with max_samples=%d", name, max_samples)
        from datasets import load_dataset

        if name == "gsm8k":
            logger.info("Loading openai/gsm8k main test split")
            ds = load_dataset("openai/gsm8k", "main", split="test")
            texts = [ex["question"] for ex in ds]
        elif name == "mbpp":
            logger.info("Loading mbpp test split")
            ds = load_dataset("mbpp", split="test")
            texts = [ex["text"] for ex in ds]
        elif name == "humaneval":
            logger.info("Loading openai_humaneval test split")
            ds = load_dataset("openai_humaneval", split="test")
            texts = [ex["prompt"] for ex in ds]
        elif name == "alpaca":
            logger.info("Loading tatsu-lab/alpaca train split")
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            texts = [ex["instruction"] for ex in ds]
        elif name == "xsum":
            logger.info("Loading EdinburghNLP/xsum test split")
            ds = load_dataset("EdinburghNLP/xsum", split="test")
            texts = [ex["document"][:512] for ex in ds]
        else:
            raise ValueError(f"Unknown dataset: {name}")

        texts = texts[:max_samples]
        logger.info("Tokenizing %d text sample(s)", len(texts))
        result = []
        for t in texts:
            ids = tokenizer.encode(t, return_tensors="pt")
            result.append((ids, ids.shape[1]))
        logger.info("Tokenization complete")
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: dict) -> None:
        name = result["config"]["name"].replace(" ", "_")
        path = os.path.join(self.output_dir, f"{name}.json")
        logger.info("Saving experiment result to %s", path)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    def _write_csv(self, results: list[dict]) -> None:
        if not results:
            return
        path = os.path.join(self.output_dir, "comparison_table.csv")
        logger.info("Writing comparison table to %s", path)
        metric_keys = list(results[0]["metrics"].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment"] + metric_keys)
            writer.writeheader()
            for r in results:
                row = {"experiment": r["config"]["name"]}
                row.update({k: r["metrics"].get(k, "") for k in metric_keys})
                writer.writerow(row)
        print(f"\nComparison table saved to: {path}")
        logger.info("Comparison table saved")


# ---------------------------------------------------------------------------
# Pre-defined ablation suite
# ---------------------------------------------------------------------------

ABLATION_SUITE = [
    ExperimentConfig(name="01_baseline", use_rule1=True, use_rule2=True),
    ExperimentConfig(name="02_+lattice", use_rule1=True, use_lattice=True),
    ExperimentConfig(name="03_+translator", use_rule1=True, use_lattice=True, use_translator=True),
    ExperimentConfig(name="04_+online_distil", use_online_distil=True),
    ExperimentConfig(
        name="05_+replay_fifo", use_online_distil=True, use_replay=True, replay_strategy="fifo"
    ),
    ExperimentConfig(
        name="06_+replay_prio",
        use_online_distil=True,
        use_replay=True,
        replay_strategy="prioritized",
    ),
    ExperimentConfig(name="07_+contrastive", use_online_distil=True, use_contrastive=True),
    ExperimentConfig(name="08_+speedup_adapt", use_speedup_adaptive=True),
    ExperimentConfig(name="09_+routing", use_dynamic_routing=True),
    ExperimentConfig(name="10_+universal", use_universal_drafter=True),
    ExperimentConfig(
        name="11_full_system",
        use_rule1=True,
        use_lattice=True,
        use_translator=True,
        use_online_distil=True,
        use_replay=True,
        replay_strategy="prioritized",
        use_contrastive=True,
        use_speedup_adaptive=True,
        use_dynamic_routing=True,
        use_universal_drafter=True,
    ),
]
