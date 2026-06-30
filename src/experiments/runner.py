"""
experiments/runner.py

Experiment runner orchestrator.

Loads models and datasets, manages GPU memory, persists results to JSON/CSV,
and optionally logs to MLflow.  The runner does NOT contain experiment logic —
that lives in ``BaseExperiment`` subclasses.

Usage
-----
>>> from experiments import ExperimentRunner
>>> from experiments.suites import ABLATION_SUITE
>>> runner = ExperimentRunner(experiments=ABLATION_SUITE)
>>> results = runner.run_all()
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field

import torch

logger = logging.getLogger(__name__)

# Global log level, set by main.py before experiments run.
# Possible values: "QUIET", "NORMAL", "VERBOSE"
_log_level: str = "QUIET"

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
    """Base configuration for experiments.

    Used by ``BaseExperiment.get_config()`` to specify model paths,
    dataset, and hyperparameters.  This dataclass is retained as a
    convenient container; experiments may extend it or construct it
    from scratch.
    """

    name: str

    # Model paths
    drafter_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_model_path: str = "Qwen/Qwen2.5-7B-Instruct"

    # Model dtype — 4-bit trades VRAM for speed
    target_use_4bit: bool = True

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
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    accum_steps: int = 8

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
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    log_every: int = 50
    log_level: str = "QUIET"  # QUIET | NORMAL | VERBOSE
    seed: int = 42


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Orchestrates a list of ``BaseExperiment`` instances.

    Responsibilities
    ----------------
    - Load models and datasets (shared utilities)
    - Manage GPU memory between experiments
    - Persist results to JSON/CSV
    - Optional MLflow logging

    Experiment logic (building components, decode loop) lives in
    ``BaseExperiment`` subclasses, not here.

    Parameters
    ----------
    experiments :
        List of ``BaseExperiment`` instances to run.
    output_dir :
        Directory for result JSON/CSV files.
    device :
        Torch device string (``"cuda"``, ``"cpu"``, etc.).
    """

    def __init__(
        self,
        experiments: list[object] | None = None,  # list[BaseExperiment]
        output_dir: str = "results",
        device: str = "cuda",
    ) -> None:
        self.experiments = experiments or []
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "ExperimentRunner initialized: output_dir=%s device=%s experiments=%d",
            output_dir,
            device,
            len(self.experiments),
        )

    @staticmethod
    def _clear_gpu_memory() -> None:
        """Aggressively free GPU memory between experiments.

        Runs multiple GC passes and clears the CUDA cache.  For 4-bit
        (bitsandbytes) models this may not release *all* memory because
        the quantization state is held in a global C++ allocator, but
        it recovers the majority of fragmentation.
        """
        import gc

        import torch

        # Multiple GC passes to break reference cycles
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                logger.warning("CUDA synchronize failed — GPU may be in error state")
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                logger.warning("CUDA empty_cache failed — GPU may be in error state")
            try:
                logger.info(
                    "GPU memory after cleanup: %.1f MB used / %.1f MB reserved / %.1f MB total",
                    torch.cuda.memory_allocated() / 1e6,
                    torch.cuda.memory_reserved() / 1e6,
                    torch.cuda.get_device_properties(0).total_memory / 1e6,
                )
            except RuntimeError:
                logger.warning("Could not query GPU memory — GPU may be in error state")

    # ------------------------------------------------------------------
    # High-level run
    # ------------------------------------------------------------------

    def run_all(self) -> list[dict]:
        """Run all experiments and collect results.

        Returns
        -------
        list[dict]
            Each dict has keys ``config`` and ``metrics`` for
            backward compatibility with existing consumers.
        """
        results = []
        total = len(self.experiments)
        logger.info("Starting run_all for %d experiment(s)", total)

        for index, exp in enumerate(self.experiments, start=1):
            exp_name = exp.meta.name
            logger.info("Starting experiment %d/%d: %s", index, total, exp_name)
            print(f"\n{'=' * 60}")
            print(f"  Running: {exp_name}")
            print(f"{'=' * 60}")

            if index > 1:
                self._clear_gpu_memory()

            try:
                exp_result = self.run_one_experiment(exp)
                # Convert to legacy dict format for unified CSV/JSON output
                legacy_result = {
                    "config": exp_result.config,
                    "metrics": exp_result.metrics,
                }
                results.append(legacy_result)
                self._save_result(legacy_result)
                logger.info(
                    "Finished experiment %d/%d: %s",
                    index,
                    total,
                    exp_name,
                )
            except Exception as e:
                logger.error(
                    "Experiment %d/%d (%s) FAILED: %s",
                    index,
                    total,
                    exp_name,
                    e,
                )
                import traceback

                traceback.print_exc()
                failed_result = {
                    "config": {"name": exp_name},
                    "metrics": {
                        "error": True,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                }
                results.append(failed_result)
                self._save_result(failed_result)

            # Release experiment-level references (models, buffers, etc.)
            try:
                exp.cleanup()
            except Exception as e:
                logger.warning("Experiment cleanup failed for %s: %s", exp_name, e)

            self._clear_gpu_memory()

        self._write_csv(results)
        logger.info("Finished run_all with %d result(s)", len(results))
        return results

    # ------------------------------------------------------------------
    # Single experiment execution
    # ------------------------------------------------------------------

    def run_one_experiment(self, exp: object) -> object:  # BaseExperiment -> ExperimentResult
        """Run a single ``BaseExperiment`` and return its result.

        Parameters
        ----------
        exp : BaseExperiment
            The experiment to run.

        Returns
        -------
        ExperimentResult
            Contains meta, config dict, metrics dict, and optional error.
        """
        return exp.run(self)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _build_models(self, cfg: ExperimentConfig) -> tuple:
        """Load drafter and target models from config.

        Returns
        -------
        tuple[DraftModel, TargetModel]
        """
        from core.models.drafter import DraftModel, TargetModel

        logger.info("Loading drafter model: %s", cfg.drafter_model_path)
        drafter = DraftModel(cfg.drafter_model_path, device=self.device)

        # Respect target_use_4bit: FP16 is ~15-20%% faster for target_verify
        # but uses more VRAM; 4-bit NF4 saves VRAM with slight speed penalty.
        load_in_4bit = getattr(cfg, "target_use_4bit", True)
        target_dtype = torch.float16
        if load_in_4bit:
            logger.info("Loading target model in 4-bit NF4 (lower VRAM)")
        else:
            logger.info("Loading target model in FP16 (faster inference)")

        logger.info("Loading target model: %s", cfg.target_model_path)
        target = TargetModel(
            cfg.target_model_path,
            device=self.device,
            dtype=target_dtype,
            load_in_4bit=load_in_4bit,
        )
        return drafter, target

    @staticmethod
    def _asdict_config(cfg: ExperimentConfig) -> dict:
        """Serialize ExperimentConfig to dict (wrapper around dataclasses.asdict)."""
        return asdict(cfg)

    def _setup_mlflow(self, cfg: ExperimentConfig) -> None:
        """Initialize MLflow run if configured."""
        if _HAS_MLFLOW and getattr(cfg, "mlflow_experiment", None):
            # End any previously active run (e.g. from a failed experiment)
            active = mlflow.active_run()
            if active is not None:
                logger.warning(
                    "Active MLflow run %s found — ending before starting new run",
                    active.info.run_id,
                )
                mlflow.end_run()
            logger.info(
                "Initializing MLflow experiment=%s run=%s",
                cfg.mlflow_experiment,
                cfg.name,
            )
            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow.start_run(run_name=cfg.name)
            mlflow.log_params(asdict(cfg))

    def _log_mlflow_final(self, cfg: ExperimentConfig, summary: dict) -> None:
        """Log final metrics to MLflow and end the run."""
        if _HAS_MLFLOW and getattr(cfg, "mlflow_experiment", None):
            logger.info("Logging final metrics to MLflow")
            metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            tags = {k: v for k, v in summary.items() if not isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))
            mlflow.end_run()

    # ------------------------------------------------------------------
    # Dataset loader
    # ------------------------------------------------------------------

    def _load_dataset(self, cfg: ExperimentConfig) -> list:
        """Load and tokenize dataset from config.

        Returns
        -------
        list[tuple[torch.Tensor, int]]
            List of (input_ids_tensor, prompt_len) tuples.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_path)
        return self._load_dataset_with_tokenizer(cfg.dataset, cfg.max_samples, tokenizer)

    @staticmethod
    def _load_dataset_with_tokenizer(name: str, max_samples: int, tokenizer) -> list:
        """Load dataset and tokenize with the given tokenizer.

        Returns
        -------
        list[tuple[torch.Tensor, int]]
            List of (input_ids_tensor, prompt_len) tuples.
        """
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
        logger.info("Tokenizing %d text sample(s) with batched encoding", len(texts))

        # Batch encoding: one call per chunk instead of per-sample loop.
        # This avoids Python-C++ boundary overhead (~50-200us per sample)
        # and enables fused kernel execution inside the tokenizer.
        chunk_size = 256
        result = []
        for chunk_start in range(0, len(texts), chunk_size):
            chunk = texts[chunk_start : chunk_start + chunk_size]

            encodings = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids_batch = encodings.input_ids  # (chunk_len, max_seq_len)

            for i in range(len(chunk)):
                ids = input_ids_batch[i].unsqueeze(0)  # (1, seq_len)
                prompt_len = ids.shape[1]
                result.append((ids, prompt_len))

        logger.info("Tokenization complete: %d samples", len(result))
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: dict) -> None:
        """Save a single experiment result to JSON."""
        name = result["config"]["name"].replace(" ", "_")
        path = os.path.join(self.output_dir, f"{name}.json")
        logger.info("Saving experiment result to %s", path)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    def _write_csv(self, results: list[dict]) -> None:
        """Write a comparison CSV from all results."""
        if not results:
            return
        path = os.path.join(self.output_dir, "comparison_table.csv")
        logger.info("Writing comparison table to %s", path)
        metric_keys = list(results[0]["metrics"].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", *metric_keys])
            writer.writeheader()
            for r in results:
                row = {"experiment": r["config"]["name"]}
                row.update({k: r["metrics"].get(k, "") for k in metric_keys})
                writer.writerow(row)
        print(f"\nComparison table saved to: {path}")
        logger.info("Comparison table saved")
