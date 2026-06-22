"""
benchmarks/metrics/collector.py

Benchmark metrics collector.

Tracks all metrics defined in the spec:
  - Acceptance Rate
  - Average Accepted Tokens
  - Draft Length
  - Cache Hit Rate
  - Translator Accuracy
  - KL Divergence
  - Training Stability (loss variance)
  - Wall-Clock Speedup
  - Tokens/sec
  - Memory Usage
  - GPU Utilization
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class DecodeRecord:
    """Single decode sequence record."""

    prompt_len: int
    total_new_tokens: int
    wall_time_s: float
    step_records: list[StepRecord] = field(default_factory=list)

    @property
    def tokens_per_sec(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_new_tokens / self.wall_time_s

    @property
    def acceptance_rate(self) -> float:
        if not self.step_records:
            return 0.0
        total_a = sum(r.accepted for r in self.step_records)
        # Use actual_draft_len when available, fall back to draft_len
        total_d = sum(r.actual_draft_len if r.actual_draft_len > 0 else r.draft_len for r in self.step_records)
        return total_a / max(1, total_d)


@dataclass
class StepRecord:
    draft_len: int
    accepted: int
    cache_hit: bool
    kl_div: float = 0.0
    actual_draft_len: int = 0  # 0 = unknown (kept for backward compat)


class BenchmarkCollector:
    """
    Collects and aggregates metrics across multiple runs.

    Usage::

        collector = BenchmarkCollector(name="Adaptive-Baseline")
        with collector.record_sequence(prompt_len=50) as rec:
            for step in decode_steps:
                rec.add_step(draft_len=5, accepted=3, cache_hit=True)
        collector.finalize()
        print(collector.summary())
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._records: list[DecodeRecord] = []
        self._gpu_mem_samples: list[float] = []
        self._loss_samples: list[float] = []
        self._kl_samples: list[float] = []

        # Baseline for speedup computation
        self._baseline_tps: float | None = None
        logger.info("BenchmarkCollector initialized: name=%s", name)

    # ------------------------------------------------------------------
    # Context manager for recording a single decode sequence
    # ------------------------------------------------------------------

    class _SequenceContext:
        def __init__(self, collector: BenchmarkCollector, prompt_len: int) -> None:
            self._col = collector
            self._rec = DecodeRecord(
                prompt_len=prompt_len,
                total_new_tokens=0,
                wall_time_s=0.0,
            )
            self._t0 = 0.0

        def __enter__(self) -> BenchmarkCollector._SequenceContext:
            self._t0 = time.perf_counter()
            return self

        def add_step(
            self,
            draft_len: int,
            accepted: int,
            cache_hit: bool = False,
            kl_div: float = 0.0,
            actual_draft_len: int = 0,
        ) -> None:
            self._rec.step_records.append(StepRecord(draft_len, accepted, cache_hit, kl_div, actual_draft_len))
            self._rec.total_new_tokens += accepted

        def __exit__(self, *args) -> None:
            self._rec.wall_time_s = time.perf_counter() - self._t0
            self._col._records.append(self._rec)
            logger.debug(
                "Sequence recorded: prompt_len=%d new_tokens=%d wall_time=%.3fs acc=%.3f",
                self._rec.prompt_len,
                self._rec.total_new_tokens,
                self._rec.wall_time_s,
                self._rec.acceptance_rate,
            )

    def record_sequence(self, prompt_len: int) -> _SequenceContext:
        return self._SequenceContext(self, prompt_len)

    # ------------------------------------------------------------------
    # Standalone add methods (for use outside context manager)
    # ------------------------------------------------------------------

    def add_loss(self, loss: float) -> None:
        self._loss_samples.append(loss)

    def add_kl(self, kl: float) -> None:
        self._kl_samples.append(kl)

    def sample_gpu_memory(self, device: str = "cuda") -> None:
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / 1024**3  # GB
            self._gpu_mem_samples.append(mem)

    def set_baseline_tps(self, tps: float) -> None:
        """Set autoregressive-only baseline for speedup computation."""
        self._baseline_tps = tps

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self._records:
            logger.warning("No records to summarize for %s", self.name)
            return {"name": self.name, "n_sequences": 0}

        all_acc_rates = [r.acceptance_rate for r in self._records]
        all_tps = [r.tokens_per_sec for r in self._records]
        all_draft_lens = [s.draft_len for r in self._records for s in r.step_records]
        all_accepted = [s.accepted for r in self._records for s in r.step_records]
        cache_hits = [s.cache_hit for r in self._records for s in r.step_records]

        # Overall TPS: total new tokens / total wall time (unbiased)
        total_new_tokens = sum(r.total_new_tokens for r in self._records)
        total_wall_time = sum(r.wall_time_s for r in self._records)
        overall_tps = total_new_tokens / max(total_wall_time, 1e-9)
        avg_tps = sum(all_tps) / len(all_tps)

        speedup = (overall_tps / self._baseline_tps) if self._baseline_tps else None

        result = {
            "name": self.name,
            "n_sequences": len(self._records),
            "acceptance_rate": _mean(all_acc_rates),
            "avg_accepted_tokens": _mean(all_accepted),
            "avg_draft_length": _mean(all_draft_lens),
            "cache_hit_rate": _mean([float(h) for h in cache_hits]),
            "tokens_per_sec": overall_tps,
            "avg_tokens_per_sec": avg_tps,
            "total_new_tokens": total_new_tokens,
            "wall_time_total_s": total_wall_time,
            "wall_time_mean_s": _mean([r.wall_time_s for r in self._records]),
            "gpu_mem_peak_gb": max(self._gpu_mem_samples, default=0.0),
            "gpu_mem_mean_gb": _mean(self._gpu_mem_samples) if self._gpu_mem_samples else 0.0,
        }
        if speedup is not None:
            result["wall_clock_speedup"] = speedup
        if self._kl_samples:
            result["mean_kl_divergence"] = _mean(self._kl_samples)
        if self._loss_samples:
            result["training_loss_mean"] = _mean(self._loss_samples)
            result["training_loss_std"] = _std(self._loss_samples)
        logger.info(
            "Benchmark summary for %s: n_seq=%d acc=%.3f tps=%.1f cache_hit=%.3f",
            self.name,
            len(self._records),
            result.get("acceptance_rate", 0),
            result.get("tokens_per_sec", 0),
            result.get("cache_hit_rate", 0),
        )
        return result

    def clear(self) -> None:
        """Release all collected records and samples.

        Call this after the experiment is fully complete to free memory
        from DecodeRecord / StepRecord objects. Does NOT clear the
        baseline_tps or name.
        """
        self._records.clear()
        self._gpu_mem_samples.clear()
        self._loss_samples.clear()
        self._kl_samples.clear()

    def compare(self, other: BenchmarkCollector) -> dict:
        """Return delta metrics between self and other.

        Both collectors' records remain intact after comparison;
        call ``clear()`` explicitly when done.
        """
        a = self.summary()
        b = other.summary()
        keys = [
            "acceptance_rate",
            "tokens_per_sec",
            "wall_clock_speedup",
            "cache_hit_rate",
            "mean_kl_divergence",
        ]
        delta = {}
        for k in keys:
            if k in a and k in b:
                delta[f"{k}_delta"] = a[k] - b[k]
        return delta


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var**0.5
