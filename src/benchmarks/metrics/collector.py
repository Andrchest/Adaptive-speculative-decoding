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
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rich.console import Console

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
        # Use accepted_draft (excludes bonus tokens) when available,
        # fall back to accepted for backward compatibility.
        total_a = sum(
            (r.accepted_draft if r.accepted_draft >= 0 else r.accepted)
            for r in self.step_records
        )
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
    accepted_draft: int = -1  # -1 = fall back to `accepted` (backward compat)


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
            accepted_draft: int = -1,
        ) -> None:
            self._rec.step_records.append(StepRecord(draft_len, accepted, cache_hit, kl_div, actual_draft_len, accepted_draft))
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

    def print_end_summary(self, result: dict) -> None:
        """Print a detailed summary block after each experiment.

        Called once per experiment, after collector.summary() is computed.
        Uses rich for colored table output (falls back to plain print if
        rich is unavailable).
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text

            console = Console(file=sys.stderr)
        except ImportError:
            # Fallback: plain text
            lines = self._format_summary_plain(result)
            for line in lines:
                sys.stderr.write(line + "\n")
            sys.stderr.write("\n")
            return

        # Build the summary panel
        lines = []
        lines.append(f"  Experiment: {self.name}")
        lines.append("")

        # --- Duration ---
        wall_total = result.get("wall_time_total_s", 0)
        wall_mean = result.get("wall_time_mean_s", 0)
        n_seq = result.get("n_sequences", 0)
        lines.append(f"  Duration: {wall_total:.3f}s  ({wall_mean:.3f}s per sample, {n_seq} sequences)")

        # --- Throughput ---
        tps = result.get("tokens_per_sec", 0)
        avg_tps = result.get("avg_tokens_per_sec", 0)
        total_tokens = result.get("total_new_tokens", 0)
        lines.append(f"  Throughput: {tps:.1f} tok/s  (avg {avg_tps:.1f} tok/s, {total_tokens} tokens)")

        # --- Acceptance ---
        acc_rate = result.get("acceptance_rate", 0)
        avg_acc = result.get("avg_accepted_tokens", 0)
        avg_draft = result.get("avg_draft_length", 0)
        lines.append(f"  Acceptance: {acc_rate:.1%}  ({avg_acc:.2f}/{avg_draft:.2f} avg accepted / draft)")

        # --- Cache ---
        cache_hit = result.get("cache_hit_rate", 0)
        lines.append(f"  Cache hit:  {cache_hit:.1%}")

        # --- GPU ---
        gpu_peak = result.get("gpu_mem_peak_gb", 0)
        gpu_mean = result.get("gpu_mem_mean_gb", 0)
        lines.append(f"  GPU: peak={gpu_peak:.2f} GB  mean={gpu_mean:.2f} GB")

        # --- Loss metrics (if distillation active) ---
        if "training_loss_mean" in result:
            loss_mean = result["training_loss_mean"]
            loss_std = result.get("training_loss_std", 0)
            kl = result.get("mean_kl_divergence", 0)
            lines.append(f"  Loss: mean={loss_mean:.2f}  kl={kl:.2f}")

        # --- Status ---
        sep = "─" * 55
        lines.append(f"  {sep}")
        lines.append("  ✓ Experiment complete")

        panel_text = "\n".join(lines)
        panel = Panel(
            panel_text,
            title="[bold]Experiment Summary[/]",
            subtitle="[dim]" + self.name + "[/]",
            border_style="bright_green",
            padding=(0, 2),
        )
        console.print(panel)
        console.print()

    def _format_summary_plain(self, result: dict) -> list[str]:
        """Plain-text summary (no rich dependency)."""
        wall_total = result.get("wall_time_total_s", 0)
        wall_mean = result.get("wall_time_mean_s", 0)
        n_seq = result.get("n_sequences", 0)
        tps = result.get("tokens_per_sec", 0)
        avg_tps = result.get("avg_tokens_per_sec", 0)
        total_tokens = result.get("total_new_tokens", 0)
        acc_rate = result.get("acceptance_rate", 0)
        avg_acc = result.get("avg_accepted_tokens", 0)
        avg_draft = result.get("avg_draft_length", 0)
        cache_hit = result.get("cache_hit_rate", 0)
        gpu_peak = result.get("gpu_mem_peak_gb", 0)
        gpu_mean = result.get("gpu_mem_mean_gb", 0)

        lines = []
        lines.append(f"\n{'=' * 55}")
        lines.append(f"  Experiment: {self.name}")
        lines.append(f"  Duration: {wall_total:.3f}s  ({wall_mean:.3f}s per sample, {n_seq} sequences)")
        lines.append(f"  Throughput: {tps:.1f} tok/s  (avg {avg_tps:.1f} tok/s, {total_tokens} tokens)")
        lines.append(f"  Acceptance: {acc_rate:.1%}  ({avg_acc:.2f}/{avg_draft:.2f} avg accepted / draft)")
        lines.append(f"  Cache hit:  {cache_hit:.1%}")
        lines.append(f"  GPU: peak={gpu_peak:.2f} GB  mean={gpu_mean:.2f} GB")
        if "training_loss_mean" in result:
            loss_mean = result["training_loss_mean"]
            kl = result.get("mean_kl_divergence", 0)
            lines.append(f"  Loss: mean={loss_mean:.2f}  kl={kl:.2f}")
        lines.append(f"  {'─' * 55}")
        lines.append("  ✓ Experiment complete")
        lines.append(f"{'=' * 55}\n")
        return lines

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
