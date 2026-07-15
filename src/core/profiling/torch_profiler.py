"""
core/profiling/torch_profiler.py

GPU kernel-level profiling with torch.profiler to identify specific bottlenecks:

1. aten::mm / aten::matmul >30% of GPU time → Rule2 dense matmul bottleneck
2. aten::to, aten::copy_, aten::_local_scalar_dense in top-10 → host sync overhead
3. aten::index_add_ with very large self tensors → Rule1/scatter overhead
4. Python wall time per step >> GPU kernel time → Python dispatch overhead
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class KernelStat:
    """Aggregated stats for a single ATen/CUDA kernel."""

    name: str
    cpu_time_us: float = 0.0
    gpu_time_us: float = 0.0
    count: int = 0
    cpu_pct: float = 0.0
    gpu_pct: float = 0.0


@dataclass
class Bottleneck:
    """A detected performance bottleneck."""

    category: str  # e.g. "rule2_dense_matmul", "host_sync", "rule1_scatter", "python_dispatch"
    severity: str  # "critical", "warning", "info"
    message: str
    detail: str = ""
    value: float = 0.0  # the measured value (e.g. percentage)
    threshold: float = 0.0  # the threshold it was compared against


@dataclass
class TorchProfilerAnalysis:
    """Complete analysis result from torch.profiler."""

    # Raw kernel stats
    kernels_by_gpu_time: list[KernelStat] = field(default_factory=list)
    kernels_by_cpu_time: list[KernelStat] = field(default_factory=list)

    # Aggregate metrics
    total_cpu_time_us: float = 0.0
    total_gpu_time_us: float = 0.0
    python_wall_time_us: float = 0.0
    cuda_kernel_time_us: float = 0.0

    # Detected bottlenecks
    bottlenecks: list[Bottleneck] = field(default_factory=list)

    # Per-step breakdown (if multiple profiler steps)
    step_count: int = 0

    # Summary flags
    has_rule2_bottleneck: bool = False
    has_host_sync_overhead: bool = False
    has_rule1_scatter_overhead: bool = False
    has_python_dispatch_overhead: bool = False

    @property
    def gpu_matmul_pct(self) -> float:
        """Combined GPU time percentage of aten::mm + aten::matmul."""
        return sum(
            k.gpu_pct for k in self.kernels_by_gpu_time if k.name in ("aten::mm", "aten::matmul")
        )

    @property
    def host_sync_kernels_in_top10(self) -> list[str]:
        """Host sync kernels found in top-10 by GPU time."""
        sync_names = {"aten::to", "aten::copy_", "aten::_local_scalar_dense"}
        return [k.name for k in self.kernels_by_gpu_time[:10] if k.name in sync_names]

    @property
    def index_add_large_self(self) -> list[KernelStat]:
        """aten::index_add_ kernels with large self tensors."""
        return [k for k in self.kernels_by_gpu_time if k.name == "aten::index_add_" and k.count > 0]

    @property
    def python_gpu_ratio(self) -> float:
        """Ratio of Python wall time to GPU kernel time."""
        if self.cuda_kernel_time_us <= 0:
            return float("inf")
        return self.python_wall_time_us / self.cuda_kernel_time_us


# ──────────────────────────────────────────────────────────────────────
# Kernel aggregation
# ──────────────────────────────────────────────────────────────────────


def _aggregate_kineto_results(prof) -> dict[str, dict]:
    """
    Parse profiler results using key_averages() (stable PyTorch API).

    Returns a dict of kernel_name -> {cpu_time_us, gpu_time_us, count}.
    """
    kernel_stats: dict[str, dict] = defaultdict(
        lambda: {"cpu_time_us": 0.0, "gpu_time_us": 0.0, "count": 0}
    )

    try:
        for kavg in prof.key_averages():
            name = kavg.key
            cpu_us = kavg.cpu_time_total  # already in microseconds
            # device_time_total is the modern API (replaces cuda_time_total)
            gpu_us = getattr(kavg, "device_time_total", 0.0) or 0.0
            count = kavg.count

            kernel_stats[name]["cpu_time_us"] += cpu_us
            kernel_stats[name]["gpu_time_us"] += gpu_us
            kernel_stats[name]["count"] += count
    except Exception as e:
        logger.warning("Failed to parse profiler key_averages: %s", e)

    return kernel_stats


def _build_kernel_lists(
    kernel_stats: dict[str, dict],
) -> tuple[list[KernelStat], list[KernelStat]]:
    """Build sorted kernel stat lists and compute percentages."""
    total_cpu = sum(v["cpu_time_us"] for v in kernel_stats.values())
    total_gpu = sum(v["gpu_time_us"] for v in kernel_stats.values())

    kernels = []
    for name, stats in kernel_stats.items():
        ks = KernelStat(
            name=name,
            cpu_time_us=stats["cpu_time_us"],
            gpu_time_us=stats["gpu_time_us"],
            count=stats["count"],
            cpu_pct=(stats["cpu_time_us"] / total_cpu * 100) if total_cpu > 0 else 0.0,
            gpu_pct=(stats["gpu_time_us"] / total_gpu * 100) if total_gpu > 0 else 0.0,
        )
        kernels.append(ks)

    by_gpu = sorted(kernels, key=lambda k: k.gpu_time_us, reverse=True)
    by_cpu = sorted(kernels, key=lambda k: k.cpu_time_us, reverse=True)

    return by_gpu, by_cpu


# ──────────────────────────────────────────────────────────────────────
# Bottleneck detection
# ──────────────────────────────────────────────────────────────────────


def _detect_bottlenecks(analysis: TorchProfilerAnalysis) -> list[Bottleneck]:
    """
    Run the 4 specific bottleneck checks:

    1. aten::mm / aten::matmul >30% GPU time → Rule2 dense matmul
    2. aten::to, aten::copy_, aten::_local_scalar_dense in top-10 → host sync
    3. aten::index_add_ with large self tensors → Rule1/scatter
    4. Python wall time >> GPU kernel time → Python dispatch overhead
    """
    bottlenecks: list[Bottleneck] = []

    # ── Check 1: Rule2 dense matmul bottleneck ──
    matmul_pct = analysis.gpu_matmul_pct
    if matmul_pct > 30.0:
        b = Bottleneck(
            category="rule2_dense_matmul",
            severity="critical",
            message=(
                f"aten::mm + aten::matmul consume {matmul_pct:.1f}% of GPU time "
                f"(threshold: 30%). This confirms Rule2 dense matmul is the bottleneck."
            ),
            detail=(
                "The CrossVocabTranslator's Rule2Mapping uses a dense "
                "(target_vocab x drafter_vocab) matrix multiplied with drafter "
                "probabilities. For OPT models (~50K vocab), this is a large "
                "matrix multiplication every decode step. Consider: "
                "(a) sparse matmul, (b) TokenizerLattice extension, "
                "(c) pre-computing the transfer once per session."
            ),
            value=matmul_pct,
            threshold=30.0,
        )
        bottlenecks.append(b)
        analysis.has_rule2_bottleneck = True
    elif matmul_pct > 15.0:
        b = Bottleneck(
            category="rule2_dense_matmul",
            severity="warning",
            message=(
                f"aten::mm + aten::matmul consume {matmul_pct:.1f}% of GPU time "
                f"(approaching 30% threshold). Rule2 dense matmul may become dominant."
            ),
            value=matmul_pct,
            threshold=30.0,
        )
        bottlenecks.append(b)

    # ── Check 2: Host sync overhead ──
    sync_in_top10 = analysis.host_sync_kernels_in_top10
    if sync_in_top10:
        sync_kernels_str = ", ".join(sync_in_top10)
        # Compute total sync GPU time
        sync_gpu_pct = sum(
            k.gpu_pct for k in analysis.kernels_by_gpu_time[:10] if k.name in sync_in_top10
        )
        b = Bottleneck(
            category="host_sync",
            severity="critical",
            message=(
                f"Host sync kernels in top-10 GPU time: {sync_kernels_str} "
                f"(combined {sync_gpu_pct:.1f}% of GPU time). "
                f"This confirms host-device synchronization overhead."
            ),
            detail=(
                "aten::_local_scalar_dense triggers a CUDA sync to read a single "
                "scalar from GPU to CPU. aten::to with non_blocking=False and "
                "aten::copy_ can also cause implicit syncs. Common sources in "
                "this codebase: .item() calls, .cpu().tolist() for accept/reject, "
                "and tensor list rebuilds in cache ops."
            ),
            value=sync_gpu_pct,
            threshold=0.0,
        )
        bottlenecks.append(b)
        analysis.has_host_sync_overhead = True

    # ── Check 3: Rule1/scatter overhead ──
    index_add_kernels = analysis.index_add_large_self
    if index_add_kernels:
        total_index_add_pct = sum(k.gpu_pct for k in index_add_kernels)
        total_index_add_count = sum(k.count for k in index_add_kernels)
        b = Bottleneck(
            category="rule1_scatter",
            severity="warning",
            message=(
                f"aten::index_add_ appears {total_index_add_count} times with "
                f"{total_index_add_pct:.1f}% of GPU time. "
                f"This confirms Rule1/scatter overhead."
            ),
            detail=(
                "Rule1Mapping.map_logits() uses scatter_add_ to map drafter "
                "probability mass to target vocab positions. With ~50K vocab, "
                "this operates on large tensors. The current implementation uses "
                "gather + scatter_add_ to avoid OOM from fancy indexing, but "
                "scatter_add_ itself can be slow for large vocab sizes."
            ),
            value=total_index_add_pct,
            threshold=0.0,
        )
        bottlenecks.append(b)
        analysis.has_rule1_scatter_overhead = True

    # Also check scatter_ and gather_ which are part of Rule1
    scatter_kernels = [
        k
        for k in analysis.kernels_by_gpu_time
        if k.name in ("aten::scatter_", "aten::scatter_add_", "aten::gather")
    ]
    if scatter_kernels:
        scatter_pct = sum(k.gpu_pct for k in scatter_kernels)
        if scatter_pct > 5.0:
            b = Bottleneck(
                category="rule1_scatter",
                severity="info",
                message=(
                    f"Related scatter/gather kernels: {', '.join(k.name for k in scatter_kernels)} "
                    f"consume {scatter_pct:.1f}% GPU time (Rule1 pipeline)."
                ),
                value=scatter_pct,
                threshold=5.0,
            )
            bottlenecks.append(b)

    # ── Check 4: Python dispatch overhead ──
    python_ratio = analysis.python_gpu_ratio
    if python_ratio > 1.5:
        b = Bottleneck(
            category="python_dispatch",
            severity="critical" if python_ratio > 3.0 else "warning",
            message=(
                f"Python wall time ({analysis.python_wall_time_us / 1000:.1f}ms) is "
                f"{python_ratio:.1f}x GPU kernel time ({analysis.cuda_kernel_time_us / 1000:.1f}ms). "
                f"Python dispatch overhead is dominant."
            ),
            detail=(
                "High Python/GPU ratio indicates the decode loop spends more time "
                "in Python code than GPU kernels. Common causes: list rebuilds "
                "(ctx_list, drafter_context_ids), cache operations (dict lookups, "
                "OrderedDict moves), .tolist()/.item() calls, and tensor "
                "creation/movement. Consider: pre-allocated buffers, fused ops."
            ),
            value=python_ratio,
            threshold=1.5,
        )
        bottlenecks.append(b)
        analysis.has_python_dispatch_overhead = True
    elif python_ratio > 1.2:
        b = Bottleneck(
            category="python_dispatch",
            severity="info",
            message=(
                f"Python wall time is {python_ratio:.1f}x GPU kernel time "
                f"(approaching overhead threshold)."
            ),
            value=python_ratio,
            threshold=1.5,
        )
        bottlenecks.append(b)

    return bottlenecks


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def run_torch_profile(
    decoder,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    draft_length: int = 5,
    warmup_steps: int = 2,
    active_steps: int = 5,
    activities: list[ProfilerActivity] | None = None,
    with_stack: bool = False,
    output_dir: str = "",
    record_shapes: bool = False,
    profile_memory: bool = True,
    distiller=None,
    adaptive_fn=None,
    rng: torch.Generator | None = None,
) -> TorchProfilerAnalysis:
    """
    Run torch.profiler on the speculative decoding loop and analyze results.

    Parameters
    ----------
    decoder : SpeculativeDecoder
        The decoder instance to profile.
    input_ids : torch.Tensor
        Input prompt token IDs (1, prompt_len).
    max_new_tokens : int
        Max new tokens to generate.
    draft_length : int
        Fixed draft length for profiling.
    warmup_steps : int
        Number of decode steps to warm up before profiling starts.
    active_steps : int
        Number of decode steps to profile.
    activities : list[ProfilerActivity], optional
        Profiler activities. Default: CPU + CUDA.
    with_stack : bool
        Whether to record source info (line numbers).
    output_dir : str
        Directory for TensorBoard trace output.
    record_shapes : bool
        Whether to record tensor shapes.
    profile_memory : bool
        Whether to profile memory usage.
    distiller : optional
        Online distiller instance.
    adaptive_fn : optional
        Adaptive draft length function.
    rng : torch.Generator, optional
        Random number generator.

    Returns
    -------
    TorchProfilerAnalysis
        Complete analysis with kernel stats and bottleneck detection.
    """
    if activities is None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    device = str(input_ids.device)

    # Ensure CUDA is ready
    if "cuda" in device:
        torch.cuda.synchronize()

    analysis = TorchProfilerAnalysis()

    logger.info(
        "Starting torch.profiler: warmup=%d active=%d max_tokens=%d draft_length=%d",
        warmup_steps,
        active_steps,
        max_new_tokens,
        draft_length,
    )

    # ── Warmup phase (no profiling) ──
    import contextlib

    grad_ctx = contextlib.nullcontext() if distiller is not None else torch.no_grad()

    if warmup_steps > 0:
        logger.info("Running %d warmup steps...", warmup_steps)
        with grad_ctx:
            _run_decode_steps(
                decoder,
                input_ids.clone(),
                max_new_tokens=min(warmup_steps * draft_length, max_new_tokens),
                draft_length=draft_length,
                distiller=distiller,
                adaptive_fn=adaptive_fn,
                rng=rng,
            )
        if "cuda" in device:
            torch.cuda.synchronize()
        decoder._drafter_kv = None
        decoder._drafter_kv_len = 0
        decoder._cached_drafter_logits = None
        decoder._target_kv = None
        decoder._step_results.clear()

    # ── Profiled phase ──
    logger.info("Running %d profiled steps...", active_steps)
    trace_dir = output_dir if output_dir else None

    kwargs = {}
    if trace_dir:
        kwargs["on_trace_ready"] = tensorboard_trace_handler(trace_dir)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        **kwargs,
    ) as prof:
        with grad_ctx:
            _run_decode_steps(
                decoder,
                input_ids.clone(),
                max_new_tokens=min(active_steps * draft_length, max_new_tokens),
                draft_length=draft_length,
                distiller=distiller,
                adaptive_fn=adaptive_fn,
                rng=rng,
            )

        if "cuda" in device:
            torch.cuda.synchronize()

    # ── Parse results ──
    kernel_stats = _aggregate_kineto_results(prof)
    by_gpu, by_cpu = _build_kernel_lists(kernel_stats)

    analysis.kernels_by_gpu_time = by_gpu
    analysis.kernels_by_cpu_time = by_cpu
    analysis.total_cpu_time_us = sum(v["cpu_time_us"] for v in kernel_stats.values())
    analysis.total_gpu_time_us = sum(v["gpu_time_us"] for v in kernel_stats.values())

    # Extract Python wall time and CUDA kernel time from profiler stats.
    # The "cuda" event key doesn't always exist. Use device_time_total from
    # all events to get actual GPU kernel time, and compute Python overhead
    # as total_cpu_time - total_gpu_time (CPU time includes GPU kernel
    # launch overhead, so the difference is a proxy for Python dispatch).
    try:
        total_device_us = 0.0
        total_cpu_us = 0.0
        for s in prof.key_averages():
            total_device_us += getattr(s, "device_time_total", 0.0) or 0.0
            total_cpu_us += s.cpu_time_total
        analysis.cuda_kernel_time_us = total_device_us
        analysis.python_wall_time_us = max(total_cpu_us - total_device_us, 0.0)
    except Exception:
        analysis.python_wall_time_us = analysis.total_cpu_time_us
        analysis.cuda_kernel_time_us = analysis.total_gpu_time_us

    analysis.step_count = active_steps

    # ── Detect bottlenecks ──
    analysis.bottlenecks = _detect_bottlenecks(analysis)

    logger.info(
        "torch.profiler analysis complete: %d kernels, %d bottlenecks detected",
        len(by_gpu),
        len(analysis.bottlenecks),
    )

    return analysis


def _run_decode_steps(
    decoder,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_length: int,
    distiller=None,
    adaptive_fn=None,
    rng=None,
):
    """Run the decode loop for a fixed number of steps.

    Maintains a separate drafter_context_ids list for cross-vocab pairs
    (e.g. pythia -> qwen) so the drafter always receives its own vocab IDs.
    """
    prompt_len = input_ids.shape[1]
    generated = input_ids.clone()
    # Drafter vocab context — must be maintained separately from generated
    # because generated contains target-vocab accepted tokens.
    drafter_context_ids: list[int] = generated[0].tolist()

    for _ in range(max_new_tokens):
        if generated.shape[1] - prompt_len >= max_new_tokens:
            break

        k = decoder._choose_draft_length(generated, adaptive_fn)

        # Build drafter context tensor from drafter vocab token IDs
        drafter_ctx = torch.tensor(
            [drafter_context_ids], dtype=generated.dtype, device=generated.device
        )

        result = decoder._decode_step(
            generated,
            k,
            drafter_context_ids[:],
            drafter_ctx=drafter_ctx,
            distiller=distiller,
            rng=rng,
        )
        decoder._step_results.append(result)
        decoder.cache.step()

        if result.accepted_tokens:
            new_ids = torch.tensor(
                result.accepted_tokens,
                dtype=torch.long,
                device=generated.device,
            ).unsqueeze(0)
            generated = torch.cat([generated, new_ids], dim=1)

            # Translate accepted tokens (target vocab) back to drafter vocab
            if not decoder._same_vocab:
                drafter_emitted = decoder.translator.translate_target_to_drafter(
                    result.accepted_tokens
                )
            else:
                drafter_emitted = result.accepted_tokens
            drafter_context_ids.extend(drafter_emitted)
        else:
            break

        if decoder._is_eos(generated[0, -1]):
            break


# ──────────────────────────────────────────────────────────────────────
# Rich rendering (for integration with existing profiler.py)
# ──────────────────────────────────────────────────────────────────────


def render_torch_profile_table(analysis: TorchProfilerAnalysis):
    """Render a rich Table with torch.profiler results. Returns (Table, Table)."""
    from rich.table import Table

    # ── Top kernels by GPU time ──
    gpu_table = Table(
        title="Top Kernels by GPU Time (torch.profiler)",
        collapse_padding=True,
    )
    gpu_table.add_column("#", justify="right", width=4)
    gpu_table.add_column("Kernel", style="cyan", width=36)
    gpu_table.add_column("GPU (ms)", justify="right", width=12)
    gpu_table.add_column("GPU %", justify="right", width=8)
    gpu_table.add_column("CPU (ms)", justify="right", width=12)
    gpu_table.add_column("Count", justify="right", width=8)

    for i, k in enumerate(analysis.kernels_by_gpu_time[:20], 1):
        gpu_ms = k.gpu_time_us / 1000
        cpu_ms = k.cpu_time_us / 1000
        style = ""
        if k.name in ("aten::mm", "aten::matmul") and k.gpu_pct > 10:
            style = "bold red"
        elif k.name in ("aten::to", "aten::copy_", "aten::_local_scalar_dense"):
            style = "bold yellow"
        elif k.name == "aten::index_add_":
            style = "bold magenta"

        gpu_table.add_row(
            str(i),
            f"[{style}]{k.name}[/{style}]" if style else k.name,
            f"{gpu_ms:.3f}",
            f"{k.gpu_pct:.1f}%",
            f"{cpu_ms:.3f}",
            str(k.count),
        )

    # ── Bottleneck summary ──
    bn_table = Table(
        title="Bottleneck Detection Results",
        collapse_padding=True,
    )
    bn_table.add_column("Severity", width=10)
    bn_table.add_column("Category", style="cyan", width=22)
    bn_table.add_column("Message", width=70)

    severity_colors = {
        "critical": "bold red",
        "warning": "bold yellow",
        "info": "dim",
    }

    if analysis.bottlenecks:
        for b in analysis.bottlenecks:
            color = severity_colors.get(b.severity, "")
            bn_table.add_row(
                f"[{color}]{b.severity.upper()}[/{color}]",
                b.category,
                b.message,
            )
    else:
        bn_table.add_row("OK", "-", "No bottlenecks detected")

    return gpu_table, bn_table


def print_torch_profile_summary(analysis: TorchProfilerAnalysis, console):
    """Print full torch.profiler summary to rich console."""
    from rich.panel import Panel

    # Summary panel
    total_gpu_ms = analysis.total_gpu_time_us / 1000
    total_cpu_ms = analysis.total_cpu_time_us / 1000
    ratio = analysis.python_gpu_ratio

    console.print(
        Panel(
            f"[green]Total GPU kernel time:[/green] {total_gpu_ms:.1f}ms\n"
            f"[green]Total CPU time:[/green] {total_cpu_ms:.1f}ms\n"
            f"[green]Python/GPU ratio:[/green] {ratio:.2f}x\n"
            f"[green]GPU kernels profiled:[/green] {len(analysis.kernels_by_gpu_time)}\n"
            f"[green]Steps profiled:[/green] {analysis.step_count}",
            title="torch.profiler Summary",
        )
    )

    # Key metrics for the 4 checks
    console.print("\n[bold white]DIAGNOSTIC CHECKS[/bold white]")

    # Check 1
    matmul_pct = analysis.gpu_matmul_pct
    color = "red" if matmul_pct > 30 else "yellow" if matmul_pct > 15 else "green"
    console.print(
        f"  [bold]Check 1 (Rule2 bottleneck):[/bold] "
        f"[{color}]aten::mm + aten::matmul = {matmul_pct:.1f}% GPU time[/{color}] "
        f"(threshold: 30%)"
    )

    # Check 2
    sync_kernels = analysis.host_sync_kernels_in_top10
    color = "red" if sync_kernels else "green"
    sync_str = ", ".join(sync_kernels) if sync_kernels else "none"
    console.print(
        f"  [bold]Check 2 (Host sync overhead):[/bold] "
        f"[{color}]Sync kernels in top-10: {sync_str}[/{color}]"
    )

    # Check 3
    index_add_kernels = analysis.index_add_large_self
    color = "yellow" if index_add_kernels else "green"
    ia_count = sum(k.count for k in index_add_kernels) if index_add_kernels else 0
    console.print(
        f"  [bold]Check 3 (Rule1/scatter overhead):[/bold] "
        f"[{color}]aten::index_add_ count={ia_count}[/{color}]"
    )

    # Check 4
    ratio = analysis.python_gpu_ratio
    color = "red" if ratio > 3 else "yellow" if ratio > 1.5 else "green"
    console.print(
        f"  [bold]Check 4 (Python dispatch overhead):[/bold] "
        f"[{color}]Python/GPU ratio = {ratio:.2f}x[/{color}] "
        f"(threshold: 1.5x)"
    )

    # Tables
    gpu_table, bn_table = render_torch_profile_table(analysis)
    console.print()
    console.print(gpu_table)
    console.print()
    console.print(bn_table)
