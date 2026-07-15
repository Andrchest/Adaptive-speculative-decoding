#!/usr/bin/env python3
"""
Profiling script for Adaptive Speculative Decoding experiments.

Breaks down wall-clock time by:
  1. High-level stages: model_loading, build_components, dataset_loading, decode_loop
  2. Per-step phases: cache_lookup, drafter_forward, translation, target_verify, accept_reject, residual
  3. GPU memory at each major stage
  4. Speculative decoding vs plain target model throughput
  5. GPU kernel-level profiling via torch.profiler (--torch-profile)

Uses small models (opt-125m / opt-350m) for fast runs.

Usage:
    python src/profiler.py --samples 10 --compare
    python src/profiler.py --suite ablation --samples 5
    python src/profiler.py --torch-profile --samples 3
    python src/profiler.py --torch-profile --torch-warmup 2 --torch-active 5
"""

from __future__ import annotations

import cProfile
import io
import json
import logging
import pstats
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
)
for noisy in ("urllib3", "httpx", "requests", "transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
console = Console()


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class StageTimings:
    stage: str = ""
    duration_s: float = 0.0
    pct_of_total: float = 0.0


@dataclass
class StepTimings:
    cache_lookup_ms: float = 0.0
    draft_forward_ms: float = 0.0
    translate_ms: float = 0.0
    target_verify_ms: float = 0.0
    accept_reject_ms: float = 0.0
    residual_sample_ms: float = 0.0
    total_ms: float = 0.0
    draft_len: int = 0
    accepted: int = 0


@dataclass
class ExperimentProfile:
    name: str = ""
    stages: list[StageTimings] = field(default_factory=list)
    step_timings: list[StepTimings] = field(default_factory=list)
    substep_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    gpu_mem_after_models_gb: float = 0.0
    gpu_mem_after_setup_gb: float = 0.0
    gpu_mem_peak_gb: float = 0.0
    gpu_mem_final_gb: float = 0.0
    total_wall_s: float = 0.0
    n_prompts: int = 0
    n_steps: int = 0
    n_accepted_total: int = 0
    n_draft_total: int = 0
    error: str = ""


@dataclass
class PlainModelProfile:
    name: str = ""
    total_wall_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    n_prompts: int = 0


# ──────────────────────────────────────────────────────────────────────
# Monkey-patch _decode_step for per-phase timing
# ──────────────────────────────────────────────────────────────────────


_original_decode_step = None


def _profiled_decode_step(
    self, context, k, ctx_list=None, drafter_ctx=None, distiller=None, rng=None
):
    """Instrumented version of SpeculativeDecoder._decode_step."""
    from core.decoder.speculative import StepResult as _StepResult

    if ctx_list is None:
        ctx_list = context[0].tolist()
    if drafter_ctx is None:
        drafter_ctx = context

    # Phase 1: cache lookup
    t0 = time.perf_counter()
    _ = self.cache.lookup(ctx_list)
    cache_ms = (time.perf_counter() - t0) * 1000

    # Phase 2: drafter forward
    t0 = time.perf_counter()
    draft_tokens_drafter, draft_logits, _ = self.drafter.draft(
        drafter_ctx, k, distill=(distiller is not None), temperature=self.temperature
    )
    draft_ms = (time.perf_counter() - t0) * 1000

    if not draft_tokens_drafter:
        return _StepResult(draft_length=0, accepted_count=0, rejected_at=-1, cache_hit=False)

    # Phase 3: translate
    t0 = time.perf_counter()
    if draft_logits is not None:
        with torch.no_grad():
            t_eff = max(self.temperature, 1e-6)
            from core.translation.vocabulary import _align_last_dim

            translated_probs = self.translator.translate(draft_logits / t_eff)
            translated_probs = _align_last_dim(translated_probs, self.translator.rule1.target_size)
    else:
        translated_probs = None

    draft_tokens_target = self._translate_draft_tokens(draft_tokens_drafter, translated_probs)
    translate_ms = (time.perf_counter() - t0) * 1000

    # Phase 4: target verify (with KV cache)
    t0 = time.perf_counter()
    target_logits, self._target_kv = self.target.verify(
        context, draft_tokens_target, past_key_values=getattr(self, "_target_kv", None)
    )
    verify_ms = (time.perf_counter() - t0) * 1000

    if translated_probs is not None:
        translated_probs = _align_last_dim(translated_probs, target_logits.shape[-1])

    # Phase 5: accept/reject
    t0 = time.perf_counter()
    accepted, rejected_at = self._accept_reject_gpu(
        draft_tokens_target, target_logits, translated_probs, rng=rng
    )
    ar_ms = (time.perf_counter() - t0) * 1000

    # Phase 6: residual sample
    t0 = time.perf_counter()
    bonus = self._residual_sample(target_logits, translated_probs, rejected_at, rng=rng)
    rs_ms = (time.perf_counter() - t0) * 1000

    accepted_count = len(accepted)  # before bonus — matches main decoder

    # Truncate target KV cache to keep only verified prefix
    kv_keep = context.shape[1] + accepted_count
    if self._target_kv is not None:
        try:
            from core.models.target_model import _truncate_pkv

            self._target_kv = _truncate_pkv(self._target_kv, kv_keep)
        except (TypeError, IndexError):
            self._target_kv = None

    if bonus is not None:
        accepted = accepted + [bonus]

    # Update cache
    self.cache.update_acceptance(ctx_list, accepted=accepted_count > 0)
    if accepted:
        self.cache.insert(
            ctx_list,
            accepted,
            logits=target_logits[: len(accepted)].detach().cpu()
            if target_logits is not None
            else None,
        )

    # Online distillation
    if distiller is not None and draft_logits is not None:
        accepted_mask = [
            (i < rejected_at) if rejected_at >= 0 else True
            for i in range(len(draft_tokens_drafter))
        ]
        distiller.step(
            draft_logits=draft_logits,
            target_logits=target_logits[: len(draft_tokens_target)],
            draft_tokens=draft_tokens_drafter,
            accepted_mask=accepted_mask,
            prompt_ids=context[0].tolist(),
        )

    # Collect step timing
    total_ms = cache_ms + draft_ms + translate_ms + verify_ms + ar_ms + rs_ms
    step_timings_collector.append(
        StepTimings(
            cache_lookup_ms=cache_ms,
            draft_forward_ms=draft_ms,
            translate_ms=translate_ms,
            target_verify_ms=verify_ms,
            accept_reject_ms=ar_ms,
            residual_sample_ms=rs_ms,
            total_ms=total_ms,
            draft_len=k,
            accepted=accepted_count,
        )
    )

    return _StepResult(
        draft_length=k,
        accepted_count=accepted_count,
        rejected_at=rejected_at,
        cache_hit=False,
        draft_tokens=draft_tokens_target,
        accepted_tokens=accepted,
    )


step_timings_collector: list[StepTimings] = []


def install_profiled_decode_step():
    """Monkey-patch SpeculativeDecoder._decode_step with profiled version."""
    from core.decoder.speculative import SpeculativeDecoder

    global _original_decode_step
    _original_decode_step = SpeculativeDecoder._decode_step
    SpeculativeDecoder._decode_step = _profiled_decode_step


def restore_decode_step():
    """Restore original _decode_step."""
    global _original_decode_step
    if _original_decode_step:
        from core.decoder.speculative import SpeculativeDecoder

        SpeculativeDecoder._decode_step = _original_decode_step
        _original_decode_step = None


# ──────────────────────────────────────────────────────────────────────
# Main profiler
# ──────────────────────────────────────────────────────────────────────


def run_profiled_experiment(
    exp,
    device: str,
    max_samples: int,
    max_new_tokens: int,
) -> ExperimentProfile:
    """Run one experiment with detailed profiling."""
    from core.decoder.speculative import SpeculativeDecoder
    from experiments.base import BuildContext
    from experiments.runner import ExperimentRunner

    profile = ExperimentProfile(name=exp.meta.name)

    cfg = exp.get_config()
    for key, value in exp._overrides.items():
        setattr(cfg, key, value)

    cfg.name = f"profile_{cfg.name}"
    cfg.max_samples = max_samples
    cfg.max_new_tokens = max_new_tokens

    runner = ExperimentRunner(
        experiments=[],
        output_dir="/tmp/profile_outputs",
        device=device,
    )

    # Global timing collector reset
    step_timings_collector.clear()

    # Enable substep timer for fine-grained bottleneck diagnosis
    from core.profiling.substep_timer import substep_timer

    substep_timer.enable()

    # ── Stage 1: Model loading ──
    t0 = time.perf_counter()
    drafter, target = runner._build_models(cfg)

    t1 = time.perf_counter()
    profile.stages.append(StageTimings(stage="model_loading", duration_s=t1 - t0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        profile.gpu_mem_after_models_gb = torch.cuda.memory_allocated(device) / 1024**3

    # ── Stage 2: Build components ──
    t0 = time.perf_counter()
    build_ctx = BuildContext(
        device=device,
        drafter=drafter,
        target=target,
        config=cfg,
        components={},
    )
    translator = exp.build_translator(build_ctx)
    build_ctx.components["translator"] = translator

    cache = exp.build_cache(build_ctx)
    build_ctx.components["cache"] = cache

    distiller = exp.build_distiller(build_ctx)
    build_ctx.components["distiller"] = distiller

    adaptive_fn = exp.build_adaptive_controller(build_ctx)
    build_ctx.components["adaptive_fn"] = adaptive_fn

    router = exp.build_router(build_ctx)
    build_ctx.components["router"] = router

    universal_adapter = exp.build_universal_drafter(build_ctx)
    if universal_adapter is not None:
        drafter = universal_adapter
        build_ctx.components["drafter"] = drafter

    t1 = time.perf_counter()
    profile.stages.append(StageTimings(stage="build_components", duration_s=t1 - t0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        profile.gpu_mem_after_setup_gb = torch.cuda.memory_allocated(device) / 1024**3

    # ── Stage 3: Load dataset ──
    t0 = time.perf_counter()
    prompts = runner._load_dataset(cfg)
    t1 = time.perf_counter()
    profile.stages.append(StageTimings(stage="dataset_loading", duration_s=t1 - t0))
    profile.n_prompts = len(prompts)
    del runner  # free runner — only needed for _build_models

    # ── Stage 4: Decode loop ──
    draft_length = getattr(cfg, "draft_length", 5)
    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target,
        translator=translator,
        cache=cache,
        draft_length=draft_length,
    )

    t0 = time.perf_counter()
    torch_rng = torch.Generator()
    torch_rng.manual_seed(getattr(cfg, "seed", 42))

    max_new_tokens = getattr(cfg, "max_new_tokens", 32)
    max_consec_zero = 5

    for prompt_idx, (input_ids, prompt_real_len) in enumerate(prompts):
        generated = input_ids.clone().to(device)
        prompt_len = generated.shape[1]
        consec_zero = 0

        # Maintain drafter vocab context separately (same as _generate_loop)
        drafter_context_ids: list[int] = generated[0].tolist()

        for step_idx in range(max_new_tokens):
            new_tokens = generated.shape[1] - prompt_len
            if new_tokens >= max_new_tokens:
                break

            k = decoder._choose_draft_length(generated, adaptive_fn)

            # Build drafter context tensor from drafter vocab token IDs
            drafter_ctx = torch.tensor(
                [drafter_context_ids], dtype=generated.dtype, device=generated.device
            )

            # Instrumented decode step
            result = decoder._decode_step(
                generated,
                k,
                drafter_context_ids[:],
                drafter_ctx=drafter_ctx,
                distiller=distiller,
                rng=torch_rng,
            )
            decoder._step_results.append(result)
            decoder.cache.step()

            # Truncate to budget
            budget = max_new_tokens - new_tokens
            emitted = result.accepted_tokens[:budget]
            if emitted:
                new_ids = torch.tensor(
                    emitted, dtype=torch.long, device=generated.device
                ).unsqueeze(0)
                generated = torch.cat([generated, new_ids], dim=1)

                # Translate accepted tokens back to drafter vocab
                if not decoder._same_vocab:
                    drafter_emitted = decoder.translator.translate_target_to_drafter(emitted)
                else:
                    drafter_emitted = emitted
                drafter_context_ids.extend(drafter_emitted)

                consec_zero = 0
            else:
                consec_zero += 1
                if consec_zero >= max_consec_zero:
                    break

            if generated.shape[1] and decoder._is_eos(generated[0, -1]):
                break

        decoder.cache.step()

    t1 = time.perf_counter()
    profile.stages.append(StageTimings(stage="decode_loop", duration_s=t1 - t0))
    profile.step_timings = list(step_timings_collector)
    profile.n_steps = len(step_timings_collector)
    profile.n_accepted_total = sum(s.accepted for s in step_timings_collector)
    profile.n_draft_total = sum(s.draft_len for s in step_timings_collector)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        profile.gpu_mem_peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        profile.gpu_mem_final_gb = torch.cuda.memory_allocated(device) / 1024**3

    profile.total_wall_s = sum(s.duration_s for s in profile.stages)

    # Disable substep timer and attach summary to profile
    substep_timer.disable()
    profile.substep_summary = substep_timer.summary()

    # Cleanup
    if hasattr(drafter, "cleanup"):
        drafter.cleanup()
    elif hasattr(drafter, "remove_hooks"):
        drafter.remove_hooks()
    del drafter, target, translator, cache, decoder
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return profile


def profile_plain_target(
    target_path: str,
    device: str,
    max_samples: int,
    max_new_tokens: int,
) -> PlainModelProfile:
    """Profile the target model running alone (no speculative decoding)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    profile = PlainModelProfile(name=f"plain_target_{target_path.split('/')[-1]}")
    profile.n_prompts = max_samples

    console.print(f"  Loading target model: {target_path}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.float16,
        device_map=device,
        load_in_4bit=True,
    )
    target.eval()

    t1 = time.perf_counter()
    console.print(f"  Model loaded in {t1 - t0:.1f}s")

    # Generate synthetic prompts
    prompts = [
        f"Question {i}: The total number of apples is {i * 3 + 7}. How many apples are there?"
        for i in range(1, max_samples + 1)
    ]

    tokens_generated = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = target.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                max_length=input_ids.shape[1] + max_new_tokens,
            )
            new_tokens = out.shape[1] - input_ids.shape[1]
            tokens_generated += max(0, new_tokens)
    t1 = time.perf_counter()

    profile.total_wall_s = t1 - t0
    profile.tokens_generated = tokens_generated
    profile.tokens_per_sec = tokens_generated / max(profile.total_wall_s, 1e-6)

    # Cleanup
    del target, tokenizer
    torch.cuda.empty_cache()

    return profile


def run_torch_profiled_experiment(
    exp,
    device: str,
    max_samples: int,
    max_new_tokens: int,
    warmup_steps: int = 2,
    active_steps: int = 5,
    output_dir: str = "",
):
    """Run one experiment with torch.profiler GPU kernel-level analysis."""
    from core.decoder.speculative import SpeculativeDecoder
    from core.profiling.torch_profiler import run_torch_profile
    from experiments.base import BuildContext
    from experiments.runner import ExperimentRunner

    runner = ExperimentRunner(
        experiments=[],
        output_dir="/tmp/profile_outputs",
        device=device,
    )

    cfg = exp.get_config()
    for key, value in exp._overrides.items():
        setattr(cfg, key, value)

    cfg.name = f"torch_profile_{cfg.name}"
    cfg.max_samples = max_samples
    cfg.max_new_tokens = max_new_tokens

    # Build models
    drafter, target = runner._build_models(cfg)

    # Build components
    build_ctx = BuildContext(
        device=device,
        drafter=drafter,
        target=target,
        config=cfg,
        components={},
    )
    translator = exp.build_translator(build_ctx)
    build_ctx.components["translator"] = translator
    cache = exp.build_cache(build_ctx)
    build_ctx.components["cache"] = cache
    distiller = exp.build_distiller(build_ctx)

    # Build decoder
    draft_length = getattr(cfg, "draft_length", 5)
    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target,
        translator=translator,
        cache=cache,
        draft_length=draft_length,
    )

    # Load one prompt for profiling
    prompts = runner._load_dataset(cfg)
    if not prompts:
        console.print("[red]No prompts available for torch profiling.[/red]")
        return None
    del runner  # free runner — only needed for _build_models

    input_ids = prompts[0][0].clone().to(device)

    # Run torch.profiler
    torch_rng = torch.Generator()
    torch_rng.manual_seed(getattr(cfg, "seed", 42))

    analysis = run_torch_profile(
        decoder,
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_length=draft_length,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        output_dir=output_dir,
        distiller=distiller,
        rng=torch_rng,
    )

    # Cleanup
    if hasattr(drafter, "cleanup"):
        drafter.cleanup()
    elif hasattr(drafter, "remove_hooks"):
        drafter.remove_hooks()
    del drafter, target, translator, cache, decoder
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return analysis


# ──────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────


def render_stage_table(profiles: list[ExperimentProfile]) -> Table:
    table = Table(title="Stage-by-Stage Timing Breakdown (seconds)", collapse_padding=True)
    table.add_column("Experiment", style="cyan", width=28)
    table.add_column("Stage", style="green", width=22)
    table.add_column("Time (s)", justify="right", width=12)
    table.add_column("% of Exp", justify="right", width=10)
    table.add_column("% of All", justify="right", width=10)

    total_time = max((p.total_wall_s for p in profiles), default=1.0)
    for p in profiles:
        first = True
        for s in p.stages:
            pct_exp = (s.duration_s / p.total_wall_s * 100) if p.total_wall_s > 0 else 0
            pct_all = (s.duration_s / total_time * 100) if total_time > 0 else 0
            table.add_row(
                p.name if first else "",
                s.stage,
                f"{s.duration_s:.3f}",
                f"{pct_exp:.1f}%",
                f"{pct_all:.1f}%",
            )
            first = False

    return table


def render_step_table(profiles: list[ExperimentProfile]) -> Table:
    table = Table(
        title="Per-Step Phase Breakdown (averaged over all steps)",
        collapse_padding=True,
    )
    table.add_column("Experiment", style="cyan", width=26)
    table.add_column("Steps", justify="right", width=6)
    table.add_column("Draft L", justify="right", width=8)
    table.add_column("Accepted", justify="right", width=8)
    table.add_column("Acc Rate", justify="right", width=9)
    table.add_column("Cache", justify="right", width=9)
    table.add_column("Drafter", justify="right", width=10)
    table.add_column("Trans.", justify="right", width=10)
    table.add_column("Verify", justify="right", width=10)
    table.add_column("Accept", justify="right", width=9)
    table.add_column("Total", justify="right", width=10)

    for p in profiles:
        if not p.step_timings:
            table.add_row(p.name, "0", "-", "-", "-", "-", "-", "-", "-", "-", "N/A")
            continue

        n = len(p.step_timings)
        avg = lambda fn: np.mean([getattr(s, fn) for s in p.step_timings]) if n else 0

        ar = (p.n_accepted_total / max(p.n_draft_total, 1)) if p.n_draft_total > 0 else 0

        table.add_row(
            p.name,
            str(p.n_steps),
            f"{avg('draft_len'):.1f}",
            f"{avg('accepted'):.1f}",
            f"{ar:.1%}",
            f"{avg('cache_lookup_ms'):.1f}ms",
            f"{avg('draft_forward_ms'):.1f}ms",
            f"{avg('translate_ms'):.1f}ms",
            f"{avg('target_verify_ms'):.1f}ms",
            f"{avg('accept_reject_ms') + avg('residual_sample_ms'):.1f}ms",
            f"{avg('total_ms'):.1f}ms",
        )

    return table


def render_percent_bars(profiles: list[ExperimentProfile]) -> None:
    for p in profiles:
        if not p.stages or p.total_wall_s == 0:
            continue
        console.print(
            f"\n[bold blue]▓ {p.name}[/bold blue]  [dim](total: {p.total_wall_s:.1f}s)[/dim]"
        )
        for s in p.stages:
            pct = (s.duration_s / p.total_wall_s * 100) if p.total_wall_s > 0 else 0
            bar_len = max(1, int(pct / 2))
            bar = "█" * bar_len + "░" * (50 - bar_len)
            color = "green" if pct > 50 else "yellow" if pct > 20 else "dim"
            console.print(f"  [{color}]┃{pct:5.1f}%┃[/]" + bar)


def render_gpu_table(profiles: list[ExperimentProfile]) -> Table:
    table = Table(title="GPU Memory Usage (GB)", collapse_padding=True)
    table.add_column("Experiment", style="cyan", width=28)
    table.add_column("After Models", justify="right", width=14)
    table.add_column("After Setup", justify="right", width=14)
    table.add_column("Peak", justify="right", width=10)
    table.add_column("Final", justify="right", width=10)

    for p in profiles:
        table.add_row(
            p.name,
            f"{p.gpu_mem_after_models_gb:.2f}",
            f"{p.gpu_mem_after_setup_gb:.2f}",
            f"{p.gpu_mem_peak_gb:.2f}",
            f"{p.gpu_mem_final_gb:.2f}",
        )

    return table


def render_substep_table(profiles: list[ExperimentProfile]) -> Table | None:
    """Render min/max/avg/sum summary for substep timings collected during profiling."""
    all_summary: dict[str, dict[str, float]] = {}
    for p in profiles:
        for name, stats in p.substep_summary.items():
            if name not in all_summary:
                all_summary[name] = dict(stats)
            else:
                all_summary[name]["min"] = min(all_summary[name]["min"], stats["min"])
                all_summary[name]["max"] = max(all_summary[name]["max"], stats["max"])
                all_summary[name]["sum"] += stats["sum"]
                all_summary[name]["count"] += stats["count"]
                all_summary[name]["avg"] = all_summary[name]["sum"] / all_summary[name]["count"]

    if not all_summary:
        return None

    table = Table(
        title="Substep Timing Summary (min / max / avg / sum ms)",
        collapse_padding=True,
    )
    table.add_column("Substep", style="cyan", width=36)
    table.add_column("Min", justify="right", width=10)
    table.add_column("Max", justify="right", width=10)
    table.add_column("Avg", justify="right", width=10)
    table.add_column("Sum", justify="right", width=12)
    table.add_column("Count", justify="right", width=7)

    # Group by parent method for readability
    groups: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for name, stats in sorted(all_summary.items()):
        parts = name.split(".", 1)
        group = parts[0] if len(parts) > 1 else "other"
        groups.setdefault(group, []).append((name, stats))

    for group_name, entries in groups.items():
        table.add_row(f"[bold]{group_name}[/bold]", "", "", "", "", "")
        for name, stats in entries:
            table.add_row(
                f"  {name}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                f"{stats['avg']:.2f}",
                f"{stats['sum']:.1f}",
                str(int(stats["count"])),
            )

    return table


def render_comparison(
    profiled: list[ExperimentProfile],
    plain: list[PlainModelProfile],
) -> Table:
    table = Table(title="Speculative Decoding vs Plain Target", collapse_padding=True)
    table.add_column("Config", style="cyan", width=32)
    table.add_column("Wall (s)", justify="right", width=10)
    table.add_column("Samples", justify="right", width=8)
    table.add_column("Steps", justify="right", width=7)
    table.add_column("Tok/sec*", justify="right", width=14)
    table.add_column("Speedup*", justify="right", width=10)

    baseline_tps = 1.0
    if plain:
        baseline_tps = plain[0].tokens_per_sec

    for p in profiled:
        if not p.total_wall_s or p.n_steps == 0:
            continue
        tps = p.n_accepted_total / p.total_wall_s
        speedup = tps / baseline_tps if baseline_tps > 0 else 0

        table.add_row(
            f"SpecDec ({p.name})",
            f"{p.total_wall_s:.2f}",
            str(p.n_prompts),
            str(p.n_steps),
            f"{tps:.1f}",
            f"{speedup:.2f}x" if speedup > 0 else "—",
        )

    if plain:
        for pl in plain:
            table.add_row(
                f"Plain ({pl.name})",
                f"{pl.total_wall_s:.2f}",
                str(pl.n_prompts),
                "—",
                f"{pl.tokens_per_sec:.1f}",
                "1.00x",
            )

    return table


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main(
    samples: Annotated[
        int, typer.Option("-n", "--samples", help="Number of samples to profile")
    ] = 10,
    max_new_tokens: Annotated[
        int, typer.Option("--max-tokens", "-m", help="Max new tokens per prompt")
    ] = 32,
    experiments: Annotated[
        str, typer.Option("--experiments", "-e", help="Comma-separated experiment names")
    ] = "01_baseline,08_speedup_adapt,11_full_system",
    suite: Annotated[
        str, typer.Option("--suite", "-s", help="Experiment suite: ablation, all")
    ] = "",
    compare_plain: Annotated[
        bool, typer.Option("--compare", "-c", help="Compare with plain target model")
    ] = False,
    compare_4bit_fp16: Annotated[
        bool, typer.Option("--compare-4bit-fp16", help="Compare 4-bit vs FP16 target model")
    ] = False,
    cprofile_flag: Annotated[
        bool, typer.Option("--cprofile", "-p", help="Run Python-level cProfile")
    ] = False,
    cprofile_top: Annotated[
        int, typer.Option("--cprofile-top", help="Top-N cProfile entries")
    ] = 20,
    torch_profile_flag: Annotated[
        bool,
        typer.Option(
            "--torch-profile", "-t", help="Run GPU kernel-level profiling with torch.profiler"
        ),
    ] = False,
    torch_warmup: Annotated[
        int, typer.Option("--torch-warmup", help="Warmup steps before torch profiling starts")
    ] = 2,
    torch_active: Annotated[
        int, typer.Option("--torch-active", help="Number of steps to profile with torch.profiler")
    ] = 5,
    torch_trace_dir: Annotated[
        str, typer.Option("--torch-trace-dir", help="Directory for TensorBoard trace output")
    ] = "",
    drafter_model: Annotated[
        str, typer.Option("--drafter-model", help="Drafter model path (overrides default)")
    ] = "",
    target_model: Annotated[
        str, typer.Option("--target-model", help="Target model path (overrides default)")
    ] = "",
    output: Annotated[str, typer.Option("--output", "-o", help="JSON output file")] = "",
):
    """Profile Adaptive Speculative Decoding experiments."""
    from experiments import ABLATION_SUITE, discover_experiments

    device = "cuda" if torch.cuda.is_available() else "cpu"

    drafter_name = drafter_model.split("/")[-1] if drafter_model else "opt-125m"
    target_name = target_model.split("/")[-1] if target_model else "opt-350m"
    console.print(
        Panel(
            f"[green]Device:[/green] {device}  "
            f"[green]GPU:[/green] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n"
            f"[green]Models:[/green] {drafter_name} (drafter) / {target_name} (target)\n"
            f"[green]Samples:[/green] {samples}  [green]Max tokens:[/green] {max_new_tokens}"
        )
    )

    # Select experiments
    if suite:
        exps = ABLATION_SUITE if suite == "ablation" else discover_experiments()
    else:
        names = [n.strip() for n in experiments.split(",")]
        exps = [e for e in ABLATION_SUITE if e.meta.name in names]

    if not exps:
        console.print("[red]No experiments matched.[/red]")
        sys.exit(1)

    console.print(
        f"[dim]Profiling {len(exps)} experiment(s): {', '.join(e.meta.name for e in exps)}[/dim]\n"
    )

    # Apply model overrides
    for exp in exps:
        if drafter_model:
            exp.set_config_override("drafter_model_path", drafter_model)
        else:
            exp.set_config_override("drafter_model_path", "facebook/opt-125m")
        if target_model:
            exp.set_config_override("target_model_path", target_model)
        else:
            exp.set_config_override("target_model_path", "facebook/opt-350m")
        exp.set_config_override("max_new_tokens", max_new_tokens)

    profiles: list[ExperimentProfile] = []

    for idx, exp in enumerate(exps, 1):
        console.print(
            f"\n[bold yellow]▓ [{idx}/{len(exps)}] Profiling: {exp.meta.name}[/bold yellow]"
        )
        try:
            install_profiled_decode_step()
            p = run_profiled_experiment(exp, device, samples, max_new_tokens)
            profiles.append(p)
            console.print(
                f"  [green]✓[/green] {p.total_wall_s:.2f}s, {p.n_steps} steps, "
                f"accepted={p.n_accepted_total}, draft={p.n_draft_total}"
            )
            if p.error:
                console.print(f"  [red]Error:[/red] {p.error}")
        except Exception as e:
            import traceback

            console.print(f"  [red]✗[/red] {e}")
            tb = traceback.format_exc()
            console.print(f"  [dim]{tb}[/dim]")
            ep = ExperimentProfile(name=exp.meta.name, error=str(e))
            profiles.append(ep)
        finally:
            restore_decode_step()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    # ── 4-bit vs FP16 comparison ──
    fp16_profiles: dict[str, ExperimentProfile] = {}
    if compare_4bit_fp16:
        console.print("\n" + "=" * 80)
        console.print("[bold yellow]Running 4-bit vs FP16 target comparison...[/bold yellow]")
        for idx, exp in enumerate(exps, 1):
            console.print(
                f"\n[bold yellow]▓ [{idx}/{len(exps)}] FP16 run: {exp.meta.name}[/bold yellow]"
            )
            # Override: load target in FP16 (no 4-bit quantization)
            exp.set_config_override("target_use_4bit", False)
            try:
                install_profiled_decode_step()
                fp16_p = run_profiled_experiment(exp, device, samples, max_new_tokens)
                fp16_profiles[exp.meta.name] = fp16_p
                console.print(
                    f"  [green]✓[/green] FP16: {fp16_p.total_wall_s:.2f}s, "
                    f"{fp16_p.n_steps} steps, gpu_mem={fp16_p.gpu_mem_peak_gb:.2f}GB"
                )
                if fp16_p.error:
                    console.print(f"  [red]Error:[/red] {fp16_p.error}")
            except Exception as e:
                import traceback

                console.print(f"  [red]✗[/red] {e}")
                tb = traceback.format_exc()
                console.print(f"  [dim]{tb}[/dim]")
            restore_decode_step()

    # ── Plain target baseline ──
    plain_profiles: list[PlainModelProfile] = []
    if compare_plain:
        console.print("\n" + "=" * 80)
        console.print("[bold yellow]Profiling plain target model...[/bold yellow]")
        try:
            pp = profile_plain_target(
                target_model or "facebook/opt-350m",
                device,
                samples,
                max_new_tokens,
            )
            plain_profiles.append(pp)
            console.print(
                f"  [green]✓[/green] {pp.total_wall_s:.2f}s, "
                f"{pp.tokens_per_sec:.1f} tok/s, {pp.tokens_generated} tokens"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] {e}")
            import traceback

            console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    # ── Render reports ──
    console.print("\n" + "=" * 80)
    console.print("[bold white]STAGE TIMING BREAKDOWN[/bold white]")
    console.print(render_stage_table(profiles))

    console.print("\n" + "=" * 80)
    console.print("[bold white]PER-STEP PHASE BREAKDOWN[/bold white]")
    console.print(render_step_table(profiles))

    console.print("\n" + "=" * 80)
    console.print("[bold white]STAGE PERCENTAGES[/bold white]")
    render_percent_bars(profiles)

    console.print("\n" + "=" * 80)
    console.print("[bold white]GPU MEMORY USAGE[/bold white]")
    console.print(render_gpu_table(profiles))

    console.print("\n" + "=" * 80)
    console.print("[bold white]SUBSTEP TIMING DETAIL[/bold white]")
    substep_tbl = render_substep_table(profiles)
    if substep_tbl is not None:
        console.print(substep_tbl)
    else:
        console.print("[dim]No substep timings recorded.[/dim]")

    # ── 4-bit vs FP16 comparison table ──
    if fp16_profiles:
        console.print("\n" + "=" * 80)
        console.print("[bold white]4-BIT vs FP16 TARGET MODEL COMPARISON[/bold white]")
        table = Table(collapse_padding=True)
        table.add_column("Experiment", style="cyan", width=28)
        table.add_column("4-bit Time (s)", justify="right", width=16)
        table.add_column("FP16 Time (s)", justify="right", width=15)
        table.add_column("Speedup", justify="right", width=10)
        table.add_column("4-bit GPU (GB)", justify="right", width=14)
        table.add_column("FP16 GPU (GB)", justify="right", width=14)
        table.add_column("Δ GPU (GB)", justify="right", width=12)

        for exp in exps:
            name = exp.meta.name
            four_bit_p = next((p for p in profiles if p.name == name), None)
            fp16_p = fp16_profiles.get(name)

            if not four_bit_p or not fp16_p:
                continue

            time_4bit = four_bit_p.total_wall_s
            time_fp16 = fp16_p.total_wall_s
            speedup = time_4bit / time_fp16 if time_fp16 > 0 else 0
            gpu_4bit = four_bit_p.gpu_mem_peak_gb
            gpu_fp16 = fp16_p.gpu_mem_peak_gb
            gpu_delta = gpu_fp16 - gpu_4bit

            table.add_row(
                name,
                f"{time_4bit:.2f}",
                f"{time_fp16:.2f}",
                f"{speedup:.2f}x{' ⬆' if speedup > 1.02 else ''}",
                f"{gpu_4bit:.2f}",
                f"{gpu_fp16:.2f}",
                f"+{gpu_delta:.2f}",
            )
        console.print(table)

    if plain_profiles:
        console.print("\n" + "=" * 80)
        console.print("[bold white]SPECULATIVE VS PLAIN COMPARISON[/bold white]")
        console.print(render_comparison(profiles, plain_profiles))

    # ── cProfile if requested ──
    if cprofile_flag and profiles:
        console.print("\n" + "=" * 80)
        console.print("[bold white]CPROFILE (Python-level)[/bold white]")
        exp = exps[0]
        try:
            install_profiled_decode_step()
            pr = cProfile.Priver()
            pr.enable()
            p = run_profiled_experiment(exp, device, samples, max_new_tokens)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats(cprofile_top)
            console.print(s.getvalue())
        except Exception as e:
            console.print(f"[red]cProfile error:[/red] {e}")
        restore_decode_step()

    # ── torch.profiler if requested ──
    if torch_profile_flag:
        console.print("\n" + "=" * 80)
        console.print("[bold white]TORCH.PROFILER (GPU kernel-level analysis)[/bold white]")
        from core.profiling.torch_profiler import print_torch_profile_summary

        torch_analyses = []
        for idx, exp in enumerate(exps, 1):
            console.print(
                f"\n[bold yellow]▓ [{idx}/{len(exps)}] torch.profiler: {exp.meta.name}[/bold yellow]"
            )
            try:
                analysis = run_torch_profiled_experiment(
                    exp,
                    device,
                    samples,
                    max_new_tokens,
                    warmup_steps=torch_warmup,
                    active_steps=torch_active,
                    output_dir=torch_trace_dir,
                )
                if analysis is not None:
                    torch_analyses.append((exp.meta.name, analysis))
                    console.print(
                        f"  [green]✓[/green] {len(analysis.kernels_by_gpu_time)} kernels profiled, "
                        f"{len(analysis.bottlenecks)} bottlenecks detected"
                    )
            except Exception as e:
                import traceback

                console.print(f"  [red]✗[/red] {e}")
                console.print(f"  [dim]{traceback.format_exc()}[/dim]")
            finally:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

        for name, analysis in torch_analyses:
            console.print(f"\n{'=' * 80}")
            console.print(f"[bold cyan]torch.profiler results: {name}[/bold cyan]")
            print_torch_profile_summary(analysis, console)

    # ── JSON output ──
    if output:
        data = []
        for p in profiles:
            data.append(
                {
                    "name": p.name,
                    "total_wall_s": p.total_wall_s,
                    "n_prompts": p.n_prompts,
                    "n_steps": p.n_steps,
                    "n_accepted_total": p.n_accepted_total,
                    "n_draft_total": p.n_draft_total,
                    "gpu_mem_after_models_gb": p.gpu_mem_after_models_gb,
                    "gpu_mem_after_setup_gb": p.gpu_mem_after_setup_gb,
                    "gpu_mem_peak_gb": p.gpu_mem_peak_gb,
                    "gpu_mem_final_gb": p.gpu_mem_final_gb,
                    "stages": [
                        {
                            "stage": s.stage,
                            "duration_s": s.duration_s,
                            "pct_of_total": s.pct_of_total,
                        }
                        for s in p.stages
                    ],
                    "error": p.error,
                }
            )
        # Add torch.profiler data if available
        if torch_profile_flag and "torch_analyses" in dir():
            for name, analysis in torch_analyses:
                data.append(
                    {
                        "name": f"torch_profile_{name}",
                        "total_gpu_time_us": analysis.total_gpu_time_us,
                        "total_cpu_time_us": analysis.total_cpu_time_us,
                        "python_gpu_ratio": analysis.python_gpu_ratio,
                        "gpu_matmul_pct": analysis.gpu_matmul_pct,
                        "host_sync_kernels_in_top10": analysis.host_sync_kernels_in_top10,
                        "has_rule2_bottleneck": analysis.has_rule2_bottleneck,
                        "has_host_sync_overhead": analysis.has_host_sync_overhead,
                        "has_rule1_scatter_overhead": analysis.has_rule1_scatter_overhead,
                        "has_python_dispatch_overhead": analysis.has_python_dispatch_overhead,
                        "top_kernels_by_gpu": [
                            {"name": k.name, "gpu_pct": k.gpu_pct, "count": k.count}
                            for k in analysis.kernels_by_gpu_time[:20]
                        ],
                        "bottlenecks": [
                            {"category": b.category, "severity": b.severity, "message": b.message}
                            for b in analysis.bottlenecks
                        ],
                    }
                )
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"\n[green]JSON saved to {output}[/green]")

    # ── Top bottleneck summary ──
    console.print("\n" + "=" * 80)
    console.print("[bold red]⚠ TOP BOTTLENECKS[/bold red]")
    for p in profiles:
        if not p.stages or p.total_wall_s == 0:
            continue
        sorted_stages = sorted(p.stages, key=lambda s: s.duration_s, reverse=True)
        for i, s in enumerate(sorted_stages[:3]):
            pct = s.duration_s / p.total_wall_s * 100
            console.print(
                f"  [red]{p.name}[/red] #{i + 1}: [bold]{s.stage}[/bold] = {pct:.1f}% "
                f"({s.duration_s:.3f}s)"
            )

    # ── Per-step bottleneck summary ──
    for p in profiles:
        if not p.step_timings or not p.step_timings[0].total_ms:
            continue
        console.print(f"\n[bold red]⚠ PER-STEP BOTTLENECKS ({p.name})[/bold red]")
        n = len(p.step_timings)
        fields = [
            "draft_forward_ms",
            "target_verify_ms",
            "translate_ms",
            "accept_reject_ms",
            "residual_sample_ms",
            "cache_lookup_ms",
        ]
        sorted_fields = sorted(
            fields, key=lambda f: np.mean([getattr(s, f) for s in p.step_timings]), reverse=True
        )
        for i, f in enumerate(sorted_fields[:4]):
            avg_val = np.mean([getattr(s, f) for s in p.step_timings])
            pct = (
                (avg_val / np.mean([s.total_ms for s in p.step_timings]) * 100)
                if np.mean([s.total_ms for s in p.step_timings]) > 0
                else 0
            )
            console.print(
                f"  [red]{p.name}[/red] step #{i + 1}: [bold]{f.replace('_ms', '')}[/bold] = "
                f"{avg_val:.1f}ms ({pct:.1f}%)"
            )

    console.print("\n" + "=" * 80)
    console.print("[green]✓ Profiling complete![/green]")


if __name__ == "__main__":
    typer.run(main)
