#!/usr/bin/env python3
"""
Memory + performance profiler for Adaptive Speculative Decoding experiments.

Measures:
  1. GPU memory allocation at key lifecycle points (per experiment & per prompt)
  2. Python gc object counts (to detect growing references)
  3. In-memory data structure sizes (collector records, distiller losses, replay buffer, cache, step results)
  4. Per-experiment wall time and TPS
  5. Per-prompt memory drift (to detect leaks within a single experiment)

Usage:
    # Profile a single experiment (fast, 5 samples)
    python scripts/profile_experiments.py --experiment 01_baseline -n 5

    # Profile multiple experiments
    python scripts/profile_experiments.py --experiment 01_baseline --experiment 04_+online_distil -n 5

    # Profile all ablation suite (fast mode)
    python scripts/profile_experiments.py --suite ablation -n 5 --tiny

    # Profile the full system
    python scripts/profile_experiments.py --experiment 11_full_system -n 3 --tiny
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Ensure the external venv's site-packages is in path
_venv_sp = "/home/andreipc/migration/Adaptive-speculative-decoding/.venv/lib/python3.12/site-packages"
if _venv_sp not in sys.path:
    sys.path.insert(0, _venv_sp)

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
for noisy in ("urllib3", "httpx", "requests", "transformers", "huggingface_hub", "datasets"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
# Profiling data structures
# =============================================================================

@dataclass
class MemorySnapshot:
    timestamp: float
    phase: str
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_util_pct: float = 0.0
    gc_counts: tuple[int, int, int] = (0, 0, 0)
    gc_total: int = 0


@dataclass
class StructureSnapshot:
    """Snapshots of key data structures that can grow."""
    timestamp: float
    collector_records: int = 0
    distiller_losses: int = 0
    distiller_kl_losses: int = 0
    distiller_nll_losses: int = 0
    replay_buffer_size: int = 0
    cache_size: int = 0
    cache_max_size: int = 0
    step_results_len: int = 0


@dataclass
class PromptMetrics:
    prompt_index: int
    wall_time_s: float
    tps: float
    draft_len: float
    accepted: int


@dataclass
class ExperimentProfile:
    name: str
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    struct_snapshots: list[StructureSnapshot] = field(default_factory=list)
    prompt_metrics: list[PromptMetrics] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    error: str | None = None


# =============================================================================
# GPU helpers
# =============================================================================

def get_gpu_info() -> dict[str, float]:
    """Return current GPU metrics."""
    info: dict[str, float] = {}
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
        info["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024 ** 3)
        # Approximate GPU util from nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            vals = [float(v.strip()) for v in result.stdout.strip().split('\n') if v.strip()]
            info["gpu_util_pct"] = vals[0] if vals else 0.0
        except Exception:
            info["gpu_util_pct"] = 0.0
    else:
        info["gpu_available"] = False
        info["gpu_allocated_gb"] = 0.0
        info["gpu_reserved_gb"] = 0.0
        info["gpu_util_pct"] = 0.0
    return info


def snapshot_memory(phase: str) -> MemorySnapshot:
    """Capture a full memory snapshot."""
    gpu = get_gpu_info()
    try:
        all_stats = gc.get_stats()
        stats0 = all_stats[0] if len(all_stats) > 0 else None
        stats1 = all_stats[1] if len(all_stats) > 1 else None
        stats2 = all_stats[2] if len(all_stats) > 2 else None
        gc_counts = (
            getattr(stats0, 'collect', 0) if stats0 else 0,
            getattr(stats1, 'collect', 0) if stats1 else 0,
            getattr(stats2, 'collections', 0) if stats2 else 0,
        )
    except (IndexError, AttributeError):
        gc_counts = (0, 0, 0)
    gc_total = gc.get_count()[0] + gc.get_count()[1] + gc.get_count()[2]
    return MemorySnapshot(
        timestamp=time.perf_counter(),
        phase=phase,
        gpu_allocated_gb=gpu["gpu_allocated_gb"],
        gpu_reserved_gb=gpu["gpu_reserved_gb"],
        gpu_util_pct=gpu["gpu_util_pct"],
        gc_counts=gc_counts,
        gc_total=gc_total,
    )


def snapshot_structures(ctx_refs: dict[str, Any] | None = None) -> StructureSnapshot:
    """Capture sizes of key data structures."""
    snap = StructureSnapshot(timestamp=time.perf_counter())
    if ctx_refs is None:
        return snap

    collector = ctx_refs.get("collector")
    if collector is not None:
        snap.collector_records = len(getattr(collector, "_records", []))

    distiller = ctx_refs.get("distiller")
    if distiller is not None:
        snap.distiller_losses = len(getattr(distiller, "losses", []))
        snap.distiller_kl_losses = len(getattr(distiller, "kl_losses", []))
        snap.distiller_nll_losses = len(getattr(distiller, "nll_losses", []))

    replay_distiller = ctx_refs.get("replay_distiller")
    if replay_distiller is not None:
        snap.replay_buffer_size = len(getattr(replay_distiller.buffer, "_buffer", []))
    else:
        distiller = ctx_refs.get("distiller")
        if distiller is not None:
            buf = getattr(distiller, "_replay_buf", None)
            if buf is not None:
                snap.replay_buffer_size = len(getattr(buf, "_buffer", []))

    cache = ctx_refs.get("cache")
    if cache is not None:
        snap.cache_size = len(cache)
        snap.cache_max_size = getattr(cache, "max_size", 0)

    decoder = ctx_refs.get("decoder")
    if decoder is not None:
        snap.step_results_len = len(getattr(decoder, "_step_results", []))

    return snap


# =============================================================================
# Monkey-patch hooks to intercept data structure growth
# =============================================================================

_profiler_state: dict[str, Any] | None = None


def _patch_benchmark_collector(collector):
    """Patch BenchmarkCollector.record_sequence to capture per-prompt metrics."""
    original_init = collector._SequenceContext.__init__
    original_add_step = collector._SequenceContext.add_step
    original_exit = collector._SequenceContext.__exit__

    def patched_init(self, col, prompt_len):
        original_init(self, col, prompt_len)
        self._prompt_idx = getattr(collector, "_current_prompt_idx", 0)

    def patched_add_step(self, draft_len, accepted, cache_hit=False, kl_div=0.0, actual_draft_len=0):
        original_add_step(self, draft_len, accepted, cache_hit, kl_div, actual_draft_len)
        # Capture after each step for fine-grained analysis
        p = _profiler_state
        if p is not None and accepted > 0:
            p["accepted_tokens_total"] += accepted
            p["draft_tokens_total"] += draft_len

    def patched_exit(self, *args):
        result = original_exit(self, *args)
        p = _profiler_state
        if p is not None:
            wall = self._rec.wall_time_s
            tps = self._rec.tokens_per_sec
            total_new = self._rec.total_new_tokens
            avg_draft = (sum(s.draft_len for s in self._rec.step_records)
                         / max(len(self._rec.step_records), 1))
            p["prompt_metrics"].append(PromptMetrics(
                prompt_index=self._prompt_idx,
                wall_time_s=wall,
                tps=tps,
                draft_len=avg_draft,
                accepted=total_new,
            ))
            # Take a structure snapshot at end of each prompt
            snap = snapshot_structures(p.get("ctx_refs"))
            snap.timestamp = time.perf_counter()
            p["struct_snapshots"].append(snap)
        return result

    collector._SequenceContext.__init__ = patched_init
    collector._SequenceContext.add_step = patched_add_step
    collector._SequenceContext.__exit__ = patched_exit


# =============================================================================
# Experiment runner with profiling
# =============================================================================

def run_profiled_experiment(
    exp_class,
    *,
    tiny_models: bool = False,
    max_samples: int = 10,
    max_new_tokens: int = 64,
    output_dir: str = "results_profile",
    device: str = "cuda",
    experiment_name: str = "unknown",
) -> ExperimentProfile:
    """Run a single experiment with comprehensive profiling."""
    profile = ExperimentProfile(name=experiment_name)
    global _profiler_state

    # Pre-run GC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # --- Phase 0: Baseline ---
    snap = snapshot_memory("00_baseline")
    profile.snapshots.append(snap)

    # --- Phase 1: Build models ---
    from experiments.runner import ExperimentRunner, ExperimentConfig

    runner = ExperimentRunner(experiments=[], output_dir=output_dir, device=device)

    # Create a single experiment instance (must reuse for stateful build methods)
    exp = exp_class()
    cfg = exp.get_config()
    if tiny_models:
        cfg.drafter_model_path = "facebook/opt-125m"
        cfg.target_model_path = "facebook/opt-350m"
        cfg.max_new_tokens = 32
    cfg.max_samples = max_samples

    logger.info("Building models for %s...", experiment_name)
    drafter, target = runner._build_models(cfg)

    snap = snapshot_memory("01_models_loaded")
    profile.snapshots.append(snap)

    # --- Phase 2: Build components ---
    from experiments.base import BuildContext

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

    snap = snapshot_memory("02_components_built")
    profile.snapshots.append(snap)

    # --- Phase 3: Build decoder ---
    from core.decoder.speculative import SpeculativeDecoder

    draft_length = getattr(cfg, "draft_length", 5)
    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target,
        translator=translator,
        cache=cache,
        draft_length=draft_length,
    )

    snap = snapshot_memory("03_decoder_built")
    profile.snapshots.append(snap)

    # --- Phase 4: Load dataset ---
    logger.info("Loading dataset (%d samples)...", cfg.max_samples)
    prompts = runner._load_dataset(cfg)

    snap = snapshot_memory("04_dataset_loaded")
    profile.snapshots.append(snap)

    # --- Phase 5: Decode loop with profiling ---
    from benchmarks.metrics.collector import BenchmarkCollector

    collector = BenchmarkCollector(name=experiment_name)

    # Detect replay distiller (stored as _replay_distiller attribute on the OnlineDistiller)
    replay_distiller = None
    if distiller is not None:
        try:
            replay_distiller = distiller._replay_distiller
        except AttributeError:
            pass

    # Patch the collector to intercept per-prompt data
    _profiler_state = {
        "ctx_refs": {
            "collector": collector,
            "distiller": distiller,
            "replay_distiller": replay_distiller,
            "cache": cache,
            "decoder": decoder,
        },
        "accepted_tokens_total": 0,
        "draft_tokens_total": 0,
        "prompt_metrics": [],
        "struct_snapshots": [],
    }

    _patch_benchmark_collector(collector)
    collector._gpu_mem_samples = []

    max_new_tokens = getattr(cfg, "max_new_tokens", 32)

    log_msg = f"Decoding {len(prompts)} prompts for {experiment_name}..."
    logger.info(log_msg)

    t_decode_start = time.perf_counter()

    for i, (input_ids, prompt_len) in enumerate(prompts):
        input_ids = input_ids.to(device)
        collector._current_prompt_idx = i

        # Memory snapshot at prompt start
        if i == 0 or i % max(1, len(prompts) // 10) == 0 or i == len(prompts) - 1:
            prompt_snap = snapshot_memory(f"05_prompt_{i:03d}")
            profile.snapshots.append(prompt_snap)

        # Router selection
        if router is not None:
            try:
                selected_drafter, _selected_idx = router.select_drafter(input_ids)
                if selected_drafter is not None:
                    decoder.drafter = selected_drafter
            except Exception:
                pass

        # Generate
        with collector.record_sequence(prompt_len=prompt_len) as seq_rec:
            decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                adaptive_length_fn=adaptive_fn,
                distiller=distiller,
                rng=None,
            )
            for sr in decoder._step_results[-max_new_tokens:]:
                seq_rec.add_step(
                    draft_len=sr.draft_length,
                    accepted=sr.accepted_count,
                    cache_hit=sr.cache_hit,
                )
        decoder._step_results.clear()

    t_decode_end = time.perf_counter()
    profile.total_wall_time_s = t_decode_end - t_decode_start
    profile.prompt_metrics = _profiler_state["prompt_metrics"]
    profile.struct_snapshots = _profiler_state["struct_snapshots"]

    # --- Phase 6: Post-decode summary ---
    summary = collector.summary()

    snap = snapshot_memory("06_decode_complete")
    profile.snapshots.append(snap)
    profile.struct_snapshots.append(
        snapshot_structures(_profiler_state.get("ctx_refs"))
    )

    # Save profile data
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, f"profile_{experiment_name}.json")

    # Make profile serializable
    profile_data = _serialize_profile(profile, summary)
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2, default=str)

    logger.info("Profile saved to %s", profile_path)
    logger.info("Summary: %s", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in summary.items()}))

    # Cleanup
    del decoder, drafter, target, translator, cache, distiller, router, collector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return profile


def _serialize_profile(profile: ExperimentProfile, summary: dict) -> dict:
    """Convert profile to JSON-serializable dict."""
    return {
        "name": profile.name,
        "total_wall_time_s": profile.total_wall_time_s,
        "snapshots": [
            {
                "phase": s.phase,
                "gpu_allocated_gb": round(s.gpu_allocated_gb, 4),
                "gpu_reserved_gb": round(s.gpu_reserved_gb, 4),
                "gpu_util_pct": round(s.gpu_util_pct, 1),
                "gc_total": s.gc_total,
                "timestamp_offset_s": round(s.timestamp - (profile.snapshots[0].timestamp if profile.snapshots else s.timestamp), 4),
            }
            for s in profile.snapshots
        ],
        "struct_snapshots": [
            {
                "collector_records": s.collector_records,
                "distiller_losses": s.distiller_losses,
                "distiller_kl_losses": s.distiller_kl_losses,
                "distiller_nll_losses": s.distiller_nll_losses,
                "replay_buffer_size": s.replay_buffer_size,
                "cache_size": s.cache_size,
                "cache_max_size": s.cache_max_size,
                "step_results_len": s.step_results_len,
                "timestamp_offset_s": round(s.timestamp - (profile.struct_snapshots[0].timestamp if profile.struct_snapshots else s.timestamp), 4),
            }
            for s in profile.struct_snapshots
        ],
        "prompt_metrics": [
            {
                "prompt_index": pm.prompt_index,
                "wall_time_s": round(pm.wall_time_s, 4),
                "tps": round(pm.tps, 2),
                "draft_len": round(pm.draft_len, 2),
                "accepted": pm.accepted,
            }
            for pm in profile.prompt_metrics
        ],
        "summary": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()},
    }


# =============================================================================
# Report generator
# =============================================================================

def generate_report(profiles: list[ExperimentProfile]) -> str:
    """Generate a human-readable profiling report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MEMORY & PERFORMANCE PROFILING REPORT")
    lines.append("=" * 80)

    # ---- GPU Memory Timeline ----
    lines.append("\n" + "─" * 80)
    lines.append("1. GPU MEMORY TIMELINE (per experiment)")
    lines.append("─" * 80)

    for prof in profiles:
        lines.append(f"\n  Experiment: {prof.name}")
        lines.append(f"  Total decode wall time: {prof.total_wall_time_s:.2f}s")
        lines.append(f"  Prompts decoded: {len(prof.prompt_metrics)}")

        # Memory delta analysis
        if prof.snapshots:
            initial_alloc = prof.snapshots[0].gpu_allocated_gb
            peak_alloc = max(s.gpu_allocated_gb for s in prof.snapshots)
            final_alloc = prof.snapshots[-1].gpu_allocated_gb
            reserved_initial = prof.snapshots[0].gpu_reserved_gb
            reserved_final = prof.snapshots[-1].gpu_reserved_gb

            lines.append(f"  GPU allocated: {initial_alloc:.2f}GB → {final_alloc:.2f}GB (Δ={final_alloc - initial_alloc:+.2f}GB)")
            lines.append(f"  GPU reserved:  {reserved_initial:.2f}GB → {reserved_final:.2f}GB (Δ={reserved_final - reserved_initial:+.2f}GB)")
            lines.append(f"  Peak allocated: {peak_alloc:.2f}GB")

            # Detect growth
            if final_alloc > initial_alloc + 0.1:
                lines.append(f"  ⚠️  MEMORY LEAK DETECTED: {final_alloc - initial_alloc:.2f}GB leaked during experiment")
            if reserved_final > reserved_initial + 0.2:
                lines.append(f"  ⚠️  GPU reserved grew: {reserved_final - reserved_initial:.2f}GB (uncollectable fragments)")

        # Per-prompt TPS
        if prof.prompt_metrics:
            tps_values = [pm.tps for pm in prof.prompt_metrics if pm.tps > 0]
            if tps_values:
                avg_tps = sum(tps_values) / len(tps_values)
                min_tps = min(tps_values)
                max_tps = max(tps_values)
                lines.append(f"  TPS: avg={avg_tps:.1f} min={min_tps:.1f} max={max_tps:.1f}")

                # Check if TPS degrades over time
                first_half = tps_values[:len(tps_values)//2]
                second_half = tps_values[len(tps_values)//2:]
                if first_half and second_half:
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    if second_avg < first_avg * 0.7:
                        lines.append(f"  ⚠️  TPS DEGRADATION: {first_avg:.1f} → {second_avg:.1f} ({(1-second_avg/first_avg)*100:.0f}% drop)")

    # ---- Data Structure Growth ----
    lines.append("\n" + "─" * 80)
    lines.append("2. DATA STRUCTURE GROWTH (per experiment)")
    lines.append("─" * 80)

    for prof in profiles:
        lines.append(f"\n  Experiment: {prof.name}")

        if prof.struct_snapshots:
            initial = prof.struct_snapshots[0]
            final = prof.struct_snapshots[-1]

            fields = [
                ("collector_records", "BenchmarkCollector records"),
                ("distiller_losses", "Distiller losses list"),
                ("distiller_kl_losses", "Distiller KL losses list"),
                ("distiller_nll_losses", "Distiller NLL losses list"),
                ("replay_buffer_size", "Replay buffer entries"),
                ("cache_size", "N-gram cache entries"),
                ("step_results_len", "Step results list"),
            ]

            lines.append(f"  {'Metric':<35} {'Initial':>10} {'Final':>10} {'Growth':>10}")
            lines.append(f"  {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 10}")

            for field_name, label in fields:
                init_val = getattr(initial, field_name, 0)
                final_val = getattr(final, field_name, 0)
                growth = final_val - init_val
                lines.append(f"  {label:<35} {init_val:>10} {final_val:>10} {growth:>+10}")

                if field_name != "cache_size" and growth > 100:  # cache is expected to fill
                    lines.append(f"    ⚠️  {label}: growing by {growth} entries (potential leak)")

    # ---- Prompt-by-Prompt TPS ----
    lines.append("\n" + "─" * 80)
    lines.append("3. PER-PROMPT TPS ANALYSIS")
    lines.append("─" * 80)

    for prof in profiles:
        lines.append(f"\n  Experiment: {prof.name}")
        if prof.prompt_metrics:
            lines.append(f"  {'Prompt':>6} {'Wall(s)':>8} {'TPS':>8} {'Draft':>8} {'Accept':>8}")
            lines.append(f"  {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
            for pm in prof.prompt_metrics[:30]:  # First 30
                lines.append(f"  {pm.prompt_index:>6} {pm.wall_time_s:>8.3f} {pm.tps:>8.1f} {pm.draft_len:>8.2f} {pm.accepted:>8}")
            if len(prof.prompt_metrics) > 30:
                lines.append(f"  ... and {len(prof.prompt_metrics) - 30} more prompts")

    # ---- Summary of Findings ----
    lines.append("\n" + "─" * 80)
    lines.append("4. AUTOMATED FINDINGS SUMMARY")
    lines.append("─" * 80)

    for prof in profiles:
        lines.append(f"\n  Experiment: {prof.name}")

        # GPU memory leak check (exclude "00_baseline" which is before models)
        if prof.snapshots and len(prof.snapshots) >= 2:
            # Use the post-models-loaded snapshot as the real baseline
            model_snap = next((s for s in prof.snapshots if s.phase == "01_models_loaded"), None)
            if model_snap is None:
                model_snap = prof.snapshots[0]

            initial_alloc = model_snap.gpu_allocated_gb
            final_alloc = prof.snapshots[-1].gpu_allocated_gb
            reserved_initial = model_snap.gpu_reserved_gb
            reserved_final = prof.snapshots[-1].gpu_reserved_gb

            if final_alloc > initial_alloc + 0.1:
                lines.append(f"  🔴 GPU allocated leaked: {final_alloc - initial_alloc:.2f}GB")
            if reserved_final > reserved_initial + 0.2:
                lines.append(f"  🟡 GPU reserved leaked: {reserved_final - reserved_initial:.2f}GB")

            # Per-prompt memory drift (after models loaded)
            prompt_snaps = [s for s in prof.snapshots if s.phase.startswith("05_prompt_")]
            if len(prompt_snaps) >= 3:
                mid_alloc = prompt_snaps[len(prompt_snaps)//2].gpu_allocated_gb
                if mid_alloc > prompt_snaps[0].gpu_allocated_gb + 0.3:
                    lines.append(f"  🟡 GPU memory grows mid-experiment: {mid_alloc - prompt_snaps[0].gpu_allocated_gb:.2f}GB")

        # Structure growth check
        if prof.struct_snapshots and len(prof.struct_snapshots) >= 2:
            init = prof.struct_snapshots[0]
            final = prof.struct_snapshots[-1]
            if final.collector_records > 50:
                lines.append(f"  🟡 BenchmarkCollector: {final.collector_records} records (unbounded)")
            if final.distiller_losses > 100:
                lines.append(f"  🟡 Distiller losses: {final.distiller_losses} entries (unbounded)")
            if final.replay_buffer_size > 0:
                lines.append(f"  🟡 Replay buffer: {final.replay_buffer_size}/{4096} entries (bounded)")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Profile memory & performance of speculative decoding experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", action="append", help="Run specific experiment(s) by name")
    group.add_argument("--suite", choices=["ablation"], help="Run full ablation suite")
    parser.add_argument("--tiny", action="store_true", help="Use tiny models (OPT-125m/OPT-350m)")
    parser.add_argument("-n", "--max-samples", type=int, default=10, help="Max samples per experiment (default: 10)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens per sequence")
    parser.add_argument("--output-dir", default="results_profile", help="Output directory for profiles")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be profiled, don't run")
    args = parser.parse_args()

    # Import experiment suites
    from experiments import ABLATION_SUITE, discover_experiments

    # Discover experiments
    if args.suite == "ablation":
        experiments = ABLATION_SUITE
    elif args.experiment:
        all_exps = discover_experiments(include_research=False)
        experiments = [e for e in all_exps if e.meta.name in args.experiment]
        if not experiments:
            logger.error("No matching experiments found")
            return
    else:
        parser.print_help()
        return

    logger.info("Profiling %d experiment(s): %s", len(experiments),
                [e.meta.name for e in experiments])

    profiles = []
    for exp in experiments:
        logger.info("=" * 60)
        logger.info("PROFILING: %s — %s", exp.meta.name, exp.meta.description)
        logger.info("=" * 60)

        try:
            profile = run_profiled_experiment(
                exp.__class__,
                tiny_models=args.tiny,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                output_dir=args.output_dir,
                device=args.device,
                experiment_name=exp.meta.name,
            )
            profiles.append(profile)
        except Exception as e:
            import traceback
            logger.error("Experiment %s FAILED: %s", exp.meta.name, e)
            traceback.print_exc()
            profiles.append(ExperimentProfile(name=exp.meta.name, error=str(e)))

    # Generate report
    report = generate_report(profiles)

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "profiling_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
