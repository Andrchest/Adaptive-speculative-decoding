#!/usr/bin/env python3
"""
Smoke test for the online distillation path.

Reproduces the exact pipeline of experiment 04_+online_distil with
minimal models and a single sample to catch OOM / shape bugs quickly.

Usage:
    cd src
    uv run python ../tests/smoke_distill.py
"""

from __future__ import annotations

import gc
import sys

import torch

# ---------------------------------------------------------------------------
# Quick GPU memory helper
# ---------------------------------------------------------------------------


def gpu_mem_mb() -> tuple[float, float]:
    """Return (allocated_MB, reserved_MB) or (0, 0) if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6,
    )


def print_mem(label: str = "") -> None:
    alloc, reserved = gpu_mem_mb()
    print(f"[MEM] {label}: allocated={alloc:.1f} MB  reserved={reserved:.1f} MB")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Use tiny models for fast testing
    drafter_path = "facebook/opt-125m"
    target_path = "facebook/opt-350m"

    print_mem("initial")

    # --- Import and build components ---
    from core.cache.ngram import NgramCache
    from core.decoder.speculative import SpeculativeDecoder
    from core.distillation.online import OnlineDistiller
    from core.models.drafter import DraftModel, TargetModel
    from core.translation.vocabulary import CrossVocabTranslator

    # Models
    print("Loading drafter...")
    drafter = DraftModel(drafter_path, device=device, dtype=torch.float32)
    print_mem("after drafter")

    print("Loading target...")
    target = TargetModel(
        target_path, device=device, dtype=torch.float32, load_in_4bit=False
    )
    print_mem("after target")

    # Translator
    print("Building translator...")
    translator = CrossVocabTranslator.from_tokenizers(
        drafter.tokenizer,
        target.tokenizer,
        device=device,
        drafter_vocab_size=drafter.model.config.vocab_size,
        target_vocab_size=target.model.config.vocab_size,
    )
    print_mem("after translator")

    # Cache
    cache = NgramCache(max_size=1024, eviction="lru")

    # Distiller
    print("Building distiller...")
    drafter.model.train()
    for p in drafter.model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(drafter.model.parameters(), lr=1e-5)
    distiller = OnlineDistiller(
        drafter_model=drafter,
        translator=translator,
        optimizer=optimizer,
        lambda_ngram=0.5,
        use_lora=False,
    )
    print_mem("after distiller")

    # Decoder
    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target,
        translator=translator,
        cache=cache,
        draft_length=5,
    )

    # --- Run generation with distillation ---
    print("Tokenizing input...")
    prompt = "Write a short story about a cat."
    input_ids = drafter.tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input shape: {input_ids.shape}")
    print_mem("before generation")

    try:
        output = decoder.generate(
            input_ids,
            max_new_tokens=32,
            distiller=distiller,
        )
        print_mem("after generation")
        print(f"Output shape: {output.shape}")

        # Decode output
        result_text = target.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated text (first 200 chars): {result_text[:200]}")

        # Print distiller stats
        stats = distiller.training_stats()
        print(f"Distiller stats: {stats}")

        print("\n[OK] Smoke test PASSED")
        sys.exit(0)

    except Exception as e:
        print_mem("at error")
        print(f"\n[FAIL] Smoke test FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_mem("final")


if __name__ == "__main__":
    main()
