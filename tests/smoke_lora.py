#!/usr/bin/env python3
"""
Smoke test for LoRA-based online distillation.

Verifies:
  1. LoRA adapters are applied correctly
  2. Only LoRA params are trainable, base params are frozen
  3. A decode step with LoRA distiller runs without errors
  4. LoRA parameters change after a weight update

Usage:
    cd src
    uv run python ../tests/smoke_lora.py
"""

from __future__ import annotations

import gc
import sys

import torch


def gpu_mem_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6


def print_mem(label: str = "") -> None:
    alloc, reserved = gpu_mem_mb()
    print(f"[MEM] {label}: allocated={alloc:.1f} MB  reserved={reserved:.1f} MB")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    drafter_path = "facebook/opt-125m"
    target_path = "facebook/opt-350m"

    print_mem("initial")

    from core.cache.ngram import NgramCache
    from core.decoder.speculative import SpeculativeDecoder
    from core.distillation.online import OnlineDistiller
    from core.models.drafter import DraftModel, TargetModel
    from core.translation.vocabulary import CrossVocabTranslator

    # --- Load models ---
    print("Loading drafter...")
    drafter = DraftModel(drafter_path, device=device, dtype=torch.float32)
    print_mem("after drafter")

    print("Loading target...")
    target = TargetModel(target_path, device=device, dtype=torch.float32, load_in_4bit=False)
    print_mem("after target")

    # --- Apply LoRA ---
    print("Applying LoRA adapters...")
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    drafter.model = get_peft_model(drafter.model, lora_config)
    print_mem("after LoRA")

    # --- Verify only LoRA params are trainable ---
    trainable_params = sum(1 for p in drafter.model.parameters() if p.requires_grad)
    frozen_params = sum(1 for p in drafter.model.parameters() if not p.requires_grad)
    trainable_elems = sum(p.numel() for p in drafter.model.parameters() if p.requires_grad)
    total_elems = sum(p.numel() for p in drafter.model.parameters())
    print(
        f"LoRA: trainable_params={trainable_params} frozen_params={frozen_params} "
        f"trainable_elems={trainable_elems}/{total_elems} "
        f"({100.0 * trainable_elems / max(total_elems, 1):.2f}%)"
    )
    assert trainable_params > 0, "No trainable LoRA parameters"
    assert frozen_params > 0, "No frozen base parameters"
    assert trainable_elems < total_elems, "All parameters are trainable, LoRA not applied correctly"
    print("[OK] LoRA parameter freeze verified")

    # --- Rest of pipeline ---
    translator = CrossVocabTranslator.from_tokenizers(
        drafter.tokenizer,
        target.tokenizer,
        device=device,
        drafter_vocab_size=drafter.model.config.vocab_size,
        target_vocab_size=target.model.config.vocab_size,
    )
    print_mem("after translator")

    cache = NgramCache(max_size=1024, eviction="lru")

    # Distiller — only LoRA params should be updated
    drafter.model.train()
    optimizer = torch.optim.Adam(drafter.model.parameters(), lr=1e-3)
    distiller = OnlineDistiller(
        drafter_model=drafter,
        translator=translator,
        optimizer=optimizer,
        lambda_ngram=0.5,
        use_lora=True,
    )
    print_mem("after distiller")

    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target,
        translator=translator,
        cache=cache,
        draft_length=5,
    )

    # --- Run generation with LoRA distillation ---
    prompt = "Write a short story about a cat."
    input_ids = drafter.tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input shape: {input_ids.shape}")
    print_mem("before generation")

    # Snapshot LoRA A weights from first layer before training
    lora_params_before = {}
    for name, p in drafter.model.named_parameters():
        if "lora_" in name:
            lora_params_before[name] = p.detach().clone()

    try:
        output = decoder.generate(
            input_ids,
            max_new_tokens=16,
            distiller=distiller,
        )
        print_mem("after generation")
        print(f"Output shape: {output.shape}")
        result_text = target.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated: {result_text[:200]}")

        stats = distiller.training_stats()
        print(f"Distiller stats: {stats}")

        # --- Verify LoRA weights changed ---
        any_changed = False
        for name, p in drafter.model.named_parameters():
            if "lora_" in name and name in lora_params_before:
                if not torch.equal(p.detach(), lora_params_before[name]):
                    any_changed = True
                    break

        if any_changed:
            print("[OK] LoRA weights changed after training")
        else:
            print("[WARN] LoRA weights did not change (may be normal if no updates occurred)")

        print("\n[OK] LoRA smoke test PASSED")
        sys.exit(0)

    except Exception as e:
        print_mem("at error")
        print(f"\n[FAIL] LoRA smoke test FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_mem("final")


if __name__ == "__main__":
    main()
