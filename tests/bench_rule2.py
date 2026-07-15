#!/usr/bin/env python3
"""Benchmark Rule2 optimization end-to-end."""

import sys
import time
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.translation.vocabulary import CrossVocabTranslator
from core.models.drafter import DraftModel, TargetModel
from core.decoder.speculative import SpeculativeDecoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEVICE = "cuda"


def main():
    drafter_tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    target_tok = AutoTokenizer.from_pretrained("facebook/opt-350m")

    drafter = DraftModel("facebook/opt-125m", device=DEVICE)
    target_model = TargetModel(
        "facebook/opt-350m", device=DEVICE, dtype=torch.float32, load_in_4bit=False
    )
    translator = CrossVocabTranslator.from_tokenizers(
        drafter_tok,
        target_model.tokenizer,
        drafter_vocab_size=drafter.model.config.vocab_size,
        target_vocab_size=target_model.model.config.vocab_size,
    )

    decoder = SpeculativeDecoder(
        drafter=drafter,
        target=target_model,
        translator=translator,
        draft_length=5,
    )

    prompt = "The capital of France is"
    input_ids = drafter_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # Warmup
    decoder.generate(input_ids, max_new_tokens=10)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = decoder.generate(input_ids, max_new_tokens=32)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = len(output[0]) - len(input_ids[0])
    tps = new_tokens / elapsed

    print(f"Generated {new_tokens} tokens in {elapsed:.2f}s")
    print(f"TPS: {tps:.1f}")
    print(f"Output: {drafter_tok.decode(output[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
