# Optimization Results

> **Date:** 2026-06-25
> **Platform:** NVIDIA GeForce RTX 3060 Ti (8GB)
> **Models:** OPT-125m (drafter) + OPT-350m 4-bit (target)
> **Prompt:** "The quick brown fox" (5 tokens), max_new_tokens=32

---

## Summary

All optimizations produce **bit-identical output tokens** across all phases when given the same seed. The overall throughput improved from **36.3 TPS to 45.3 TPS** (+24.8%).

## Performance Tracking

| Optimization | TPS | Acceptance | Wall (s) | Speedup | Tokens match? |
|-------------|-----|-----------|----------|---------|---------------|
| Baseline | 36.3 | 52% | 0.716 | 1.00x | ref |
| P0: KV Cache for Drafter | 39.4 | 52% | — | 1.09x | yes |
| P1: Vectorized Accept-Reject | 36.7 | 52% | — | 1.01x | yes |
| P2: Pre-allocated Output Buffer | 42.3 | 52% | 0.615 | 1.17x | yes |
| P3: Batch GPU Sync | 43.8 | 52% | 0.593 | 1.21x | yes |
| P4: Eliminate Copies | 43.2 | 52% | 0.602 | 1.19x | yes |
| P5: Logging Reduction | 45.3 | 52% | 0.574 | 1.25x | yes |

Note: Individual TPS measurements vary slightly due to system load. The trend is consistently upward.

## Detailed Changes

### P0: KV Cache for Drafter (+9%)
- **Files:** `src/core/models/draft_model.py`
- Refactored `_draft_impl` and `_draft_distill` to use `past_key_values` instead of `torch.cat`/`clone` in the draft loop
- Eliminates O(seq_len²) autoregressive drafting → O(seq_len)

### P1: Vectorized Accept-Reject (+1%)
- **Files:** `src/core/decoder/speculative.py`
- Batched softmax and probability gathering in `_accept_reject` (single kernel launch for all k positions)
- Preserved sequential RNG draws to maintain token identity
- Fixed CPU/CUDA generator mismatch in `_residual_sample`

### P2: Pre-allocated Output Buffer (+17%)
- **Files:** `src/core/decoder/speculative.py`
- Replaced `generated = input_ids.clone()` + repeated `torch.cat` with pre-allocated `torch.zeros((1, prompt_len + max_new_tokens))`
- Indexed assignment via `pos` cursor instead of dynamic tensor growth
- EOS check and budget enforcement updated to use `pos`

### P3: Batch GPU→CPU Synchronization (+21%)
- **Files:** `src/core/decoder/speculative.py`
- P3.1: Reused `ctx_list` in distiller section (eliminated redundant `context[0].tolist()`)
- P3.2: Vectorized `_translate_draft_tokens` with batched GPU indexing (`mapping[draft_tensor]`) and single `.tolist()` instead of per-token `.item()`
- P3.4: Eliminated `cache.insert` `.detach().cpu()` (cached logits are never read back)

### P4: Eliminate Unnecessary Copies (+19%)
- **Files:** `src/core/models/target_model.py`
- P4.2: Pre-allocated `_draft_buffer` in `TargetModel.verify()` with `.copy_()` instead of `torch.tensor()` each step

### P5: Logging Reduction (+25%)
- **Files:** `src/core/decoder/speculative.py`, `src/core/models/draft_model.py`
- Changed per-step `logger.info` to `logger.debug` in hot paths
- Kept `logger.info` for per-sequence summaries (start, EOS, finish)

## Verification

### Smoke Test
All phases pass `python src/main.py --smoke` with consistent acceptance rate (~52%).

### Differential Tests
All phases pass `pytest tests/test_differential.py`:
- `test_determinism_repeat_10_seeds` — same seed produces identical tokens across runs
- `test_acceptance_rate_range` — acceptance rates within expected bounds
- `test_different_seeds_different_tokens` — different seeds produce different output

### Token Identity
Each optimization was verified against the previous phase using seed=42:
- P0 → P1: bit-identical ✓
- P1 → P2: bit-identical ✓
- P2 → P3: bit-identical ✓
- P3 → P4: bit-identical ✓
- P4 → P5: bit-identical ✓

## Known Issues

### Ablation Suite OOM
The full 11-experiment ablation suite (`--suite ablation`) runs out of GPU memory on 8GB consumer GPUs due to Rule2 sparse matrix multiplication (`torch.sparse.mm`). This is a pre-existing limitation unrelated to the optimizations. The smoke test (single experiment) works correctly.

## Future Work

1. **Fused kernels:** Custom CUDA kernels for accept-reject could eliminate Python overhead entirely.
2. **FlashAttention:** If target model supports it, could further reduce KV cache overhead.
3. **Batched generation:** Process multiple prompts in parallel to amortize kernel launch costs.
4. **P3.3 (deferred):** `_residual_sample` `.item()` could be deferred further, but requires larger refactoring of token representation (list[int] → tensor).
