# SpeculativeDecoder

The core speculative decoding engine.

## Interface

```python
class SpeculativeDecoder:
    def generate(self, input_ids, max_new_tokens, adaptive_length_fn, distiller)
```

## Components

- **DraftModel** — small model for token proposal
- **TargetModel** — large model for verification
- **CrossVocabTranslator** — vocab mapping
- **NgramCache** — cache layer

## Algorithm

1. Drafter generates k tokens
2. Target scores all k tokens + the next position
3. Acceptance/rejection with residual sampling
4. Cache update + optional distillation

See [drafter.py](../../../src/omnidraft/core/drafter.py) and [speculative.py](../../../src/omnidraft/core/speculative.py).
