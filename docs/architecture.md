# Architecture Overview

## System Design

This project implements adaptive speculative decoding for accelerating LLM inference.

### Core Pipeline

```
Prompt → [Drafter] → Draft k tokens → [Translator] → Map to target vocab
                                                      ↓
                                             [Target Model] → Verify all k tokens
                                                      ↓
                                             Accept / Reject + Residual sample
                                                      ↓
                                             Update Cache + Distill
```

### Key Components

1. **SpeculativeDecoder** — orchestrates the main loop (draft → verify → accept)
2. **CrossVocabTranslator** — maps probabilities from drafter vocab to target vocab
3. **NgramCache** — caches previous decoding traces for faster lookup
4. **OnlineDistiller** — continuously improves the drafter during inference
5. **Extensions** — modular plugins for advanced features

See individual module docs for details:
- [Core](modules/core/speculative.md)
- [Translation](modules/translation/cross_vocab.md)
- [Cache](modules/cache/ngram.md)
- [Distillation](modules/distillation/online.md)
