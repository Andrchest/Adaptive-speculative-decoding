# Glossary

## Core Terms

- **Drafter** — small fast model that proposes k tokens
- **Target** — large slow model that verifies drafter's proposals
- **Speculative Decoding** — technique where target verifies multiple tokens in one pass
- **Acceptance Rate** — fraction of draft tokens accepted by target
- **Draft Length (k)** — number of tokens proposed by drafter
- **Cross-vocabulary Translation** — mapping draft tokens from drafter vocab to target vocab
- **Online Distillation** — training the drafter during inference using target's feedback
- **N-gram Cache** — cache of previous token sequences for faster lookup
- **Eviction Strategy** — policy for removing entries from the cache (LRU, LFU, etc.)
- **Residual Sampling** — sampling a bonus token from the rejected distribution
- **InfoNCE** — contrastive loss function used in training
- **LoRA** — Low-Rank Adaptation for efficient fine-tuning

## Metrics

- **Acceptance Rate** — fraction of draft tokens accepted
- **Tokens/sec** — generation speed
- **Speedup** — ratio vs autoregressive baseline
- **Cache Hit Rate** — fraction of cache lookups that resulted in a hit
