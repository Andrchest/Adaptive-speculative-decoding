# OnlineDistiller

Performs distillation during inference.

## Loss

```
L = KL_divergence + λ * NLL(accepted_tokens)
```

## Features

- Gradient accumulation
- LoRA support
- Running loss tracking
- Periodic weight updates

## ReplayDistiller

Wraps OnlineDistiller with a replay buffer for continual learning.

- **FIFO** — uniform random sampling
- **Prioritized** — weighted by (1 - acceptance_rate)
