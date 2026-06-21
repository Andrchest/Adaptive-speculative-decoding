"""
extensions/adaptive/speedup_predictor.py

Speedup-Aware Adaptive Drafting.

Instead of predicting acceptance probability, directly predict expected
tokens/sec speedup for each candidate draft length k in {1, …, K_max}.

Inputs : last hidden state of drafter (d_hidden,)
Outputs: predicted speedup for k = 1 … K_max  (K_max,)

Inference policy:
  k* = argmax_{k} predicted_speedup[k]

Training:
  Collect (hidden, k, observed_speedup) tuples during decoding.
  Fit with MSE regression.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpeedupSample:
    hidden: torch.Tensor  # (d_hidden,) — drafter's last hidden state
    draft_len: int
    speedup: float  # measured tokens/sec / baseline tokens/sec


class SpeedupPredictor(nn.Module):
    """
    Small MLP that predicts speedup for each k.

    Architecture:
      hidden_state → LayerNorm → Linear(d, 128) → GELU → Linear(128, K_max)
    """

    def __init__(self, d_hidden: int, k_max: int = 8) -> None:
        super().__init__()
        self.k_max = k_max
        self.net = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 128),
            nn.GELU(),
            nn.Linear(128, k_max),
        )
        self._buffer: deque[SpeedupSample] = deque(maxlen=8192)
        self._optimizer: torch.optim.Optimizer | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns predicted speedup per k: (K_max,) or (B, K_max)."""
        out = self.net(hidden)
        if out.isnan().any() or out.isinf().any():
            logger.warning("SpeedupPredictor output contains NaN/Inf — zeroing them")
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        return out

    def select_k(self, hidden: torch.Tensor) -> int:
        """Return the k with the highest predicted speedup."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(hidden.unsqueeze(0)).squeeze(0)
        k = int(preds.argmax().item()) + 1
        logger.debug("SpeedupPredictor selected k=%d (scores=%s)", k, preds.tolist())
        return k

    def record(self, hidden: torch.Tensor, draft_len: int, speedup: float) -> None:
        """Store an observation for later training."""
        self._buffer.append(SpeedupSample(hidden.detach().cpu(), draft_len, speedup))

    def train_on_buffer(
        self,
        n_steps: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        rng: torch.Generator | None = None,
    ) -> float:
        """
        Fit on collected samples; returns mean loss.

        Each sample provides a speedup observation for exactly ONE
        draft length ``k`` (the one actually used). The other ``K_max-1``
        columns of the target are unknown and MUST be masked out of the
        MSE — otherwise the predictor is trained to output 0 for every
        unobserved ``k``, which contaminates the regression target and
        breaks ``select_k`` (it would return whichever column happened
        to get the most non-zero observations, not the genuinely best k).

        Parameters
        ----------
        rng : optional torch.Generator for deterministic sampling.
        """
        if len(self._buffer) < batch_size:
            logger.debug(
                "SpeedupPredictor: buffer too small (%d < %d), skipping train",
                len(self._buffer),
                batch_size,
            )
            return 0.0
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        device = next(self.parameters()).device
        total_loss = 0.0
        n_valid_steps = 0

        for _ in range(n_steps):
            if rng is not None:
                indices = torch.randint(
                    len(self._buffer), (batch_size,), generator=rng
                )
            else:
                indices = torch.randint(len(self._buffer), (batch_size,))
            samples = [self._buffer[i] for i in indices.tolist()]

            hidden_batch = torch.stack([s.hidden for s in samples]).to(device)
            target = torch.zeros(batch_size, self.k_max, device=device)
            obs_mask = torch.zeros(
                batch_size, self.k_max, device=device, dtype=torch.bool
            )
            for i, s in enumerate(samples):
                if s.draft_len >= 1:
                    k_idx = min(s.draft_len - 1, self.k_max - 1)
                    speedup = min(max(s.speedup, 0.0), 10.0)
                    target[i, k_idx] = speedup
                    obs_mask[i, k_idx] = True

            if not obs_mask.any():
                continue

            self._optimizer.zero_grad()
            pred = self.forward(hidden_batch)
            # MSE only on observed (sample, k) positions.
            loss = F.mse_loss(pred[obs_mask], target[obs_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self._optimizer.step()
            total_loss += loss.item()
            n_valid_steps += 1

            self._find_invalid_weights()

        mean_loss = total_loss / max(n_valid_steps, 1)
        logger.info(
            "SpeedupPredictor trained: n_steps=%d batch_size=%d buffer_size=%d mean_loss=%.6f",
            n_valid_steps,
            batch_size,
            len(self._buffer),
            mean_loss,
        )
        return mean_loss

    def _find_invalid_weights(self) -> None:
        for name, param in self.net.named_parameters():
            if torch.isnan(param).any():
                logger.warning("NaN found in %s — reinitializing", name)
                param.data.copy_(torch.randn_like(param) * 0.01)
            if torch.isinf(param).any():
                logger.warning("Inf found in %s — reinitializing", name)
                param.data.copy_(torch.randn_like(param) * 0.01)

class AdaptiveDraftController:
    """
    Manages dynamic draft length selection.

    Wraps SpeedupPredictor and provides the callable interface expected by
    SpeculativeDecoder: fn(context_tensor) → int.
    """

    def __init__(
        self,
        predictor: SpeedupPredictor,
        drafter_model,  # to extract hidden states
        k_min: int = 1,
        k_max: int = 8,
        baseline_tokens_per_sec: float = 1.0,
    ) -> None:
        self.predictor = predictor
        self.drafter = drafter_model
        self.k_min = k_min
        self.k_max = k_max
        self.baseline_tps = baseline_tokens_per_sec

        self._last_start: float | None = None
        self._last_k: int | None = None
        self._last_hidden: torch.Tensor | None = None

    def __call__(self, context: torch.Tensor) -> int:
        hidden = self._get_hidden(context)
        self._last_hidden = hidden
        k = self.predictor.select_k(hidden)
        k = max(self.k_min, min(k, self.k_max))
        self._last_k = k
        self._last_start = time.perf_counter()
        logger.debug(
            "AdaptiveDraftController: selected k=%d (clamped to [%d, %d])",
            k,
            self.k_min,
            self.k_max,
        )
        return k

    def record_result(self, accepted_count: int) -> None:
        """Call after each decode step to record actual speedup."""
        if self._last_start is None or self._last_k is None:
            return
        elapsed = time.perf_counter() - self._last_start
        elapsed = max(elapsed, 1e-6)  # avoid division by zero
        tps = accepted_count / elapsed
        speedup = tps / max(self.baseline_tps, 1e-6)
        speedup = min(max(speedup, 0.0), 10.0)  # clamp to prevent training instability
        if self._last_hidden is not None:
            self.predictor.record(self._last_hidden, self._last_k, speedup)
            logger.debug(
                "AdaptiveDraftController: k=%d accepted=%d tps=%.1f speedup=%.2f",
                self._last_k,
                accepted_count,
                tps,
                speedup,
            )

    def _get_hidden(self, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.drafter.model(context, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()  # (d_hidden,)
