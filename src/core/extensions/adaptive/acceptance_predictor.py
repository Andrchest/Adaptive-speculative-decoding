"""
ADDITIONS to core/extensions/adaptive/speedup_predictor.py

Implements acceptance-rate prediction as an alternative training signal
to wall-clock speedup, per the adaptive-draft-length approach in the
original paper. Designed as a drop-in sibling to SpeedupPredictor so
AdaptiveDraftController can be built with either objective and the two
can be A/B compared via the existing ExperimentResult / comparison_table
machinery.

Why a separate predictor instead of multi-task head on SpeedupPredictor:
  - Acceptance rate is bounded in [0, 1] and is a property of the
    (drafter, target, context) triple alone — it doesn't depend on
    system load, kernel launch overhead, or GPU contention the way
    wall-clock speedup does. Mixing the two objectives in one network
    would let timing noise leak gradient into the acceptance head.
  - Keeping them separate lets you log acceptance_rate_mse and
    speedup_mse independently and decide empirically which transfers
    better to k-selection, which is the comparison being asked for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AcceptanceSample:
    hidden: torch.Tensor
    draft_len: int
    acceptance_rate: float  # accepted_count / draft_len for this step, in [0, 1]


class AcceptanceRatePredictor(nn.Module):
    """
    Predicts, for each candidate draft length k in [1, k_max], the
    expected acceptance rate of drafting k tokens from the current
    hidden state.

    Architecturally identical to SpeedupPredictor (same d_hidden -> 128
    -> k_max MLP) so the two are comparable controllers differing only
    in training target and output activation. Output passes through
    sigmoid since acceptance rate is bounded in [0, 1] — SpeedupPredictor
    has no such bound (speedup can exceed 1), so this is the one
    architectural difference and it matters: an unbounded linear head
    regressing a [0,1] target wastes capacity learning the boundary.
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
        # Same pessimistic-prior fix as SpeedupPredictor: bias toward a
        # LOW logit (sigmoid(-2.0) ≈ 0.12 acceptance) so heads that have
        # never been trained for a given hidden-state region don't look
        # competitive against heads with real, even mediocre, signal.
        # See SpeedupPredictor for the full rationale — this predictor
        # has the identical missing-action problem since it's trained
        # the same way (one observed head per sample).
        nn.init.constant_(self.net[-1].bias, -2.0)
        self._buffer: deque[AcceptanceSample] = deque(maxlen=8192)
        self._optimizer: torch.optim.Optimizer | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (batch, k_max), pre-sigmoid."""
        out = self.net(hidden)
        if out.isnan().any() or out.isinf().any():
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        return out

    def predict_rates(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns predicted acceptance rates (batch, k_max) in [0, 1]."""
        return torch.sigmoid(self.forward(hidden))

    def select_k(self, hidden: torch.Tensor, expected_tokens: bool = True, epsilon: float = 0.10) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(1, self.k_max + 1, (1,)).item()

        self.eval()
        with torch.no_grad():
            rates = self.predict_rates(hidden.unsqueeze(0)).squeeze(0)  # (k_max,)
        if expected_tokens:
            k_values = torch.arange(1, self.k_max + 1, device=rates.device, dtype=rates.dtype)
            scores = rates * k_values
        else:
            scores = rates
        k = int(scores.argmax().item()) + 1
        return k

    def record(self, hidden: torch.Tensor, draft_len: int, acceptance_rate: float) -> None:
        self._buffer.append(
            AcceptanceSample(hidden.detach().cpu(), draft_len, acceptance_rate)
        )

    def train_on_buffer(
        self,
        n_steps: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        rng: torch.Generator | None = None,
    ) -> float:
        """
        Same masked-regression scheme as SpeedupPredictor.train_on_buffer,
        but with BCE-with-logits instead of MSE since the target is a
        rate in [0, 1] rather than an unbounded speedup ratio. BCE gives
        a steeper gradient near 0/1 than MSE would, which matters here
        because acceptance rates cluster near the extremes (mostly-accept
        or mostly-reject) far more than speedup ratios do.
        """
        if len(self._buffer) < batch_size:
            return 0.0
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        device = next(self.parameters()).device
        total_loss = 0.0
        n_valid_steps = 0

        for _ in range(n_steps):
            if rng is not None:
                indices = torch.randint(len(self._buffer), (batch_size,), generator=rng)
            else:
                indices = torch.randint(len(self._buffer), (batch_size,))
            samples = [self._buffer[i] for i in indices.tolist()]

            hidden_batch = torch.stack([s.hidden for s in samples]).to(device)
            target = torch.zeros(batch_size, self.k_max, device=device)
            obs_mask = torch.zeros(batch_size, self.k_max, device=device, dtype=torch.bool)
            for i, s in enumerate(samples):
                if s.draft_len >= 1:
                    k_idx = min(s.draft_len - 1, self.k_max - 1)
                    rate = min(max(s.acceptance_rate, 0.0), 1.0)
                    target[i, k_idx] = rate
                    obs_mask[i, k_idx] = True

            if not obs_mask.any():
                continue

            self._optimizer.zero_grad()
            pred_logits = self.forward(hidden_batch)
            loss = F.binary_cross_entropy_with_logits(
                pred_logits[obs_mask], target[obs_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self._optimizer.step()
            total_loss += loss.item()
            n_valid_steps += 1

        return total_loss / max(n_valid_steps, 1)


class AcceptanceAdaptiveController:
    """
    Sibling to AdaptiveDraftController, identical lifecycle and hidden-
    state-caching strategy (reuses the decoder's bonus-token forward —
    see SpeedupPredictor module docstring for why that avoids a
    redundant drafter forward per step), but driven by
    AcceptanceRatePredictor instead of SpeedupPredictor.

    Kept as a separate class rather than a mode flag on
    AdaptiveDraftController so the two can be wired into DIFFERENT
    experiments (SpeedupAdaptiveExperiment vs a new
    AcceptanceAdaptiveExperiment) and their results land in separate
    rows of comparison_table.csv without one experiment's config
    silently controlling which loss the other reports.
    """

    def __init__(
        self,
        predictor: AcceptanceRatePredictor,
        drafter_model,
        k_min: int = 1,
        k_max: int = 8,
    ) -> None:
        self.predictor = predictor
        self.drafter = drafter_model
        self.k_min = k_min
        self.k_max = k_max

        self._last_k: int | None = None
        self._last_hidden: torch.Tensor | None = None

    def __call__(self, context: torch.Tensor) -> int:
        if self._last_hidden is None:
            hidden = self._get_hidden(context)
            self._last_hidden = hidden
        else:
            hidden = self._last_hidden

        k = self.predictor.select_k(hidden)
        k = max(self.k_min, min(k, self.k_max))
        self._last_k = k
        return k

    def update_hidden(self, hidden: torch.Tensor) -> None:
        if hidden is not None:
            self._last_hidden = hidden.detach().float()

    def record_result(self, accepted_count: int) -> None:
        """
        Unlike AdaptiveDraftController.record_result, this needs NO wall-
        clock timing — acceptance rate is accepted_count / k, known the
        instant verification finishes, with no perf_counter() bookkeeping
        and no dependency on when this method happens to get called
        relative to a stored _last_start. That timing-independence is
        the main practical advantage of this objective over speedup:
        it can't be skewed by how long record_result took to get wired
        in, or by GPU scheduling jitter between draft and verify.
        """
        if self._last_k is None or self._last_hidden is None:
            return
        rate = accepted_count / max(self._last_k, 1)
        self.predictor.record(self._last_hidden, self._last_k, rate)

    def _get_hidden(self, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.drafter.model(context, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()

    def reset(self) -> None:
        self._last_hidden = None
        self._last_k = None
        self._last_start = None
