"""
extensions/routing/router.py

Dynamic Multi-Drafter Router.

Given multiple drafter models of different sizes (e.g. 32M, 68M, 160M),
a lightweight router selects the most appropriate drafter per prompt.

Router:
  Input  : prompt embedding (mean-pooled embeddings)
  Output : difficulty score → select drafter

Training:
  For each (prompt, drafter) pair, record acceptance rate.
  Drafter that maximises acceptance_rate * (1 / drafter_size_penalty)
  is the label.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class DrafterSpec:
    name: str
    model: object  # DraftModel instance
    n_params: int  # approximate parameter count
    size_penalty: float = 1.0  # larger → more expensive

    def efficiency_score(self, acceptance_rate: float) -> float:
        return acceptance_rate / max(self.size_penalty, 1e-6)


class RouterModel(nn.Module):
    """
    Small MLP router that maps a prompt embedding to a drafter index.

    Parameters
    ----------
    d_input   : input embedding dimension
    n_drafters: number of drafter choices
    """

    def __init__(self, d_input: int, n_drafters: int, d_hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_drafters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (n_drafters,) or (B, n_drafters)."""
        return self.net(x)

    def select(self, prompt_embedding: torch.Tensor) -> int:
        """Return index of the selected drafter."""
        with torch.no_grad():
            param_dtype = next(self.parameters()).dtype
            x = prompt_embedding.to(dtype=param_dtype).unsqueeze(0)
            logits = self.forward(x)
        return int(logits.squeeze(0).argmax().item())


class DynamicRouter:
    """
    Orchestrates multi-drafter selection.

    Parameters
    ----------
    drafter_specs : ordered list of DrafterSpec (small → large)
    router_model  : RouterModel (optional; falls back to smallest if None)
    embedder      : callable(input_ids) → embedding (d,)
    """

    def __init__(
        self,
        drafter_specs: list[DrafterSpec],
        router_model: RouterModel | None = None,
        embedder=None,
    ) -> None:
        self.specs = drafter_specs
        self.router = router_model
        self.embedder = embedder
        self.n_drafters = len(drafter_specs)

        # Training buffer — bounded deque to prevent unbounded memory growth.
        # maxlen=5000 covers typical experiment sizes; oldest samples drop
        # off automatically when the buffer fills.
        self._train_X: deque[torch.Tensor] = deque(maxlen=5000)
        self._train_y: deque[int] = deque(maxlen=5000)

        # Lazy cache: rebuild torch.Tensor batch only when data changes.
        self._cached_X: torch.Tensor | None = None
        self._cached_y: list[int] | None = None
        self._cache_dirty: bool = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object, int]:
        """
        Select the best drafter for this input.

        Returns (DraftModel, drafter_index).
        Falls back to spec 0 if the selected spec has no model loaded.
        """
        if self.router is None or self.embedder is None:
            logger.info('Selecting default (0) drafter')
            return self.specs[0].model, 0

        emb = self.embedder(input_ids)  # (d,)
        idx = self.router.select(emb)
        if self.specs[idx].model is None:
            logger.warning(
                "Router selected drafter %d/%d but model is None; falling back to spec 0",
                idx, self.n_drafters,
            )
            idx = 0
        logger.debug("Router selected drafter %d/%d", idx, self.n_drafters)
        return self.specs[idx].model, idx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def record(
        self,
        input_ids: torch.Tensor,
        drafter_idx: int,
        acceptance_rate: float,
    ) -> None:
        """Store a training observation (oldest dropped when buffer is full)."""
        if self.embedder is None:
            return
        with torch.no_grad():
            emb = self.embedder(input_ids)
        efficiency = self.specs[drafter_idx].efficiency_score(acceptance_rate)
        self._train_X.append(emb.cpu())
        self._train_y.append(drafter_idx)
        self._cache_dirty = True  # invalidate cached batch

    def _get_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazy-compute the (X, y) training batch from deque buffers.

        The result is cached and only recomputed when new data is recorded.
        """
        if not self._cache_dirty and self._cached_X is not None:
            assert self._cached_y is not None
            return self._cached_X, torch.tensor(self._cached_y, dtype=torch.long)

        self._cached_X = torch.stack(list(self._train_X))
        self._cached_y = list(self._train_y)
        self._cache_dirty = False
        return self._cached_X, torch.tensor(self._cached_y, dtype=torch.long)

    def train_router(
        self,
        n_epochs: int = 10,
        lr: float = 1e-3,
    ) -> float:
        """Train the router on collected observations (uses lazy caching)."""
        if self.router is None or not self._train_X:
            return 0.0

        device = next(self.router.parameters()).device
        X, y = self._get_train_batch()
        X, y = X.to(device), y.to(device)

        opt = torch.optim.Adam(self.router.parameters(), lr=lr)
        total_loss = 0.0
        for _ in range(n_epochs):
            opt.zero_grad()
            logits = self.router(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        mean_loss = total_loss / n_epochs
        logger.info(
            "Router trained: n_epochs=%d samples=%d mean_loss=%.4f",
            n_epochs,
            len(self._train_X),
            mean_loss,
        )
        return mean_loss

    def stats(self) -> dict:
        return {
            "n_drafters": self.n_drafters,
            "drafter_names": [s.name for s in self.specs],
            "n_train_samples": len(self._train_X),
        }
