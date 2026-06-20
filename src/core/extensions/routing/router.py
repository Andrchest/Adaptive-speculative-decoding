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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            logits = self.forward(prompt_embedding.unsqueeze(0))
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

        # Training buffer
        self._train_X: list[torch.Tensor] = []
        self._train_y: list[int] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_drafter(self, input_ids: torch.Tensor) -> tuple[object, int]:
        """
        Select the best drafter for this input.

        Returns (DraftModel, drafter_index).
        """
        if self.router is None or self.embedder is None:
            logger.info('Selecting default (0) drafter')
            return self.specs[0].model, 0

        emb = self.embedder(input_ids)  # (d,)
        idx = self.router.select(emb)
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
        """Store a training observation."""
        if self.embedder is None:
            return
        with torch.no_grad():
            emb = self.embedder(input_ids)
        efficiency = self.specs[drafter_idx].efficiency_score(acceptance_rate)
        self._train_X.append(emb.cpu())
        self._train_y.append(drafter_idx)

    def train_router(
        self,
        n_epochs: int = 10,
        lr: float = 1e-3,
    ) -> float:
        """Train the router on collected observations."""
        if self.router is None or not self._train_X:
            return 0.0

        device = next(self.router.parameters()).device
        X = torch.stack(self._train_X).to(device)
        y = torch.tensor(self._train_y, dtype=torch.long, device=device)

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
