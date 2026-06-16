"""
extensions/translator/model.py

Lightweight Transformer Encoder translator.

Input : sequence of drafter subtoken ids (variable length)
Output: probability distribution over target vocabulary

Architecture:
  Embedding → Transformer Encoder (2-4 layers) → Pool → Linear → Softmax

Parameters: < 10M

Training:
  - Offline from speculative decoding traces
  - Online updates (small lr, few steps)

Hybrid mode:
  if cache_hit: use cache
  else:         use translator
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TranslatorModel(nn.Module):
    """
    Lightweight Transformer encoder that maps a sequence of drafter subtoken
    embeddings to a probability distribution over the target vocabulary.

    Parameters
    ----------
    drafter_vocab_size  : size of drafter vocabulary
    target_vocab_size   : size of target vocabulary
    d_model             : embedding / model dimension (default 256)
    n_heads             : attention heads (default 4)
    n_layers            : number of encoder layers (default 2)
    max_seq_len         : maximum input sequence length (default 16)
    dropout             : dropout probability (default 0.1)
    """

    def __init__(
        self,
        drafter_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drafter_vocab_size = drafter_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(drafter_vocab_size, d_model, padding_idx=0)
        self.pos_enc = _PositionalEncoding(d_model, max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, target_vocab_size)

        self._init_weights()
        logger.info(
            "TranslatorModel initialized: d_model=%d n_heads=%d n_layers=%d params=%d",
            d_model,
            n_heads,
            n_layers,
            self.param_count(),
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq)
        padding_mask: torch.Tensor | None = None,  # (batch, seq) bool
    ) -> torch.Tensor:
        """Returns logits (batch, target_vocab_size)."""
        x = self.embedding(input_ids)  # (B, S, d)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # Mean pooling over non-padding positions
        if padding_mask is not None:
            mask_f = (~padding_mask).float().unsqueeze(-1)  # (B, S, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (B, d)
        return self.proj(x)  # (B, target_vocab)

    # ------------------------------------------------------------------
    # High-level API used by CrossVocabTranslator
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        draft_logits: torch.Tensor,  # (k, drafter_vocab)
        topk_tokens: int = 8,
    ) -> torch.Tensor:
        """
        For each of the k draft positions, predict a probability vector over
        the target vocabulary.

        This works by treating the top-k drafter tokens as the "subtoken sequence"
        for that position.

        Returns (k, target_vocab) probability tensor.
        """
        self.eval()
        k = draft_logits.shape[0]
        results = []
        for i in range(k):
            probs = F.softmax(draft_logits[i].float(), dim=-1)
            top_ids = probs.topk(topk_tokens).indices.unsqueeze(0)  # (1, topk)
            logits = self.forward(top_ids)  # (1, target_vocab)
            results.append(F.softmax(logits.squeeze(0), dim=-1))
        return torch.stack(results, dim=0)  # (k, target_vocab)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,  # (B, S)
        target_ids: torch.Tensor,  # (B,) — ground-truth target token ids
        optimizer: torch.optim.Optimizer,
        padding_mask: torch.Tensor | None = None,
    ) -> float:
        """Single supervised training step. Returns loss value."""
        self.train()
        optimizer.zero_grad()
        logits = self.forward(input_ids, padding_mask)  # (B, target_vocab)
        loss = F.cross_entropy(logits, target_ids)
        loss.backward()
        optimizer.step()
        logger.debug("Train step loss=%.4f", loss.item())
        return loss.item()

    def online_update(
        self,
        drafter_subtoken_ids: list[int],
        target_token_id: int,
        lr: float = 1e-5,
    ) -> float:
        """Single-example online update."""
        if not hasattr(self, "_online_opt"):
            self._online_opt = torch.optim.Adam(self.parameters(), lr=lr)
        ids = torch.tensor(
            [drafter_subtoken_ids], dtype=torch.long, device=next(self.parameters()).device
        )
        return self.train_step(
            ids,
            torch.tensor([target_token_id], device=ids.device),
            self._online_opt,
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]
