"""
extensions/lattice/tokenizer_lattice.py

Exact cross-tokenizer probability mapping via a tokenizer lattice DAG.

Replaces the approximate Rule 2 heuristic with DP-exact computation.

Algorithm:
----------
Given a target token string s_t (e.g. "flake"):
  1. Enumerate all valid drafter tokenisations of s_t using the drafter's
     vocabulary (i.e. all ways the drafter can cover s_t exactly).
  2. Build a DAG where:
       - node i = character offset i in s_t
       - edge (i→j) labelled with drafter token covering s_t[i:j]
  3. The probability of target token s_t under the drafter is:
       P(s_t) = Σ_{paths p in DAG} Π_{tokens e in p} q(e)
     where q(e) is the drafter's probability for token e.
  4. This is computed exactly via forward DP on the DAG.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    _HAS_NX = True
except ImportError:
    _HAS_NX = False


class TokenizerLattice:
    """
    Builds and caches character-level DAGs for target tokens,
    then computes exact probability mass via DP.

    Parameters
    ----------
    drafter_tokenizer : HF tokenizer for the drafter
    target_tokenizer  : HF tokenizer for the target
    max_cache_size    : number of (target_token_string → lattice) to cache
    """

    def __init__(
        self,
        drafter_tokenizer,
        target_tokenizer,
        max_cache_size: int = 8192,
        drafter_vocab_size: int | None = None,
        target_vocab_size: int | None = None,
    ) -> None:
        self.drafter_tok = drafter_tokenizer
        self.target_tok = target_tokenizer

        # String look-ups: idx → clean string
        self._drafter_vocab = self._build_vocab(drafter_tokenizer)  # idx → str
        self._drafter_by_string: dict[str, int] = {v: k for k, v in self._drafter_vocab.items()}
        self._target_vocab = self._build_vocab(target_tokenizer)  # idx → str

        # Prefer model.config.vocab_size (lm_head output dim) over the
        # tokenizer dict size — see Rule1Mapping for why these can differ.
        self.drafter_size = (
            drafter_vocab_size if drafter_vocab_size is not None else len(self._drafter_vocab)
        )
        self.target_size = (
            target_vocab_size if target_vocab_size is not None else len(self._target_vocab)
        )

        # Lattice cache: target_string → list of drafter-token-id paths
        # Each "path" is a list of drafter token ids whose strings concatenate to target_string
        self._lattice_cache: dict[str, list[list[int]]] = {}
        self._max_cache = max_cache_size
        logger.info(
            "TokenizerLattice initialized: drafter_vocab=%d target_vocab=%d max_cache=%d",
            self.drafter_size,
            self.target_size,
            max_cache_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, target_string: str) -> list[list[int]]:
        """
        Return all valid drafter tokenisation paths for *target_string*.
        Each path is a list of drafter token ids.
        """
        if target_string in self._lattice_cache:
            return self._lattice_cache[target_string]

        paths = self._enumerate_paths(target_string)
        logger.debug(
            "Lattice build for '%s': %d paths (cache size=%d)",
            target_string[:20],
            len(paths),
            len(self._lattice_cache),
        )

        if len(self._lattice_cache) >= self._max_cache:
            # Simple FIFO eviction
            oldest = next(iter(self._lattice_cache))
            del self._lattice_cache[oldest]
        self._lattice_cache[target_string] = paths
        return paths

    def forward(
        self,
        drafter_probs: torch.Tensor,  # (drafter_vocab,)
        target_token_id: int,
    ) -> torch.Tensor:
        """
        Compute the exact probability mass assigned by the drafter to a
        given target token (scalar tensor).

        P(target_token) = Σ_paths Π_{token in path} drafter_probs[token]
        """
        t_str = self._target_vocab.get(target_token_id, "")
        if not t_str:
            return torch.tensor(0.0, device=drafter_probs.device)

        paths = self.build(t_str)
        if not paths:
            return torch.tensor(0.0, device=drafter_probs.device)

        total = torch.tensor(0.0, device=drafter_probs.device, dtype=drafter_probs.dtype)
        for path in paths:
            path_prob = torch.tensor(1.0, device=drafter_probs.device, dtype=drafter_probs.dtype)
            for tok_id in path:
                path_prob = path_prob * drafter_probs[tok_id]
            total = total + path_prob
        return total

    def backward(
        self,
        drafter_probs: torch.Tensor,  # (drafter_vocab,)
        target_probs: torch.Tensor,  # (target_vocab,)
    ) -> torch.Tensor:
        """
        Compute the KL divergence KL(target || lattice_approx) over the
        target vocabulary.

        This drives the lattice-aware distillation loss.
        """
        target_vocab_size = target_probs.shape[-1]
        lattice_probs = torch.zeros(
            target_vocab_size, device=drafter_probs.device, dtype=drafter_probs.dtype
        )
        for t_idx in range(target_vocab_size):
            lattice_probs[t_idx] = self.forward(drafter_probs, t_idx)

        # Normalise (may not sum to 1 if drafter can't cover all target tokens)
        lattice_probs = lattice_probs / lattice_probs.sum().clamp(min=1e-8)
        log_lattice = lattice_probs.clamp(min=1e-8).log()
        kl = F.kl_div(log_lattice, target_probs.clamp(min=1e-8), reduction="sum", log_target=False)
        return kl

    def exact_map_logits(self, drafter_logits: torch.Tensor) -> torch.Tensor:
        """
        Map drafter logits → exact target probability vector.

        drafter_logits : (k, drafter_vocab)  or  (drafter_vocab,)
        returns        : (k, target_vocab)   or  (target_vocab,)
        """
        squeeze = drafter_logits.dim() == 1
        if squeeze:
            drafter_logits = drafter_logits.unsqueeze(0)

        k = drafter_logits.shape[0]
        logger.debug(
            "Lattice exact_map_logits: k=%d drafter_vocab=%d target_size=%d",
            k,
            drafter_logits.shape[-1],
            self.target_size,
        )

        k = drafter_logits.shape[0]
        target_size = self.target_size
        drafter_probs = F.softmax(drafter_logits.float(), dim=-1)  # (k, Vd)
        result = torch.zeros(k, target_size, device=drafter_logits.device)

        # Iterate over each batch step
        for step in range(k):
            dp = drafter_probs[step]
            # forward() returns 0.0 for target_token_id outside the
            # tokenizer's string table (i.e. padding region of target_size),
            # so this naturally yields zeros there.
            for t_idx in range(target_size):
                result[step, t_idx] = self.forward(dp, t_idx)

        if squeeze:
            return result.squeeze(0)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vocab(tokenizer) -> dict[int, str]:
        vocab = tokenizer.get_vocab()  # str → int
        rev = {}
        for tok_str, idx in vocab.items():
            clean = tok_str.replace("▁", " ").replace("Ġ", " ").replace("##", "")
            rev[idx] = clean.strip()
        return rev

    def _enumerate_paths(self, target_str: str) -> list[list[int]]:
        """
        DP over character positions to find all valid drafter tokenisations.

        dp[i] = list of partial paths reaching offset i
        """
        n = len(target_str)
        dp: dict[int, list[list[int]]] = {0: [[]]}

        for i in range(n):
            if i not in dp:
                continue
            for j in range(i + 1, n + 1):
                sub = target_str[i:j]
                if sub in self._drafter_by_string:
                    d_id = self._drafter_by_string[sub]
                    if j not in dp:
                        dp[j] = []
                    for path in dp[i]:
                        dp[j].append(path + [d_id])

        return dp.get(n, [])
