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
import time

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
        self._last_access: dict[str, float] = {}
        self._max_cache = max_cache_size

        # Precomputed match index for fast DP: target_string → [(start, end, d_id)]
        self._match_index = self._build_match_index()

        # Precompute match tensors and filtered str_to_ids
        self._precompute_match_tensors()

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

        Uses LRU eviction to keep hot entries longer.
        """
        if target_string in self._lattice_cache:
            self._last_access[target_string] = time.time()
            return self._lattice_cache[target_string]

        paths = self._enumerate_paths(target_string)
        logger.debug(
            "Lattice build for '%s': %d paths (cache size=%d)",
            target_string[:20],
            len(paths),
            len(self._lattice_cache),
        )

        if len(self._lattice_cache) >= self._max_cache:
            # LRU eviction: remove the least recently accessed entry
            oldest_key = min(self._last_access, key=self._last_access.get)
            del self._lattice_cache[oldest_key]
            del self._last_access[oldest_key]

        self._lattice_cache[target_string] = paths
        self._last_access[target_string] = time.time()
        return paths

    def forward(
        self,
        drafter_probs: torch.Tensor,  # (drafter_vocab,)
        target_token_id: int,
    ) -> torch.Tensor:
        """
        Compute the exact probability mass assigned by the drafter to a
        given target token using forward DP over character positions.

        Replaces exponential path enumeration with O(n²) DP where n is the
        string length. Uses precomputed match index from _build_match_index().

        P(target_token) = Σ_paths Π_{token in path} drafter_probs[token]
        """
        t_str = self._target_vocab.get(target_token_id, "")
        if not t_str:
            return torch.tensor(0.0, device=drafter_probs.device, dtype=drafter_probs.dtype)

        matches = self._match_index.get(t_str)
        if not matches:
            return torch.tensor(0.0, device=drafter_probs.device, dtype=drafter_probs.dtype)

        n = len(t_str)
        fwd = torch.zeros(n + 1, device=drafter_probs.device, dtype=drafter_probs.dtype)
        fwd[0] = 1.0

        # Process matches sorted by start position (already sorted from _build_match_index)
        for start, end, d_id in matches:
            if fwd[start] > 0:
                fwd[end] += fwd[start] * drafter_probs[d_id]

        return fwd[n]

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
        squeeze = drafter_logits.dim() == 1
        if squeeze:
            drafter_logits = drafter_logits.unsqueeze(0)

        k = drafter_logits.shape[0]
        target_size = self.target_size
        drafter_probs = F.softmax(drafter_logits.float(), dim=-1)
        result = torch.zeros(k, target_size, device=drafter_logits.device)

        for step in range(k):
            dp = drafter_probs[step]

            for t_str, t_ids in self._str_to_ids.items():
                starts, ends, d_ids = self._match_tensors[t_str]
                if starts.device != dp.device:
                    starts = starts.to(dp.device, non_blocking=True)
                    ends = ends.to(dp.device, non_blocking=True)
                    d_ids = d_ids.to(dp.device, non_blocking=True)
                    self._match_tensors[t_str] = (starts, ends, d_ids)
                n = len(t_str)
                fwd = torch.zeros(n + 1, device=dp.device, dtype=dp.dtype)
                fwd[0] = 1.0

                for pos in range(n):
                    mask = starts == pos
                    if mask.any():
                        fwd.index_add_(0, ends[mask], fwd[pos] * dp[d_ids[mask]])

                prob = fwd[-1]
                if prob > 0:
                    result[step, t_ids] = prob

        if squeeze:
            return result.squeeze(0)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _precompute_match_tensors(self) -> None:
        self._match_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        filtered_str_to_ids: dict[str, list[int]] = {}

        for t_str, t_ids in self._build_str_to_ids().items():
            matches = self._match_index.get(t_str)

            if matches is None:
                continue

            filtered_str_to_ids[t_str] = t_ids
            m = torch.tensor(matches, dtype=torch.long)
            self._match_tensors[t_str] = (m[:, 0], m[:, 1], m[:, 2])

        self._str_to_ids = filtered_str_to_ids

    def _build_match_index(self) -> dict[str, list[tuple[int, int, int]]]:
        """Precompute drafter token matches for fast forward DP.

        For each target string, find all drafter token substrings and their
        character positions. Returns a sorted list of (start, end, d_id) tuples.
        Sorting is by (start, end) so the forward DP processes matches in
        topological order.
        """
        match_index: dict[str, list[tuple[int, int, int]]] = {}
        for t_idx in range(self.target_size):
            t_str = self._target_vocab.get(t_idx, "")
            if not t_str:
                continue
            matches: list[tuple[int, int, int]] = []
            n = len(t_str)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    sub = t_str[i:j]
                    if sub in self._drafter_by_string:
                        matches.append((i, j, self._drafter_by_string[sub]))
            if matches:
                match_index[t_str] = matches
        return match_index

    def _build_str_to_ids(self) -> dict[str, list[int]]:
        """Map each unique non-empty string to its target token ids.

        Used for deduplication in exact_map_logits: compute probability once
        per unique string, then broadcast to all token ids sharing that string.
        """
        str_to_ids: dict[str, list[int]] = {}
        for t_idx, t_str in self._target_vocab.items():
            if t_str:
                str_to_ids.setdefault(t_str, []).append(t_idx)
        return str_to_ids

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
