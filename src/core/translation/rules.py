"""
core/translation/rules.py

Rule 1  — Direct token probability mapping.
           If a drafter token exactly matches a target token (same string),
           its probability is transferred 1-to-1.

Rule 2  — Approximate probability redistribution.
           For drafter tokens whose strings are sub-strings of target tokens,
           distribute probability mass proportionally.

Both rules return a dense probability vector over the *target* vocabulary.

See also: core/extensions/lattice/ for the exact replacement of Rule 2.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import ahocorasick
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _process_target_worker(args):
    (
        t_idx,
        t_str,
        contains_matches,
        contained_by_matches,
        prefix_matches,
    ) = args

    if not t_str:
        return None

    contributors = defaultdict(float)

    # d_str in t_str
    for d_idx, d_len in contains_matches:
        contributors[d_idx] = max(contributors[d_idx], float(d_len))

    # t_str in d_str
    for d_idx, overlap_len in contained_by_matches:
        contributors[d_idx] = max(
            contributors[d_idx],
            float(overlap_len),
        )

    # longest common prefix
    for d_idx, prefix_len in prefix_matches:
        contributors[d_idx] = max(
            contributors[d_idx],
            float(prefix_len),
        )

    if not contributors:
        return None

    total = sum(contributors.values())

    return (
        t_idx,
        [(d_idx, w / total) for d_idx, w in contributors.items()],
    )


class Rule1Mapping:
    """
    Exact-match probability mapping from drafter → target vocabulary.

    Builds a (drafter_vocab_size,) → (target_vocab_size,) index mapping.
    """

    def __init__(
        self,
        drafter_tokenizer,
        target_tokenizer,
        device: str = "cpu",
        drafter_vocab_size: int | None = None,
        target_vocab_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        drafter_vocab_size, target_vocab_size : optional int
            Pass ``model.config.vocab_size`` here (the actual lm_head output
            dimension), NOT ``len(tokenizer.get_vocab())``. Many model
            families (OPT, GPT-2, etc.) pad the embedding matrix for
            hardware alignment, so the tokenizer's vocab dict can be smaller
            than the model's logits dimension (e.g. 50265 vs 50272 for OPT).
            If omitted, falls back to the tokenizer vocab size for backward
            compatibility (fine for fake/test tokenizers).
        """
        self.drafter_vocab = drafter_tokenizer.get_vocab()  # str → int
        self.target_vocab = target_tokenizer.get_vocab()  # str → int
        self.drafter_size = (
            drafter_vocab_size if drafter_vocab_size is not None else len(self.drafter_vocab)
        )
        self.target_size = (
            target_vocab_size if target_vocab_size is not None else len(self.target_vocab)
        )

        # Build mapping: drafter_idx → target_idx  (-1 if no match)
        self._mapping: torch.Tensor = torch.full(
            (self.drafter_size,), -1, dtype=torch.long, device=device
        )
        for tok_str, d_idx in self.drafter_vocab.items():
            if d_idx >= self.drafter_size:
                continue  # outside the model's actual embedding range
            t_idx = self.target_vocab.get(tok_str, -1)
            if t_idx >= self.target_size:
                t_idx = -1
            self._mapping[d_idx] = t_idx

        self._valid_mask = self._mapping >= 0  # (drafter_vocab,)
        self.device = device
        n_mapped = self._valid_mask.sum().item()
        logger.info(
            "Rule1Mapping: %d/%d drafter tokens have exact target match (%.1f%%)",
            n_mapped,
            self.drafter_size,
            100.0 * n_mapped / max(self.drafter_size, 1),
        )

    def map_logits(self, drafter_logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        drafter_logits : (batch, drafter_vocab) or (drafter_vocab,)

        Returns
        -------
        target_probs   : (batch, target_vocab) or (target_vocab,)
                         Un-normalised; caller may need softmax.
        """
        squeeze = drafter_logits.dim() == 1
        if squeeze:
            drafter_logits = drafter_logits.unsqueeze(0)

        device = drafter_logits.device
        drafter_probs = F.softmax(drafter_logits.float(), dim=-1)  # (B, Vd)
        batch = drafter_probs.shape[0]
        target_probs = torch.zeros(
            batch, self.target_size, dtype=torch.float32, device=device
        )

        # Ensure index tensors are on the same device as the data tensors
        # to avoid implicit cross-device copies that can trigger OOM under
        # memory fragmentation (see #index_add_ OOM with CUDA + CPU indices).
        valid_mask = self._valid_mask.to(device, non_blocking=True)
        mapping = self._mapping.to(device, non_blocking=True)

        # Pre-compute index tensors once (avoid repeated torch.where calls)
        source_d_indices = torch.where(valid_mask)[0]  # (M,)
        target_d_indices = mapping[source_d_indices]    # (M,) — target vocab indices

        # scatter_add: for each valid drafter token, accumulate its prob at target index
        target_probs.index_add_(1, target_d_indices, drafter_probs[:, source_d_indices])
        if squeeze:
            return target_probs.squeeze(0)
        return target_probs


class Rule2Mapping:
    """
    Approximate probability redistribution for tokens without direct matches.

    Algorithm:
      For each target token t with string s_t:
        Find all drafter tokens whose strings are sub-strings of s_t
        (or vice-versa).  Redistribute probability proportional to the
        length of overlap.

    This is the heuristic baseline; replace with TokenizerLattice for
    exact computation.
    """

    def __init__(
        self,
        drafter_tokenizer,
        target_tokenizer,
        device: str = "cpu",
        drafter_vocab_size: int | None = None,
        target_vocab_size: int | None = None,
    ) -> None:
        self.drafter_tok = drafter_tokenizer
        self.target_tok = target_tokenizer
        # See Rule1Mapping docstring: prefer model.config.vocab_size over
        # len(tokenizer.get_vocab()) so output tensors line up with logits.
        self.drafter_size = (
            drafter_vocab_size
            if drafter_vocab_size is not None
            else len(drafter_tokenizer.get_vocab())
        )
        self.target_size = (
            target_vocab_size
            if target_vocab_size is not None
            else len(target_tokenizer.get_vocab())
        )
        self.device = device

        # Pre-build string maps
        self._drafter_strings: list[str] = self._build_string_list(drafter_tokenizer)
        self._target_strings: list[str] = self._build_string_list(target_tokenizer)

        # Sparse transfer matrix: (target_idx) → list[(drafter_idx, weight)]
        self._transfer: dict[int, list[tuple[int, float]]] = {}
        self._build_transfer_matrix()
        logger.info(
            "Rule2Mapping built: drafter_size=%d target_size=%d transfer_entries=%d",
            self.drafter_size,
            self.target_size,
            len(self._transfer),
        )

    # ------------------------------------------------------------------

    def map_logits(
        self,
        drafter_logits: torch.Tensor,
        rule1_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        drafter_logits : (batch, drafter_vocab) or (drafter_vocab,)
        rule1_mask     : optional bool tensor (target_vocab,) marking tokens
                         already handled by Rule 1 (to avoid double-counting).

        Returns
        -------
        rule2_target_probs : (batch, target_vocab) or (target_vocab,)
        """
        squeeze = drafter_logits.dim() == 1
        if squeeze:
            drafter_logits = drafter_logits.unsqueeze(0)

        drafter_probs = F.softmax(drafter_logits.float(), dim=-1)  # (B, Vd)
        batch = drafter_probs.shape[0]
        target_probs = torch.zeros(
            batch,
            self.target_size,
            dtype=torch.float32,
            device=drafter_logits.device,
        )

        for t_idx, contrib in self._transfer.items():
            if rule1_mask is not None and rule1_mask[t_idx]:
                continue
            d_indices = torch.tensor(
                [d for d, _ in contrib], dtype=torch.long, device=drafter_logits.device
            )
            weights = torch.tensor(
                [w for _, w in contrib], dtype=torch.float32, device=drafter_logits.device
            )
            target_probs[:, t_idx] = (drafter_probs[:, d_indices] * weights).sum(dim=-1)

        if squeeze:
            return target_probs.squeeze(0)
        return target_probs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_string_list(tokenizer) -> list[str]:
        vocab = tokenizer.get_vocab()  # str → int
        strings = [""] * len(vocab)
        for tok_str, idx in vocab.items():
            # Strip SentencePiece / BPE artefacts for overlap comparison
            clean = tok_str.replace("▁", " ").replace("Ġ", " ").strip()
            strings[idx] = clean
        return strings

    def _build_transfer_matrix(self):

        drafter_strings = self._drafter_strings
        target_strings = self._target_strings

        #
        # 1. Build Aho-Corasick automaton
        #
        automaton = ahocorasick.Automaton()

        for d_idx, d_str in enumerate(drafter_strings):
            if d_str:
                automaton.add_word(
                    d_str,
                    (d_idx, len(d_str)),
                )

        automaton.make_automaton()

        #
        # 2. Build reverse containment lookup
        #
        contained_by = defaultdict(list)

        for d_idx, d_str in enumerate(drafter_strings):
            if not d_str:
                continue

            contained_by[d_str].append((d_idx, len(d_str)))

        #
        # 3. Prefix index
        #
        prefix_index = defaultdict(list)

        for d_idx, d_str in enumerate(drafter_strings):
            if not d_str:
                continue

            max_prefix = min(len(d_str), 16)

            for k in range(1, max_prefix + 1):
                prefix_index[d_str[:k]].append((d_idx, k))

        #
        # 4. Prepare work
        #
        work_items = []

        for t_idx, t_str in enumerate(target_strings):
            if not t_str:
                continue

            #
            # d_str in t_str
            #
            contains_matches = []

            for _, (d_idx, d_len) in automaton.iter(t_str):
                contains_matches.append((d_idx, d_len))

            #
            # t_str in d_str
            #
            contained_by_matches = contained_by.get(t_str, [])

            #
            # longest common prefix
            #
            prefix_matches = []

            max_prefix = min(len(t_str), 16)

            for k in range(max_prefix, 0, -1):
                pref = t_str[:k]

                if pref in prefix_index:
                    prefix_matches.extend(prefix_index[pref])
                    break

            work_items.append(
                (
                    t_idx,
                    t_str,
                    contains_matches,
                    contained_by_matches,
                    prefix_matches,
                )
            )

        #
        # 5. Parallel reduction
        #
        with ProcessPoolExecutor() as pool:
            for result in pool.map(
                _process_target_worker,
                work_items,
                chunksize=256,
            ):
                if result is None:
                    continue

                t_idx, contribs = result
                self._transfer[t_idx] = contribs

    @staticmethod
    def _overlap(a: str, b: str) -> int:
        """Length of common substring (simple heuristic)."""
        if a == b:
            return len(a)
        if a in b:
            return len(a)
        if b in a:
            return len(b)
        # Longest common prefix
        n = min(len(a), len(b))
        for i in range(n, 0, -1):
            if a[:i] == b[:i]:
                return i
        return 0
