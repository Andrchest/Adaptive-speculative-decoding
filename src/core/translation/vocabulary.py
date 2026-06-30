"""
core/translation/vocabulary.py

CrossVocabTranslator — orchestrates Rule1, Rule2, and optionally the
learned TranslatorModel (from extensions/).

translate(draft_logits) → target_probs  (k, target_vocab)
"""

from __future__ import annotations

import logging

import torch

from .rules import Rule1Mapping, Rule2Mapping

import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _align_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Pad with zeros or truncate the last dimension of *x* to *size*.

    Defensive guard against vocab-size mismatches between a tokenizer's
    reported vocabulary (len(tokenizer.get_vocab())) and a model's actual
    lm_head output dimension (model.config.vocab_size). These commonly
    diverge for OPT/GPT-2-style models whose embedding matrices are padded
    for hardware alignment (e.g. 50265 vs 50272), and would otherwise cause
    shape errors when comparing translated probabilities against raw target
    logits (e.g. in SpeculativeDecoder._residual_sample).
    """
    cur = x.shape[-1]
    if cur == size:
        return x
    if cur < size:
        pad_shape = (*x.shape[:-1], size - cur)
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    return x[..., :size]


class SameVocabTranslator:
    """
    Fast probability translator for identical vocabularies.

    When drafter and target share the same tokenizer
    the translation is trivially identity: P_target(t) = P_draft(t) for every token.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def translate(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        draft_logits : (batch, vocab) or (vocab,)

        Returns
        -------
        target_probs : (batch, vocab) or (vocab,)  — identical to draft probs
        """
        squeeze = draft_logits.dim() == 1
        if squeeze:
            draft_logits = draft_logits.unsqueeze(0)

        # Remove extra dims: (B, 1, V) -> (B, V)
        if draft_logits.dim() == 3 and draft_logits.shape[1] == 1:
            draft_logits = draft_logits.squeeze(1)

        probs = F.softmax(draft_logits.float(), dim=-1)

        if squeeze:
            return probs.squeeze(0)
        return probs

    @property
    def rule1(self):
        """Compatibility shim: same vocab means Rule1 covers everything."""
        # Return a minimal object so existing code that accesses
        # translator.rule1.drafter_size / _valid_mask doesn't break.
        return _IdentityRule1Adapter(self.vocab_size)


class _IdentityRule1Adapter:
    """Thin adapter providing rule1-like interface for SameVocabTranslator."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.drafter_size = vocab_size
        self.target_size = vocab_size

    @property
    def _mapping(self):
        # Identity mapping: each drafter token maps to itself in target vocab
        if not hasattr(self, '_map'):
            self._map = torch.arange(self.vocab_size, dtype=torch.long)
        return self._map

    @property
    def _valid_mask(self):
        # All tokens are valid (identity mapping)
        if not hasattr(self, '_mask'):
            self._mask = torch.ones(self.vocab_size, dtype=torch.bool)
        return self._mask


class CrossVocabTranslator:
    """
    Combines Rule1 and Rule2 into a single probability vector over the target
    vocabulary.

    Optionally accepts a learned translator (extensions/translator/) that
    handles tokens not covered by the cache.

    Parameters
    ----------
    rule1           : Rule1Mapping
    rule2           : Rule2Mapping
    learned_model   : optional TranslatorModel from extensions/translator/
    learned_weight  : weight for blending learned output (0 = disabled)
    """

    def __init__(
        self,
        rule1: Rule1Mapping,
        rule2: Rule2Mapping,
        learned_model=None,
        learned_weight: float = 0.0,
        lattice=None,
    ) -> None:
        self.rule1 = rule1
        self.rule2 = rule2
        self.learned_model = learned_model
        self.learned_weight = learned_weight
        self.lattice = lattice
        logger.info(
            "CrossVocabTranslator initialized: rule1=%s rule2=%s learned=%s weight=%.2f",
            type(rule1).__name__,
            type(rule2).__name__,
            "yes" if learned_model else "no",
            learned_weight,
        )

    def translate(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        draft_logits : (k, drafter_vocab) float tensor

        Returns
        -------
        target_probs : (k, target_vocab) probability tensor (sums to ≤ 1 per row)
        """
        logger.debug("Translating %d draft logits", draft_logits.shape[0])
        # Rule 1 — exact matches
        r1 = self.rule1.map_logits(draft_logits)  # (k, target_vocab)
        r1_mask = r1.sum(dim=0) > 0  # target tokens covered by R1

        # Rule 2 — either lattice exact mapping or approximate redistribution
        if self.lattice is not None:
            # Lattice: exact probability mass via DAG forward-DP
            r2 = self.lattice.exact_map_logits(draft_logits)  # (k, target_vocab)
            # Zero out R1-covered positions to avoid double-counting
            r2 = r2 * (1 - r1_mask.float().unsqueeze(0))
            logger.debug("Using lattice for Rule 2 (exact mapping)")
        else:
            # Fallback: approximate redistribution via Rule 2 heuristic
            r2 = self.rule2.map_logits(draft_logits, rule1_mask=r1_mask)

        # Combine: R1 has priority; R2 fills the rest
        combined = r1 + r2

        # Learned translator blend
        if self.learned_model is not None and self.learned_weight > 0:
            learned = self.learned_model.predict(draft_logits)  # (k, target_vocab)
            combined = (1 - self.learned_weight) * combined + self.learned_weight * learned

        # Normalise each row (may have residual un-mapped mass)
        row_sums = combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        combined = combined / row_sums

        logger.debug(
            "Translation complete: %d positions -> target_vocab=%d r1_covered=%d",
            draft_logits.shape[0],
            self.rule1.target_size,
            r1_mask.sum().item(),
        )

        # Final defensive alignment — guarantees the output width always
        # matches rule1.target_size even if an extension (learned model /
        # lattice) returned a differently-sized tensor.
        return _align_last_dim(combined, self.rule1.target_size)

    @classmethod
    def from_tokenizers(
        cls,
        drafter_tokenizer,
        target_tokenizer,
        device: str = "cpu",
        learned_model=None,
        learned_weight: float = 0.0,
        lattice=None,
        drafter_vocab_size: int | None = None,
        target_vocab_size: int | None = None,
    ) -> CrossVocabTranslator:
        """
        drafter_vocab_size / target_vocab_size should be the *model's*
        ``config.vocab_size`` (the lm_head output dimension), not
        ``len(tokenizer.get_vocab())``. These frequently differ — e.g. OPT
        and GPT-2 pad their embedding matrices to 50272 while the tokenizer
        reports a 50265-entry vocab — and passing the tokenizer size here
        will cause shape mismatches against raw model logits downstream.

        Optimization: if vocabularies are identical (same tokenizer), returns a
        fast SameVocabTranslator — no mapping tables, just softmax. ~30-50x faster.
        """
        # ── Fast path: identical vocabularies ──────────────────────────
        if drafter_vocab_size is not None and target_vocab_size is not None:
            if drafter_vocab_size == target_vocab_size:
                d_vocab = drafter_tokenizer.get_vocab()
                t_vocab = target_tokenizer.get_vocab()
                if len(d_vocab) == len(t_vocab) and d_vocab == t_vocab:
                    logger.info(
                        "SameVocabTranslator: vocabularies identical (size=%d), "
                        "using fast softmax-only path (no mapping)",
                        drafter_vocab_size,
                    )
                    return SameVocabTranslator(drafter_vocab_size)

        # ── Slow path: different vocabularies ──────────────────────────
        r1 = Rule1Mapping(
            drafter_tokenizer,
            target_tokenizer,
            device=device,
            drafter_vocab_size=drafter_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        r2 = Rule2Mapping(
            drafter_tokenizer,
            target_tokenizer,
            device=device,
            drafter_vocab_size=drafter_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        logger.info(
            "Built CrossVocabTranslator: drafter_vocab=%d target_vocab=%d device=%s",
            r1.drafter_size,
            r1.target_size,
            device,
        )
        return cls(r1, r2, learned_model=learned_model, learned_weight=learned_weight, lattice=lattice)
