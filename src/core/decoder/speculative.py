"""
core/decoder/speculative.py

Core speculative decoding engine.

Implements:
  1. Draft generation (drafter produces k tokens)
  2. Target verification (target scores the full draft in one pass)
  3. Acceptance / rejection with residual sampling
  4. Cache lookup + update
  5. Online distillation trigger

The public interface is SpeculativeDecoder.generate().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from core.cache.ngram import NgramCache
from core.translation.vocabulary import CrossVocabTranslator, _align_last_dim

# ---------------------------------------------------------------------------
# Step stats (returned per decode step for logging / adaptive drafting)
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    draft_length: int
    accepted_count: int
    rejected_at: int  # index of first rejection (-1 = all accepted)
    wall_time_ms: float = 0.0
    cache_hit: bool = False
    draft_tokens: list[int] = field(default_factory=list)
    accepted_tokens: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.draft_length == 0:
            return 0.0
        return self.accepted_count / self.draft_length


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """
    Model-agnostic speculative decoding loop.

    Parameters
    ----------
    drafter        : DraftModel  — fast small model
    target         : TargetModel — slow large model
    translator     : CrossVocabTranslator — maps drafter probs → target vocab
    cache          : NgramCache
    draft_length   : default number of speculative tokens per step
    temperature    : sampling temperature (0 = greedy)
    """

    def __init__(
        self,
        drafter,
        target,
        translator: CrossVocabTranslator,
        cache: NgramCache | None = None,
        draft_length: int = 5,
        temperature: float = 1.0,
    ) -> None:
        self.drafter = drafter
        self.target = target
        self.translator = translator
        self.cache = cache or NgramCache()
        self.draft_length = draft_length
        self.temperature = temperature

        # Accumulated stats
        self._step_results: list[StepResult] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,  # (1, seq_len) on correct device
        max_new_tokens: int = 128,
        adaptive_length_fn=None,  # callable(hidden) → int, optional
        distiller=None,  # OnlineDistiller, optional
    ) -> torch.Tensor:
        """
        Full generation loop. Returns (1, seq_len + new_tokens).
        """
        import contextlib

        logger.info(
            "Starting speculative generation: prompt_len=%d max_new_tokens=%d draft_length=%d",
            input_ids.shape[1],
            max_new_tokens,
            self.draft_length,
        )
        generated = input_ids.clone()
        self.cache.step()

        # Only disable no_grad when distillation is active
        grad_ctx = contextlib.nullcontext() if distiller is not None else torch.no_grad()
        with grad_ctx:
            return self._generate_loop(input_ids, max_new_tokens, adaptive_length_fn, distiller)

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        adaptive_length_fn,
        distiller,
    ) -> torch.Tensor:
        generated = input_ids.clone()

        for step_idx in range(max_new_tokens):
            k = self._choose_draft_length(generated, adaptive_length_fn)
            logger.debug(
                "Decode step %d/%d selected draft length k=%d", step_idx + 1, max_new_tokens, k
            )

            result = self._decode_step(generated, k, distiller=distiller)
            self._step_results.append(result)
            self.cache.step()

            # Append accepted tokens
            if result.accepted_tokens:
                logger.debug(
                    "Decode step %d accepted %d token(s): %s",
                    step_idx + 1,
                    len(result.accepted_tokens),
                    result.accepted_tokens,
                )
                new_ids = torch.tensor(
                    result.accepted_tokens, dtype=torch.long, device=generated.device
                ).unsqueeze(0)
                generated = torch.cat([generated, new_ids], dim=1)
            else:
                logger.debug("Decode step %d accepted no draft tokens", step_idx + 1)

            # Stop if EOS produced
            if generated.shape[1] and self._is_eos(generated[0, -1]):
                logger.info(
                    "Stopping generation at step %d/%d after EOS token",
                    step_idx + 1,
                    max_new_tokens,
                )
                break

        logger.info(
            "Finished speculative generation: generated_tokens=%d steps=%d cache_hit_rate=%.3f",
            generated.shape[1] - input_ids.shape[1],
            len(self._step_results),
            self.cache.hit_rate(),
        )
        return generated

    def stats(self) -> dict:
        if not self._step_results:
            return {}
        total_draft = sum(r.draft_length for r in self._step_results)
        total_accepted = sum(r.accepted_count for r in self._step_results)
        return {
            "steps": len(self._step_results),
            "total_draft": total_draft,
            "total_accepted": total_accepted,
            "mean_draft_len": total_draft / len(self._step_results),
            "acceptance_rate": total_accepted / max(1, total_draft),
            "cache_hit_rate": self.cache.hit_rate(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _choose_draft_length(self, context: torch.Tensor, fn) -> int:
        if fn is not None:
            k = fn(context)
            logger.debug("Adaptive draft controller selected k=%d", k)
            return k
        logger.debug("Using fixed draft length k=%d", self.draft_length)
        return self.draft_length

    def _decode_step(
        self,
        context: torch.Tensor,
        k: int,
        distiller=None,
    ) -> StepResult:
        ctx_list = context[0].tolist()

        # 1. Try cache lookup
        logger.debug("Cache lookup for context length %d", len(ctx_list))
        cache_entry = self.cache.lookup(ctx_list)
        if cache_entry is not None and len(cache_entry.token_ids) >= 1:
            draft_tokens = cache_entry.token_ids[:k]
            draft_logits = None  # cache has no per-step logits
            cache_hit = True
            logger.info("Cache hit: reusing %d token(s)", len(draft_tokens))
        else:
            # 2. Drafter generates k tokens autoregressively
            logger.info("Cache miss: running drafter for k=%d", k)
            draft_tokens, draft_logits = self.drafter.draft(
                context, k, distill=(distiller is not None)
            )
            cache_hit = False

        # 3. Target verifies the full draft in one forward pass
        logger.info("Running target verification for %d draft token(s)", len(draft_tokens))
        target_logits = self.target.verify(context, draft_tokens)
        # target_logits: (k+1, target_vocab_size) — the +1 is the bonus token

        # 4. Translate drafter logits to target vocab space (if we have them)
        # When distillation is active, draft_logits has requires_grad=True.
        # The translator is NOT trained, so wrap in no_grad to prevent
        # intermediate activations (softmax, index_add_) from being retained
        # on the computation graph and contributing to OOM under cumulative
        # memory pressure from drafter forward-pass activations.
        if draft_logits is not None:
            logger.debug("Translating drafter logits to target vocab")
            with torch.no_grad():
                translated_probs = self.translator.translate(draft_logits)  # (k, target_vocab)
                # Defensive: guarantee this matches target_logits' actual width
                # regardless of how the translator was constructed (e.g. if it
                # was built from len(tokenizer.get_vocab()) rather than
                # model.config.vocab_size — see translation/vocabulary.py).
                translated_probs = _align_last_dim(translated_probs, target_logits.shape[-1])
        else:
            logger.debug("Skipping translation because draft tokens came from cache")
            translated_probs = None

        # 5. Acceptance / rejection
        logger.debug("Running acceptance test for %d draft token(s)", len(draft_tokens))
        accepted, rejected_at = self._accept_reject(draft_tokens, target_logits, translated_probs)
        logger.info("Acceptance result: accepted=%d rejected_at=%d", len(accepted), rejected_at)
        accepted_count = len(accepted)

        # 6. Bonus token from residual distribution at rejection point
        bonus = self._residual_sample(target_logits, translated_probs, rejected_at)
        if bonus is not None:
            logger.debug("Sampled residual bonus token: %d", bonus)
            accepted = accepted + [bonus]

        # 7. Update cache
        accepted_ctx = ctx_list + accepted
        logger.debug("Updating cache acceptance for context length %d", len(accepted_ctx))
        self.cache.update_acceptance(ctx_list, accepted=len(accepted) > 0)
        if accepted:
            logger.debug("Inserting %d accepted token(s) into cache", len(accepted))
            self.cache.insert(
                ctx_list,
                accepted,
                logits=target_logits[: len(accepted)].detach().cpu()
                if target_logits is not None
                else None,
            )

        # 8. Optional online distillation
        if distiller is not None and draft_logits is not None:
            logger.debug("Running online distillation step")
            distiller.step(
                draft_logits=draft_logits,
                target_logits=target_logits[: len(draft_tokens)],
                draft_tokens=draft_tokens,
                accepted_mask=[
                    i < rejected_at if rejected_at >= 0 else True for i in range(len(draft_tokens))
                ],
                prompt_ids=context[0].tolist(),
            )
            logger.debug("Online distillation step complete")

        return StepResult(
            draft_length=k,
            accepted_count=accepted_count,
            rejected_at=rejected_at,
            cache_hit=cache_hit,
            draft_tokens=draft_tokens,
            accepted_tokens=accepted,
        )

    def _accept_reject(
        self,
        draft_tokens: list[int],
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
    ) -> tuple[list[int], int]:
        """
        Standard speculative decoding acceptance test.

        For each position i:
          p(x) = target_prob(draft_tokens[i] | context + draft[:i])
          q(x) = translated drafter prob (or uniform fallback)
          Accept with prob min(1, p/q).

        Returns (accepted_list, first_rejection_index).
        """
        accepted: list[int] = []
        for i, tok in enumerate(draft_tokens):
            t_logit = target_logits[i]  # (target_vocab,)
            p = F.softmax(t_logit.float() / max(self.temperature, 1e-6), dim=-1)
            p_tok = p[tok].item()

            if translated_probs is not None:
                q_tok = translated_probs[i, tok].item()
                q_tok = max(q_tok, 1e-8)
                accept_prob = min(1.0, p_tok / q_tok)
            else:
                accept_prob = p_tok  # no drafter info → use target directly

            logger.debug(
                "Acceptance token %d/%d: token=%d accept_prob=%.4f",
                i + 1,
                len(draft_tokens),
                tok,
                accept_prob,
            )
            if torch.rand(1).item() < accept_prob:
                accepted.append(tok)
            else:
                logger.info(
                    "Rejected draft token %d: token=%d accept_prob=%.4f", i + 1, tok, accept_prob
                )
                return accepted, i

        logger.info("All %d draft token(s) accepted", len(draft_tokens))
        return accepted, -1  # all accepted

    def _residual_sample(
        self,
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
        rejected_at: int,
    ) -> int | None:
        """Sample one bonus token from the residual distribution at rejection point."""
        # The bonus token comes from target_logits[rejected_at] (or last position if all accepted)
        pos = rejected_at if rejected_at >= 0 else len(target_logits) - 1
        if pos >= target_logits.shape[0]:
            logger.debug("Skipping residual sample: position %d out of range", pos)
            return None

        t_logit = target_logits[pos]
        p = F.softmax(t_logit.float() / max(self.temperature, 1e-6), dim=-1)

        if translated_probs is not None and rejected_at >= 0:
            q = translated_probs[rejected_at]
            residual = F.relu(p - q)
            if residual.sum() > 1e-8:
                residual = residual / residual.sum()
                token = torch.multinomial(residual, 1).item()
                logger.debug(
                    "Residual sample from positive mass at position %d: token=%d", pos, token
                )
                return token
            logger.debug("Residual mass empty at position %d; falling back to target", pos)

        token = torch.multinomial(p, 1).item()
        logger.debug(
            "Residual sample from target distribution at position %d: token=%d", pos, token
        )
        return token

    def _is_eos(self, token_id: torch.Tensor) -> bool:
        eos_ids = getattr(self.target.model.config, "eos_token_id", None)
        if eos_ids is None:
            logger.debug("No EOS token id configured")
            return False
        if isinstance(eos_ids, int):
            is_eos = token_id.item() == eos_ids
            logger.debug("EOS check for token %d against %d: %s", token_id.item(), eos_ids, is_eos)
            return is_eos
        is_eos = token_id.item() in eos_ids
        logger.debug("EOS check for token %d against %s: %s", token_id.item(), eos_ids, is_eos)
        return is_eos
