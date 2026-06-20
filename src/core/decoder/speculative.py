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
        """
        Generate up to ``max_new_tokens`` NEW tokens.

        Note: each decode step may emit 1..k+1 tokens (accepted draft
        prefix + bonus), so we cannot simply iterate ``max_new_tokens``
        times. The loop runs until the token budget is exhausted or EOS
        is produced, and the per-step emission is truncated to the
        remaining budget before being appended.
        """
        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]

        for step_idx in range(max_new_tokens):
            # Stop if we've already produced the requested number of tokens.
            new_tokens = generated.shape[1] - prompt_len
            if new_tokens >= max_new_tokens:
                break

            k = self._choose_draft_length(generated, adaptive_length_fn)
            logger.debug(
                "Decode step %d selected draft length k=%d (new_tokens=%d/%d)",
                step_idx + 1,
                k,
                new_tokens,
                max_new_tokens,
            )

            result = self._decode_step(generated, k, distiller=distiller)
            self._step_results.append(result)
            self.cache.step()

            # Truncate the step's emission to the remaining token budget.
            # Without this, the loop could emit up to (k+1)*max_new_tokens
            # tokens when the user asked for max_new_tokens.
            budget = max_new_tokens - new_tokens
            emitted = result.accepted_tokens[:budget]
            if emitted:
                logger.debug(
                    "Decode step %d appending %d token(s) (budget=%d): %s",
                    step_idx + 1,
                    len(emitted),
                    budget,
                    emitted,
                )
                new_ids = torch.tensor(
                    emitted, dtype=torch.long, device=generated.device
                ).unsqueeze(0)
                generated = torch.cat([generated, new_ids], dim=1)
            else:
                logger.debug("Decode step %d emitted no tokens", step_idx + 1)

            # Stop if EOS produced
            if generated.shape[1] and self._is_eos(generated[0, -1]):
                logger.info(
                    "Stopping generation at step %d after EOS token (new_tokens=%d)",
                    step_idx + 1,
                    generated.shape[1] - prompt_len,
                )
                break

        logger.info(
            "Finished speculative generation: generated_tokens=%d steps=%d cache_hit_rate=%.3f",
            generated.shape[1] - prompt_len,
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
        """
        One step of speculative decoding.

        Vocabulary handling
        -------------------
        The drafter emits token ids in its OWN vocabulary. The target
        model and the translated drafter probabilities operate in TARGET
        vocabulary space. We therefore:

          1. Run the drafter → get drafter-vocab tokens + drafter-vocab logits.
          2. Translate drafter logits to target vocab → ``translated_probs``
             (used as ``q`` in the acceptance test).
          3. Translate the drafter-vocab token ids to target-vocab ids
             (``draft_tokens_target``) via Rule1's direct mapping, falling
             back to the argmax of ``translated_probs`` for unmapped tokens.
          4. Run target verification with the target-vocab tokens.
          5. Run acceptance test in target-vocab space (so ``p[tok]`` and
             ``q[tok]`` are both indexed correctly).
          6. Online distillation uses the ORIGINAL drafter-vocab tokens
             (since ``draft_logits`` is in drafter-vocab space).

        For same-tokenizer drafter/target pairs (the common case),
        Rule1 maps every token to itself, so steps 1-3 are no-ops.
        """
        ctx_list = context[0].tolist()

        # 1. Cache lookup (stats only — see C2 fix note below).
        # NOTE: Previously, on a cache hit we reused the cached token_ids
        # as the draft. This is unsound because the cache does not store
        # the drafter's proposal distribution ``q`` for those tokens, so
        # the acceptance rule cannot preserve the target distribution.
        # The cache still tracks acceptance rates for the eviction policy
        # and for adaptive drafting; we simply no longer fast-path the
        # draft itself.
        # We still call lookup() so the cache can update its hit_count
        # and recency stats (which feed the eviction policy), but we
        # ignore the returned entry for drafting.
        logger.debug("Cache lookup for context length %d", len(ctx_list))
        _ = self.cache.lookup(ctx_list)
        cache_hit = False  # stats-only; always re-draft

        # 2. Drafter generates k tokens autoregressively.
        #    Pass the decoder's temperature so the drafter samples from
        #    the SAME distribution we use as ``q`` in the acceptance test
        #    (C1 fix: a greedy drafter + softmax-``q`` acceptance does
        #    NOT preserve the target distribution).
        logger.info("Running drafter for k=%d temperature=%s", k, self.temperature)
        draft_tokens_drafter, draft_logits = self.drafter.draft(
            context,
            k,
            distill=(distiller is not None),
            temperature=self.temperature,
        )

        # 3. Translate drafter logits to target vocab space to obtain ``q``.
        #    Apply the decoder's temperature to the drafter logits BEFORE
        #    translation so ``q`` is at the same temperature as ``p``
        #    (H1 fix: previously ``p`` was at temperature T but ``q`` was
        #    at temperature 1, distorting the acceptance ratio p/q).
        #    The translator is NOT trained, so wrap in no_grad to prevent
        #    intermediate activations from being retained on the graph.
        if draft_logits is not None:
            logger.debug("Translating drafter logits to target vocab (T=%s)", self.temperature)
            with torch.no_grad():
                t_eff = max(self.temperature, 1e-6)
                translated_probs = self.translator.translate(draft_logits / t_eff)
                # Defensive alignment to the translator's declared target
                # vocab size (which may differ from a target model's actual
                # lm_head dim by a few padded rows for OPT/GPT-2 — we'll
                # re-align to the actual target_logits width after
                # target.verify() returns).
                translated_probs = _align_last_dim(
                    translated_probs, self.translator.rule1.target_size
                )
        else:
            logger.debug("No drafter logits; skipping translation")
            translated_probs = None

        # 4. Translate drafter-vocab token ids → target-vocab token ids.
        #    Required for cross-tokenizer setups (C3 fix). For same-tokenizer
        #    pairs this is a no-op (Rule1 maps every token to itself).
        draft_tokens_target = self._translate_draft_tokens(
            draft_tokens_drafter, translated_probs
        )

        # 5. Target verifies the (target-vocab) draft tokens in one pass.
        logger.info(
            "Running target verification for %d draft token(s)", len(draft_tokens_target)
        )
        target_logits = self.target.verify(context, draft_tokens_target)
        # target_logits: (k+1, target_vocab_size) — the +1 is the bonus token

        # Re-align translated_probs to the target_logits width (defensive:
        # translator width may differ from target model's actual lm_head
        # dim by a few padded rows for OPT/GPT-2).
        if translated_probs is not None:
            translated_probs = _align_last_dim(translated_probs, target_logits.shape[-1])

        # 6. Acceptance / rejection in target-vocab space.
        logger.debug(
            "Running acceptance test for %d draft token(s)", len(draft_tokens_target)
        )
        accepted, rejected_at = self._accept_reject(
            draft_tokens_target, target_logits, translated_probs
        )
        logger.info(
            "Acceptance result: accepted=%d rejected_at=%d",
            len(accepted),
            rejected_at,
        )
        accepted_count = len(accepted)

        # 7. Bonus token from residual distribution at rejection point.
        bonus = self._residual_sample(target_logits, translated_probs, rejected_at)
        if bonus is not None:
            logger.debug("Sampled residual bonus token: %d", bonus)
            accepted = accepted + [bonus]

        # 8. Update cache (stats + acceptance EMA).
        #    Use accepted_count (without bonus) so the EMA reflects how
        #    often the DRAFT was accepted, not whether a bonus was sampled
        #    (the bonus is always sampled, so including it would inflate
        #    the acceptance rate toward 1).
        logger.debug(
            "Updating cache acceptance for context length %d (accepted=%d)",
            len(ctx_list),
            accepted_count,
        )
        self.cache.update_acceptance(ctx_list, accepted=accepted_count > 0)
        if accepted:
            logger.debug("Inserting %d accepted token(s) into cache", len(accepted))
            self.cache.insert(
                ctx_list,
                accepted,
                logits=target_logits[: len(accepted)].detach().cpu()
                if target_logits is not None
                else None,
            )

        # 9. Optional online distillation.
        #    Pass the ORIGINAL drafter-vocab tokens (draft_tokens_drafter)
        #    because draft_logits is in drafter vocab space. The NLL loss
        #    indexes draft_logits with these tokens, so they MUST match
        #    the drafter vocab.
        if distiller is not None and draft_logits is not None:
            logger.debug("Running online distillation step")
            accepted_mask = [
                (i < rejected_at) if rejected_at >= 0 else True
                for i in range(len(draft_tokens_drafter))
            ]
            distiller.step(
                draft_logits=draft_logits,
                target_logits=target_logits[: len(draft_tokens_target)],
                draft_tokens=draft_tokens_drafter,
                accepted_mask=accepted_mask,
                prompt_ids=context[0].tolist(),
            )
            logger.debug("Online distillation step complete")

        return StepResult(
            draft_length=k,
            accepted_count=accepted_count,
            rejected_at=rejected_at,
            cache_hit=cache_hit,
            draft_tokens=draft_tokens_target,
            accepted_tokens=accepted,
        )

    def _translate_draft_tokens(
        self,
        draft_tokens_drafter: list[int],
        translated_probs: torch.Tensor | None,
    ) -> list[int]:
        """
        Map drafter-vocab token ids → target-vocab token ids.

        For each drafter token:
          - If Rule1 has a direct mapping, use it.
          - Otherwise, fall back to the argmax of ``translated_probs`` at
            that position (the most likely target token under the
            translated drafter distribution).

        For same-tokenizer drafter/target pairs, Rule1 maps every token
        to itself, so this is a no-op.
        """
        if not draft_tokens_drafter:
            return []
        mapping = self.translator.rule1._mapping  # (drafter_vocab,) → target_idx or -1
        result: list[int] = []
        for i, d_idx in enumerate(draft_tokens_drafter):
            t_idx = -1
            if 0 <= d_idx < mapping.shape[0]:
                t_idx = int(mapping[d_idx].item())
            if t_idx < 0:
                # No direct Rule1 mapping: use translated argmax at this position.
                if translated_probs is not None and i < translated_probs.shape[0]:
                    t_idx = int(translated_probs[i].argmax().item())
                else:
                    # Last-resort fallback: pass through the drafter id.
                    # target.verify may clip or raise on out-of-range ids;
                    # this branch should only trigger in degenerate test setups.
                    t_idx = d_idx
            result.append(t_idx)
        return result

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
          q(x) = translated drafter prob (or uniform 1/V fallback)
          Accept with prob min(1, p/q).

        The uniform 1/V fallback preserves the target distribution when
        the drafter's proposal is unknown (defensive path; in normal
        operation ``translated_probs`` is always provided).

        Returns (accepted_list, first_rejection_index).
        """
        accepted: list[int] = []
        V = target_logits.shape[-1]

        for i, tok in enumerate(draft_tokens):
            t_logit = target_logits[i]  # (target_vocab,)
            p = F.softmax(t_logit.float() / max(self.temperature, 1e-6), dim=-1)
            p_tok = p[tok].item()

            if translated_probs is not None:
                q_tok = translated_probs[i, tok].item()
                q_tok = max(q_tok, 1e-8)
                accept_prob = min(1.0, p_tok / q_tok)
            else:
                # Uniform-proposal fallback: q = 1/V.
                # accept_prob = min(1, p_tok / (1/V)) = min(1, p_tok * V).
                # Combined with the matching uniform residual in
                # _residual_sample, this preserves the target distribution.
                accept_prob = min(1.0, p_tok * V)

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
        """
        Sample one bonus token from the residual distribution at the
        rejection point (or the position after a fully-accepted draft).

        Residual = norm(max(0, p - q)).

        When ``translated_probs`` is None (defensive path), use the
        uniform proposal q = 1/V so the marginal of the produced token
        is still exactly ``p``. When all draft tokens are accepted
        (``rejected_at < 0``), there is no rejection residual to sample
        from, so we sample directly from ``p`` at the bonus position
        (this is the standard "bonus token" rule and preserves ``p``).
        """
        pos = rejected_at if rejected_at >= 0 else len(target_logits) - 1
        if pos >= target_logits.shape[0]:
            logger.debug("Skipping residual sample: position %d out of range", pos)
            return None

        t_logit = target_logits[pos]
        p = F.softmax(t_logit.float() / max(self.temperature, 1e-6), dim=-1)

        if rejected_at >= 0:
            # Rejection occurred at position `pos`; sample from residual.
            if translated_probs is not None:
                q = translated_probs[rejected_at]
            else:
                V = p.shape[-1]
                q = torch.full_like(p, 1.0 / max(V, 1))
            residual = F.relu(p - q)
            s = residual.sum()
            if s > 1e-8:
                residual = residual / s
                token = torch.multinomial(residual, 1).item()
                logger.debug(
                    "Residual sample at position %d: token=%d", pos, token
                )
                return token
            logger.debug(
                "Residual mass empty at position %d; falling back to target", pos
            )

        token = torch.multinomial(p, 1).item()
        logger.debug(
            "Bonus sample from target distribution at position %d: token=%d", pos, token
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
