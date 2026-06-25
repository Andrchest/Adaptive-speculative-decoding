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
# Device-agnostic RNG helpers
# ---------------------------------------------------------------------------


def _multinomial_with_rng(
    probs: torch.Tensor, num_samples: int, rng: torch.Generator | None
) -> int:
    """
    Run ``torch.multinomial(probs, num_samples)`` with an optional RNG,
    handling the case where the generator lives on a different device than
    the tensor (``torch.multinomial`` requires device match).

    When devices differ, move the tensor to the generator's device for
    sampling, then return the result. This preserves exact RNG state
    consumption so output tokens are bit-identical to the reference.
    """
    if rng is not None and str(rng.device) != str(probs.device):
        return torch.multinomial(probs.to(rng.device), num_samples, generator=rng).item()
    return torch.multinomial(probs, num_samples, generator=rng).item()


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
        rng: torch.Generator | None = None,  # optional RNG for reproducibility
    ) -> torch.Tensor:
        """
        Full generation loop. Returns (1, seq_len + new_tokens).

        Parameters
        ----------
        rng : optional torch.Generator for deterministic sampling.
              When provided, acceptance tests and residual sampling use
              this generator for reproducible results.
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
            return self._generate_loop(
                input_ids, max_new_tokens, adaptive_length_fn, distiller, rng
            )

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        adaptive_length_fn,
        distiller,
        rng=None,
    ) -> torch.Tensor:
        """
        Generate up to ``max_new_tokens`` NEW tokens.

        Note: each decode step may emit 1..k+1 tokens (accepted draft
        prefix + bonus), so we cannot simply iterate ``max_new_tokens``
        times. The loop runs until the token budget is exhausted or EOS
        is produced, and the per-step emission is truncated to the
        remaining budget before being appended.
        """
        prompt_len = input_ids.shape[1]
        # Pre-allocate output buffer to avoid repeated torch.cat / reallocation.
        # Only the first `pos` columns are valid; the rest is zero padding.
        output = torch.zeros(
            (input_ids.shape[0], prompt_len + max_new_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output[:, :prompt_len] = input_ids
        pos = prompt_len  # write cursor
        max_consec_zero = 5  # stop after N consecutive steps with 0 accepted
        consec_zero = 0

        for step_idx in range(max_new_tokens):
            # Stop if we've already produced the requested number of tokens.
            new_tokens = pos - prompt_len
            if new_tokens >= max_new_tokens:
                break

            # Pass only the filled portion to the model (avoids zero padding
            # in the attention mask / positional embeddings).
            generated = output[:, :pos]

            k = self._choose_draft_length(generated, adaptive_length_fn)
            logger.debug(
                "Decode step %d selected draft length k=%d (new_tokens=%d/%d)",
                step_idx + 1,
                k,
                new_tokens,
                max_new_tokens,
            )

            result = self._decode_step(generated, k, distiller=distiller, rng=rng)
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
                # Indexed assignment into pre-allocated buffer (no torch.cat).
                for i, tok in enumerate(emitted):
                    output[0, pos + i] = tok
                pos += len(emitted)
                consec_zero = 0
            else:
                consec_zero += 1
                logger.debug(
                    "Decode step %d emitted no tokens (consecutive zeros=%d)",
                    step_idx + 1,
                    consec_zero,
                )
                if consec_zero >= max_consec_zero:
                    logger.warning(
                        "Stopping after %d consecutive steps with zero accepted tokens",
                        max_consec_zero,
                    )
                    break

            # Stop if EOS produced
            if pos > prompt_len and self._is_eos(output[0, pos - 1]):
                logger.info(
                    "Stopping generation at step %d after EOS token (new_tokens=%d)",
                    step_idx + 1,
                    pos - prompt_len,
                )
                break

        logger.info(
            "Finished speculative generation: generated_tokens=%d steps=%d cache_hit_rate=%.3f",
            pos - prompt_len,
            len(self._step_results),
            self.cache.hit_rate(),
        )
        return output[:, :pos]

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
    # Cleanup
    # ------------------------------------------------------------------

    def clear_step_results(self) -> None:
        """Clear accumulated step results after collecting metrics."""
        self._step_results.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _choose_draft_length(self, context: torch.Tensor, fn) -> int:
        if fn is not None:
            k = fn(context)
            if not (1 <= k <= 10):
                logger.warning("Adaptive draft controller returned invalid k=%d; using default", k)
                return self.draft_length
            logger.debug("Adaptive draft controller selected k=%d", k)
            return k
        logger.debug("Using fixed draft length k=%d", self.draft_length)
        return self.draft_length

    def _decode_step(
        self,
        context: torch.Tensor,
        k: int,
        distiller=None,
        rng=None,
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
        logger.debug("Running drafter for k=%d temperature=%s", k, self.temperature)
        draft_tokens_drafter, draft_logits = self.drafter.draft(
            context,
            k,
            distill=(distiller is not None),
            temperature=self.temperature,
        )

        # Normalize drafter logits to 2D: (k, Vd).
        # Some model families emit 3D logits (k, 1, Vd) due to device_map
        # behavior; squeeze the intermediate dimension so all downstream
        # modules (translator, distiller, etc.) receive clean 2D tensors.
        if draft_logits.dim() == 3 and draft_logits.shape[1] == 1:
            draft_logits = draft_logits.squeeze(1)

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

        # --- Defensive: truncate draft_tokens to k (can exceed k due to
        # 3D drafter logits or mapping anomalies). This prevents IndexError
        # in _accept_reject where translated_probs has exactly k rows.
        if len(draft_tokens_target) != k:
            logger.warning(
                "Draft tokens length mismatch: got %d, expected %d — truncating",
                len(draft_tokens_target), k,
            )
            draft_tokens_target = draft_tokens_target[:k]

        # 5. Target verifies the (target-vocab) draft tokens in one pass.
        logger.debug(
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
            draft_tokens_target, target_logits, translated_probs, rng=rng
        )
        logger.debug(
            "Acceptance result: accepted=%d rejected_at=%d",
            len(accepted),
            rejected_at,
        )
        accepted_count = len(accepted)

        # 7. Bonus token from residual distribution at rejection point.
        bonus = self._residual_sample(target_logits, translated_probs, rejected_at, rng=rng)
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
            # P3.4: Skip storing logits — cache hit fast-path was removed,
            # so cached logits are never read back. Saves a .detach().cpu() sync.
            self.cache.insert(ctx_list, accepted, logits=None)

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
                prompt_ids=ctx_list,  # reuse cached .tolist() — P3.1
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

        Optimized (P3.2): batched GPU indexing with a single .tolist()
        instead of per-token .item() calls.
        """
        if not draft_tokens_drafter:
            return []

        k = len(draft_tokens_drafter)
        mapping = self.translator.rule1._mapping  # (drafter_vocab,) → target_idx or -1

        # Ensure mapping is on the same device as translated_probs (or cuda).
        device = "cuda"
        if mapping.device.type != "cuda":
            mapping = mapping.cuda()

        # Batched lookup: map all drafter tokens at once.
        draft_tensor = torch.tensor(
            draft_tokens_drafter, dtype=torch.long, device=device
        )
        # Clamp to valid range so indexing doesn't fail on out-of-range tokens.
        safe_indices = draft_tensor.clamp(0, mapping.shape[0] - 1)
        mapped = mapping[safe_indices]  # (k,) — target indices or -1

        # Determine which positions need fallback (mapped == -1).
        need_fallback = mapped < 0

        if need_fallback.any() and translated_probs is not None:
            # Fill fallback positions with argmax of translated_probs.
            fallback_mask = need_fallback & (safe_indices < translated_probs.shape[0])
            if fallback_mask.any():
                argmax_vals = translated_probs.argmax(dim=-1)  # (k,)
                mapped[fallback_mask] = argmax_vals[fallback_mask]

        # Remaining -1 (no probs available): pass through drafter id.
        still_negative = mapped < 0
        if still_negative.any():
            mapped[still_negative] = draft_tensor[still_negative]

        return mapped.tolist()  # one GPU→CPU sync

    def _accept_reject(
        self,
        draft_tokens: list[int],
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
        rng: torch.Generator | None = None,
    ) -> tuple[list[int], int]:
        """
        Vectorized speculative decoding acceptance test.

        For each position i:
          p(x) = target_prob(draft_tokens[i] | context + draft[:i])
          q(x) = translated drafter prob (or uniform 1/V fallback)
          Accept with prob min(1, p/q).

        Optimizations over the scalar loop:
          - One batched softmax over the entire (k, V) target logit tensor
            instead of k separate softmax calls.
          - Gather p[tok_i] and q[tok_i] via advanced indexing.
          - Acceptance probabilities computed as a single vector operation.
          - Random draws consumed one-by-one (``torch.rand(1)``) to preserve
            the exact RNG consumption order of the reference implementation,
            so output tokens are bit-identical for a given seed.

        This eliminates ~k softmax kernel launches and ~2k GPU→CPU syncs
        from the hot path (k = draft_length).

        Parameters
        ----------
        rng : optional torch.Generator for deterministic acceptance.

        The uniform 1/V fallback preserves the target distribution when
        the drafter's proposal is unknown (defensive path; in normal
        operation ``translated_probs`` is always provided).

        Returns (accepted_list, first_rejection_index).
        """
        k = len(draft_tokens)
        if k == 0:
            return [], -1

        V = target_logits.shape[-1]
        t_eff = max(self.temperature, 1e-6)

        # --- Batched softmax over all k positions (single kernel launch) ---
        t_logits = target_logits[:k].float() / t_eff  # (k, V)
        if t_logits.isnan().any() or t_logits.isinf().any():
            logger.warning(
                "Target logits contain NaN/Inf — replacing with safe values."
            )
            t_logits = torch.nan_to_num(t_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        target_probs = F.softmax(t_logits, dim=-1)  # (k, V)

        # --- Gather p[tok_i] and q[tok_i] for all positions ---
        device = target_logits.device
        tok_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device)
        idx = torch.arange(k, device=device)
        p_tok_vec = target_probs[idx, tok_tensor]  # (k,)

        if translated_probs is not None:
            q_tok_vec = translated_probs[idx, tok_tensor].clamp(min=1e-8)  # (k,)
            accept_probs = (p_tok_vec / q_tok_vec).clamp(max=1.0)  # (k,)
        else:
            # Uniform-proposal fallback: q = 1/V.
            accept_probs = (p_tok_vec * V).clamp(max=1.0)  # (k,)

        # --- Sequential random draws to preserve RNG consumption order ---
        # Each ``torch.rand(1, generator=rng)`` call advances the RNG state
        # by exactly one draw, matching the reference implementation.
        accepted: list[int] = []
        for i in range(k):
            ap = accept_probs[i].item()
            logger.debug(
                "Acceptance token %d/%d: token=%d accept_prob=%.4f",
                i + 1, k, draft_tokens[i], ap,
            )
            if rng is not None:
                draw = torch.rand(1, generator=rng).item()
            else:
                draw = torch.rand(1).item()
            if draw < ap:
                accepted.append(draft_tokens[i])
            else:
                logger.debug(
                    "Rejected draft token %d: token=%d accept_prob=%.4f",
                    i + 1, draft_tokens[i], ap,
                )
                return accepted, i

        logger.debug("All %d draft token(s) accepted", k)
        return accepted, -1

    def _residual_sample(
        self,
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
        rejected_at: int,
        rng: torch.Generator | None = None,
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

        Parameters
        ----------
        rng : optional torch.Generator for deterministic sampling.
        """
        pos = rejected_at if rejected_at >= 0 else len(target_logits) - 1
        if pos >= target_logits.shape[0]:
            logger.debug("Skipping residual sample: position %d out of range", pos)
            return None

        t_logit = target_logits[pos]
        if t_logit.isnan().any() or t_logit.isinf().any():
            logger.warning(
                "Target logits at position %d contain NaN/Inf — zeroing. "
                "This usually indicates the input sequence was corrupted "
                "by prior draft output errors.",
                pos,
            )
            t_logit = torch.nan_to_num(t_logit, nan=0.0, posinf=1e6, neginf=-1e6)
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
                token = _multinomial_with_rng(residual, 1, rng)
                logger.debug(
                    "Residual sample at position %d: token=%d", pos, token
                )
                return token
            logger.debug(
                "Residual mass empty at position %d; falling back to target", pos
            )

        token = _multinomial_with_rng(p, 1, rng)
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
