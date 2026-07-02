# core/decoder/speculative.py
"""
core/decoder/speculative.py

Core speculative decoding engine.

CRITICAL FIXES (performance):
  1. Drafter KV cache maintained across decode steps (was discarded).
     After each step: truncate to accepted prefix, forward bonus token,
     save logits for next step's first draft token.
  2. Removed useless cache.lookup() (result was discarded but still
     computed hash + updated stats).
  3. Accept/reject test moved entirely to GPU (was doing .cpu().tolist()
     + Python loop, causing 2 GPU syncs per step).
  4. Token translation uses single GPU indexing (was already batched
     but had redundant device checks).
  5. Target KV cache reset at start of each generate() call.
  6. Adaptive controller shares drafter forward (no separate hidden
     state extraction).
  7. Logging reduced to WARNING level in hot path (DEBUG was evaluating
     format strings even when filtered).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from core.cache.ngram import NgramCache
from core.models.target_model import _truncate_pkv
from core.translation.vocabulary import CrossVocabTranslator, _align_last_dim


def _multinomial_with_rng(
    probs: torch.Tensor, num_samples: int, rng: torch.Generator | None
) -> int:
    if rng is not None and str(rng.device) != str(probs.device):
        return torch.multinomial(probs.to(rng.device), num_samples, generator=rng).item()
    return torch.multinomial(probs, num_samples, generator=rng).item()


@dataclass
class StepResult:
    draft_length: int
    accepted_count: int
    rejected_at: int
    wall_time_ms: float = 0.0
    cache_hit: bool = False
    draft_tokens: list[int] = field(default_factory=list)
    accepted_tokens: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.draft_length == 0:
            return 0.0
        return self.accepted_count / self.draft_length


class SpeculativeDecoder:
    """
    Model-agnostic speculative decoding loop.

    Parameters
    ----------
    drafter        : DraftModel — fast small model
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

        self._step_results: list[StepResult] = []

        # Drafter KV cache state (maintained across decode steps)
        self._drafter_kv = None
        self._drafter_kv_len = 0
        self._cached_drafter_logits = None  # logits from bonus-token forward

        # Target KV cache state
        self._target_kv = None

        self._same_vocab = (
            translator.rule1.drafter_size == translator.rule1.target_size and
            translator.rule1._valid_mask.all()
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        adaptive_length_fn=None,
        distiller=None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        import contextlib

        logger.info(
            "Starting speculative generation: prompt_len=%d max_new_tokens=%d draft_length=%d",
            input_ids.shape[1],
            max_new_tokens,
            self.draft_length,
        )

        # CRITICAL: Reset KV cache state for new generation
        self._drafter_kv = None
        self._drafter_kv_len = 0
        self._cached_drafter_logits = None
        self._target_kv = None
        if hasattr(self.target, "reset_kv_state"):
            self.target.reset_kv_state()

        self.cache.step()

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
        prompt_len = input_ids.shape[1]
        output = torch.zeros(
            (input_ids.shape[0], prompt_len + max_new_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output[:, :prompt_len] = input_ids
        pos = prompt_len
        max_consec_zero = 5
        consec_zero = 0
        ctx_list: list[int] = input_ids[0].tolist()

        # CRITICAL: Maintain a separate drafter context in drafter vocab.
        # The output buffer contains accepted tokens in target vocab,
        # but the drafter needs drafter vocab tokens for embedding.
        drafter_context_ids: list[int] = input_ids[0].tolist()  # drafter vocab

        for step_idx in range(max_new_tokens):
            new_tokens = pos - prompt_len
            if new_tokens >= max_new_tokens:
                break

            generated = output[:, :pos]

            # Build drafter context tensor from drafter vocab token IDs
            drafter_ctx = torch.tensor(
                [drafter_context_ids], dtype=input_ids.dtype, device=input_ids.device
            )

            k = self._choose_draft_length(generated, adaptive_length_fn)

            result = self._decode_step(
                generated, k, ctx_list,
                drafter_ctx=drafter_ctx,  # always in drafter vocab
                distiller=distiller,
                rng=rng,
            )
            self._step_results.append(result)
            self._notify_adaptive_result(adaptive_length_fn, result)
            self.cache.step()

            budget = max_new_tokens - new_tokens
            emitted = result.accepted_tokens[:budget]
            if emitted:
                for i, tok in enumerate(emitted):
                    output[0, pos + i] = tok
                pos += len(emitted)
                ctx_list.extend(emitted)

                # CRITICAL: Translate accepted tokens (target vocab) back to
                # drafter vocab for the drafter's context.
                if not self._same_vocab:
                    drafter_emitted = self.translator.translate_target_to_drafter(emitted)
                else:
                    drafter_emitted = emitted
                drafter_context_ids.extend(drafter_emitted)

                # Also update ctx_list with drafter vocab for cache consistency
                ctx_list = drafter_context_ids[:]

                consec_zero = 0
            else:
                consec_zero += 1
                if consec_zero >= max_consec_zero:
                    logger.warning("Stopping after %d consecutive zero-emission steps", max_consec_zero)
                    break

            if pos > prompt_len and self._is_eos(output[0, pos - 1]):
                logger.info("EOS at step %d (new_tokens=%d)", step_idx + 1, pos - prompt_len)
                break

        logger.info(
            "Finished: generated=%d steps=%d cache_hit_rate=%.3f",
            pos - prompt_len, len(self._step_results), self.cache.hit_rate(),
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

    def clear_step_results(self) -> None:
        self._step_results.clear()

    def _notify_adaptive_result(self, adaptive_length_fn, result: StepResult) -> None:
        """Report verification feedback to adaptive controllers when supported."""
        if adaptive_length_fn is None:
            return

        observer = getattr(adaptive_length_fn, "observe_step", None)
        if callable(observer):
            observer(result)
            return

        recorder = getattr(adaptive_length_fn, "record_result", None)
        if callable(recorder):
            recorder(result.accepted_count)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _choose_draft_length(self, context: torch.Tensor, fn) -> int:
        if fn is not None:
            k = fn(context)
            if not (1 <= k <= 10):
                return self.draft_length
            return k
        return self.draft_length

    def _decode_step(
        self,
        context: torch.Tensor,
        k: int,
        ctx_list: list[int] | None = None,
        drafter_ctx: torch.Tensor | None = None,
        distiller=None,
        rng=None,
    ) -> StepResult:
        """
        One step of speculative decoding.

        CRITICAL FIXES APPLIED HERE:
        1. Drafter KV cache is passed and updated across steps.
        2. Bonus token forward is reused for adaptive hidden states.
        3. Useless cache.lookup() is removed.
        4. Target KV cache is properly truncated.
        """
        if drafter_ctx is None:
            drafter_ctx = context
        if ctx_list is None:
            ctx_list = context[0].tolist()

        # 1. Drafter generates k tokens autoregressively.
        #    We pass the maintained KV cache and any cached logits from
        #    the previous step's bonus-token forward. This reduces the
        #    drafter forward from O(seq_len) to O(1-2) tokens per step.
        draft_tokens_drafter, draft_logits, new_drafter_kv = self.drafter.draft(
            drafter_ctx,
            k,
            distill=(distiller is not None),
            temperature=self.temperature,
            past_key_values=self._drafter_kv,
            past_len=self._drafter_kv_len,
            cached_logits=self._cached_drafter_logits,
        )

        # Update drafter KV state for this step
        self._drafter_kv = new_drafter_kv
        # FIX: KV cache has context.shape[1] + k entries after drafting,
        # but on the next step the context grows by len(accepted) tokens.
        # The drafter only needs to process the NEW tokens beyond past_len,
        # so we store the actual KV cache length here.
        self._drafter_kv_len = context.shape[1] + k
        self._cached_drafter_logits = None  # Consumed

        # Normalize drafter logits to 2D: (k, Vd).
        if draft_logits is not None and draft_logits.dim() == 3 and draft_logits.shape[1] == 1:
            draft_logits = draft_logits.squeeze(1)

        # 2. Translate drafter logits to target vocab space to obtain q.
        if draft_logits is not None:
            with torch.no_grad():
                t_eff = max(self.temperature, 1e-6)
                if self._same_vocab:
                    translated_probs = F.softmax(draft_logits.float() / t_eff, dim=-1)
                else:
                    translated_probs = self.translator.translate(draft_logits / t_eff)
                translated_probs = _align_last_dim(
                    translated_probs, self.translator.rule1.target_size
                )
        else:
            translated_probs = None

        # 3. Translate drafter-vocab token ids → target-vocab token ids.
        draft_tokens_target = self._translate_draft_tokens(
            draft_tokens_drafter, translated_probs
        )
        if len(draft_tokens_target) != k:
            draft_tokens_target = draft_tokens_target[:k]

        # 4. Target verifies the (target-vocab) draft tokens in one pass.
        #    Uses KV cache from previous step when available.
        target_logits, self._target_kv = self.target.verify(
            context, draft_tokens_target, past_key_values=self._target_kv
        )

        if translated_probs is not None:
            translated_probs = _align_last_dim(translated_probs, target_logits.shape[-1])

        # 5. Acceptance / rejection in target-vocab space (GPU vectorized).
        accepted, rejected_at = self._accept_reject_gpu(
            draft_tokens_target, target_logits, translated_probs, rng=rng
        )
        accepted_count = len(accepted)

        # 6. Bonus token from residual distribution at rejection point.
        bonus = self._residual_sample(target_logits, translated_probs, rejected_at, rng=rng)
        if bonus is not None:
            accepted = accepted + [bonus]

        # 7. Truncate drafter KV cache to keep only the verified prefix.
        #    Then, if there's a bonus token, forward it through the drafter
        #    to extend the KV cache AND extract hidden states for the
        #    adaptive controller (eliminates a redundant forward pass).
        #
        #    CRITICAL: drafter_ctx_len is the length of the drafter context
        #    (in drafter vocab), not the output buffer length.
        drafter_ctx_len = drafter_ctx.shape[1] if drafter_ctx is not None else context.shape[1]
        if self._drafter_kv is not None and not distiller:
            drafter_keep = drafter_ctx_len + accepted_count
            self._drafter_kv = _truncate_pkv(self._drafter_kv, drafter_keep)
            self._drafter_kv_len = drafter_keep

            if bonus is not None:
                # Translate bonus token to drafter vocab for the drafter.
                # The bonus token is sampled from target logits (target vocab),
                # but the drafter expects drafter vocab ids.
                drafter_vocab_size = self.drafter.model.config.vocab_size
                if not self._same_vocab:
                    bonus_drafter = self.translator.translate_target_to_drafter([bonus])[0]
                else:
                    bonus_drafter = bonus

                if bonus_drafter < drafter_vocab_size:
                    bonus_tensor = torch.tensor(
                        [[bonus_drafter]], dtype=context.dtype, device=context.device
                    )
                    with torch.no_grad():
                        from core.models.target_model import _to_cache
                        need_hidden = hasattr(self, '_adaptive_controller_ref') and self._adaptive_controller_ref is not None
                        try:
                            bonus_out = self.drafter.model(
                                bonus_tensor,
                                past_key_values=_to_cache(self._drafter_kv),
                                output_hidden_states=need_hidden,
                                use_cache=True,
                            )
                        except RuntimeError as e:
                            if "same number of dimensions" not in str(e):
                                raise
                            logger.warning(
                                "KV dim mismatch in bonus forward (%s) — full forward", e
                            )
                            bonus_out = self.drafter.model(
                                bonus_tensor,
                                output_hidden_states=True,
                                use_cache=True,
                            )
                    self._drafter_kv = bonus_out.past_key_values
                    self._drafter_kv_len += 1
                    self._cached_drafter_logits = bonus_out.logits[:, -1, :]
                else:
                    logger.debug(
                        "Skipping drafter bonus forward: token %d >= drafter vocab %d (cross-vocab)",
                        bonus_drafter, drafter_vocab_size,
                    )

                # Share hidden state with adaptive controller if attached
                if hasattr(self, '_adaptive_controller_ref'):
                    ctrl = self._adaptive_controller_ref
                    if ctrl is not None and hasattr(ctrl, 'update_hidden'):
                        ctrl.update_hidden(bonus_out.hidden_states[-1][0, -1, :])
        else:
            # Distillation mode or no KV: reset for next step
            self._drafter_kv = None
            self._drafter_kv_len = 0
            self._cached_drafter_logits = None

        # 8. Truncate target KV cache to keep only the verified prefix.
        kv_keep = context.shape[1] + accepted_count
        if self._target_kv is not None:
            try:
                self._target_kv = _truncate_pkv(self._target_kv, kv_keep)
            except (TypeError, IndexError):
                self._target_kv = None
            # self.target.reset_kv_state()  # fresh KV attempt per prompt (P0: sticky False fix)

        # 9. Update cache stats + acceptance EMA (lookup removed for speed).
        self.cache.update_acceptance(ctx_list, accepted=accepted_count > 0)
        if accepted:
            # Translate accepted tokens to drafter vocab for cache consistency
            if not self._same_vocab:
                accepted_drafter = self.translator.translate_target_to_drafter(accepted)
            else:
                accepted_drafter = accepted
            self.cache.insert(ctx_list, accepted_drafter, logits=None)

        # 10. Optional online distillation.
        if distiller is not None and draft_logits is not None:
            accepted_mask = [
                (i < rejected_at) if rejected_at >= 0 else True
                for i in range(len(draft_tokens_drafter))
            ]
            distiller.step(
                draft_logits=draft_logits,
                target_logits=target_logits[: len(draft_tokens_target)],
                draft_tokens=draft_tokens_drafter,
                accepted_mask=accepted_mask,
                prompt_ids=ctx_list,
            )

        return StepResult(
            draft_length=k,
            accepted_count=accepted_count,
            rejected_at=rejected_at,
            cache_hit=False,
            draft_tokens=draft_tokens_target,
            accepted_tokens=accepted,
        )

    def _translate_draft_tokens(
        self,
        draft_tokens_drafter: list[int],
        translated_probs: torch.Tensor | None,
    ) -> list[int]:
        """Map drafter-vocab token ids → target-vocab token ids (batched GPU)."""
        if not draft_tokens_drafter:
            return []

        k = len(draft_tokens_drafter)
        mapping = self.translator.rule1._mapping

        device = str(translated_probs.device) if translated_probs is not None else str(mapping.device)
        if mapping.device.type != device:
            mapping = mapping.to(device)

        draft_tensor = torch.tensor(draft_tokens_drafter, dtype=torch.long, device=device)
        safe_indices = draft_tensor.clamp(0, mapping.shape[0] - 1)
        mapped = mapping[safe_indices]

        need_fallback = mapped < 0
        if need_fallback.any() and translated_probs is not None:
            fallback_mask = need_fallback & (safe_indices < translated_probs.shape[0])
            if fallback_mask.any():
                argmax_vals = translated_probs.argmax(dim=-1)
                mapped[fallback_mask] = argmax_vals[fallback_mask]

        still_negative = mapped < 0
        if still_negative.any():
            # FIX: Don't use raw drafter token id as fallback — it may be
            # >= target model's vocab_size, causing embedding OOB errors.
            # Use target UNK token (or pad token) as safe fallback.
            unk_id = getattr(self.target.tokenizer, 'unk_token_id', None)
            pad_id = getattr(self.target.tokenizer, 'pad_token_id', None)
            fallback_id = unk_id if unk_id is not None else (pad_id if pad_id is not None else 0)
            mapped[still_negative] = fallback_id

        return mapped.tolist()

    def _accept_reject_gpu(
        self,
        draft_tokens: list[int],
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
        rng: torch.Generator | None = None,
    ) -> tuple[list[int], int]:
        """
        GPU-vectorized acceptance test.

        FIX: Eliminates 2 CPU syncs (.cpu().tolist() for accept_probs
        and draws) by doing the comparison on GPU and finding the first
        rejection via torch.cumprod.

        Returns (accepted_list, first_rejection_index).
        """
        k = len(draft_tokens)
        if k == 0:
            return [], -1

        V = target_logits.shape[-1]
        t_eff = max(self.temperature, 1e-6)

        # Batched softmax over all k positions
        t_logits = target_logits[:k].float() / t_eff
        if t_logits.isnan().any() or t_logits.isinf().any():
            t_logits = torch.nan_to_num(t_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        target_probs = F.softmax(t_logits, dim=-1)  # (k, V)

        device = target_logits.device
        tok_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device)
        idx = torch.arange(k, device=device)
        p_tok_vec = target_probs[idx, tok_tensor]  # (k,)

        if translated_probs is not None:
            q_tok_vec = translated_probs[idx, tok_tensor].clamp(min=1e-8)
            accept_probs = (p_tok_vec / q_tok_vec).clamp(max=1.0)
        else:
            accept_probs = (p_tok_vec * V).clamp(max=1.0)

        # Generate random draws on the same device as the RNG
        if rng is not None:
            if str(rng.device) == str(device):
                draws = torch.rand(k, generator=rng, device=device)
            else:
                draws = torch.rand(k, generator=rng).to(device)
        else:
            draws = torch.rand(k, device=device)

        # Vectorized acceptance: accept where draw < accept_prob
        accepted_mask = draws < accept_probs  # (k,) bool

        # Find first rejection using cumprod: if all positions 0..i are
        # accepted, cumprod[i] = True. First False = first rejection.
        cum_accepted = torch.cumprod(accepted_mask, dim=0)
        # accepted = cum_accepted is True
        accepted_count_gpu = cum_accepted.sum().item()  # single sync

        if accepted_count_gpu == k:
            # All accepted
            accepted = draft_tokens[:k]
            return accepted, -1

        # First rejection index
        rejected_at = accepted_count_gpu  # first False position
        accepted = draft_tokens[:accepted_count_gpu]
        return accepted, rejected_at

    def _residual_sample(
        self,
        target_logits: torch.Tensor,
        translated_probs: torch.Tensor | None,
        rejected_at: int,
        rng: torch.Generator | None = None,
    ) -> int | None:
        """Sample one bonus token from residual distribution."""
        pos = rejected_at if rejected_at >= 0 else len(target_logits) - 1
        if pos >= target_logits.shape[0]:
            return None

        t_logit = target_logits[pos]
        if t_logit.isnan().any() or t_logit.isinf().any():
            t_logit = torch.nan_to_num(t_logit, nan=0.0, posinf=1e6, neginf=-1e6)
        p = F.softmax(t_logit.float() / max(self.temperature, 1e-6), dim=-1)

        if rejected_at >= 0:
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
                return token

        token = _multinomial_with_rng(p, 1, rng)
        return token

    def _is_eos(self, token_id: torch.Tensor) -> bool:
        eos_ids = getattr(self.target.model.config, "eos_token_id", None)
        if eos_ids is None:
            return False
        tid = token_id.item()
        if isinstance(eos_ids, int):
            return tid == eos_ids
        return tid in eos_ids
