# core/models/draft_model.py
"""
core/models/draft_model.py

Wraps a small causal LM as a drafter for speculative decoding.

CRITICAL FIX: KV cache is now reused across decode steps.
Previously, the drafter re-processed the ENTIRE context (up to 64
tokens via sliding window) every step. Now it only processes the
1-2 new tokens per step, reducing drafter cost by ~10-60×.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)


def _normalize_cache(cache: object) -> object:
    """Ensure all key/value tensors in a Cache are 4D in-place.
    Never squeezes the heads dimension (dim=1).
    """
    if not isinstance(cache, Cache):
        return cache
    # transformers 5.x Cache — has .layers attribute
    if hasattr(cache, 'layers'):
        for layer in cache.layers:
            if layer.is_initialized:
                if layer.keys is not None and layer.keys.ndim == 5:
                    # Extra dim between heads and seq_len: [B, H, 1, T, D]
                    if layer.keys.shape[2] == 1:
                        layer.keys = layer.keys.squeeze(2)
                        layer.values = layer.values.squeeze(2)
                    # Extra leading dim: [1, B, H, T, D]
                    elif layer.keys.shape[0] == 1:
                        layer.keys = layer.keys.squeeze(0)
                        layer.values = layer.values.squeeze(0)
    # transformers 4.x DynamicCache — has .key_cache and .value_cache
    elif hasattr(cache, 'key_cache'):
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None and cache.key_cache[i].ndim == 5:
                if cache.key_cache[i].shape[2] == 1:
                    cache.key_cache[i] = cache.key_cache[i].squeeze(2)
                    cache.value_cache[i] = cache.value_cache[i].squeeze(2)
                elif cache.key_cache[i].shape[0] == 1:
                    cache.key_cache[i] = cache.key_cache[i].squeeze(0)
                    cache.value_cache[i] = cache.value_cache[i].squeeze(0)
    return cache


def _load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Load tokenizer with automatic fallback to slow if fast fails.

    Some models (e.g. JackFram/llama-68m) have a broken fast tokenizer
    config (use_fast=true) but a SentencePiece protobuf tokenizer.model.
    This function tries fast first, then falls back to slow on error.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.warning(
            "Fast tokenizer failed for %s: %s. Falling back to slow tokenizer.",
            model_name_or_path, e,
        )
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


class DraftModel:
    """
    Wraps a small causal LM as a drafter.

    draft(context, k) -> (token_ids, logits)
    draft(context, k, past_key_values=..., past_len=..., cached_logits=...)
        -> (token_ids, logits, new_past_key_values)

    When past_key_values is provided, only new tokens (context[:, past_len:])
    are forwarded. When cached_logits is also provided (from a previous
    bonus-token forward), even that forward is skipped — the cached logits
    are used directly for the first draft token.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **model_kwargs,
    ) -> None:
        logger.info("Loading drafter tokenizer from %s", model_name_or_path)
        self.tokenizer = _load_tokenizer(model_name_or_path)
        logger.info("Loading drafter model from %s on %s", model_name_or_path, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            **model_kwargs,
        )
        self.model.eval()
        self.device = device
        logger.info("Drafter model ready: %s", model_name_or_path)

    def prepare_for_training(self, dtype: torch.dtype = torch.float32) -> None:
        if self.model.dtype != dtype:
            logger.info("Upcasting drafter %s -> %s for training", self.model.dtype, dtype)
            self.model = self.model.to(dtype)

    def draft(
        self,
        context: torch.Tensor,
        k: int,
        distill: bool = False,
        temperature: float = 1.0,
        past_key_values=None,
        past_len: int = 0,
        cached_logits: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor, object]:
        """
        Autoregressively generate k tokens.

        KV CACHE REUSE: When past_key_values is provided, only
        context[:, past_len:] is forwarded (typically 0-2 tokens).
        When cached_logits is also provided (from the previous step's
        bonus-token forward), even that forward is skipped.

        Returns: (token_ids, logits, new_past_key_values)
        """
        if not distill:
            with torch.no_grad():
                return self._draft_impl_kv(
                    context,
                    k,
                    temperature,
                    past_key_values,
                    past_len,
                    cached_logits,
                )
        else:
            return self._draft_distill(context, k, temperature)

    def _draft_impl_kv(
        self, context, k, temperature, past_key_values=None, past_len=0, cached_logits=None
    ):
        greedy = temperature <= 1e-6
        result_tokens, logits_list = [], []

        if cached_logits is not None:
            logits = cached_logits.squeeze(0) if cached_logits.dim() > 1 else cached_logits
            out_pkv = past_key_values
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())
            logits_list.append(logits.unsqueeze(0))
            new_input = next_token.unsqueeze(0).unsqueeze(0)
            remaining = k - 1
        elif past_key_values is not None and past_len > 0:
            # FIX: past_len may exceed context.shape[1] when the context
            # grows by fewer tokens than were drafted. Clamp to avoid
            # empty/negative slicing.
            actual_past = min(past_len, context.shape[1])
            new_input = context[:, actual_past:]   # only the unseen tail
            if new_input.shape[1] == 0:
                # Nothing new to process — just use last token from context
                new_input = context[:, -1:]
            out = self.model(new_input, past_key_values=past_key_values, use_cache=True)
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())
            logits_list.append(logits.unsqueeze(0))
            new_input = next_token.unsqueeze(0).unsqueeze(0)
            remaining = k - 1
        else:
            # FIX: Pass DynamicCache() explicitly — without it, some models
            # return a plain tuple as past_key_values which crashes later.
            from transformers.cache_utils import DynamicCache
            init_pkv = DynamicCache()
            out = self.model(context, past_key_values=init_pkv, use_cache=True)
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())
            logits_list.append(logits.unsqueeze(0))
            new_input = next_token.unsqueeze(0).unsqueeze(0)
            remaining = k - 1

        for _ in range(remaining):
            out = self.model(new_input, past_key_values=out_pkv, use_cache=True)
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
            logits_list.append(logits.unsqueeze(0))
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())
            new_input = next_token.unsqueeze(0).unsqueeze(0)

        logits_to_return = torch.stack(logits_list, dim=0)
        if logits_to_return.dim() == 3 and logits_to_return.shape[1] == 1:
            logits_to_return = logits_to_return.squeeze(1)
        return result_tokens, logits_to_return, out_pkv

    def _draft_distill(
        self,
        context: torch.Tensor,
        k: int,
        temperature: float,
        past_key_values=None,
        past_len: int = 0,
        cached_logits: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor, None]:
        """
        Distillation-aware drafting.

        FIX: Steps 0..k-2 use KV cache under no_grad (same as _draft_impl_kv).
        Only the FINAL step (k-1) runs with gradients — and it forwards
        ONLY the last few tokens (not the full context) by detaching
        the cached KV and using it as initial state.

        This reduces the gradient-enabled forward from O(L+k) to O(2-3),
        a major speedup when distillation is active.
        """
        greedy = temperature <= 1e-6
        result_tokens: list[int] = []
        logits_list: list[torch.Tensor] = []
        sampled_tokens: list[torch.Tensor] = []

        # --- Steps 0..k-2: KV cache + no_grad ---
        # FIX: Pass DynamicCache() explicitly — without it, some models
        # (e.g. pythia) return a plain tuple as past_key_values, which
        # lacks get_seq_length() and crashes on subsequent forward calls.
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache()
        with torch.no_grad():
            out = self.model(context, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

        for i in range(k - 1):
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]
            logits_list.append(logits.unsqueeze(0).detach())
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())
            sampled_tokens.append(next_token)

            with torch.no_grad():
                out = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values

        # --- Step k-1: gradient-enabled forward ---
        # FIX: Only forward the LAST sampled token (not full context).
        # The KV cache from no_grad steps is detached and used as
        # initial state — this is valid because the gradient only
        # needs to flow through the LAST token's logits.
        if sampled_tokens:
            last_token = sampled_tokens[-1].unsqueeze(0).unsqueeze(0)
            # Detach past_key_values to prevent graph corruption
            detached_kv = self._detach_pkv(past_key_values)
            out = self.model(
                last_token,
                past_key_values=detached_kv,
                use_cache=False,
            )
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]
        else:
            # k==1: no previous steps, forward full context with gradients
            out = self.model(context, use_cache=False)
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]

        logits_list.append(logits.unsqueeze(0))
        next_token = self._sample_next_token(logits, temperature, greedy)
        result_tokens.append(next_token.item())

        logits_to_return = torch.stack(logits_list, dim=0)
        if logits_to_return.dim() == 3 and logits_to_return.shape[1] == 1:
            logits_to_return = logits_to_return.squeeze(1)
        logger.debug(
            "Drafter (distill) generated %d tokens, logits shape: %s",
            k,
            tuple(logits_to_return.shape),
        )
        return result_tokens, logits_to_return, None  # KV cache not returned for distill

    @staticmethod
    def _detach_pkv(pkv):
        """Detach all tensors in past_key_values to prevent graph corruption."""
        if pkv is None:
            return None
        # Old-style DynamicCache (transformers 4.x) — key_cache / value_cache
        if hasattr(pkv, "key_cache"):
            for i in range(len(pkv.key_cache)):
                if pkv.key_cache[i] is not None:
                    pkv.key_cache[i] = pkv.key_cache[i].detach()
                if pkv.value_cache[i] is not None:
                    pkv.value_cache[i] = pkv.value_cache[i].detach()
            return pkv
        # transformers 5.x Cache — detach each layer's keys/values
        if isinstance(pkv, Cache) and hasattr(pkv, "layers"):
            for layer in pkv.layers:
                if layer.is_initialized:
                    if layer.keys is not None:
                        layer.keys = layer.keys.detach()
                    if layer.values is not None:
                        layer.values = layer.values.detach()
            return pkv
        # Legacy tuple-of-tuples
        return tuple(tuple(kv.detach() for kv in layer) for layer in pkv)

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float, greedy: bool) -> torch.Tensor:
        if greedy:
            return logits.argmax(dim=-1)

        if logits.isnan().any() or logits.isinf().any():
            logger.warning("Drafter logits contain NaN/Inf — falling back to argmax.")
            return logits.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6).argmax(dim=-1)

        probs = F.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits.squeeze(0)

    def cleanup(self) -> None:
        """Release model from GPU memory.

        Moves the model to CPU and deletes the reference.  After calling
        this the DraftModel is no longer usable until reloaded.
        """
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            self.model = None
        self.tokenizer = None

    @staticmethod
    def _forward_cached(model, input_ids, past_key_values, **kwargs):
        """Forward with KV cache, normalizing dims before and after."""
        past_key_values = _normalize_cache(past_key_values)
        try:
            out = model(input_ids, past_key_values=past_key_values, **kwargs)
        except RuntimeError as e:
            if "same number of dimensions" not in str(e) or past_key_values is None:
                raise
            logger.warning("KV dim mismatch (%s) — falling back to full forward", e)
            out = model(input_ids, **kwargs)
        if hasattr(out, "past_key_values"):
            out.past_key_values = _normalize_cache(out.past_key_values)
        return out
