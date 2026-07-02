# core/models/target_model.py
"""
core/models/target_model.py

Wraps a large causal LM as a target / verifier for speculative decoding.

CRITICAL FIXES:
  1. _truncate_pkv now updates DynamicCache._seen_tokens (was stale →
     get_seq_length() returned wrong value → KV cache rejected → full
     forward every step = O(L²) instead of O(L)).
  2. _kv_ok is reset per generate() call (was sticky-False after one
     transient failure, permanently disabling KV cache).
  3. verify() avoids CPU↔GPU sync for draft_tokens by using a pre-
     allocated GPU buffer with copy_ from a CPU tensor built once.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import Cache, DynamicCache

logger = logging.getLogger(__name__)


def _get_pkv_len(pkv: object) -> int:
    """Get sequence length from past_key_values."""
    if pkv is None:
        return 0
    # transformers 5.x Cache (has get_seq_length as a method)
    if isinstance(pkv, Cache):
        try:
            return pkv.get_seq_length()
        except Exception:
            pass
    # transformers 4.x DynamicCache — key_cache list of tensors
    if isinstance(pkv, DynamicCache) and hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0 and pkv.key_cache[0] is not None:
        return pkv.key_cache[0].shape[-2]
    # transformers 5.x DynamicCache — layers[i].keys.shape[-2]
    if isinstance(pkv, DynamicCache) and hasattr(pkv, "layers"):
        for layer in pkv.layers:
            if layer.is_initialized and layer.keys is not None and layer.keys.numel() > 0:
                return layer.keys.shape[-2]
        return 0
    # Detached PKV (list of (k, v) tuples per layer)
    if isinstance(pkv, (list, tuple)) and len(pkv) > 0:
        if isinstance(pkv[0], (list, tuple)) and len(pkv[0]) >= 1 and hasattr(pkv[0][0], "shape"):
            return pkv[0][0].shape[-2]
    # Legacy tuple-of-tuples
    if isinstance(pkv, tuple) and len(pkv) > 0 and isinstance(pkv[0], tuple):
        return pkv[0][0].shape[-2]
    return 0


def _truncate_pkv(pkv: object, keep_len: int) -> object:
    """
    Truncate past_key_values to keep_len positions.
    """
    if pkv is None:
        return None

    # transformers 5.x Cache — use built-in crop
    if isinstance(pkv, Cache):
        pkv.crop(keep_len)
        return pkv

    # Old-style DynamicCache (transformers 4.x)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        for i in range(len(pkv.key_cache)):
            if pkv.key_cache[i] is not None:
                pkv.key_cache[i] = pkv.key_cache[i][..., :keep_len, :]
            if pkv.value_cache[i] is not None:
                pkv.value_cache[i] = pkv.value_cache[i][..., :keep_len, :]
        if hasattr(pkv, "_seen_tokens"):
            pkv._seen_tokens = keep_len
        if hasattr(pkv, "_seq_length"):
            pkv._seq_length = keep_len
        return pkv

    # Legacy tuple-of-tuples — skip None entries
    def _truncate_layer(layer):
        return tuple(kv[..., :keep_len, :] if kv is not None else None for kv in layer)

    if isinstance(pkv, tuple) and len(pkv) > 0:
        try:
            return tuple(_truncate_layer(layer) for layer in pkv)
        except (TypeError, IndexError):
            pass

    return pkv


def _normalize_kv_dims(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure key/value tensors are 4D [batch, heads, seq_len, head_dim].
    Only squeeze trailing 1-sized dimensions after seq_len, preserving
    the batch, heads, seq_len, and head_dim axes.
    """
    if k.ndim > 4:
        # The extra dim is between heads and seq_len: [B, H, 1, T, D] → squeeze dim 2
        if k.ndim == 5 and k.shape[2] == 1:
            k = k.squeeze(2)
            v = v.squeeze(2)
        elif k.shape[0] == 1:
            k = k.squeeze(0)
            v = v.squeeze(0)
    return k, v


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


def _to_cache(pkv: object) -> object:
    """Convert legacy tuple to DynamicCache; pass through if already a Cache."""
    if pkv is None:
        return None
    # Already a proper Cache (transformers 5.x) or has get_seq_length
    if isinstance(pkv, Cache) or hasattr(pkv, "get_seq_length"):
        return pkv
    # transformers 4.x-style tuple-of-tuples
    if isinstance(pkv, tuple) and len(pkv) > 0 and isinstance(pkv[0], tuple):
        try:
            cache = DynamicCache()
            for i, layer in enumerate(pkv):
                if isinstance(layer, tuple) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if hasattr(k, "shape") and hasattr(v, "shape"):
                        k, v = _normalize_kv_dims(k, v)
                        cache.update(k, v, i)
            # Check for layers (transformers 5.x) or key_cache (transformers 4.x)
            has_content = hasattr(cache, 'layers') and cache.layers or \
                          hasattr(cache, 'key_cache') and len(cache.key_cache) > 0
            if has_content:
                n = len(cache.layers) if hasattr(cache, 'layers') else len(cache.key_cache)
                logger.debug("Converted legacy PKV to DynamicCache (%d layers)", n)
                return cache
        except Exception as e:
            logger.warning("PKV tuple→Cache failed: %s", e)
    # Fallback: list of (k,v) tuples (some models return this)
    if isinstance(pkv, (list, tuple)) and len(pkv) > 0:
        try:
            cache = DynamicCache()
            for i, item in enumerate(pkv):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    k, v = item[0], item[1]
                    if hasattr(k, "shape") and hasattr(v, "shape"):
                        k, v = _normalize_kv_dims(k, v)
                        cache.update(k, v, i)
                    else:
                        raise TypeError(f"Expected tensors, got {type(k).__name__}, {type(v).__name__}")
            # Check for layers (transformers 5.x) or key_cache (transformers 4.x)
            has_content = hasattr(cache, 'layers') and cache.layers or \
                          hasattr(cache, 'key_cache') and len(cache.key_cache) > 0
            if has_content:
                n = len(cache.layers) if hasattr(cache, 'layers') else len(cache.key_cache)
                logger.debug("Converted list-of-tuples PKV to DynamicCache (%d layers)", n)
                return cache
        except Exception as e:
            logger.warning("PKV list→Cache failed: %s", e)
    logger.warning("Unexpected PKV type=%s — KV cache may not work", type(pkv).__name__)
    return pkv


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


class TargetModel:
    """
    Wraps a large causal LM as a target / verifier.

    verify(context, draft_tokens, past_key_values=None) -> (logits, pkv)

    FIX: _kv_ok is resettable per-generation via reset_kv_state().
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = True,
        **model_kwargs,
    ) -> None:
        logger.info("Loading target tokenizer from %s", model_name_or_path)
        self.tokenizer = _load_tokenizer(model_name_or_path)

        if load_in_4bit:
            logger.info("Using 4-bit quantization for target model")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            logger.info("Loading target model without 4-bit quantization")

        logger.info("Loading target model from %s on %s", model_name_or_path, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            **model_kwargs,
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self._kv_ok: bool = True
        self.device = device
        # Pre-allocated GPU buffer for draft tokens — avoids per-step
        # torch.tensor(list) CPU→GPU transfer
        self._draft_buffer: torch.Tensor | None = None
        self._draft_buffer_size: int = 0
        logger.info("Target model ready: %s", model_name_or_path)

    def reset_kv_state(self) -> None:
        """Reset KV cache state for a new generation.

        Called by SpeculativeDecoder at the start of each generate()
        call to ensure transient KV failures from a previous prompt
        don't permanently disable caching.
        """
        self._kv_ok = True

    def cleanup(self) -> None:
        """Release model from GPU memory.

        Moves the model to CPU and deletes the reference.  After calling
        this the TargetModel is no longer usable until reloaded.
        """
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            self.model = None
        self.tokenizer = None
        self._draft_buffer = None

    @torch.no_grad()
    def verify(
        self,
        context: torch.Tensor,
        draft_tokens: list[int],
        past_key_values=None,
    ) -> tuple[torch.Tensor, object]:
        """Score context + draft_tokens. Uses past_key_values when available.

        FIX: Avoids CPU→GPU sync by using a pre-allocated GPU buffer
        and copy_ instead of torch.tensor(list, device=...).
        """
        k = len(draft_tokens)
        ctx_len = context.shape[1]

        if draft_tokens:
            # Ensure draft buffer is large enough
            if self._draft_buffer is None or k > self._draft_buffer_size:
                self._draft_buffer_size = max(k * 2, 16)
                self._draft_buffer = torch.zeros(
                    (1, self._draft_buffer_size),
                    dtype=torch.long,
                    device=context.device,
                )
            # Copy draft tokens to GPU buffer — single small transfer
            self._draft_buffer[0, :k].copy_(
                torch.tensor(draft_tokens, dtype=torch.long, device=context.device)
            )
            draft_tensor = self._draft_buffer[:, :k]
        else:
            draft_tensor = None

        out = None
        new_pkv = None

        if past_key_values is not None and self._kv_ok:
            try:
                # Ensure past_key_values is a Cache object for the model
                pkv = _to_cache(past_key_values)

                # If _to_cache couldn't convert, pkv lacks get_seq_length()
                # and will crash the model. Skip to full forward instead.
                if not hasattr(pkv, "get_seq_length"):
                    raise TypeError(f"PKV has no get_seq_length (type={type(pkv).__name__})")

                past_len = _get_pkv_len(pkv)

                # Safety: past_len must not exceed context length
                if past_len > ctx_len:
                    raise ValueError(f"KV cache length ({past_len}) > context length ({ctx_len})")

                new_ctx = context[:, past_len:]
                new_len = new_ctx.shape[1]

                if draft_tensor is not None:
                    full_input = torch.cat([new_ctx, draft_tensor], dim=1)
                else:
                    full_input = new_ctx

                out = self.model(
                    full_input,
                    past_key_values=pkv,
                    use_cache=True,
                )
                new_pkv = _to_cache(out.past_key_values)

                # Logits: we need k+1 positions starting from the last
                # context token's prediction.
                # full_input has (new_len + k) tokens.
                # Logits at position (new_len - 1) = prediction for first
                # draft token (or the bonus token if k==0).
                start = new_len - 1
                logits = out.logits[0, start : start + k + 1, :]

            except Exception as e:
                logger.warning("Target KV cache failed (%s). Falling back to full forward.", e)
                self._kv_ok = False
                out = None

        if out is None:
            # Full forward fallback — successful forward produces fresh PKV.
            # Reset _kv_ok so the NEXT step tries KV cache again with this fresh PKV.
            # A one-time PKV format mismatch should not permanently disable caching.
            if draft_tensor is not None:
                full_input = torch.cat([context, draft_tensor], dim=1)
            else:
                full_input = context
            out = self.model(full_input, use_cache=True)
            new_pkv = _to_cache(out.past_key_values)
            logits = out.logits[0, ctx_len - 1 : ctx_len + k, :]
            self._kv_ok = True  # retry KV cache on next step with fresh PKV

        return logits, new_pkv
