"""
core/models/target_model.py

Wraps a large causal LM as a target / verifier for speculative decoding.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class TargetModel:
    """
    Wraps a large causal LM as a target / verifier.

    verify(context, draft_tokens) -> target_logits (k+1, target_vocab)

    The k+1-th logit corresponds to the token after the full accepted prefix
    (used for the bonus token).
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
        self.device = device
        # P4.2: Pre-allocated buffer for draft tokens to avoid per-step
        # torch.tensor() allocation. Grows on demand as draft length increases.
        self._draft_buffer: torch.Tensor | None = None
        self._draft_buffer_size: int = 0
        logger.info("Target model ready: %s", model_name_or_path)

    @torch.no_grad()
    def verify(
        self,
        context: torch.Tensor,  # (1, seq_len)
        draft_tokens: list[int],
    ) -> torch.Tensor:
        """
        Score context ++ draft_tokens in a single forward pass.

        Returns logits at each draft position + the position after the last draft
        token, shape (len(draft_tokens) + 1, target_vocab_size).
        """
        logger.info(
            "Target verification context_len=%d draft_tokens=%d",
            context.shape[1],
            len(draft_tokens),
        )
        k = len(draft_tokens)
        if draft_tokens:
            # P4.2: Reuse pre-allocated buffer instead of torch.tensor() each step.
            if self._draft_buffer is None or k > self._draft_buffer_size:
                self._draft_buffer_size = max(k * 2, 16)  # grow with headroom
                self._draft_buffer = torch.zeros(
                    (1, self._draft_buffer_size),
                    dtype=torch.long,
                    device=context.device,
                )
            # Copy list into GPU buffer via a small tensor (avoids GPU alloc).
            self._draft_buffer[0, :k].copy_(
                torch.tensor(draft_tokens, dtype=torch.long)
            )
            draft_tensor = self._draft_buffer[:, :k]
            full_input = torch.cat([context, draft_tensor], dim=1)
        else:
            full_input = context

        out = self.model(full_input)
        ctx_len = context.shape[1]
        logits = out.logits[0, ctx_len - 1 : ctx_len + k, :]  # (k+1, vocab)
        logger.debug("Target verification logits shape: %s", tuple(logits.shape))
        return logits
