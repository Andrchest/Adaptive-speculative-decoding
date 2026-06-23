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
        if draft_tokens:
            draft_tensor = torch.tensor(
                draft_tokens, dtype=torch.long, device=context.device
            ).unsqueeze(0)
            full_input = torch.cat([context, draft_tensor], dim=1)
        else:
            full_input = context

        out = self.model(full_input)
        ctx_len = context.shape[1]
        k = len(draft_tokens)
        logits = out.logits[0, ctx_len - 1 : ctx_len + k, :]  # (k+1, vocab)
        logger.debug("Target verification logits shape: %s", tuple(logits.shape))
        return logits
