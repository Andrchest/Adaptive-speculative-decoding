"""
core/models/drafter.py

Thin wrappers around HuggingFace models to standardise the interface
used by SpeculativeDecoder.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class DraftModel:
    """
    Wraps a small causal LM as a drafter.

    draft(context, k) → (token_ids, logits)
      token_ids : List[int] of length k
      logits    : (k, drafter_vocab_size) float tensor
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **model_kwargs,
    ) -> None:
        logger.info("Loading drafter tokenizer from %s", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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

    def draft(
        self,
        context: torch.Tensor,
        k: int,
        distill: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Autoregressively generate k tokens.
        Returns token ids and the corresponding logits.

        Parameters
        ----------
        distill : if True, gradient information is preserved for
                  online distillation. Steps 0..k-2 are run with
                  torch.no_grad() and their logits are discarded,
                  keeping only the last step's logits for the loss.
                  This reduces activation memory by ~80%.
        """
        logger.info("Drafting %d token(s) from context length %d", k, context.shape[1])
        tokens: list[int] = []
        cur = context.clone()

        for i in range(k):
            is_last = i == k - 1

            if distill and not is_last:
                # Intermediate steps: no gradient, no logits stored.
                # The context stays connected to the original tensor
                # (through torch.cat), so the final step can still
                # compute gradients through the model parameters.
                with torch.no_grad():
                    out = self.model(cur, use_cache=True)
                _ = out.logits[:, -1, :]
            else:
                out = self.model(cur, use_cache=True)

            logits = out.logits[:, -1, :]  # (1, vocab)
            next_tok = logits.argmax(dim=-1)  # greedy
            tokens.append(next_tok.item())
            cur = torch.cat([cur, next_tok.unsqueeze(0)], dim=1)
            logger.debug("Draft token %d/%d: %d", i + 1, k, tokens[-1])

        # Only the last step has gradient tracking (when distill=True),
        # so we only return its logits. For non-distillation mode we
        # return all k logits for the full sequence.
        if distill:
            logits_to_return = logits.unsqueeze(0)  # (1, vocab)
        else:
            logits_to_return = torch.stack(
                [self._get_logits_at(context, cur, k)], 0
            )  # (k, vocab)

        logger.info("Draft complete: generated %d token(s)", len(tokens))
        return tokens, logits_to_return

    def _get_logits_at(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Recompute the full sequence logits (non-gradient) for return."""
        with torch.no_grad():
            cur = start.clone()
            all_logits: list[torch.Tensor] = []
            for i in range(k):
                out = self.model(cur, use_cache=True)
                logits = out.logits[:, -1, :].squeeze(0)
                all_logits.append(logits)
                next_tok = logits.argmax(dim=-1)
                cur = torch.cat([cur, next_tok.unsqueeze(0)], dim=1)
        return torch.stack(all_logits, dim=0)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass; returns logits (seq, vocab)."""
        return self.model(input_ids).logits.squeeze(0)


class TargetModel:
    """
    Wraps a large causal LM as a target / verifier.

    verify(context, draft_tokens) → target_logits (k+1, target_vocab)

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
