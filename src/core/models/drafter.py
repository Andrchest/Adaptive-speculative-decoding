"""
core/models/drafter.py

Thin wrappers around HuggingFace models to standardise the interface
used by SpeculativeDecoder.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
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
            dtype=dtype,
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
        temperature: float = 1.0,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Autoregressively generate k tokens.
        Returns token ids and the corresponding logits.

        Parameters
        ----------
        distill : if True, gradient information is preserved for
                  online distillation. Steps 0..k-2 are run with
                  torch.no_grad() to save activation memory; their
                  logits are detached before stacking.  The final
                  step runs with gradients so the distiller can
                  backpropagate through the model parameters.
                  ``use_cache`` is disabled when distill=True so the
                  cached key/value states (created under no_grad) do
                  not corrupt the gradient-enabled final forward pass.
        temperature : sampling temperature for the drafter. The drafter
                  MUST sample from the SAME distribution ``q`` that the
                  decoder uses in its acceptance test, otherwise the
                  speculative-sampling theorem does not apply and the
                  output distribution is biased.

                  - ``temperature <= 1e-6`` → greedy argmax (one-hot q;
                    only valid when the decoder also uses greedy target
                    decoding — see note below).
                  - ``temperature > 0`` → sample from
                    ``softmax(logits / temperature)``.

        Note on greedy mode
        -------------------
        For the standard speculative-sampling theorem (Leviathan et al.
        2023, Chen et al. 2023) to preserve the target distribution,
        the drafter must actually *sample* from ``q``. Greedy drafting
        (argmax) is only distribution-preserving when paired with a
        greedy target AND a "strict" acceptance rule (accept iff
        ``argmax(p) == draft_token``). The default ``temperature=1.0``
        here ensures the theorem holds for stochastic decoding.
        """
        logger.info(
            "Drafting %d token(s) from context length %d (temperature=%s)",
            k,
            context.shape[1],
            temperature,
        )
        tokens: list[int] = []
        all_logits: list[torch.Tensor] = []
        cur = context.clone()
        step_logits: list[torch.Tensor] = []
        greedy = temperature <= 1e-6

        for i in range(k):
            is_last = i == k - 1

            if distill and not is_last:
                with torch.no_grad():
                    out = self.model(cur, use_cache=True)
            else:
                out = self.model(cur, use_cache=not distill)
                all_logits.append(out.logits[0, -1, :].detach())

            logits = out.logits[:, -1, :]  # (1, vocab)
            step_logits.append(logits)
            next_tok = self._sample_next_token(logits, temperature, greedy)
            tokens.append(next_tok.item())
            cur = torch.cat([cur, next_tok.unsqueeze(0)], dim=1)
            logger.debug("Draft token %d/%d: %d", i + 1, k, tokens[-1])

        if distill:
            logits_to_return = torch.cat(step_logits, dim=0)  # (k, vocab)
        else:
            # Reuse logits already collected during the autoregressive loop.
            # This avoids the redundant k forward passes that _get_logits_at() used to do.
            logits_to_return = torch.stack(all_logits, dim=0)  # (k, vocab)

        logger.info("Draft complete: generated %d token(s)", len(tokens))

        # Sanitize returned logits: NaN/Inf breaks the entire downstream
        # chain (translation, acceptance test, residual sampling).
        # The model producing NaN is abnormal — log a clear warning.
        if logits_to_return.isnan().any() or logits_to_return.isinf().any():
            n_nan = logits_to_return.isnan().sum().item()
            n_inf = logits_to_return.isinf().sum().item()
            logger.warning(
                "Drafter produced %d NaN + %d Inf logits out of %d — "
                "this indicates a model-level issue (fp16 instability, "
                "corrupt weights, or incompatible tokenizer). "
                "Zeroing them to prevent downstream crashes.",
                n_nan, n_inf, logits_to_return.numel(),
            )
            logits_to_return = torch.nan_to_num(logits_to_return, nan=0.0, posinf=1e6, neginf=-1e6)

        return tokens, logits_to_return

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor, temperature: float, greedy: bool
    ) -> torch.Tensor:
        """
        Sample (or argmax) the next token from ``logits`` of shape (1, V).
        Falls back to argmax if logits contain NaN/Inf.

        Returns a tensor of shape (1,) so that ``.unsqueeze(0)`` yields
        (1, 1) for concatenation with the running context.
        """
        if greedy:
            return logits.argmax(dim=-1)  # (1,)

        if logits.isnan().any() or logits.isinf().any():
            logger.warning(
                "Drafter logits contain NaN/Inf at sampling step — "
                "falling back to argmax. This indicates a model-level issue."
            )
            return logits.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6).argmax(dim=-1)

        probs = F.softmax(logits.float() / max(temperature, 1e-6), dim=-1)  # (1, V)
        return torch.multinomial(probs, 1).squeeze(-1)  # (1,)

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
