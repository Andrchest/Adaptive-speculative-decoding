"""
core/models/draft_model.py

Wraps a small causal LM as a drafter for speculative decoding.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class DraftModel:
    """
    Wraps a small causal LM as a drafter.

    draft(context, k) -> (token_ids, logits)
      token_ids : List[int] of length k
      logits    : (k, drafter_vocab_size) float tensor
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
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

    def prepare_for_training(self, dtype: torch.dtype = torch.float32) -> None:
        """
        Upcast the drafter to a training-stable dtype.

        Full-parameter fine-tuning directly on fp16 weights is unstable:
        Adam's default eps=1e-8 underflows to 0 in float16, so for any
        parameter whose exp_avg_sq is also small, the update denominator
        sqrt(exp_avg_sq) + eps rounds to exactly 0, producing a 0/0 = NaN
        step that corrupts the whole weight tensor on the very next
        forward pass. fp32 eps=1e-8 is well within representable
        precision, so this removes the failure mode entirely.
        """
        if self.model.dtype != dtype:
            logger.info("Upcasting drafter %s -> %s for training", self.model.dtype, dtype)
            self.model = self.model.to(dtype)

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

                  - ``temperature <= 1e-6`` -> greedy argmax (one-hot q;
                    only valid when the decoder also uses greedy target
                    decoding — see note below).
                  - ``temperature > 0`` -> sample from
                    ``softmax(logits / temperature)``.

        Note on greedy mode
        -------------------
        For the standard speculative-sampling theorem (Leviathan et al.
        2023), ``q`` (drafter distribution) and ``p`` (target
        distribution) must both be proper probability distributions.
        Greedy decoding is a degenerate distribution (one-hot) so the
        theorem technically still applies, but the acceptance probability
        becomes ``min(1, p_token / 1) = min(1, p_token)`` which is just
        the target's own probability. This is rarely useful in
        practice — use temperature=1.0 (or any > 0) for correct
        speculative sampling.
        """
        logger.debug(
            "Drafter generating %d tokens with temperature=%.2f distill=%s",
            k, temperature, distill,
        )
        if not distill:
            with torch.no_grad():
                return self._draft_impl(context, k, temperature)
        else:
            return self._draft_distill(context, k, temperature)

    def _draft_impl(
        self, context: torch.Tensor, k: int, temperature: float
    ) -> tuple[list[int], torch.Tensor]:
        """
        Autoregressively generate k tokens using KV cache.

        Instead of re-processing the entire context at each step, we:
          1. Run a single forward pass on the full context to obtain
             ``past_key_values`` (the KV cache).
          2. For each of the k steps, feed only the newly sampled token
             together with the cached key/value states.

        This reduces the drafter's per-step complexity from O(seq_len²)
        to O(seq_len) and eliminates all ``context.clone()`` / ``torch.cat``
        allocations in the hot loop.
        """
        greedy = temperature <= 1e-6
        result_tokens: list[int] = []
        logits_list: list[torch.Tensor] = []

        # Initial forward pass: process the full context and obtain KV cache.
        with torch.no_grad():
            out = self.model(context, use_cache=True)
        past_key_values = out.past_key_values

        for _ in range(k):
            # Extract logits from the last position.
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]  # (V,)
            logits_list.append(logits.unsqueeze(0))  # (1, V) for stacking
            next_token = self._sample_next_token(logits, temperature, greedy)
            result_tokens.append(next_token.item())

            # Forward pass on the single new token using the KV cache.
            with torch.no_grad():
                out = self.model(
                    next_token.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values

        logits_to_return = torch.stack(logits_list, dim=0)  # (k, V)
        # Ensure logits are 2D: (k, V) not (k, 1, V)
        if logits_to_return.dim() == 3 and logits_to_return.shape[1] == 1:
            logits_to_return = logits_to_return.squeeze(1)
        logger.debug(
            "Drafter generated %d tokens, logits shape: %s", k, tuple(logits_to_return.shape),
        )
        return result_tokens, logits_to_return

    def _draft_distill(
        self, context: torch.Tensor, k: int, temperature: float
    ) -> tuple[list[int], torch.Tensor]:
        """
        Distillation-aware drafting with KV cache.

        Steps 0..k-2 use KV cache under ``torch.no_grad()`` to avoid
        retaining activations.  The final step (k-1) runs with gradients
        enabled so the distiller can backpropagate through the model
        parameters.

        Because the cached key/value states are created under ``no_grad``,
        they must not be passed into the gradient-enabled final forward
        pass (the autograd graph would be corrupted).  Instead we do a
        single full-sequence forward without cache for the last step.
        This is acceptable because it happens only once per draft.
        """
        greedy = temperature <= 1e-6
        result_tokens: list[int] = []
        logits_list: list[torch.Tensor] = []
        sampled_tokens: list[torch.Tensor] = []

        logger.debug(
            "DRAFTER DISTILL ctx_shape=%s k=%d",
            tuple(context.shape), k,
        )

        # --- Steps 0..k-2: KV cache + no_grad ---
        with torch.no_grad():
            out = self.model(context, use_cache=True)
        past_key_values = out.past_key_values

        for i in range(k - 1):
            logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]
            logits_list.append(logits.unsqueeze(0))
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

        # --- Step k-1: gradients enabled, no cache ---
        # Reconstruct the full input: context + all sampled tokens.
        if sampled_tokens:
            full_input = torch.cat(
                [context] + [t.unsqueeze(0).unsqueeze(0) for t in sampled_tokens],
                dim=1,
            )
        else:
            full_input = context

        out = self.model(full_input, use_cache=False)
        # Logits for the last sampled position (position k-1 relative to context end).
        logits = out.logits.reshape(-1, out.logits.shape[-1])[-1, :]
        logits_list.append(logits.unsqueeze(0))
        next_token = self._sample_next_token(logits, temperature, greedy)
        result_tokens.append(next_token.item())

        logits_to_return = torch.stack(logits_list, dim=0)  # (k, V)
        # Ensure logits are 2D: (k, V) not (k, 1, V)
        if logits_to_return.dim() == 3 and logits_to_return.shape[1] == 1:
            logits_to_return = logits_to_return.squeeze(1)
        logger.debug(
            "Drafter (distill) generated %d tokens, logits shape: %s, result_tokens len=%d",
            k, tuple(logits_to_return.shape), len(result_tokens),
        )
        return result_tokens, logits_to_return

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
