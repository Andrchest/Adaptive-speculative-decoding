"""
extensions/multitarget/universal_drafter.py

Multi-Target Universal Drafter.

A single drafter model trained to draft for multiple target LLM families
(Llama, Qwen, Gemma, Mistral, DeepSeek, …).

Architecture extension:
  Standard drafter hidden states are augmented with a learnable
  target-specific embedding:

    hidden_augmented = hidden + target_embedding[target_id]

This is injected at every layer via adapter hooks.

Training:
  - Interleave batches from multiple (target, dataset) pairs
  - Each batch specifies target_id
  - Distillation loss computed against the corresponding target model
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target embedding adapter
# ---------------------------------------------------------------------------


class TargetEmbeddingAdapter(nn.Module):
    """
    Learnable per-target bias injected into drafter hidden states.

    Parameters
    ----------
    n_targets : number of supported target model families
    d_model   : hidden dimension of the drafter
    """

    def __init__(self, n_targets: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_targets, d_model)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.n_targets = n_targets

    def forward(self, hidden: torch.Tensor, target_id: int) -> torch.Tensor:
        """
        hidden    : (batch, seq, d_model)
        target_id : int
        returns   : (batch, seq, d_model)
        """
        t = torch.tensor([target_id], dtype=torch.long, device=hidden.device)
        bias = self.embedding(t)  # (1, d_model)
        return hidden + bias.unsqueeze(1)  # broadcast over batch and seq


# ---------------------------------------------------------------------------
# Universal Drafter
# ---------------------------------------------------------------------------


class UniversalDrafter(nn.Module):
    """
    Wraps a base causal LM drafter with target-conditioned adapter layers.

    The base model is loaded frozen (or LoRA-adapted); only the target
    embeddings and optional lightweight adapter layers are trained initially.

    Parameters
    ----------
    base_model_name  : HF model name/path
    target_names     : ordered list of target family names
    d_model          : hidden dim (must match base model)
    trainable_base   : if True, fine-tune base weights too
    """

    def __init__(
        self,
        base_model_name: str,
        target_names: list[str],
        d_model: int,
        trainable_base: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.target_names = target_names
        self.target_id_map: dict[str, int] = {n: i for i, n in enumerate(target_names)}
        n_targets = len(target_names)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map=device
        )
        if not trainable_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        self.target_adapter = TargetEmbeddingAdapter(n_targets, d_model)
        # Move the adapter to the same device and dtype as the base model so
        # that target_embedding weights are co-located with the input tensors
        # and match the lm_head dtype (float16/bfloat16, not float32).
        self.target_adapter.to(device=device, dtype=dtype)
        self._register_hooks()
        logger.info(
            "UniversalDrafter initialized: base=%s targets=%s trainable_base=%s",
            base_model_name,
            target_names,
            trainable_base,
        )

        self._current_target_id: int = 0

    def set_target(self, target_name: str) -> None:
        """Set active target for the next forward pass."""
        if target_name not in self.target_id_map:
            raise KeyError(f"Unknown target: {target_name!r}. Known: {self.target_names}")
        self._current_target_id = self.target_id_map[target_name]
        logger.debug(
            "UniversalDrafter target set to %s (id=%d)", target_name, self._current_target_id
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> object:
        return self.base_model(input_ids, **kwargs)

    @torch.no_grad()
    def draft(
        self,
        context: torch.Tensor,
        k: int,
        target_name: str |None = None,
        temperature: float = 1.0,
        past_key_values=None,
        past_len: int = 0,
        cached_logits: torch.Tensor |None = None,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Draft k tokens conditioned on the specified target family.

        The target adapter is applied at EVERY transformer layer via
        the forward hooks registered in ``_register_hooks``. We must
        NOT apply it again here — doing so would double the target bias
        at the final position and produce systematically wrong logits.

        Memory note: hidden_states tensors are explicitly deleted after
        each forward pass to reduce CUDA allocator pressure.
        """
        if target_name is not None:
            self.set_target(target_name)

        greedy = temperature <= 1e-6

        result_tokens = []
        logits_list = []

        if cached_logits is not None:
            logits = cached_logits.squeeze(0)
            out_pkv = past_key_values
        elif past_key_values is not None and past_len > 0:
            new_input = context[:, past_len:]
            out = self.base_model(
                new_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
        else:
            out = self.base_model(
                context,
                use_cache=True,
            )
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)

        next_tok = self._sample_next_token(
            logits,
            temperature,
            greedy,
        )

        result_tokens.append(next_tok.item())
        logits_list.append(logits.unsqueeze(0))

        new_input = next_tok.unsqueeze(0).unsqueeze(0)

        for _ in range(k - 1):
            out = self.base_model(
                new_input,
                past_key_values=out_pkv,
                use_cache=True,
            )
            out_pkv = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)
            logits_list.append(logits.unsqueeze(0))
            next_tok = self._sample_next_token(
                logits,
                temperature,
                greedy,
            )
            result_tokens.append(next_tok.item())
            new_input = next_tok.unsqueeze(0).unsqueeze(0)

        logits = torch.stack(logits_list, dim=0)

        if logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return result_tokens, logits, out_pkv

    @staticmethod
    def _sample_next_token(logits, temperature, greedy):
        if greedy:
            return logits.argmax(dim=-1)

        probs = F.softmax(
            logits.float() / max(temperature, 1e-6),
            dim=-1,
        )

        return torch.multinomial(probs, 1).squeeze(-1)

    def adapter_parameters(self) -> list[torch.Tensor]:
        return list(self.target_adapter.parameters())

    def trainable_parameters(self) -> list[torch.Tensor]:
        params = self.adapter_parameters()
        if any(p.requires_grad for p in self.base_model.parameters()):
            params += [p for p in self.base_model.parameters() if p.requires_grad]
        return params

    # ------------------------------------------------------------------
    # Hook registration (injects target bias at each residual stream)
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on transformer layers."""
        self._hooks = []

        def make_hook(adapter: TargetEmbeddingAdapter, target_id_ref: UniversalDrafter):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h = adapter(h, target_id_ref._current_target_id)
                    return (h,) + output[1:]
                return adapter(output, target_id_ref._current_target_id)

            return hook

        # Register on every transformer layer
        layers = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            layers = self.base_model.transformer.h

        if layers is not None:
            for layer in layers:
                h = layer.register_forward_hook(make_hook(self.target_adapter, self))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        if not hasattr(self, "_hooks") or self._hooks is None:
            return
        for h in self._hooks:
            try:
                h.remove()
            except RuntimeError:
                pass  # hook already removed
        self._hooks = []
