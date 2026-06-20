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
        self._hooks_standalone: list = []  # hooks not tied to layers

    def __enter__(self) -> "UniversalDrafter":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()

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

    def draft(
        self,
        context: torch.Tensor,
        k: int,
        target_name: str | None = None,
        distill: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Draft k tokens conditioned on the specified target family.

        Parameters
        ----------
        distill : if True, enable gradient flow (no hooks, no cache)
                  for online distillation. If False (default), use KV-cache
                  and run without gradients for inference speed.
        """
        if target_name is not None:
            self.set_target(target_name)

        # Remove hooks when distilling to allow clean gradient flow
        if distill:
            self.remove_hooks()
        else:
            # Re-register hooks for inference (use_cache + gradient-free)
            self._register_hooks()

        grad_ctx = torch.no_grad() if not distill else torch.enable_grad()
        if not distill:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        tokens: list[int] = []
        all_logits: list[torch.Tensor] = []
        cur = context.clone()

        with grad_ctx:
            for i in range(k):
                is_last = i == k - 1
                use_cache = not distill and not is_last

                out = self.base_model(
                    cur,
                    output_hidden_states=True,
                    use_cache=use_cache,
                )
                # Inject target adaptation into last hidden state before lm_head
                last_hidden = out.hidden_states[-1]  # (1, seq, d)
                adapted = self.target_adapter(last_hidden, self._current_target_id)
                # Re-compute logits with adapted hidden
                logits = self.base_model.lm_head(adapted[:, -1, :])  # (1, vocab)
                all_logits.append(logits.squeeze(0))
                next_tok = logits.argmax(dim=-1)
                tokens.append(next_tok.item())
                cur = torch.cat([cur, next_tok.unsqueeze(0)], dim=1)

        # Re-register hooks after distillation pass
        if distill:
            self._register_hooks()

        return tokens, torch.stack(all_logits, dim=0)  # (k, vocab)

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
        self._hooks_standalone = []
