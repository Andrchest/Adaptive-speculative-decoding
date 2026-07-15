"""
core/distillation/online.py

Online distillation during speculative decoding.

Loss:
  L = L_direct (KL divergence on Rule1-matched tokens)
    + λ * L_ngram (NLL on accepted n-gram sequences)

Supports:
  - Full fine-tuning
  - LoRA fine-tuning (via PEFT)
  - Gradient accumulation
"""

from __future__ import annotations

import logging
from collections import deque

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OnlineDistiller:
    """
    Accumulates training signal from accepted speculative steps and
    periodically updates the drafter weights.

    Parameters
    ----------
    drafter_model   : DraftModel whose weights will be updated
    translator      : CrossVocabTranslator (used to get Rule1 mask)
    optimizer       : torch Optimizer (pre-configured with drafter params)
    lambda_ngram    : weight for N-gram NLL loss
    accum_steps     : gradient accumulation steps before weight update
    use_lora        : whether PEFT LoRA adapters are active
    """

    def __init__(
        self,
        drafter_model,
        translator,
        optimizer: torch.optim.Optimizer,
        lambda_ngram: float = 0.5,
        accum_steps: int = 8,
        use_lora: bool = False,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.drafter = drafter_model
        self.translator = translator
        self.optimizer = optimizer
        self.lambda_ngram = lambda_ngram
        self.accum_steps = accum_steps
        self.use_lora = use_lora
        self.max_grad_norm = max_grad_norm

        self._accum_loss = torch.tensor(0.0)
        self._accum_count = 0
        self._step_count = 0

        # Running stats — bounded deques to prevent unbounded memory growth.
        # maxlen=10000 covers ~800 steps at accum_steps=8, more than enough
        # for any practical experiment. Summaries use [-100:] windows.
        self.losses: deque[float] = deque(maxlen=10000)
        self.kl_losses: deque[float] = deque(maxlen=10000)
        self.nll_losses: deque[float] = deque(maxlen=10000)
        self.cont_losses: deque[float] = deque(maxlen=10000)

        # Optional contrastive loss module (set by runner when use_contrastive=True)
        self._contrastive_loss = None  # ContrastiveLoss | None
        logger.info(
            "OnlineDistiller initialized: lambda_ngram=%.2f accum_steps=%d use_lora=%s lr=%s",
            lambda_ngram,
            accum_steps,
            use_lora,
            optimizer.param_groups[0]["lr"] if optimizer.param_groups else "N/A",
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_contrastive_loss(self, loss_module) -> None:
        """
        Attach a ContrastiveLoss module to be added to subsequent steps.

        This is the public API for the contrastive-loss ablation. The
        runner should call this instead of assigning to the (private)
        ``_contrastive_loss`` attribute directly — a previous version of
        the runner assigned to ``contrastive_loss`` (no underscore),
        which silently never reached the underscore-prefixed attribute
        read by ``_compute_loss`` and so the entire contrastive-loss
        ablation was a no-op.
        """
        self._contrastive_loss = loss_module
        logger.info("Contrastive loss attached: %s", type(loss_module).__name__)

    def step(
        self,
        draft_logits: torch.Tensor,  # (k, drafter_vocab)
        target_logits: torch.Tensor,  # (k, target_vocab)
        draft_tokens: list[int],
        accepted_mask: list[bool],
        prompt_ids: list | None = None,
    ) -> float | None:
        """
        Accumulate one step of distillation loss.

        Returns the combined loss value if a weight update was performed,
        else None.
        """
        # Defensive: wrap the entire distillation step in try/except to handle
        # unexpected CUDA errors gracefully (e.g. index-out-of-bounds, OOM).
        # This prevents a single distillation error from crashing the entire experiment.
        try:
            loss = self._compute_loss(draft_logits, target_logits, draft_tokens, accepted_mask)
            if loss is None:
                return None

            (loss / self.accum_steps).backward()
            # Detach immediately after backward to free the computation graph.
            # Storing only the scalar loss value on CPU to avoid GPU memory leaks.
            loss_scalar = loss.detach().item()
            self._accum_loss = self._accum_loss + torch.tensor(loss_scalar, dtype=torch.float32)
            self._accum_count += 1

            if self._accum_count >= self.accum_steps:
                return self._update_weights()
            return None
        except Exception as e:
            logger.warning(
                "Distillation step failed: %s — skipping distillation for this step",
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
        accepted_mask: list[bool],
    ) -> torch.Tensor | None:
        """
        Hybrid loss:
          L_direct  = KL(target_prob_in_drafter_space || drafter_prob)
                      computed only for tokens that have direct Rule1 mappings
          L_ngram   = NLL of accepted tokens under drafter
        """
        if not any(accepted_mask):
            logger.debug("No accepted tokens, skipping distillation loss")
            return None

        # Handle 3D logits (e.g. (k, 1, Vd)) — squeeze intermediate dims
        if draft_logits.dim() == 3 and draft_logits.shape[1] == 1:
            draft_logits = draft_logits.squeeze(1)
        if target_logits.dim() == 3 and target_logits.shape[1] == 1:
            target_logits = target_logits.squeeze(1)

        # Compute accepted indices for diagnostics
        accepted_indices = [i for i, a in enumerate(accepted_mask) if a]

        # Translate target logits back to drafter vocab for KL comparison
        # (We use the inverse of Rule1: direct-mapped tokens only)
        direct_mask = self._get_direct_mask(
            draft_logits.shape[-1], device=draft_logits.device
        )  # (drafter_vocab,)

        # --- KL divergence loss (direct mappings) ---
        # Both distributions must be normalized over the SAME support
        # (the Rule1-mappable subset) for the KL to be a valid divergence.
        # Previously, the unnormalized target projection was passed
        # directly to F.kl_div, producing a biased surrogate whose value
        # depended on vocab size rather than on learning.
        drafter_log_probs = F.log_softmax(draft_logits.float(), dim=-1)  # (k, Vd)
        # Target probs projected back to drafter space via Rule1 transpose
        target_in_drafter = self._project_target_to_drafter(target_logits)  # (k, Vd)

        # Only compute KL where we have direct mappings
        if direct_mask.any():
            # Renormalize both distributions over the direct_mask support
            # so the KL is well-defined.
            drafter_masked_log = drafter_log_probs[:, direct_mask]  # (k, M)
            drafter_masked_log = drafter_masked_log - torch.logsumexp(
                drafter_masked_log, dim=-1, keepdim=True
            )
            target_masked = target_in_drafter[:, direct_mask]  # (k, M)
            target_masked = target_masked / target_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = F.kl_div(
                drafter_masked_log,
                target_masked,
                reduction="batchmean",
                log_target=False,
            )
        else:
            kl = torch.tensor(0.0, device=draft_logits.device)

        # --- N-gram NLL loss (accepted tokens) ---
        if accepted_indices:
            acc_tokens = torch.tensor(
                [draft_tokens[i] for i in accepted_indices],
                dtype=torch.long,
                device=draft_logits.device,
            )
            nll = F.cross_entropy(
                draft_logits[accepted_indices].float(),
                acc_tokens,
            )
        else:
            nll = torch.tensor(0.0, device=draft_logits.device)

        total = kl + self.lambda_ngram * nll
        self.kl_losses.append(kl.item())
        self.nll_losses.append(nll.item())

        # --- Contrastive loss (optional) ---
        if self._contrastive_loss is not None:
            cont_loss, cont_stats = self._contrastive_loss(
                draft_logits=draft_logits,
                target_logits=target_logits,
                accepted_mask=accepted_mask,
                draft_tokens=draft_tokens,
                target_to_draft_mapping=self._get_target_to_draft_mapping(
                    target_logits.shape[-1], device=target_logits.device
                ),
            )
            total = total + cont_loss
            self.cont_losses.append(cont_stats["contrastive"])
            logger.debug(
                "Contrastive loss added: cont=%.4f total=%.4f",
                cont_loss.item(),
                total.item(),
            )

        return total

    def _update_weights(self) -> float:
        torch.nn.utils.clip_grad_norm_(self.drafter.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss_val = (self._accum_loss / self._accum_count).item()
        self.losses.append(loss_val)
        self._accum_loss = torch.tensor(0.0)
        self._accum_count = 0
        self._step_count += 1
        # deque slicing not supported in Python 3.12; use list() for the recent window
        kl_recent = list(self.kl_losses)[-self.accum_steps :]
        nll_recent = list(self.nll_losses)[-self.accum_steps :]
        logger.info(
            "Weight update step=%d loss=%.6f kl=%.6f nll=%.6f",
            self._step_count,
            loss_val,
            sum(kl_recent) / max(len(kl_recent), 1),
            sum(nll_recent) / max(len(nll_recent), 1),
        )
        return loss_val

    def _get_direct_mask(self, drafter_vocab_size: int, device=None) -> torch.Tensor:
        mapping = self.translator.rule1._mapping  # (drafter_vocab,)
        mask = mapping >= 0
        if device is not None and mask.device != device:
            mask = mask.to(device, non_blocking=True)
        return mask

    def _project_target_to_drafter(self, target_logits: torch.Tensor) -> torch.Tensor:
        """
        Project target logits back to drafter vocab using Rule1 inverse mapping.
        Tokens without a mapping get zero probability mass.
        """
        device = target_logits.device
        mapping = self.translator.rule1._mapping.to(device, non_blocking=True)
        drafter_vocab = mapping.shape[0]
        k = target_logits.shape[0]

        target_probs = F.softmax(target_logits.float(), dim=-1)  # (k, Vt)
        drafter_proj = torch.zeros(k, drafter_vocab, device=device)

        valid = mapping >= 0
        d_idx = torch.where(valid)[0]  # (M,)
        t_idx = mapping[d_idx]  # (M,)
        valid_t = t_idx < target_probs.shape[-1]
        d_idx = d_idx[valid_t]
        t_idx = t_idx[valid_t]

        # --- FIX: use explicit loop to avoid CUDA fancy-indexing issues ---
        # The original `drafter_proj[:, d_idx] = target_probs[:, t_idx]`
        # can trigger CUDA OOM or index-out-of-bounds on certain GPU/PyTorch
        # combinations. Using per-row assignment which is safer.
        if d_idx.numel() > 0:
            for b in range(k):
                drafter_proj[b, d_idx] = target_probs[b, t_idx]

        return drafter_proj

    def _get_target_to_draft_mapping(self, target_vocab_size: int, device=None) -> torch.Tensor:
        """
        Build a tensor mapping target vocab indices → drafter vocab indices.

        Returns (target_vocab_size,) where:
          mapping[t] = drafter_idx if target token t has a direct Rule1 match,
          mapping[t] = -1           otherwise.
        """
        mapping = self.translator.rule1._mapping  # (drafter_vocab,) → target_vocab
        # mapping[d] = t means drafter token d maps to target token t
        # We need the inverse: target → drafter
        t2d = torch.full((target_vocab_size,), -1, dtype=torch.long, device=device)
        d_indices = torch.where(mapping >= 0)[0]
        t_indices = mapping[d_indices]
        t2d[t_indices] = d_indices.to(device)
        return t2d

    def training_stats(self) -> dict:
        if not self.losses:
            return {}
        # deque doesn't support slicing in Python 3.12; convert to list
        losses_list = list(self.losses)
        kl_list = list(self.kl_losses)
        nll_list = list(self.nll_losses)
        stats: dict = {
            "update_steps": self._step_count,
            "mean_loss": sum(losses_list[-100:]) / max(len(losses_list[-100:]), 1),
            "mean_kl": sum(kl_list[-100:]) / max(len(kl_list[-100:]), 1),
            "mean_nll": sum(nll_list[-100:]) / max(len(nll_list[-100:]), 1),
        }
        if self.cont_losses:
            cont_list = list(self.cont_losses)
            stats["mean_contrastive"] = sum(cont_list[-100:]) / max(len(cont_list[-100:]), 1)
        return stats
