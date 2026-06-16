"""
extensions/contrastive/loss.py

Contrastive Rejection Learning.

Rejected draft tokens serve as hard negative examples.

Loss:
  L = KL_divergence + λ1 * NLL + λ2 * InfoNCE

InfoNCE:
  For each accepted token (positive) and rejected tokens (negatives),
  compute the InfoNCE loss to push the drafter away from rejected tokens
  and towards accepted ones.

  L_InfoNCE = -log [ exp(sim(z, z+)) / (exp(sim(z,z+)) + Σ exp(sim(z,z-))) ]
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def infonce_loss(
    anchor_logits: torch.Tensor,  # (k, drafter_vocab) — drafter logits
    positive_ids: torch.Tensor,  # (m,) — accepted token ids
    negative_ids: torch.Tensor,  # (n,) — rejected token ids
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    For each anchor position with an accepted token, treat its drafter
    embedding as the anchor, the accepted token's embedding as the positive,
    and the rejected tokens' embeddings as negatives.

    Parameters
    ----------
    anchor_logits : drafter output at positions with accepted tokens (m, Vd)
    positive_ids  : target token ids for accepted positions (m,)
    negative_ids  : token ids of rejected tokens (pooled across batch) (n,)

    Returns
    -------
    Scalar InfoNCE loss.
    """
    if len(positive_ids) == 0 or len(negative_ids) == 0:
        return torch.tensor(0.0, device=anchor_logits.device)

    m = len(positive_ids)

    # Use log-softmax scores as similarity
    log_probs = F.log_softmax(anchor_logits.float(), dim=-1)  # (m, Vd)

    # Positive scores: log p(positive_id | anchor)
    pos_scores = log_probs[torch.arange(m), positive_ids]  # (m,)

    # Negative scores: log p(negative_id | anchor) averaged over negatives
    # Shape: (m, n)
    neg_scores = log_probs[:, negative_ids]  # (m, n)

    # InfoNCE: -log [ exp(pos) / (exp(pos) + Σ exp(neg)) ]
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores / temperature], dim=1)  # (m, 1+n)
    loss = -F.log_softmax(all_scores, dim=1)[:, 0].mean()
    return loss


class ContrastiveLoss(torch.nn.Module):
    """
    Combined distillation + contrastive loss.

    L = KL + λ1 * NLL + λ2 * InfoNCE
    """

    def __init__(
        self,
        lambda_nll: float = 0.5,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_cont = lambda_contrastive
        self.temperature = temperature

    def forward(
        self,
        draft_logits: torch.Tensor,  # (k, drafter_vocab)
        target_logits: torch.Tensor,  # (k, target_vocab)
        accepted_mask: list[bool],
        draft_tokens: list[int],
        target_to_draft_mapping: torch.Tensor,  # (target_vocab,) → drafter_vocab idx or -1
    ) -> torch.Tensor:
        """
        Compute combined loss.
        """
        k = len(draft_tokens)
        acc_idx = [i for i, a in enumerate(accepted_mask) if a]
        rej_idx = [i for i, a in enumerate(accepted_mask) if not a]

        # --- KL divergence ---
        target_probs = F.softmax(target_logits.float(), dim=-1)
        draft_log = F.log_softmax(draft_logits.float(), dim=-1)

        # Project target probs back to drafter space via mapping
        drafter_vocab = draft_logits.shape[-1]
        target_in_drafter = torch.zeros(k, drafter_vocab, device=draft_logits.device)
        valid = target_to_draft_mapping >= 0
        d_idx = target_to_draft_mapping[valid]
        t_idx = torch.where(valid)[0]
        target_in_drafter[:, d_idx] = target_probs[:, t_idx]

        kl = F.kl_div(
            draft_log, target_in_drafter.clamp(min=1e-8), reduction="batchmean", log_target=False
        )

        # --- NLL on accepted tokens ---
        if acc_idx:
            acc_tokens = torch.tensor(
                [draft_tokens[i] for i in acc_idx],
                dtype=torch.long,
                device=draft_logits.device,
            )
            nll = F.cross_entropy(draft_logits[acc_idx], acc_tokens)
        else:
            nll = torch.tensor(0.0, device=draft_logits.device)

        # --- InfoNCE contrastive ---
        if acc_idx and rej_idx:
            pos_ids = torch.tensor(
                [draft_tokens[i] for i in acc_idx],
                dtype=torch.long,
                device=draft_logits.device,
            )
            neg_ids = torch.tensor(
                [draft_tokens[i] for i in rej_idx],
                dtype=torch.long,
                device=draft_logits.device,
            )
            cont = infonce_loss(
                draft_logits[acc_idx],
                pos_ids,
                neg_ids,
                temperature=self.temperature,
            )
        else:
            cont = torch.tensor(0.0, device=draft_logits.device)

        total = kl + self.lambda_nll * nll + self.lambda_cont * cont
        logger.debug(
            "ContrastiveLoss: kl=%.4f nll=%.4f cont=%.4f total=%.4f",
            kl.item(),
            nll.item(),
            cont.item(),
            total.item(),
        )
        return total, {"kl": kl.item(), "nll": nll.item(), "contrastive": cont.item()}
