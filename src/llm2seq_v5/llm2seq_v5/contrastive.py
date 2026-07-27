"""Counterfactual objectives that measure source-memory utilization.

The decoder-side representation is taken at the final fixed prompt token,
before teacher forcing exposes any target-summary token.  A complementary
source-swap loss compares the same target under its correct and a deranged
source memory.  Consequently, neither objective can be solved from target
content alone.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool hidden states using an attention mask.

    Args:
        hidden_states: [B, S, D] tensor of hidden states.
        mask: [B, S] tensor where 1 indicates valid positions.

    Returns:
        [B, D] tensor of pooled representations.
    """
    expanded = mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
    summed = (hidden_states * expanded).sum(dim=1)  # [B, D]
    counts = expanded.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return summed / counts


def masked_last_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the final valid token, matching Qwen3-Embedding EOS pooling."""
    if hidden_states.ndim != 3 or mask.ndim != 2 or hidden_states.shape[:2] != mask.shape:
        raise ValueError("masked_last_pool expects [B, S, D] states and a matching [B, S] mask")
    if not bool(mask.bool().any(dim=1).all()):
        raise ValueError("Every source example must contain at least one valid token")
    positions = mask.long().sum(dim=1) - 1
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, positions]


def source_memory_for_mining(
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    bank_weights: torch.Tensor | None = None,
    pooling: str = "mean_last",
) -> torch.Tensor:
    """Create target-free source vectors for in-batch hard-negative mining."""
    if memory.ndim == 4:
        bank_count = memory.shape[1]
        if bank_weights is None:
            bank_weights = memory.new_full((bank_count,), 1.0 / bank_count)
        if bank_weights.ndim != 1 or bank_weights.numel() != bank_count:
            raise ValueError(f"bank_weights must contain {bank_count} values")
        weights = bank_weights.to(memory.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        memory = torch.einsum("k,bksd->bsd", weights, memory)
    if memory.ndim != 3:
        raise ValueError("Source memory must be [B, S, D] or [B, K, S, D]")
    if pooling == "mean":
        pooled = masked_mean_pool(memory, memory_mask)
    elif pooling == "mean_last":
        pooled = 0.5 * (masked_mean_pool(memory, memory_mask) + masked_last_pool(memory, memory_mask))
    else:
        raise ValueError("Source mining pooling must be 'mean' or 'mean_last'")
    return F.normalize(pooled.float(), dim=-1)


@torch.no_grad()
def hard_negative_indices(source_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the most similar non-self source for every batch row."""
    if source_repr.ndim != 2 or source_repr.shape[0] <= 1:
        raise ValueError("Hard-negative mining requires [B, D] with B > 1")
    normalized = F.normalize(source_repr.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    indices = similarity.argmax(dim=1)
    rows = torch.arange(similarity.shape[0], device=similarity.device)
    selected_similarity = similarity[rows, indices]
    return indices, selected_similarity


class SourceAlignmentHead(nn.Module):
    """Project source memory and a target-free prompt state into a shared space.

    All examples use the same decoder prompt.  Therefore, if cross-attention is
    ignored, every prompt representation is source-independent and InfoNCE
    cannot identify the matching document.
    """

    def __init__(self, hidden_size: int, projection_size: int = 256, pooling: str = "mean_last"):
        super().__init__()
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "mean_last"}:
            raise ValueError("SourceAlignmentHead pooling must be 'mean' or 'mean_last'")
        if self.pooling == "mean_last":
            # Per-channel convex fusion starts at 50/50. The last valid token
            # carries Qwen3-Embedding's pretrained document representation;
            # mean pooling keeps distributed token evidence available.
            self.pool_gate = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter("pool_gate", None)
        self.memory_proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        self.decoder_proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        nn.init.xavier_uniform_(self.memory_proj[-1].weight)
        nn.init.xavier_uniform_(self.decoder_proj[-1].weight)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        prompt_states: torch.Tensor,
        bank_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute normalized representations for contrastive comparison.

        Args:
            memory: Adapter output. Either [B, S, D] or [B, K, S, D] for
                depth-routed HiRoute memory. If 4D, banks are combined with
                the decoder's mean routing weights before token pooling.
            memory_mask: [B, S] attention mask for source tokens.
            prompt_states: [B, D] decoder hidden states at the final prompt
                position, before any target-summary token is visible.
            bank_weights: Optional [K] mean routing distribution. It is
                required to reflect the actual routed memory when available;
                uniform weights are the safe fallback for standalone use.

        Returns:
            Dict with "memory_repr" and "decoder_repr", each [B, proj_dim]
            and L2-normalized.
        """
        # Handle depth-routed memory: [B, K, S, D] → [B, S, D]. The caller
        # passes the differentiable mean route from the positive-source pass,
        # so InfoNCE trains both the memory banks and the dynamic router.
        if memory.ndim == 4:
            bank_count = memory.shape[1]
            if bank_weights is None:
                bank_weights = memory.new_full((bank_count,), 1.0 / bank_count)
            if bank_weights.ndim != 1 or bank_weights.numel() != bank_count:
                raise ValueError(
                    f"bank_weights must be [{bank_count}] for routed memory; received {tuple(bank_weights.shape)}"
                )
            # Decoder routing weights already sum to one. Re-normalizing by
            # their sum preserves that distribution without treating them as
            # a second set of logits.
            normalized_weights = bank_weights.to(memory.dtype)
            normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-8)
            memory = torch.einsum("k,bksd->bsd", normalized_weights, memory)

        memory_pooled = masked_mean_pool(memory, memory_mask)  # [B, D]
        if self.pool_gate is not None:
            memory_last = masked_last_pool(memory, memory_mask)
            last_weight = torch.sigmoid(self.pool_gate.float()).to(memory.dtype)
            memory_pooled = (1.0 - last_weight) * memory_pooled + last_weight * memory_last
        if prompt_states.ndim != 2:
            raise ValueError(
                "SourceAlignmentHead requires one target-free prompt state per example; "
                f"received {tuple(prompt_states.shape)}"
            )

        memory_repr = F.normalize(self.memory_proj(memory_pooled), dim=-1)
        decoder_repr = F.normalize(self.decoder_proj(prompt_states), dim=-1)

        return {
            "memory_repr": memory_repr,
            "decoder_repr": decoder_repr,
        }


def info_nce_loss(
    memory_repr: torch.Tensor,
    decoder_repr: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss with in-batch negatives.

    Encourages (source_i, decoder_i) pairs to be similar while pushing
    apart cross-pairs (source_i, decoder_j) for j != i.

    Because decoder representations are collected before target tokens, a
    collapsed cross-attention path makes them source-independent.

    Args:
        memory_repr: [B, proj_dim] L2-normalized memory representations.
        decoder_repr: [B, proj_dim] L2-normalized decoder representations.
        temperature: Scaling factor for logits (lower = harder negatives).

    Returns:
        Scalar loss tensor.
    """
    batch = memory_repr.shape[0]
    if batch <= 1:
        # Cannot compute contrastive loss with a single example
        return memory_repr.new_zeros(())

    # Similarity matrix: [B, B]
    logits = decoder_repr @ memory_repr.T / temperature

    labels = torch.arange(batch, device=logits.device)

    # Symmetric: decoder→memory and memory→decoder
    loss_d2m = F.cross_entropy(logits, labels)
    loss_m2d = F.cross_entropy(logits.T, labels)

    return (loss_d2m + loss_m2d) / 2.0


def last_prompt_states(
    decoder_states: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Gather the decoder state that predicts the first supervised token.

    In the shifted decoder input this position contains the final prompt token,
    while ``labels`` already contains the first target token.  It is therefore
    the latest decoder state that cannot leak target-summary content.
    """
    if decoder_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("decoder_states must be [B, T, D] and labels must be [B, T]")
    if decoder_states.shape[:2] != labels.shape:
        raise ValueError("decoder_states and labels must have matching batch/sequence dimensions")
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain at least one supervised target token")
    positions = supervised.to(torch.int64).argmax(dim=1)
    batch_indices = torch.arange(decoder_states.shape[0], device=decoder_states.device)
    return decoder_states[batch_indices, positions]


def per_example_nll(
    supervised_logits: torch.Tensor,
    labels: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    """Reduce flattened supervised token logits to one mean NLL per example."""
    if supervised_logits.ndim != 2 or labels.ndim != 2 or supervised.shape != labels.shape:
        raise ValueError("Expected flattened logits plus [B, T] labels/supervision mask")
    expected_tokens = int(supervised.sum().item())
    if supervised_logits.shape[0] != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} supervised logits, received {supervised_logits.shape[0]}")
    token_losses = F.cross_entropy(
        supervised_logits.float(),
        labels[supervised],
        reduction="none",
    )
    batch_indices = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(labels)[supervised]
    sums = token_losses.new_zeros(labels.shape[0])
    sums.scatter_add_(0, batch_indices, token_losses)
    counts = supervised.sum(dim=1).to(token_losses.dtype).clamp_min(1.0)
    return sums / counts


def source_swap_contrastive_loss(
    positive_nll: torch.Tensor,
    negative_nll: torch.Tensor,
    margin: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Prefer the correct source over a deranged source for the same target.

    The loss approaches zero only when the wrong-source NLL exceeds the
    correct-source NLL by at least ``margin``.  Softplus provides a stable,
    bounded-gradient alternative to directly maximizing negative NLL.
    """
    if positive_nll.shape != negative_nll.shape:
        raise ValueError("positive_nll and negative_nll must have equal shapes")
    if temperature <= 0.0:
        raise ValueError("source-swap temperature must be positive")
    logits = (positive_nll - negative_nll + float(margin)) / float(temperature)
    return F.softplus(logits).mean()
