"""Training-only alignment between summary slots and ordered gold response chunks."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ordered_response_alignment_loss(
    summary_prefix: torch.Tensor,
    labels: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Align each prospective-summary slot with its ordered response segment.

    Valid target labels are split into at most ``K`` contiguous, nearly equal
    chunks, where ``K`` is the number of latent slots.  Chunk targets are means
    of the decoder's *detached* token embeddings.  The objective combines
    diagonal cosine regression with an in-example InfoNCE term so a slot must
    match both the content and the position of its own chunk.

    Gradients intentionally flow only to ``summary_prefix``.  This avoids
    distorting the decoder embedding space merely to make the bridge objective
    easier.
    """

    if summary_prefix.ndim != 3:
        raise ValueError(
            f"summary_prefix must be [B, K, D], got {tuple(summary_prefix.shape)}"
        )
    if labels.ndim != 2 or labels.shape[0] != summary_prefix.shape[0]:
        raise ValueError(
            "labels must be [B, T] and share the summary_prefix batch dimension"
        )
    if embedding_weight.ndim != 2 or embedding_weight.shape[1] != summary_prefix.shape[2]:
        raise ValueError(
            "embedding_weight must be [V, D] with D equal to the summary-prefix width"
        )
    if temperature <= 0.0:
        raise ValueError("response-alignment temperature must be positive")

    batch, slot_count, hidden_size = summary_prefix.shape
    device = summary_prefix.device

    # Build all ordered target chunks with two batched scatter operations.
    # The previous implementation looped over every example and slot and used
    # ``.item()`` on CUDA tensors.  That was mathematically correct but forced
    # repeated GPU/CPU synchronizations and left a B200 waiting on Python.
    valid_tokens = labels.ge(0)
    token_counts = valid_tokens.sum(dim=1)
    active_slots = token_counts.clamp(max=slot_count)
    token_rank = valid_tokens.long().cumsum(dim=1) - 1

    # Match the old boundaries exactly:
    #   start_s=floor(s*N/A), end_s=floor((s+1)*N/A).
    # For a valid zero-based token rank r, its chunk is
    # floor((((r+1)*A)-1)/N).  Invalid/padded tokens are sent to slot zero but
    # multiplied by zero before scatter_add, so they cannot affect a target.
    safe_counts = token_counts.clamp_min(1)
    chunk_ids = (
        ((token_rank.clamp_min(0) + 1) * active_slots[:, None] - 1)
        // safe_counts[:, None]
    ).clamp(min=0, max=slot_count - 1)

    detached_embeddings = embedding_weight.detach()
    token_ids = labels.clamp_min(0)
    embedded = F.embedding(token_ids, detached_embeddings).to(summary_prefix.dtype)
    embedded = embedded * valid_tokens.unsqueeze(-1).to(embedded.dtype)

    targets = summary_prefix.new_zeros((batch, slot_count, hidden_size))
    targets.scatter_add_(
        1,
        chunk_ids.unsqueeze(-1).expand(-1, -1, hidden_size),
        embedded,
    )
    chunk_counts = summary_prefix.new_zeros((batch, slot_count))
    chunk_counts.scatter_add_(1, chunk_ids, valid_tokens.to(summary_prefix.dtype))
    valid_mask = chunk_counts.gt(0)
    targets = (targets / chunk_counts.clamp_min(1).unsqueeze(-1)).detach()

    zero = summary_prefix.float().sum() * 0.0
    valid_count = valid_mask.sum()
    if int(valid_count.item()) == 0:
        return {
            "loss": zero,
            "cosine": zero.detach(),
            "accuracy": zero.detach(),
            "valid_slots": valid_count,
        }

    normalized_slots = F.normalize(summary_prefix.float(), dim=-1)
    normalized_targets = F.normalize(targets.float(), dim=-1)
    diagonal_cosine = (normalized_slots * normalized_targets).sum(dim=-1)
    cosine = diagonal_cosine[valid_mask].mean()
    cosine_loss = 1.0 - cosine

    logits = normalized_slots @ normalized_targets.transpose(1, 2)
    logits = logits / float(temperature)
    # Valid chunks always occupy the first A slots. Keep one harmless sentinel
    # target for empty examples so softmax never receives an all-masked row;
    # empty query rows are excluded immediately afterwards.
    valid_targets = valid_mask.clone()
    valid_targets[:, 0] = True
    logits = logits.masked_fill(
        ~valid_targets[:, None, :],
        torch.finfo(logits.dtype).min,
    )
    expected = torch.arange(slot_count, device=device).unsqueeze(0).expand(batch, -1)
    per_slot_ce = F.cross_entropy(
        logits.reshape(batch * slot_count, slot_count),
        expected.reshape(-1),
        reduction="none",
    ).view(batch, slot_count)
    contrastive_loss = per_slot_ce[valid_mask].mean()
    correct = logits.argmax(dim=-1).eq(expected)[valid_mask].float().mean().detach()
    # Equal weighting keeps the absolute scale stable when K changes.
    loss = 0.5 * (cosine_loss + contrastive_loss)
    return {
        "loss": loss,
        "cosine": cosine.detach(),
        "accuracy": correct,
        "valid_slots": valid_count,
    }
