"""Evidence-focused hard contrastive learning for EviSeq V2.

This module replaces the document-level InfoNCE with within-document
evidence InfoNCE.  The key difference: positives are oracle evidence
sentences and negatives are semantically similar but non-evidence
sentences from the *same* document.  This forces the encoder/bridge
to distinguish "sounds related" from "actually needs summarising".

The original document-level InfoNCE code is preserved but disabled
by default (use_contrastive=False in config).

References:
- Focus-Driven Contrastive Learning for medical question summarisation
- BRIO sequence-level ranking (separate mechanism, see model.py)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pooling utilities (unchanged from EviSeq V1)
# ---------------------------------------------------------------------------


def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid source tokens in FP32.

    Long-document runs pool up to 4,096 BF16 states.  Accumulating before the
    cast loses avoidable precision exactly at the representation consumed by
    InfoNCE, so only the inexpensive pooled path is promoted to FP32.
    """

    expanded = mask.unsqueeze(-1).float()
    states = hidden_states.float()
    return (states * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


def masked_last_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the final valid source token."""

    if hidden_states.ndim != 3 or mask.ndim != 2 or hidden_states.shape[:2] != mask.shape:
        raise ValueError("masked_last_pool expects [B, S, D] states and a matching [B, S] mask")
    if not bool(mask.bool().any(dim=1).all()):
        raise ValueError("Every source example must contain at least one valid token")
    positions = mask.long().sum(dim=1) - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions]


def last_prompt_states(decoder_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather the state predicting the first supervised target token.

    At this position the shifted decoder input still contains the final fixed
    prompt token.  No reference-summary token is visible to the query.
    """

    if decoder_states.ndim != 3 or labels.ndim != 2 or decoder_states.shape[:2] != labels.shape:
        raise ValueError("decoder_states and labels must be matching [B, T, ...] tensors")
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain at least one supervised target token")
    positions = supervised.long().argmax(dim=1)
    rows = torch.arange(decoder_states.shape[0], device=decoder_states.device)
    return decoder_states[rows, positions]


def exact_duplicate_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mark off-diagonal examples with byte-identical tokenized sources.

    Duplicate sources are alternative positives, not valid InfoNCE negatives.
    The comparison is intentionally exact; near-duplicate mining would add a
    second model and an unaudited threshold to the training objective.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be matching [B, S] tensors")
    valid_ids = input_ids.masked_fill(~attention_mask.bool(), 0)
    same_ids = valid_ids[:, None, :].eq(valid_ids[None, :, :]).all(dim=-1)
    same_mask = attention_mask[:, None, :].eq(attention_mask[None, :, :]).all(dim=-1)
    duplicates = same_ids & same_mask
    duplicates.fill_diagonal_(False)
    return duplicates


# ---------------------------------------------------------------------------
# Original document-level alignment head (kept for backward compatibility)
# ---------------------------------------------------------------------------


class SourcePromptAlignmentHead(nn.Module):
    """Project source memory and target-free decoder prompts to one space."""

    def __init__(self, hidden_size: int, projection_size: int = 256, pooling: str = "mean_last"):
        super().__init__()
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "mean_last"}:
            raise ValueError("contrastive pooling must be 'mean' or 'mean_last'")
        self.pool_gate: Optional[nn.Parameter]
        if self.pooling == "mean_last":
            self.pool_gate = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter("pool_gate", None)
        self.source_projection = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        self.prompt_projection = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        nn.init.xavier_uniform_(self.source_projection[-1].weight)
        nn.init.xavier_uniform_(self.prompt_projection[-1].weight)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        prompt_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if memory.ndim != 3:
            raise ValueError("EviSeq contrastive learning expects one [B, S, D] source memory")
        source = masked_mean_pool(memory, memory_mask)
        if self.pool_gate is not None:
            last = masked_last_pool(memory, memory_mask)
            last_weight = torch.sigmoid(self.pool_gate.float()).to(source.dtype)
            source = (1.0 - last_weight) * source + last_weight * last
        if prompt_states.ndim != 2:
            raise ValueError("prompt_states must be [B, D]")
        return {
            "source_repr": F.normalize(self.source_projection(source).float(), dim=-1),
            "prompt_repr": F.normalize(self.prompt_projection(prompt_states).float(), dim=-1),
        }


def info_nce_loss(
    source_repr: torch.Tensor,
    prompt_repr: torch.Tensor,
    temperature: float,
    duplicate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric in-batch InfoNCE and source-retrieval accuracy."""

    if source_repr.shape != prompt_repr.shape or source_repr.ndim != 2:
        raise ValueError("source_repr and prompt_repr must be matching [B, D] tensors")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    batch_size = source_repr.shape[0]
    if batch_size <= 1:
        zero = source_repr.sum() * 0.0
        return zero, zero.detach()
    logits = prompt_repr @ source_repr.T / float(temperature)
    if duplicate_mask is not None:
        if duplicate_mask.shape != (batch_size, batch_size):
            raise ValueError("duplicate_mask must be [B, B]")
        logits = logits.masked_fill(duplicate_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(batch_size, device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    accuracy = logits.detach().argmax(dim=1).eq(labels).float().mean()
    return loss, accuracy


# ---------------------------------------------------------------------------
# NEW: Evidence-focused hard contrastive learning
# ---------------------------------------------------------------------------


def decoder_summary_representation(
    decoder_states: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool decoder hidden states over supervised (non-padding) target tokens.

    During teacher forcing the decoder produces hidden states for every target
    position.  We pool only the positions where labels != -100 (i.e. actual
    summary tokens, excluding the fixed prompt prefix).

    Returns: [B, D] summary representation in FP32.
    """

    if decoder_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("decoder_states must be [B, T, D] and labels [B, T]")
    if decoder_states.shape[:2] != labels.shape:
        raise ValueError("decoder_states and labels must have matching [B, T]")

    supervised = labels.ne(-100).float()  # [B, T]
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain at least one supervised target token")

    # [B, T, 1] mask
    mask = supervised.unsqueeze(-1)
    states_fp32 = decoder_states.float()
    pooled = (states_fp32 * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled  # [B, D]


class EvidenceContrastiveHead(nn.Module):
    """Projection heads for within-document evidence contrastive learning.

    Projects:
    - Decoder summary representation → query q = norm(W_q @ z_y)
    - Encoder sentence representations → keys k_i = norm(W_h @ h_i)

    These projection heads are training-only; they do not participate in
    inference.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        projection_size: int = 256,
    ):
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.RMSNorm(decoder_hidden_size),
            nn.Linear(decoder_hidden_size, projection_size, bias=False),
        )
        self.key_projection = nn.Sequential(
            nn.RMSNorm(encoder_hidden_size),
            nn.Linear(encoder_hidden_size, projection_size, bias=False),
        )
        nn.init.xavier_uniform_(self.query_projection[-1].weight)
        nn.init.xavier_uniform_(self.key_projection[-1].weight)

    def forward(
        self,
        summary_repr: torch.Tensor,
        sentence_reprs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize summary query and sentence keys.

        Args:
            summary_repr: [B, D_dec] decoder summary representation
            sentence_reprs: [B, U, D_enc] encoder sentence-level representations

        Returns:
            q: [B, P] normalized query
            keys: [B, U, P] normalized keys
        """
        if summary_repr.ndim != 2:
            raise ValueError(f"summary_repr must be [B, D], got {summary_repr.shape}")
        if sentence_reprs.ndim != 3:
            raise ValueError(f"sentence_reprs must be [B, U, D], got {sentence_reprs.shape}")

        q = F.normalize(self.query_projection(summary_repr).float(), dim=-1)  # [B, P]
        keys = F.normalize(self.key_projection(sentence_reprs).float(), dim=-1)  # [B, U, P]
        return q, keys


def _evidence_masks_and_hard_negatives(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
    salience_boost: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive and hard-negative masks without per-example CPU syncs.

    Mining is deliberately detached: Top-K chooses the competing sentences,
    while the final contrastive similarities remain differentiable.  A
    continuous salience boost prioritises current false positives instead of
    the old thresholded Python loop.
    """

    if query.ndim != 2 or keys.ndim != 3 or query.shape[0] != keys.shape[0]:
        raise ValueError("query must be [B,P] and keys must be [B,U,P]")
    if evidence_labels.ndim != 2 or valid_units.ndim != 2:
        raise ValueError("evidence labels and valid units must be [B,U]")
    if num_hard_negatives <= 0:
        raise ValueError("num_hard_negatives must be positive")
    if salience_boost < 0.0:
        raise ValueError("salience_boost must be non-negative")

    width = min(keys.shape[1], evidence_labels.shape[1], valid_units.shape[1])
    labels = evidence_labels[:, :width]
    valid = valid_units[:, :width].bool()
    positive_mask = valid & labels.gt(0.5)
    negative_mask = valid & labels.ge(0.0) & labels.le(0.5)

    detached_similarity = torch.einsum(
        "bup,bp->bu",
        keys[:, :width].detach(),
        query.detach(),
    )
    hard_score = detached_similarity
    if salience_logits is not None and salience_boost > 0.0:
        salience_width = min(width, salience_logits.shape[1])
        salience_probability = torch.zeros_like(hard_score)
        salience_probability[:, :salience_width] = torch.sigmoid(
            salience_logits[:, :salience_width].detach().float()
        ).to(hard_score.dtype)
        hard_score = hard_score + float(salience_boost) * salience_probability

    hard_score = hard_score.masked_fill(~negative_mask, -torch.inf)
    hard_negative_mask = torch.zeros_like(negative_mask)
    if width > 0:
        k = min(int(num_hard_negatives), width)
        top_values, top_indices = hard_score.topk(k, dim=1)
        selected_is_valid = torch.isfinite(top_values)
        hard_negative_mask.scatter_(1, top_indices, selected_is_valid)
    return positive_mask, hard_negative_mask, detached_similarity


def mine_hard_negatives(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
    salience_boost: float = 0.1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Mine hard negatives within each document independently.

    For each document in the batch:
    1. Identify positive (evidence) and negative (non-evidence) sentences
    2. Compute similarity between detached query and detached keys
    3. Select top-K negatives by similarity (hardest negatives)
    4. Optionally include false-positive mining (high salience + gold=0)

    Args:
        query: [B, P] normalized summary query (will be detached)
        keys: [B, U, P] normalized sentence keys (will be detached)
        evidence_labels: [B, U] with 1.0=positive, 0.0=negative, -1.0=invalid
        valid_units: [B, U] boolean mask for valid units
        num_hard_negatives: number of hard negatives to mine per example
        salience_logits: [B, U] optional salience predictions for false-positive mining
        salience_boost: continuous boost assigned to predicted-salient negatives

    Returns:
        hard_neg_indices: list of [K_i] tensors, one per batch item
        positive_indices: list of [P_i] tensors, one per batch item
    """
    positive_mask, hard_negative_mask, _ = _evidence_masks_and_hard_negatives(
        query,
        keys,
        evidence_labels,
        valid_units,
        num_hard_negatives=num_hard_negatives,
        salience_logits=salience_logits,
        salience_boost=salience_boost,
    )
    hard_neg_indices = [row.nonzero(as_tuple=False).squeeze(-1) for row in hard_negative_mask]
    positive_indices = [row.nonzero(as_tuple=False).squeeze(-1) for row in positive_mask]
    return hard_neg_indices, positive_indices


def evidence_info_nce_loss(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    temperature: float = 0.07,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
    salience_boost: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Vectorised multi-positive, within-document hard evidence InfoNCE.

    For each document:
    - Positives: every oracle-evidence sentence, kept as a separate positive
    - Negatives: hard-mined non-evidence sentence representations

    Each positive is contrasted against the same within-document hard-negative
    set, then losses are averaged.  Keeping positives separate avoids the
    cancellation and single-evidence shortcut caused by mean-pooling them.

    Args:
        query: [B, P] normalized summary query
        keys: [B, U, P] normalized sentence keys
        evidence_labels: [B, U] with 1.0=positive, 0.0=negative, -1.0=invalid
        valid_units: [B, U] boolean mask for valid units
        temperature: contrastive temperature
        num_hard_negatives: number of hard negatives per example
        salience_logits: [B, U] optional for false-positive mining
        salience_boost: continuous hard-mining boost for predicted false positives

    Returns:
        Dict with: loss, top1_accuracy, positive_similarity, hard_negative_similarity,
                    similarity_gap, valid_examples
    """
    if temperature <= 0.0:
        raise ValueError("evidence contrastive temperature must be positive")

    zero = query.sum() * 0.0
    positive_mask, hard_negative_mask, _ = _evidence_masks_and_hard_negatives(
        query,
        keys,
        evidence_labels,
        valid_units,
        num_hard_negatives=num_hard_negatives,
        salience_logits=salience_logits,
        salience_boost=salience_boost,
    )

    width = positive_mask.shape[1]
    similarity = torch.einsum("bup,bp->bu", keys[:, :width], query)
    logits = similarity / float(temperature)
    negative_logits = logits.masked_fill(~hard_negative_mask, -torch.inf)
    negative_logsumexp = torch.logsumexp(negative_logits, dim=1)

    positive_count = positive_mask.sum(dim=1)
    negative_count = hard_negative_mask.sum(dim=1)
    valid_examples = positive_count.gt(0) & negative_count.gt(0)
    safe_negative_logsumexp = torch.where(valid_examples, negative_logsumexp, torch.zeros_like(negative_logsumexp))

    # One-vs-hard-negative-set NCE for every positive. Unlike putting all
    # positives in one softmax denominator, positives never compete with one
    # another and the optimum remains zero regardless of evidence count.
    positive_losses = F.softplus(safe_negative_logsumexp.unsqueeze(1) - logits)
    loss_per_example = (positive_losses * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    valid_float = valid_examples.to(loss_per_example.dtype)
    valid_count = valid_float.sum()
    loss = (loss_per_example * valid_float).sum() / valid_count.clamp_min(1.0)
    loss = loss + zero

    with torch.no_grad():
        positive_float = positive_mask.to(similarity.dtype)
        negative_float = hard_negative_mask.to(similarity.dtype)
        positive_similarity_per_example = (similarity * positive_float).sum(dim=1) / positive_count.clamp_min(1)
        negative_similarity_per_example = (similarity * negative_float).sum(dim=1) / negative_count.clamp_min(1)
        mean_pos_sim = (positive_similarity_per_example * valid_float).sum() / valid_count.clamp_min(1.0)
        mean_neg_sim = (negative_similarity_per_example * valid_float).sum() / valid_count.clamp_min(1.0)
        hardest_negative = negative_logits.max(dim=1).values
        positive_beats_hardest = positive_mask & logits.gt(hardest_negative.unsqueeze(1))
        accuracy_per_example = positive_beats_hardest.sum(dim=1).float() / positive_count.clamp_min(1)
        accuracy = (accuracy_per_example * valid_float).sum() / valid_count.clamp_min(1.0)

    return {
        "evidence_contrastive_loss": loss,
        "evidence_top1_accuracy": accuracy.detach(),
        "positive_similarity": mean_pos_sim.detach(),
        "hard_negative_similarity": mean_neg_sim.detach(),
        "evidence_similarity_gap": (mean_pos_sim - mean_neg_sim).detach(),
        "evidence_valid_examples": valid_count.detach(),
    }


# ---------------------------------------------------------------------------
# Candidate ranking utilities (BRIO-like)
# ---------------------------------------------------------------------------


def length_normalized_log_prob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute length-normalized log probability of a candidate sequence.

    f(y) = sum_t log P(y_t | x, y_{<t}) / L^alpha

    Args:
        logits: [T, V] logits from teacher forcing
        labels: [T] target token ids (-100 for ignored positions)
        alpha: length penalty exponent

    Returns:
        scalar length-normalized log probability
    """
    supervised = labels.ne(-100)
    if not bool(supervised.any()):
        return logits.sum() * 0.0

    valid_logits = logits[supervised].float()
    valid_labels = labels[supervised]
    log_probs = F.log_softmax(valid_logits, dim=-1)
    token_log_probs = log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
    total_log_prob = token_log_probs.sum()
    length = supervised.sum().float()
    return total_log_prob / length.pow(alpha)


def pairwise_ranking_loss(
    scores: torch.Tensor,
    quality_scores: torch.Tensor,
    margin: float = 0.01,
    minimum_quality_gap: float = 0.5,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise margin ranking loss over candidate summaries.

    For each pair (i, j) where Q(y_i) > Q(y_j):
      L_ij = max(0, f(y_j) - f(y_i) + m_ij)
    where m_ij = (rank_j - rank_i) * margin

    Args:
        scores: [N] or [B, N] length-normalized model log probabilities
        quality_scores: same shape, external quality (e.g. weighted ROUGE)
        margin: base margin per rank distance
        minimum_quality_gap: skip pairs with quality difference below this

    Returns:
        loss, pair_accuracy, valid_pair_count
    """
    if scores.shape != quality_scores.shape or scores.ndim not in {1, 2}:
        raise ValueError("scores and quality_scores must be matching [N] or [B,N] tensors")
    squeeze = scores.ndim == 1
    if squeeze:
        scores = scores.unsqueeze(0)
        quality_scores = quality_scores.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    elif valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must match scores")
    valid_mask = valid_mask.bool() & torch.isfinite(scores) & torch.isfinite(quality_scores)

    batch_size, count = scores.shape
    if count < 2:
        zero = scores.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    # Invalid/padded candidates sort to the end and are masked from all pairs.
    sortable_quality = quality_scores.masked_fill(~valid_mask, -torch.inf)
    order = sortable_quality.argsort(dim=1, descending=True)
    sorted_scores = scores.gather(1, order)
    sorted_quality = quality_scores.gather(1, order)
    sorted_valid = valid_mask.gather(1, order)

    upper = torch.triu(
        torch.ones((count, count), device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    quality_gap = sorted_quality[:, :, None] - sorted_quality[:, None, :]
    pair_mask = (
        upper.unsqueeze(0)
        & sorted_valid[:, :, None]
        & sorted_valid[:, None, :]
        & quality_gap.ge(float(minimum_quality_gap))
    )
    ranks = torch.arange(count, device=scores.device, dtype=scores.dtype)
    rank_distance = ranks[None, :] - ranks[:, None]
    margins = float(margin) * rank_distance.clamp_min(0)
    pair_losses = F.relu(sorted_scores[:, None, :] - sorted_scores[:, :, None] + margins.unsqueeze(0))

    # Give every document equal weight, while excluding documents without a
    # usable quality-separated pair instead of diluting the objective with 0.
    pairs_per_document = pair_mask.sum(dim=(1, 2))
    valid_documents = pairs_per_document.gt(0)
    loss_per_document = (pair_losses * pair_mask).sum(dim=(1, 2)) / pairs_per_document.clamp_min(1)
    valid_float = valid_documents.to(loss_per_document.dtype)
    valid_document_count = valid_float.sum()
    loss = (loss_per_document * valid_float).sum() / valid_document_count.clamp_min(1.0)
    loss = loss + scores.sum() * 0.0

    with torch.no_grad():
        correct = (sorted_scores[:, :, None] > sorted_scores[:, None, :]) & pair_mask
        correct_per_document = correct.sum(dim=(1, 2)).float() / pairs_per_document.clamp_min(1)
        accuracy = (correct_per_document * valid_float).sum() / valid_document_count.clamp_min(1.0)
        pair_count = pairs_per_document.sum().to(loss.dtype)
    return loss, accuracy.detach(), pair_count.detach()
