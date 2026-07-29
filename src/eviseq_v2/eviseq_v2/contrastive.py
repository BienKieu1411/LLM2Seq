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


def mine_hard_negatives(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
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

    Returns:
        hard_neg_indices: list of [K_i] tensors, one per batch item
        positive_indices: list of [P_i] tensors, one per batch item
    """
    batch_size = query.shape[0]
    unit_count = keys.shape[1]

    # Detach everything for mining — no gradient through TopK selection
    q_det = query.detach()  # [B, P]
    k_det = keys.detach()  # [B, U, P]

    # Similarity: [B, U]
    similarity = torch.bmm(k_det, q_det.unsqueeze(-1)).squeeze(-1)  # [B, U]

    hard_neg_indices: List[torch.Tensor] = []
    positive_indices: List[torch.Tensor] = []

    for b in range(batch_size):
        width = min(unit_count, evidence_labels.shape[1])
        v = valid_units[b, :width]
        el = evidence_labels[b, :width]

        # Positive: valid & label > 0.5
        pos_mask = v & el.gt(0.5)
        # Negative: valid & label == 0 (not invalid -1)
        neg_mask = v & el.ge(0) & el.le(0.5)

        pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
        neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(-1)

        positive_indices.append(pos_idx)

        if neg_idx.numel() == 0 or pos_idx.numel() == 0:
            # No valid negatives or positives — return empty
            hard_neg_indices.append(neg_idx[:0])
            continue

        # Similarity scores for negatives
        neg_sims = similarity[b, neg_idx]

        # False-positive mining boost: if salience predicts high but gold=0
        if salience_logits is not None:
            sal_width = min(salience_logits.shape[1], width)
            if sal_width > 0:
                sal = torch.sigmoid(salience_logits[b, :sal_width].float())
                # Boost score for false positives (high salience + negative gold)
                for i, nidx in enumerate(neg_idx):
                    if nidx.item() < sal_width and sal[nidx].item() > 0.5:
                        # Add small boost to make false positives more likely selected
                        neg_sims[i] = neg_sims[i] + 0.1

        # Select top-K hardest negatives
        k = min(num_hard_negatives, neg_idx.numel())
        _, topk_local = neg_sims.topk(k, dim=0)
        hard_neg_indices.append(neg_idx[topk_local])

    return hard_neg_indices, positive_indices


def evidence_info_nce_loss(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    temperature: float = 0.07,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Within-document evidence InfoNCE loss.

    For each document:
    - Positive: mean of evidence sentence key representations
    - Negatives: hard-mined non-evidence sentence representations

    L_evi = -log [ exp(sim(q, k+)/τ) / (exp(sim(q, k+)/τ) + Σ exp(sim(q, k_j-)/τ)) ]

    Args:
        query: [B, P] normalized summary query
        keys: [B, U, P] normalized sentence keys
        evidence_labels: [B, U] with 1.0=positive, 0.0=negative, -1.0=invalid
        valid_units: [B, U] boolean mask for valid units
        temperature: contrastive temperature
        num_hard_negatives: number of hard negatives per example
        salience_logits: [B, U] optional for false-positive mining

    Returns:
        Dict with: loss, top1_accuracy, positive_similarity, hard_negative_similarity,
                    similarity_gap, valid_examples
    """
    if temperature <= 0.0:
        raise ValueError("evidence contrastive temperature must be positive")

    device = query.device
    zero = query.sum() * 0.0

    # Mine hard negatives
    hard_neg_idx, pos_idx = mine_hard_negatives(
        query,
        keys,
        evidence_labels,
        valid_units,
        num_hard_negatives=num_hard_negatives,
        salience_logits=salience_logits,
    )

    losses: List[torch.Tensor] = []
    pos_sims: List[float] = []
    neg_sims: List[float] = []
    correct_count = 0
    valid_count = 0

    for b in range(query.shape[0]):
        p_idx = pos_idx[b]
        n_idx = hard_neg_idx[b]

        if p_idx.numel() == 0 or n_idx.numel() == 0:
            continue

        valid_count += 1

        # k+ = norm(mean(k_i for i in E+))
        k_pos = keys[b, p_idx]  # [|E+|, P]
        k_pos_pooled = F.normalize(k_pos.mean(dim=0, keepdim=True), dim=-1)  # [1, P]

        # k_j- for j in E-_hard
        k_neg = keys[b, n_idx]  # [K, P]

        # Similarities
        q_b = query[b].unsqueeze(0)  # [1, P]
        sim_pos = (q_b * k_pos_pooled).sum(dim=-1) / temperature  # [1]
        sim_neg = (q_b * k_neg).sum(dim=-1) / temperature  # [K]

        # InfoNCE: -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
        logits = torch.cat([sim_pos, sim_neg], dim=0)  # [1 + K]
        target = torch.zeros(1, dtype=torch.long, device=device)
        loss_b = F.cross_entropy(logits.unsqueeze(0), target)
        losses.append(loss_b)

        # Diagnostics (detached)
        with torch.no_grad():
            raw_sim_pos = (q_b * k_pos_pooled).sum(dim=-1).item()
            raw_sim_neg = (q_b * k_neg).sum(dim=-1).mean().item()
            pos_sims.append(raw_sim_pos)
            neg_sims.append(raw_sim_neg)
            if logits.argmax().item() == 0:
                correct_count += 1

    if not losses:
        return {
            "evidence_contrastive_loss": zero,
            "evidence_top1_accuracy": zero.detach(),
            "positive_similarity": zero.detach(),
            "hard_negative_similarity": zero.detach(),
            "evidence_similarity_gap": zero.detach(),
            "evidence_valid_examples": zero.detach(),
        }

    loss = torch.stack(losses).mean()
    mean_pos_sim = sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
    mean_neg_sim = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0

    return {
        "evidence_contrastive_loss": loss,
        "evidence_top1_accuracy": loss.new_tensor(correct_count / max(1, valid_count)).detach(),
        "positive_similarity": loss.new_tensor(mean_pos_sim).detach(),
        "hard_negative_similarity": loss.new_tensor(mean_neg_sim).detach(),
        "evidence_similarity_gap": loss.new_tensor(mean_pos_sim - mean_neg_sim).detach(),
        "evidence_valid_examples": loss.new_tensor(valid_count).detach(),
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise margin ranking loss over candidate summaries.

    For each pair (i, j) where Q(y_i) > Q(y_j):
      L_ij = max(0, f(y_j) - f(y_i) + m_ij)
    where m_ij = (rank_j - rank_i) * margin

    Args:
        scores: [N] length-normalized log probabilities from model
        quality_scores: [N] external quality scores (e.g. weighted ROUGE)
        margin: base margin per rank distance
        minimum_quality_gap: skip pairs with quality difference below this

    Returns:
        loss: scalar pairwise ranking loss
        pair_accuracy: fraction of correctly ordered pairs
    """
    n = scores.shape[0]
    if n < 2:
        zero = scores.sum() * 0.0
        return zero, zero.detach()

    # Sort by quality (descending) and get rank ordering
    sorted_indices = quality_scores.argsort(descending=True)
    sorted_scores = scores[sorted_indices]
    sorted_quality = quality_scores[sorted_indices]

    losses: List[torch.Tensor] = []
    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            # quality[i] >= quality[j] by sort order
            quality_gap = (sorted_quality[i] - sorted_quality[j]).item()
            if quality_gap < minimum_quality_gap:
                continue

            rank_distance = j - i
            m_ij = float(rank_distance) * margin
            # f(y_i) should be > f(y_j), so loss = max(0, f(y_j) - f(y_i) + m)
            pair_loss = F.relu(sorted_scores[j] - sorted_scores[i] + m_ij)
            losses.append(pair_loss)
            total += 1

            with torch.no_grad():
                if sorted_scores[i].item() > sorted_scores[j].item():
                    correct += 1

    if not losses:
        zero = scores.sum() * 0.0
        return zero, zero.new_tensor(0.0).detach()

    loss = torch.stack(losses).mean()
    accuracy = loss.new_tensor(correct / max(1, total)).detach()
    return loss, accuracy
