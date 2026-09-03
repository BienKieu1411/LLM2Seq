"""Contrastive objectives for source, prompt and sentence evidence alignment."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def pairwise_geometry_preservation_loss(
    source_units: torch.Tensor,
    projected_units: torch.Tensor,
    valid_units: torch.Tensor,
) -> torch.Tensor:
    """Preserve relative source-unit geometry across an encoder bridge.

    The PPLX and Qwen hidden spaces are not assumed to share coordinates, so
    matching each projected vector to its raw input would over-constrain the
    bridge.  Instead, the detached source-space pairwise cosine matrix is the
    target and the projected bridge-space matrix receives the gradient.  A
    document with fewer than two visible units contributes an exact zero.
    """

    if source_units.ndim != 3 or projected_units.ndim != 3 or valid_units.ndim != 2:
        raise ValueError("geometry inputs must be [B,U,D], [B,U,D'] and [B,U]")
    if source_units.shape[:2] != projected_units.shape[:2] or source_units.shape[:2] != valid_units.shape:
        raise ValueError("geometry inputs must agree on batch and unit dimensions")
    source = F.normalize(source_units.float().detach(), dim=-1)
    projected = F.normalize(projected_units.float(), dim=-1)
    source_sim = torch.matmul(source, source.transpose(1, 2))
    projected_sim = torch.matmul(projected, projected.transpose(1, 2))
    valid = valid_units.bool()
    pair_mask = valid[:, :, None] & valid[:, None, :]
    pair_mask &= ~torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)[None]
    if not bool(pair_mask.any()):
        return projected_units.float().sum() * 0.0
    return (source_sim - projected_sim).abs()[pair_mask].mean()


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

    Duplicate sources are ignored as negatives rather than treated as
    alternative positives, because their target summaries may differ.
    The equality check is intentionally exact; near-duplicate mining would add a
    second model and an extra threshold to the training objective.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be matching [B, S] tensors")
    valid_ids = input_ids.masked_fill(~attention_mask.bool(), 0)
    same_ids = valid_ids[:, None, :].eq(valid_ids[None, :, :]).all(dim=-1)
    same_mask = attention_mask[:, None, :].eq(attention_mask[None, :, :]).all(dim=-1)
    duplicates = same_ids & same_mask
    duplicates.fill_diagonal_(False)
    return duplicates


def source_memory_for_mining(
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    *,
    pooling: str = "mean_last",
) -> torch.Tensor:
    """Build a target-free normalized source vector for hard-negative mining."""

    if memory.ndim != 3 or memory_mask.ndim != 2 or memory.shape[:2] != memory_mask.shape:
        raise ValueError("source memory mining expects [B,S,D] memory and a matching [B,S] mask")
    if pooling == "mean":
        pooled = masked_mean_pool(memory, memory_mask)
    elif pooling == "mean_last":
        pooled = 0.5 * (masked_mean_pool(memory, memory_mask) + masked_last_pool(memory, memory_mask))
    else:
        raise ValueError("source mining pooling must be 'mean' or 'mean_last'")
    return F.normalize(pooled.float(), dim=-1)


@torch.no_grad()
def hard_negative_indices(source_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the most similar non-self source in a physical batch."""

    if source_repr.ndim != 2 or source_repr.shape[0] <= 1:
        raise ValueError("Hard-negative mining requires [B,D] with B > 1")
    normalized = F.normalize(source_repr.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    indices = similarity.argmax(dim=1)
    rows = torch.arange(similarity.shape[0], device=similarity.device)
    return indices, similarity[rows, indices]


def per_example_nll(
    supervised_logits: torch.Tensor,
    labels: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    """Reduce flattened teacher-forced token losses to one NLL per example."""

    if supervised_logits.ndim != 2 or labels.ndim != 2 or supervised.shape != labels.shape:
        raise ValueError("Expected flattened logits plus [B,T] labels/supervision mask")
    expected = int(supervised.sum().item())
    if supervised_logits.shape[0] != expected:
        raise ValueError(f"Expected {expected} supervised logits, received {supervised_logits.shape[0]}")
    token_losses = F.cross_entropy(supervised_logits.float(), labels[supervised], reduction="none")
    rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(labels)[supervised]
    totals = token_losses.new_zeros(labels.shape[0])
    totals.scatter_add_(0, rows, token_losses)
    counts = supervised.sum(dim=1).to(token_losses.dtype).clamp_min(1.0)
    return totals / counts


def source_swap_contrastive_loss(
    positive_nll: torch.Tensor,
    negative_nll: torch.Tensor,
    *,
    margin: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Require the correct source to explain the same target better than a hard swap."""

    if positive_nll.shape != negative_nll.shape:
        raise ValueError("positive_nll and negative_nll must have equal shapes")
    if temperature <= 0.0:
        raise ValueError("source-swap temperature must be positive")
    logits = (positive_nll - negative_nll + float(margin)) / float(temperature)
    return F.softplus(logits).mean()


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
        prompt_states = prompt_states.float()
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

    supervised = labels.ne(-100).float()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain at least one supervised target token")

    mask = supervised.unsqueeze(-1)
    states_fp32 = decoder_states.float()
    pooled = (states_fp32 * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled


def decoder_sentence_representations(
    decoder_states: torch.Tensor,
    labels: torch.Tensor,
    sentence_ids: torch.Tensor,
    sentence_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool teacher-forced decoder states separately for every target sentence.

    Each reference-summary sentence receives an independent evidence query.
    Sentence ids are aligned with decoder labels by the dataset and use ``0``
    for the fixed prompt.
    """

    if decoder_states.ndim != 3 or labels.ndim != 2 or sentence_ids.ndim != 2:
        raise ValueError("decoder states, labels, and sentence ids must be [B, T, ...]/[B, T]")
    if decoder_states.shape[:2] != labels.shape or labels.shape != sentence_ids.shape:
        raise ValueError("decoder states, labels, and sentence ids must have matching batch/time dimensions")
    if sentence_count <= 0:
        raise ValueError("sentence_count must be positive")

    supervised = labels.ne(-100) & sentence_ids.gt(0)
    sentence_index = sentence_ids.clamp(min=1, max=sentence_count).long() - 1
    states = decoder_states.float()
    pooled = states.new_zeros((states.shape[0], sentence_count, states.shape[-1]))
    counts = states.new_zeros((states.shape[0], sentence_count))
    weighted = states * supervised.unsqueeze(-1).to(states.dtype)
    pooled.scatter_add_(1, sentence_index.unsqueeze(-1).expand_as(weighted), weighted)
    counts.scatter_add_(1, sentence_index, supervised.to(states.dtype))
    valid = counts.gt(0)
    return pooled / counts.unsqueeze(-1).clamp_min(1.0), valid


class EvidenceContrastiveHead(nn.Module):
    """Projection heads for within-document evidence contrastive learning.

    Projects:
    - Decoder summary representation → query q = norm(W_q @ z_y)
    - Bridge-memory sentence representations → keys k_i = norm(W_h @ h_i)

    These projection heads are training-only; they do not participate in
    inference.
    """

    def __init__(
        self,
        key_hidden_size: int,
        decoder_hidden_size: int,
        projection_size: int = 256,
    ):
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.RMSNorm(decoder_hidden_size),
            nn.Linear(decoder_hidden_size, projection_size, bias=False),
        )
        self.key_projection = nn.Sequential(
            nn.RMSNorm(key_hidden_size),
            nn.Linear(key_hidden_size, projection_size, bias=False),
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
            sentence_reprs: [B, U, D_key] bridge-memory sentence representations

        Returns:
            q: [B, P] normalized query
            keys: [B, U, P] normalized keys
        """
        if summary_repr.ndim != 2:
            raise ValueError(f"summary_repr must be [B, D], got {summary_repr.shape}")
        if sentence_reprs.ndim != 3:
            raise ValueError(f"sentence_reprs must be [B, U, D], got {sentence_reprs.shape}")

        q = F.normalize(self.query_projection(summary_repr.float()).float(), dim=-1)
        keys = F.normalize(self.key_projection(sentence_reprs.float()).float(), dim=-1)
        return q, keys


class PromptConditionedEvidenceHead(EvidenceContrastiveHead):
    """Evidence query formed from an inference-available prompt/source state.

    A decoder prompt state alone can be weak because every example shares the
    same instruction.  Conversely, a global mean of source memory alone
    ignores *which* generation task is requested.  This head keeps both
    signals in a common contrastive space:

    ``q = normalize(W_prompt(h_prompt) + sigmoid(g) * W_source(mean(U)))``,
    where ``U`` is the set of pooled, visible source units.  Passing an
    equal-weight unit mean rather than a token mean is the caller's contract:
    it prevents a long sentence from silently dominating the document context
    of an otherwise unit-invariant evidence bridge.

    ``h_prompt`` is the decoder state immediately before the first generated
    summary token, and ``M`` is the same bridge memory consumed by decoder
    cross-attention.  Neither contains a reference-summary token, so this is
    available under the same conditioning at greedy inference.  In the
    static PCEB recipe the head is training-only.  DualBridge additionally
    uses it in one short prompt prefill to construct a *single fused* source
    prior before ordinary greedy decoding; it is not a second encoder,
    decoder, reranker, or generation pass.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 256,
        context_gate_init: float = 0.5,
    ):
        if not 0.0 < float(context_gate_init) < 1.0:
            raise ValueError("prompt-context gate init must be in (0, 1)")
        super().__init__(
            key_hidden_size=hidden_size,
            decoder_hidden_size=hidden_size,
            projection_size=projection_size,
        )
        self.source_projection = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, projection_size, bias=False),
        )
        nn.init.xavier_uniform_(self.source_projection[-1].weight)
        initial_logit = torch.logit(torch.tensor(float(context_gate_init), dtype=torch.float32))
        self.context_gate_logit = nn.Parameter(initial_logit)

    def context_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.context_gate_logit.float())

    def forward(
        self,
        prompt_state: torch.Tensor,
        source_context: torch.Tensor,
        sentence_reprs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prompt_state.ndim != 2 or source_context.ndim != 2:
            raise ValueError("prompt_state and source_context must be [B, D]")
        if prompt_state.shape != source_context.shape:
            raise ValueError("prompt_state and source_context must share [B, D]")
        prompt = self.query_projection(prompt_state.float()).float()
        source = self.source_projection(source_context.float()).float()
        query = F.normalize(prompt + self.context_gate() * source, dim=-1)
        keys = F.normalize(self.key_projection(sentence_reprs.float()).float(), dim=-1)
        return query, keys


def _evidence_masks_and_hard_negatives(
    query: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
    salience_boost: float = 0.1,
    attention_prior_energy: Optional[torch.Tensor] = None,
    attention_mining_boost: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive and hard-negative masks without per-example CPU syncs.

    Mining is deliberately detached: Top-K chooses the competing sentences,
    while the final contrastive similarities remain differentiable.  A
    continuous salience boost prioritises current false positives without a
    thresholded Python loop.
    """

    if query.ndim != 2 or keys.ndim != 3 or query.shape[0] != keys.shape[0]:
        raise ValueError("query must be [B,P] and keys must be [B,U,P]")
    if evidence_labels.ndim != 2 or valid_units.ndim != 2:
        raise ValueError("evidence labels and valid units must be [B,U]")
    if num_hard_negatives <= 0:
        raise ValueError("num_hard_negatives must be positive")
    if salience_boost < 0.0:
        raise ValueError("salience_boost must be non-negative")
    if attention_mining_boost < 0.0:
        raise ValueError("attention_mining_boost must be non-negative")

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
    if attention_prior_energy is not None and attention_mining_boost > 0.0:
        if attention_prior_energy.ndim != 2 or attention_prior_energy.shape[0] != query.shape[0]:
            raise ValueError("attention_prior_energy must be [B, U]")
        attention_width = min(width, attention_prior_energy.shape[1])
        detached_energy = torch.zeros_like(hard_score)
        detached_energy[:, :attention_width] = (
            attention_prior_energy[:, :attention_width].detach().float().to(hard_score.dtype)
        )
        hard_score = hard_score + float(attention_mining_boost) * detached_energy

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
    attention_prior_energy: Optional[torch.Tensor] = None,
    attention_mining_boost: float = 0.0,
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
        attention_prior_energy=attention_prior_energy,
        attention_mining_boost=attention_mining_boost,
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
    salience_logit_bias: float = 0.0,
    attention_prior_energy: Optional[torch.Tensor] = None,
    attention_mining_boost: float = 0.0,
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
        salience_logit_bias: differentiable coupling coefficient.  With
            ``attention_prior_energy`` it is applied inside the same
            temperature-scaled score as cosine similarity.
        attention_prior_energy: [B, U] optional pre-BF16 relative unit energy
            actually consumed by the bridge.  It must include the bridge
            gate, scale, and the same logit clipping used for SDPA.  This
            makes the contrastive gradient reach the live attention gate as
            well as the unit logits.  Without it, raw-logit coupling is
            retained for compatible runs.
        attention_mining_boost: detached coefficient for aligning hard-negative
            selection with ``attention_prior_energy``. Zero keeps cosine and
            salience-based mining.

    Returns:
        Dict with: loss, top1_accuracy, positive_similarity, hard_negative_similarity,
                    similarity_gap, valid_examples
    """
    if temperature <= 0.0:
        raise ValueError("evidence contrastive temperature must be positive")
    if salience_logit_bias < 0.0:
        raise ValueError("salience_logit_bias must be non-negative")
    if attention_mining_boost < 0.0:
        raise ValueError("attention_mining_boost must be non-negative")

    zero = query.sum() * 0.0
    positive_mask, hard_negative_mask, _ = _evidence_masks_and_hard_negatives(
        query,
        keys,
        evidence_labels,
        valid_units,
        num_hard_negatives=num_hard_negatives,
        salience_logits=salience_logits,
        salience_boost=salience_boost,
        attention_prior_energy=attention_prior_energy,
        attention_mining_boost=attention_mining_boost,
    )

    width = positive_mask.shape[1]
    similarity = torch.einsum("bup,bp->bu", keys[:, :width], query)
    if attention_prior_energy is not None:
        if attention_prior_energy.ndim != 2 or attention_prior_energy.shape[0] != similarity.shape[0]:
            raise ValueError("attention_prior_energy must be [B, U]")
        attention_width = min(width, attention_prior_energy.shape[1])
        aligned_energy = torch.zeros_like(similarity)
        aligned_energy[:, :attention_width] = attention_prior_energy[:, :attention_width].float()
        selected_units = positive_mask | hard_negative_mask
        logits = (
            similarity + float(salience_logit_bias) * aligned_energy * selected_units.to(similarity.dtype)
        ) / float(temperature)
    else:
        logits = similarity / float(temperature)
    if attention_prior_energy is None and salience_logits is not None and salience_logit_bias > 0.0:
        salience_width = min(width, salience_logits.shape[1])
        aligned_salience = torch.zeros_like(logits)
        aligned_salience[:, :salience_width] = salience_logits[:, :salience_width].float().clamp(-4.0, 4.0)
        selected_units = positive_mask | hard_negative_mask
        logits = logits + float(salience_logit_bias) * aligned_salience * selected_units.to(logits.dtype)
    negative_logits = logits.masked_fill(~hard_negative_mask, -torch.inf)
    negative_logsumexp = torch.logsumexp(negative_logits, dim=1)

    positive_count = positive_mask.sum(dim=1)
    negative_count = hard_negative_mask.sum(dim=1)
    valid_examples = positive_count.gt(0) & negative_count.gt(0)
    safe_negative_logsumexp = torch.where(valid_examples, negative_logsumexp, torch.zeros_like(negative_logsumexp))

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


def sentence_evidence_info_nce_loss(
    query: torch.Tensor,
    query_valid: torch.Tensor,
    keys: torch.Tensor,
    evidence_labels: torch.Tensor,
    valid_units: torch.Tensor,
    temperature: float = 0.07,
    num_hard_negatives: int = 4,
    salience_logits: Optional[torch.Tensor] = None,
    salience_boost: float = 0.1,
    salience_logit_bias: float = 0.0,
    attention_prior_energy: Optional[torch.Tensor] = None,
    global_evidence_labels: Optional[torch.Tensor] = None,
    attention_mining_boost: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Sentence-conditioned, attention-aligned evidence InfoNCE.

    ``query`` is one representation per reference-summary sentence and every
    sentence receives its own oracle evidence labels.  Hard-negative *choice*
    is detached, but the optional salience term remains differentiable in the
    final contrastive score.  A unit that is evidence for *another* target
    sentence never receives a contradictory negative salience gradient: it is
    still a q/k hard negative, but the global attention-bias term is masked.
    """

    if temperature <= 0.0:
        raise ValueError("evidence contrastive temperature must be positive")
    if salience_logit_bias < 0.0:
        raise ValueError("salience_logit_bias must be non-negative")
    if attention_mining_boost < 0.0:
        raise ValueError("attention_mining_boost must be non-negative")
    if query.ndim != 3 or keys.ndim != 3:
        raise ValueError("sentence evidence contrastive expects query [B,R,P] and keys [B,U,P]")
    if query_valid.shape != query.shape[:2]:
        raise ValueError("query_valid must be [B, R]")
    if evidence_labels.ndim != 3 or evidence_labels.shape[:2] != query.shape[:2]:
        raise ValueError("sentence evidence labels must be [B, R, U]")
    if keys.shape[0] != query.shape[0] or keys.shape[-1] != query.shape[-1]:
        raise ValueError("query/key batch and projection dimensions must match")
    if global_evidence_labels is not None and (
        global_evidence_labels.ndim != 2 or global_evidence_labels.shape[0] != query.shape[0]
    ):
        raise ValueError("global evidence labels must be [B, U]")

    zero = query.sum() * 0.0
    width = min(keys.shape[1], evidence_labels.shape[2], valid_units.shape[1])
    if width == 0:
        detached_zero = zero.detach()
        return {
            "evidence_contrastive_loss": zero,
            "evidence_top1_accuracy": detached_zero,
            "positive_similarity": detached_zero,
            "hard_negative_similarity": detached_zero,
            "evidence_similarity_gap": detached_zero,
            "evidence_valid_examples": detached_zero,
        }
    labels = evidence_labels[:, :, :width]
    valid = valid_units[:, None, :width].bool()
    positive_mask = valid & labels.gt(0.5) & query_valid[:, :, None]
    negative_mask = valid & labels.ge(0.0) & labels.le(0.5) & query_valid[:, :, None]

    similarity = torch.einsum("brp,bup->bru", query, keys[:, :width])
    mining_score = similarity.detach()
    if salience_logits is not None and salience_boost > 0.0:
        salience_width = min(width, salience_logits.shape[1])
        detached_salience = torch.zeros_like(mining_score)
        detached_salience[:, :, :salience_width] = torch.sigmoid(
            salience_logits[:, None, :salience_width].detach().float()
        ).to(mining_score.dtype)
        mining_score = mining_score + float(salience_boost) * detached_salience
    if attention_prior_energy is not None and attention_mining_boost > 0.0:
        if attention_prior_energy.ndim != 2 or attention_prior_energy.shape[0] != query.shape[0]:
            raise ValueError("attention_prior_energy must be [B, U]")
        attention_width = min(width, attention_prior_energy.shape[1])
        detached_energy = torch.zeros_like(mining_score)
        detached_energy[:, :, :attention_width] = (
            attention_prior_energy[:, None, :attention_width].detach().float().to(mining_score.dtype)
        )
        mining_score = mining_score + float(attention_mining_boost) * detached_energy
    mining_score = mining_score.masked_fill(~negative_mask, -torch.inf)
    hard_negative_mask = torch.zeros_like(negative_mask)
    if width > 0:
        k = min(int(num_hard_negatives), width)
        top_values, top_indices = mining_score.topk(k, dim=2)
        selected = torch.isfinite(top_values)
        hard_negative_mask.scatter_(2, top_indices, selected)

    if attention_prior_energy is not None:
        if attention_prior_energy.ndim != 2 or attention_prior_energy.shape[0] != query.shape[0]:
            raise ValueError("attention_prior_energy must be [B, U]")
        attention_width = min(width, attention_prior_energy.shape[1])
        aligned_energy = torch.zeros_like(similarity)
        aligned_energy[:, :, :attention_width] = attention_prior_energy[:, None, :attention_width].float()
        if global_evidence_labels is None:
            global_positive = positive_mask.any(dim=1)
        else:
            global_positive = global_evidence_labels[:, :width].gt(0.5) & valid_units[:, :width].bool()
        attention_coupling = positive_mask | (hard_negative_mask & ~global_positive[:, None, :])
        logits = (
            similarity + float(salience_logit_bias) * aligned_energy * attention_coupling.to(similarity.dtype)
        ) / float(temperature)
    else:
        logits = similarity / float(temperature)
    if attention_prior_energy is None and salience_logits is not None and salience_logit_bias > 0.0:
        salience_width = min(width, salience_logits.shape[1])
        aligned_salience = torch.zeros_like(logits)
        aligned_salience[:, :, :salience_width] = salience_logits[:, None, :salience_width].float().clamp(-4.0, 4.0)
        if global_evidence_labels is None:
            global_positive = positive_mask.any(dim=1)
        else:
            global_positive = global_evidence_labels[:, :width].gt(0.5) & valid_units[:, :width].bool()
        salience_coupling = positive_mask | (hard_negative_mask & ~global_positive[:, None, :])
        logits = logits + float(salience_logit_bias) * aligned_salience * salience_coupling.to(logits.dtype)
    negative_logits = logits.masked_fill(~hard_negative_mask, -torch.inf)
    negative_logsumexp = torch.logsumexp(negative_logits, dim=2)

    positive_count = positive_mask.sum(dim=2)
    negative_count = hard_negative_mask.sum(dim=2)
    valid_queries = query_valid & positive_count.gt(0) & negative_count.gt(0)
    safe_negative_logsumexp = torch.where(valid_queries, negative_logsumexp, torch.zeros_like(negative_logsumexp))
    positive_losses = F.softplus(safe_negative_logsumexp.unsqueeze(-1) - logits)
    loss_per_query = (positive_losses * positive_mask).sum(dim=2) / positive_count.clamp_min(1)
    valid_float = valid_queries.to(loss_per_query.dtype)
    valid_count = valid_float.sum()
    loss = (loss_per_query * valid_float).sum() / valid_count.clamp_min(1.0)
    loss = loss + zero

    with torch.no_grad():
        positive_float = positive_mask.to(similarity.dtype)
        negative_float = hard_negative_mask.to(similarity.dtype)
        positive_similarity = (similarity * positive_float).sum(dim=2) / positive_count.clamp_min(1)
        negative_similarity = (similarity * negative_float).sum(dim=2) / negative_count.clamp_min(1)
        mean_positive = (positive_similarity * valid_float).sum() / valid_count.clamp_min(1.0)
        mean_negative = (negative_similarity * valid_float).sum() / valid_count.clamp_min(1.0)
        hardest_negative = negative_logits.max(dim=2).values
        beats_hardest = positive_mask & logits.gt(hardest_negative.unsqueeze(-1))
        accuracy_per_query = beats_hardest.sum(dim=2).float() / positive_count.clamp_min(1)
        accuracy = (accuracy_per_query * valid_float).sum() / valid_count.clamp_min(1.0)

    return {
        "evidence_contrastive_loss": loss,
        "evidence_top1_accuracy": accuracy.detach(),
        "positive_similarity": mean_positive.detach(),
        "hard_negative_similarity": mean_negative.detach(),
        "evidence_similarity_gap": (mean_positive - mean_negative).detach(),
        "evidence_valid_examples": valid_count.detach(),
    }
