"""Stateful source-phrase continuation for LLM2Seq-v5.

The module adds a very small, source-grounded output path to the pretrained
decoder.  At every target position it chooses among three modes:

``generate``
    Use the decoder's normal vocabulary distribution.
``new_span``
    Copy from a newly selected source position.
``continue_span``
    Prefer the position immediately after the source position that explained
    the previous emitted token.

The continuation state is a distribution over source positions, not a hard
pointer, so repeated phrases and subword ambiguity are marginalized rather
than assigned an arbitrary single alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically safe softmax over the final source dimension."""

    if scores.shape[:-1] != mask.shape[:-1] or scores.shape[-1] != mask.shape[-1]:
        mask = mask.expand_as(scores)
    valid = mask.bool()
    no_valid = ~valid.any(dim=-1, keepdim=True)
    safe_valid = valid
    if bool(no_valid.any()):
        safe_valid = valid.clone()
        safe_valid[..., 0] |= no_valid.squeeze(-1)
    masked = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
    probabilities = F.softmax(masked.float(), dim=-1).to(scores.dtype)
    probabilities = probabilities.masked_fill(~valid, 0)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _shift_right(probabilities: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(probabilities)
    shifted[..., 1:] = probabilities[..., :-1]
    return shifted


def _gold_copy_probability(
    attention: torch.Tensor,
    source_token_ids: torch.Tensor,
    gold_token_ids: torch.Tensor,
) -> torch.Tensor:
    compatible = source_token_ids[:, None, :].eq(gold_token_ids[:, :, None])
    return (attention.float() * compatible.float()).sum(dim=-1)


def balanced_phrase_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Balanced BCE over 2/3/4-gram start labels.

    ``labels == -1`` marks source positions that are prompt/special/padding or
    cannot begin the requested n-gram without crossing a source-unit boundary.
    Positive and negative terms are averaged separately for each order so the
    much larger negative class cannot trivialize the phrase head.
    """

    if logits.shape != labels.shape or logits.ndim != 3:
        raise ValueError("phrase logits/labels must have identical [B, S, N] shapes")
    terms = []
    for order_index in range(logits.shape[-1]):
        valid = labels[..., order_index].ge(0)
        positive = valid & labels[..., order_index].gt(0.5)
        negative = valid & ~positive
        order_terms = []
        if bool(positive.any()):
            order_terms.append(F.softplus(-logits[..., order_index][positive].float()).mean())
        if bool(negative.any()):
            order_terms.append(F.softplus(logits[..., order_index][negative].float()).mean())
        if order_terms:
            terms.append(torch.stack(order_terms).mean())
    return torch.stack(terms).mean() if terms else logits.float().sum() * 0.0


@dataclass
class PointerStep:
    log_probabilities: torch.Tensor
    source_pointer_mass: torch.Tensor
    mode_probabilities: torch.Tensor
    new_attention: torch.Tensor
    continuation_attention: torch.Tensor


class StatefulPhrasePointer(nn.Module):
    """Low-rank phrase pointer with a soft continuation state."""

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        rank: int = 128,
        phrase_hidden_size: int = 256,
        dropout: float = 0.1,
        phrase_bias_scale: float = 0.5,
        continuation_strength: float = 1.0,
        generate_probability_init: float = 0.98,
        use_continuation: bool = True,
        detach_recurrent_state: bool = True,
    ):
        super().__init__()
        if hidden_size <= 0 or vocabulary_size <= 0 or rank <= 0:
            raise ValueError("hidden/vocabulary/rank sizes must be positive")
        if not 0.0 < generate_probability_init < 1.0:
            raise ValueError("generate_probability_init must be in (0, 1)")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.rank = int(rank)
        self.query_norm = nn.RMSNorm(hidden_size)
        self.memory_norm = nn.RMSNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, rank, bias=False)
        self.key_projection = nn.Linear(hidden_size, rank, bias=False)
        self.phrase_head = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, phrase_hidden_size, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(phrase_hidden_size, 3, bias=True),
        )
        self.mode_norm = nn.RMSNorm(hidden_size * 3)
        self.mode_gate = nn.Linear(hidden_size * 3, 3, bias=True)
        self.continuation_gate = nn.Linear(hidden_size, 1, bias=True)
        # Non-negative and initially small. At initialization phrase scores are
        # uniform, so this path cannot perturb source ranking.
        self.phrase_bias_gate = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.phrase_bias_scale = float(phrase_bias_scale)
        self.continuation_strength = float(continuation_strength)
        self.use_continuation = bool(use_continuation)
        self.detach_recurrent_state = bool(detach_recurrent_state)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.phrase_head[1].weight)
        nn.init.zeros_(self.phrase_head[-1].weight)
        nn.init.zeros_(self.phrase_head[-1].bias)
        # Start as the pretrained LM, then learn to allocate probability mass to
        # new/continuation copy modes without destabilizing early warm-up.
        nn.init.zeros_(self.mode_gate.weight)
        copy_probability = (1.0 - float(generate_probability_init)) / 2.0
        self.mode_gate.bias.data.copy_(
            torch.tensor(
                [
                    math.log(float(generate_probability_init)),
                    math.log(copy_probability),
                    math.log(copy_probability),
                ],
                dtype=self.mode_gate.bias.dtype,
            )
        )
        nn.init.zeros_(self.continuation_gate.weight)
        nn.init.zeros_(self.continuation_gate.bias)
        self._last_generation_modes: Optional[torch.Tensor] = None
        self._last_generation_observations = 0
        self._cached_memory_pointer: Optional[int] = None
        self._cached_keys: Optional[torch.Tensor] = None
        self._cached_phrase_logits: Optional[torch.Tensor] = None

    def set_generation_statistics(self, modes: torch.Tensor, observations: int) -> None:
        self._last_generation_modes = modes.detach().float()
        self._last_generation_observations = int(observations)

    def generation_statistics(self) -> Tuple[torch.Tensor, int]:
        if self._last_generation_modes is None:
            return self.phrase_bias_gate.new_tensor([1.0, 0.0, 0.0]), 0
        return self._last_generation_modes, self._last_generation_observations

    @torch.no_grad()
    def prepare_source_cache(self, memory: torch.Tensor) -> None:
        summary = self.summary_memory(memory)
        self._cached_memory_pointer = int(summary.data_ptr())
        self._cached_keys = self.key_projection(self.memory_norm(summary))
        self._cached_phrase_logits = self.phrase_head(summary)

    def clear_source_cache(self) -> None:
        self._cached_memory_pointer = None
        self._cached_keys = None
        self._cached_phrase_logits = None

    @staticmethod
    def summary_memory(memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim == 3:
            return memory
        if memory.ndim == 4:
            # HiRoute order is lexical/semantic/summary. Phrase realization is
            # grounded in the fully bidirectional summary memory.
            return memory[:, -1]
        raise ValueError("pointer memory must be [B,S,D] or [B,K,S,D]")

    def phrase_logits(self, memory: torch.Tensor) -> torch.Tensor:
        return self.phrase_head(self.summary_memory(memory))

    def _scores(
        self,
        decoder_states: torch.Tensor,
        memory: torch.Tensor,
        copy_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.summary_memory(memory)
        query = self.query_projection(self.query_norm(decoder_states))
        use_cache = (
            self._cached_memory_pointer == int(memory.data_ptr())
            and self._cached_keys is not None
            and self._cached_phrase_logits is not None
        )
        key = self._cached_keys if use_cache else self.key_projection(self.memory_norm(memory))
        scores = torch.einsum("btr,bsr->bts", query, key) / math.sqrt(self.rank)
        phrase_logits = self._cached_phrase_logits if use_cache else self.phrase_head(memory)
        phrase_start = torch.sigmoid(phrase_logits.float()).max(dim=-1).values.to(scores.dtype)
        phrase_gate = torch.sigmoid(self.phrase_bias_gate.float()).to(scores.dtype)
        scores = scores + phrase_gate * self.phrase_bias_scale * phrase_start[:, None].to(scores.dtype)
        if attention_bias is not None:
            scores = scores + attention_bias[:, None].to(scores.dtype)
        new_attention = _masked_softmax(scores, copy_mask[:, None, :])
        return scores, new_attention, phrase_logits

    def _continuation_attention(
        self,
        scores: torch.Tensor,
        new_attention: torch.Tensor,
        previous_responsibility: torch.Tensor,
        continuation_edge_mask: torch.Tensor,
        decoder_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Continue from the Bayes source responsibility of the last token."""

        prior = _shift_right(previous_responsibility.float())
        prior = prior * continuation_edge_mask.float()
        prior_mass = prior.sum(dim=-1, keepdim=True).clamp(max=1.0)
        has_prior = prior_mass.gt(0)
        normalized_prior = prior / prior_mass.clamp_min(1e-12)
        if decoder_states is None:
            strength = scores.new_full(scores.shape[:-1] + (1,), self.continuation_strength)
        else:
            learned = torch.sigmoid(self.continuation_gate(decoder_states).float()).to(scores.dtype)
            strength = self.continuation_strength * (0.5 + learned)
        continued_scores = scores + strength * torch.log(normalized_prior.clamp_min(1e-8)).to(scores.dtype)
        continuation = _masked_softmax(continued_scores, prior.gt(0))
        continuation = torch.where(has_prior, continuation, new_attention)
        return continuation, has_prior.squeeze(-1), prior_mass.squeeze(-1)

    @staticmethod
    def continuation_edge_mask(source_unit_ids: torch.Tensor, source_copy_mask: torch.Tensor) -> torch.Tensor:
        if source_unit_ids.shape != source_copy_mask.shape:
            raise ValueError("source_unit_ids and source_copy_mask must share [B,S]")
        edges = torch.zeros_like(source_copy_mask, dtype=torch.bool)
        edges[:, 1:] = (
            source_copy_mask[:, 1:]
            & source_copy_mask[:, :-1]
            & source_unit_ids[:, 1:].eq(source_unit_ids[:, :-1])
            & source_unit_ids[:, 1:].gt(0)
        )
        return edges

    def teacher_forced_loss(
        self,
        *,
        decoder_states: torch.Tensor,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_memory: torch.Tensor,
        source_token_ids: torch.Tensor,
        source_unit_ids: torch.Tensor,
        source_copy_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
        phrase_labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute mixture/copy/phrase losses without materializing [B,T,V]."""

        if labels.shape != decoder_input_ids.shape or labels.shape != decoder_states.shape[:2]:
            raise ValueError("labels, decoder inputs and decoder states must share [B,T]")
        supervised = labels.ge(0)
        scores, new_attention, phrase_logits = self._scores(
            decoder_states,
            source_memory,
            source_copy_mask,
            attention_bias,
        )
        memory = self.summary_memory(source_memory)
        new_context = torch.einsum("bts,bsd->btd", new_attention.to(memory.dtype), memory)
        gold = labels.clamp_min(0)
        new_gold = _gold_copy_probability(new_attention, source_token_ids, gold)
        vocab_log_gold = torch.zeros_like(labels, dtype=torch.float32)
        vocab_log_gold[supervised] = -F.cross_entropy(
            lm_logits.float(),
            labels[supervised],
            reduction="none",
        )
        continuation_edges = self.continuation_edge_mask(source_unit_ids, source_copy_mask)
        copy_available = source_copy_mask.any(dim=-1)
        responsibility = decoder_states.new_zeros(source_copy_mask.shape, dtype=torch.float32)
        continuation_rows = []
        continuation_gold_rows = []
        continuation_available_rows = []
        mode_rows = []
        mixed_gold_rows = []
        for step in range(decoder_states.shape[1]):
            continuation, continuation_available, continuation_prior_mass = self._continuation_attention(
                scores[:, step],
                new_attention[:, step],
                responsibility,
                continuation_edges,
                decoder_states=decoder_states[:, step],
            )
            continuation_context = torch.einsum(
                "bs,bsd->bd",
                continuation.to(memory.dtype),
                memory,
            )
            mode_features = torch.cat(
                [decoder_states[:, step], new_context[:, step], continuation_context],
                dim=-1,
            )
            mode_logits = self.mode_gate(self.mode_norm(mode_features)).float()
            mode_logits[:, 1] = mode_logits[:, 1].masked_fill(~copy_available, float("-inf"))
            if self.use_continuation:
                mode_logits[:, 2] = mode_logits[:, 2] + torch.log(continuation_prior_mass.clamp_min(1e-30))
            else:
                continuation_available = torch.zeros_like(continuation_available)
            mode_logits[:, 2] = mode_logits[:, 2].masked_fill(~continuation_available, float("-inf"))
            mode = F.softmax(mode_logits, dim=-1)
            step_gold = gold[:, step]
            compatible = source_token_ids.eq(step_gold[:, None]) & source_copy_mask
            continuation_gold = (continuation.float() * compatible.float()).sum(dim=-1)
            log_components = torch.stack(
                [
                    torch.log(mode[:, 0].clamp_min(1e-30)) + vocab_log_gold[:, step],
                    torch.log(mode[:, 1].clamp_min(1e-30)) + torch.log(new_gold[:, step].clamp_min(1e-30)),
                    torch.log(mode[:, 2].clamp_min(1e-30)) + torch.log(continuation_gold.clamp_min(1e-30)),
                ],
                dim=-1,
            )
            log_mixed_gold = torch.logsumexp(log_components, dim=-1)
            pointer_numerator = (
                mode[:, 1, None] * new_attention[:, step].float() + mode[:, 2, None] * continuation.float()
            ) * compatible.float()
            next_responsibility = (
                torch.where(
                    pointer_numerator.gt(0),
                    torch.exp(torch.log(pointer_numerator.clamp_min(1e-30)) - log_mixed_gold[:, None]),
                    torch.zeros_like(pointer_numerator),
                )
                * compatible.float()
            )
            mass = next_responsibility.sum(dim=-1, keepdim=True)
            next_responsibility = next_responsibility / mass.clamp_min(1.0)
            responsibility = torch.where(
                supervised[:, step, None],
                next_responsibility,
                torch.zeros_like(next_responsibility),
            )
            if self.detach_recurrent_state:
                responsibility = responsibility.detach()
            continuation_rows.append(continuation)
            continuation_gold_rows.append(continuation_gold)
            continuation_available_rows.append(continuation_available)
            mode_rows.append(mode)
            mixed_gold_rows.append(log_mixed_gold)
        continuation_attention = torch.stack(continuation_rows, dim=1)
        continuation_gold = torch.stack(continuation_gold_rows, dim=1)
        continuation_available = torch.stack(continuation_available_rows, dim=1)
        mode_probabilities = torch.stack(mode_rows, dim=1)
        log_mixed_gold = torch.stack(mixed_gold_rows, dim=1)
        loss_mixture = -log_mixed_gold[supervised].mean()

        gold_matches_source = source_token_ids[:, None, :].eq(gold[:, :, None])
        copyable = supervised & (gold_matches_source & source_copy_mask[:, None, :]).any(dim=-1)
        if bool(copyable.any()):
            # Train the source alignment independently of the mode gate. The
            # mixture NLL below is responsible for deciding whether copying is
            # preferable to normal generation; forcing copy mode for every
            # source-covered function word would otherwise encourage extractive
            # collapse on a summarization task.
            loss_copy = -torch.log(new_gold[copyable].clamp_min(1e-8)).mean()
            predicted_source_tokens = source_token_ids.gather(
                1,
                new_attention.argmax(dim=-1),
            )
            copy_accuracy = predicted_source_tokens[copyable].eq(gold[copyable]).float().mean().detach()
        else:
            loss_copy = decoder_states.float().sum() * 0.0
            copy_accuracy = loss_copy.detach()

        continuable = supervised & continuation_available & continuation_gold.gt(0)
        if bool(continuable.any()):
            loss_continue = -torch.log(
                (mode_probabilities[..., 2] * continuation_gold)[continuable].clamp_min(1e-8)
            ).mean()
        else:
            loss_continue = decoder_states.float().sum() * 0.0

        pointer_mass = (
            mode_probabilities[..., 1, None] * new_attention.float()
            + mode_probabilities[..., 2, None] * continuation_attention.float()
        )
        pointer_mass = pointer_mass * supervised[..., None].float()
        coverage_before = pointer_mass.cumsum(dim=1) - pointer_mass
        overlap = torch.minimum(pointer_mass, coverage_before).sum(dim=-1)
        loss_coverage = overlap[supervised].mean() if bool(supervised.any()) else overlap.sum() * 0.0

        if phrase_labels is not None:
            loss_phrase = balanced_phrase_loss(phrase_logits, phrase_labels)
        else:
            loss_phrase = phrase_logits.float().sum() * 0.0

        supervised_modes = mode_probabilities[supervised]
        return {
            "loss_phrase_mixture": loss_mixture,
            "loss_phrase_copy": loss_copy,
            "loss_phrase_continue": loss_continue,
            "loss_phrase_labels": loss_phrase,
            "loss_phrase_coverage": loss_coverage,
            "phrase_copyable_rate": copyable.float().sum().detach() / supervised.float().sum().clamp_min(1.0),
            "phrase_continuation_available_rate": (
                continuable.float().sum().detach() / supervised.float().sum().clamp_min(1.0)
            ),
            "phrase_mode_generate": supervised_modes[:, 0].mean().detach(),
            "phrase_mode_new": supervised_modes[:, 1].mean().detach(),
            "phrase_mode_continue": supervised_modes[:, 2].mean().detach(),
            "phrase_copy_support_accuracy": copy_accuracy,
        }

    def generation_step(
        self,
        *,
        decoder_state: torch.Tensor,
        lm_logits: torch.Tensor,
        source_memory: torch.Tensor,
        source_token_ids: torch.Tensor,
        source_unit_ids: torch.Tensor,
        source_copy_mask: torch.Tensor,
        previous_responsibility: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
    ) -> PointerStep:
        if decoder_state.ndim != 2 or lm_logits.ndim != 2:
            raise ValueError("generation_step expects [B,D] state and [B,V] logits")
        states = decoder_state.unsqueeze(1)
        scores, new_attention, _ = self._scores(
            states,
            source_memory,
            source_copy_mask,
            attention_bias,
        )
        if previous_responsibility is None:
            previous_responsibility = new_attention.new_zeros(source_copy_mask.shape)
        continuation, continuation_available, continuation_prior_mass = self._continuation_attention(
            scores[:, 0],
            new_attention[:, 0],
            previous_responsibility,
            self.continuation_edge_mask(source_unit_ids, source_copy_mask),
            decoder_states=decoder_state,
        )
        memory = self.summary_memory(source_memory)
        new_context = torch.einsum("bts,bsd->btd", new_attention.to(memory.dtype), memory)
        continuation_context = torch.einsum("bs,bsd->bd", continuation.to(memory.dtype), memory)
        mode_features = torch.cat([decoder_state, new_context[:, 0], continuation_context], dim=-1)
        mode_logits = self.mode_gate(self.mode_norm(mode_features)).float()
        copy_available = source_copy_mask.any(dim=-1)
        mode_logits[:, 1] = mode_logits[:, 1].masked_fill(~copy_available, float("-inf"))
        if self.use_continuation:
            mode_logits[:, 2] = mode_logits[:, 2] + torch.log(continuation_prior_mass.clamp_min(1e-30))
        else:
            continuation_available = torch.zeros_like(continuation_available)
        mode_logits[:, 2] = mode_logits[:, 2].masked_fill(~continuation_available, float("-inf"))
        mode = F.softmax(mode_logits, dim=-1)

        batch = lm_logits.shape[0]
        if lm_logits.shape[1] != self.vocabulary_size:
            raise ValueError(f"LM vocabulary {lm_logits.shape[1]} != phrase-pointer vocabulary {self.vocabulary_size}")
        pointer_vocab = lm_logits.new_zeros((batch, self.vocabulary_size), dtype=torch.float32)
        invalid = source_copy_mask & (source_token_ids.lt(0) | source_token_ids.ge(self.vocabulary_size))
        if bool(invalid.any()):
            raise ValueError("Copyable source token ID is outside decoder vocabulary")
        ids = source_token_ids.masked_fill(~source_copy_mask, 0)
        vocab = F.softmax(lm_logits.float(), dim=-1)
        pointer_mass = mode[:, 1, None] * new_attention[:, 0].float() + mode[:, 2, None] * continuation.float()
        pointer_vocab.scatter_add_(1, ids, pointer_mass)
        mixture = mode[:, 0, None] * vocab + pointer_vocab
        return PointerStep(
            log_probabilities=torch.log(mixture.clamp_min(1e-12)),
            source_pointer_mass=pointer_mass,
            mode_probabilities=mode,
            new_attention=new_attention[:, 0],
            continuation_attention=continuation,
        )

    @staticmethod
    def posterior_source_responsibility(
        step: PointerStep,
        emitted_token_ids: torch.Tensor,
        source_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Bayes responsibility that the emitted token came from each source position."""

        emitted = emitted_token_ids.view(-1)
        compatible = source_token_ids.eq(emitted[:, None])
        numerator = step.source_pointer_mass.float() * compatible.float()
        emitted_probability = step.log_probabilities.exp().gather(1, emitted[:, None])
        posterior = numerator / emitted_probability.clamp_min(1e-12)
        # Numerical roundoff must never make total copy responsibility exceed 1.
        return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1.0)
