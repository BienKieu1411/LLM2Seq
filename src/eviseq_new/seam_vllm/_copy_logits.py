"""Single-token copy mixture without dense vocabulary-sized copy buffers."""

from __future__ import annotations

import torch
from eviseq_afmr.modeling.grounded_copy import CopyState, GroundedCopyHead


class CopyLogitsMixer:
    """Cache source destinations for one immutable, ordered copy batch."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.state = None
        self.vocab_size = None
        self.destinations = None
        self.inverse = None
        self.rows = None

    def mix(self, head: GroundedCopyHead, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState):
        # The vLLM sampler selects one hidden row per request, including prefill.
        batch, vocab_size = logits.shape
        if self.state is not state or self.vocab_size != vocab_size:
            rows = torch.arange(batch, device=state.token_ids.device)[:, None]
            destinations = state.token_ids + rows * vocab_size
            self.destinations, self.inverse = torch.unique(destinations.reshape(-1), return_inverse=True)
            self.rows = self.destinations.div(vocab_size, rounding_mode="floor")
            self.state, self.vocab_size = state, vocab_size

        log_attention, log_copy, log_generate = head.distribution(hidden[:, None, :], state)
        probability = torch.zeros_like(self.destinations, dtype=torch.float32).scatter_add_(
            0, self.inverse, log_attention.exp().reshape(-1)
        )
        normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        generator = logits.float() - normalizer + log_generate[:, 0, :]
        candidate_generator = generator.reshape(-1).index_select(0, self.destinations)
        floor = torch.finfo(torch.float32).min
        # Keep the reference's finite floor even for a generator logit of -inf.
        # Logits may be a strided view after vLLM removes vocabulary padding.
        # The flattened in-place update below must write into the returned tensor.
        mixed = torch.logaddexp(generator, floor + log_copy[:, 0, :]).contiguous()
        mixed.add_(normalizer)
        log_source = probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_source.masked_fill_(probability.eq(0), floor)
        candidate_mixed = torch.logaddexp(
            candidate_generator, log_source + log_copy[:, 0, 0].index_select(0, self.rows)
        ) + normalizer[:, 0].index_select(0, self.rows)
        mixed.reshape(-1).index_copy_(0, self.destinations, candidate_mixed)
        return torch.where(state.mask.any(-1)[:, None], mixed, logits.float())
