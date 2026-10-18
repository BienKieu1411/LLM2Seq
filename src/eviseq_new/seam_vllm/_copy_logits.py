"""Single-token copy mixture without dense vocabulary-sized copy buffers."""

from __future__ import annotations

import os

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
        self.all_sourced = False
        self.columns = None

    _fused = None
    _fuse_enabled = os.environ.get("SEAM_VLLM_FUSE_COPY_MIX", "1") != "0"

    @classmethod
    def _fused_dense(cls, logits, normalizer, log_generate, absent_floor):
        """Return max((l - n) + g, floor) + n in one kernel, or None to use the eager path."""
        if not cls._fuse_enabled or logits.device.type == "cpu" or logits.dtype == torch.float64:
            return None
        try:
            if cls._fused is None:
                cls._fused = torch.compile(lambda l, n, g, s: torch.maximum((l - n) + g, s) + n, dynamic=True)
            mixed = cls._fused(logits, normalizer, log_generate, absent_floor)
            # Inductor may preserve a strided input layout; the scatter below needs a flat view.
            return mixed if mixed.is_contiguous() else mixed.contiguous()
        except Exception:
            cls._fuse_enabled = False
            return None

    def mix(self, head: GroundedCopyHead, hidden: torch.Tensor, logits: torch.Tensor, state: CopyState):
        # The vLLM sampler selects one hidden row per request, including prefill.
        batch, vocab_size = logits.shape
        if self.state is not state or self.vocab_size != vocab_size:
            rows = torch.arange(batch, device=state.token_ids.device)[:, None]
            destinations = state.token_ids + rows * vocab_size
            self.destinations, self.inverse = torch.unique(destinations.reshape(-1), return_inverse=True)
            self.rows = self.destinations.div(vocab_size, rounding_mode="floor")
            self.columns = self.destinations - self.rows * vocab_size
            self.state, self.vocab_size = state, vocab_size
            self.all_sourced = bool(state.mask.any(-1).all())

        log_attention, log_copy, log_generate = head.distribution(hidden[:, None, :], state)
        probability = torch.zeros_like(self.destinations, dtype=torch.float32).scatter_add_(
            0, self.inverse, log_attention.exp().reshape(-1)
        )
        # Write subtraction directly into one private contiguous FP32 buffer.
        # This avoids first copying all logits and then reading/writing them a
        # second time for normalization. Keep the reference's operation order;
        # the two pointwise ops remain separate and are not fused.
        # Reduce the original layout, too: changing a reduction's strides can
        # select a different GPU reduction kernel and rounding order.
        normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        floor = torch.finfo(torch.float32).min
        mixed = self._fused_dense(logits, normalizer, log_generate[:, 0, :], floor + log_copy[:, 0, :])
        if mixed is not None:
            # Same fp32 elementwise ops as the eager sequence, evaluated only at the candidates.
            rows = self.rows
            candidate_generator = (logits[rows, self.columns].float() - normalizer[:, 0].index_select(0, rows)) + (
                log_generate[:, 0, 0].index_select(0, rows)
            )
        else:
            if logits.dtype == torch.float64:
                # vLLM uses FP32/BF16 logits, but preserve the reference's broader
                # dtype behavior for callers outside the serving path.
                mixed = logits.float().to(memory_format=torch.contiguous_format)
                mixed.sub_(normalizer)
            else:
                mixed = torch.empty_like(logits, dtype=torch.float32, memory_format=torch.contiguous_format)
                torch.sub(logits, normalizer, out=mixed)
            mixed.add_(log_generate[:, 0, :])
            candidate_generator = mixed.reshape(-1).index_select(0, self.destinations)
            # Keep the reference's finite floor even for a generator logit of -inf.
            # Logits may be a strided view after vLLM removes vocabulary padding.
            # The flattened in-place update below must write into the returned tensor.
            # The absent-copy term is either float32.min or -inf. At that scale
            # logaddexp rounds exactly to max in FP32, including generator=-inf;
            # avoid exp/log work over the entire vocabulary for this default branch.
            torch.maximum(mixed, floor + log_copy[:, 0, :], out=mixed)
            mixed.add_(normalizer)
        log_source = probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_source.masked_fill_(probability.eq(0), floor)
        candidate_mixed = torch.logaddexp(
            candidate_generator, log_source + log_copy[:, 0, 0].index_select(0, self.rows)
        ) + normalizer[:, 0].index_select(0, self.rows)
        mixed.reshape(-1).scatter_(0, self.destinations, candidate_mixed)
        if self.all_sourced:
            # torch.where(all-True, mixed, logits) is exactly mixed; skip a dense [B, V] pass.
            return mixed
        torch.where(state.mask.any(-1)[:, None], mixed, logits.float(), out=mixed)
        return mixed
