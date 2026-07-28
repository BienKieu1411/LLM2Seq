"""Pure dual-mask mixing primitives used by the native Qwen encoder."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Variant = Literal["causal", "full", "dec2enc", "evidence"]


def align_trainable_sdpa_bias_heads(attention_bias: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Materialize a trainable broadcast bias on SDPA's query-head axis.

    CUDA memory-efficient backward indexes differentiable masks by query head.
    EviSeq's compact evidence bias is ``[B,1,1,K]``; expanding only its
    singleton head axis costs ``B*H*K`` rather than a quadratic ``B*H*Q*K``.
    """

    if attention_bias.ndim != 4:
        raise ValueError("SDPA attention bias must have [B,H,Q,K] rank")
    if attention_bias.shape[1] not in (1, query_heads):
        raise ValueError("SDPA attention-bias heads must be 1 or match query heads")
    if attention_bias.requires_grad and attention_bias.shape[1] == 1 and query_heads > 1:
        return attention_bias.expand(-1, query_heads, -1, -1).contiguous()
    return attention_bias


def ensure_sdpa_lse_for_bias_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Keep fused SDPA backward valid when only its bias needs gradients.

    During interface warm-up the pretrained Q/K/V path is frozen, while the
    learned evidence-key bias remains differentiable.  Some CUDA
    memory-efficient SDPA versions decide whether to retain log-sum-exp (LSE)
    state from Q/K/V alone.  A bias-only backward then fails with
    ``LSE is not correctly aligned (strideH)``.  Giving the query a leaf
    gradient in exactly that case makes the kernel retain its backward state;
    the disposable query gradient does not update any pretrained parameter or
    alter the forward values.
    """

    needs_bias_backward = (
        torch.is_grad_enabled()
        and attention_bias is not None
        and attention_bias.requires_grad
        and not (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    return query.detach().requires_grad_(True) if needs_bias_backward else query


def pool_units(
    token_states: torch.Tensor,
    unit_ids: torch.Tensor,
    unit_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool source units; id 0 is task prefix/padding and is excluded."""

    batch, _, hidden = token_states.shape
    if unit_count <= 0:
        return (
            token_states.new_zeros((batch, 0, hidden)),
            torch.zeros((batch, 0), dtype=torch.bool, device=token_states.device),
        )
    indices = unit_ids.clamp(min=0, max=unit_count)
    sums = token_states.new_zeros((batch, unit_count + 1, hidden))
    counts = token_states.new_zeros((batch, unit_count + 1, 1))
    sums.scatter_add_(1, indices.unsqueeze(-1).expand(-1, -1, hidden), token_states)
    counts.scatter_add_(1, indices.unsqueeze(-1), torch.ones_like(token_states[..., :1]))
    pooled = sums[:, 1:] / counts[:, 1:].clamp_min(1.0)
    valid = counts[:, 1:, 0].gt(0)
    return pooled, valid


def unit_evidence_token_bias(
    unit_logits: torch.Tensor,
    valid_units: torch.Tensor,
    unit_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute unit evidence over tokens without a sentence-length prior.

    For unit ``u`` with ``n_u`` visible subwords, every token receives
    ``scale * logit_u - log(n_u)``. Consequently, if token-content scores are
    equal, the unit contributes total unnormalised mass
    ``exp(scale * logit_u)`` regardless of its token count.
    """

    if unit_logits.ndim != 2 or unit_logits.shape != valid_units.shape:
        raise ValueError("unit logits and validity must have the same [B,U] shape")
    if unit_ids.ndim != 2 or attention_mask.shape != unit_ids.shape:
        raise ValueError("unit_ids and attention_mask must have the same [B,K] shape")
    if unit_logits.shape[0] != unit_ids.shape[0]:
        raise ValueError("unit logits and token tensors must have the same batch size")

    unit_count = unit_logits.shape[1]
    capped_ids = unit_ids.clamp(min=0, max=unit_count)
    padded_logits = torch.cat(
        [unit_logits.float().new_zeros(unit_logits.shape[0], 1), unit_logits.float()],
        dim=1,
    )
    padded_valid = torch.cat(
        [torch.zeros(valid_units.shape[0], 1, dtype=torch.bool, device=valid_units.device), valid_units.bool()],
        dim=1,
    )
    source_keys = attention_mask.bool() & capped_ids.gt(0) & padded_valid.gather(1, capped_ids)
    token_counts = torch.zeros(
        (unit_ids.shape[0], unit_count + 1),
        dtype=torch.float32,
        device=unit_ids.device,
    )
    token_counts.scatter_add_(1, capped_ids, source_keys.float())
    length_penalty = token_counts.gather(1, capped_ids).clamp_min(1.0).log()
    key_logits = padded_logits.gather(1, capped_ids).clamp(-5.0, 5.0)
    return float(scale) * key_logits - length_penalty, source_keys


def evidence_key_attention_bias(
    unit_logits: torch.Tensor,
    valid_units: torch.Tensor,
    unit_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    """Build a differentiable key bias for the noncausal evidence view.

    Each source key receives the predicted log-odds of its source unit, so
    positive evidence units receive more attention and negative units receive
    less. Prefix, EOS, and padding positions (unit id 0) are excluded from this
    noncausal view; the separate causal view still retains them normally.

    The returned additive SDPA mask has broadcast shape ``[B, 1, 1, K]``.
    Avoiding a materialized ``Q x K`` mask is essential at 3072--4096 tokens.
    Unlike the old scalar suffix-mass gate, it preserves *where* evidence is:
    two documents with the same total evidence but different selected units
    therefore induce different noncausal attention distributions.
    """

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("evidence attention bias requires a floating-point dtype")
    bias, source_keys = unit_evidence_token_bias(
        unit_logits,
        valid_units,
        unit_ids,
        attention_mask,
        scale=scale,
    )
    bias = bias.masked_fill(~source_keys, torch.finfo(dtype).min)
    return bias.to(dtype=dtype)[:, None, None, :]


def mix_attention_outputs(
    causal: torch.Tensor,
    full: torch.Tensor,
    variant: Variant,
    head_gate: torch.Tensor,
    *,
    generic_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mix before o_proj; zero head_gate is exactly the causal pretrained path."""

    if causal.shape != full.shape or causal.ndim != 4:
        raise ValueError("attention outputs must share [B,S,H,D] shape")
    if variant == "causal":
        return causal
    if variant == "full":
        return full
    signed = torch.tanh(head_gate.float()).to(causal.dtype).view(1, 1, -1, 1)
    if variant == "evidence":
        token_gate = 1.0
    elif variant == "dec2enc":
        if generic_logits is None or generic_logits.shape != causal.shape[:3]:
            raise ValueError("dec2enc variant requires [B,S,H] generic logits")
        token_gate = torch.sigmoid(generic_logits.float()).to(causal.dtype).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown dual-mask variant: {variant}")
    return causal + token_gate * signed * (full - causal)


def sdpa_mask(attention_mask: torch.Tensor, causal: bool, query_length: int) -> torch.Tensor:
    """Boolean allowed-attention mask for the audited SDPA fallback."""

    key_valid = attention_mask.bool()[:, None, None, :]
    if not causal:
        return key_valid
    triangle = torch.ones(
        (query_length, attention_mask.shape[1]),
        dtype=torch.bool,
        device=attention_mask.device,
    ).tril()
    return key_valid & triangle[None, None, :, :]
