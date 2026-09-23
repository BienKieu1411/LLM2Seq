"""Research-only XOV repairs, loading the recorded historical class from Git.

Synthetic tensors only. This is not a production port or a ROUGE experiment.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class EncoderState:
    final: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor


@dataclass
class BridgeState:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    content_mask: torch.Tensor
    value_memory: torch.Tensor
    copy_memory: torch.Tensor


def validate_architecture(config):
    # Historical configuration validation is intentionally outside this tensor probe.
    pass


def load_historical():
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        ["git", "show", "efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py"], cwd=root, text=True
    )
    source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
    namespace = dict(EncoderState=EncoderState, BridgeState=BridgeState, validate_architecture=validate_architecture)
    exec(compile(source, "historical_xov.py", "exec"), namespace)
    return namespace["CrossTokenizerOrderedValueBridge"]


def gap_aware_conv(compact, mask, original_positions, conv):
    """Width-three depthwise cross-correlation; no mixing across discarded tokens."""
    x = compact.masked_fill(~mask[..., None], 0)
    neighbor = mask[:, :-1] & mask[:, 1:] & (original_positions[:, 1:] == original_positions[:, :-1] + 1)
    left = F.pad(x[:, :-1] * neighbor[..., None], (0, 0, 1, 0))
    right = F.pad(x[:, 1:] * neighbor[..., None], (0, 0, 0, 1))
    w = conv.weight[:, 0, :]
    out = left * w[:, 0] + x * w[:, 1] + right * w[:, 2]
    return out.masked_fill(~mask[..., None], 0)


def repaired_forward(bridge, state, embedding, ids, mask, positions, destinations):
    memory = bridge.direct_projection(state.final)
    memory = memory.masked_fill(~state.attention_mask[..., None], 0)
    bank = bridge.lexical_down(bridge.lexical_norm(embedding(ids)))
    ordered = gap_aware_conv(bank, mask, positions, bridge.phrase_conv)
    # Activate BEFORE many-to-one pooling.
    pooled, aligned = bridge._reverse_scatter(
        F.silu(ordered),
        state.content_mask & state.attention_mask,
        mask,
        destinations,
        torch.arange(ids.shape[1])[None].expand_as(ids),
        mask.float(),
    )
    residual = bridge._unit_capped(bridge.lexical_up(pooled), memory)
    residual = residual.masked_fill(~aligned[..., None], 0)
    gate = bridge.value_gate_max * bridge.value_gate_raw.sigmoid()
    return BridgeState(memory, state.attention_mask, state.content_mask, memory + gate * residual, memory)


def main():
    torch.manual_seed(256)
    bridge = load_historical()(16, 16, {"lexical_rank": 8})
    embedding = torch.nn.Embedding(32, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    state = EncoderState(torch.randn(1, 4, 16), mask, mask)
    ids = torch.tensor([[1, 2, 3, 4]])
    pos = torch.arange(4)[None]
    destinations = torch.zeros_like(ids)

    initial = repaired_forward(bridge, state, embedding, ids, mask, pos, destinations)
    assert torch.equal(initial.memory, initial.value_memory)
    optimizer = torch.optim.SGD(bridge.parameters(), lr=0.1)
    read = torch.randn(16, 7)

    def loss_and_backward():
        output = repaired_forward(bridge, state, embedding, ids, mask, pos, destinations)
        # A toy categorical reader, NOT the actual cross-attention decoder.
        logits = F.rms_norm(output.value_memory[:, 0], (16,)) @ read
        F.cross_entropy(logits, torch.tensor([2])).backward()
        return output

    loss_and_backward()
    grad_first = {name: float(p.grad.abs().sum()) for name, p in bridge.named_parameters() if p.grad is not None}
    assert grad_first["lexical_up.weight"] > 0
    assert grad_first["lexical_down.weight"] == 0
    optimizer.step()
    bridge.zero_grad(set_to_none=True)
    embedding.zero_grad(set_to_none=True)
    active = loss_and_backward()
    grad_second = {name: float(p.grad.abs().sum()) for name, p in bridge.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in bridge.parameters() if p.grad is not None)
    for name in ("lexical_down.weight", "phrase_conv.weight", "lexical_up.weight", "value_gate_raw"):
        assert grad_second[name] > 0, name

    with torch.no_grad():
        bridge.lexical_up.weight.normal_(0, 0.2)
        original = repaired_forward(bridge, state, embedding, ids, mask, pos, destinations)
        swapped = repaired_forward(bridge, state, embedding, ids[:, [0, 2, 1, 3]], mask, pos, destinations)
        order_delta = float((original.value_memory - swapped.value_memory).abs().max())
        assert order_delta > 1e-5
        assert torch.equal(original.memory, original.copy_memory)

        empty = repaired_forward(bridge, state, embedding, ids, mask & False, pos, destinations)
        assert torch.equal(empty.value_memory, empty.memory)
        cap_ratio = float(
            (
                (original.value_memory - original.memory).square().mean(-1).sqrt()
                / original.memory.square().mean(-1).sqrt().clamp_min(1e-9)
            ).max()
        )
        assert cap_ratio <= bridge.value_gate_max + 1e-6

        x = torch.randn(1, 4, 8)
        contiguous = gap_aware_conv(x, mask, pos, bridge.phrase_conv)
        plain = bridge.phrase_conv(x.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(contiguous, plain, atol=1e-6, rtol=1e-6)
        # Original stream positions [0,2,3,4]: position 1 was discarded.
        gaps = torch.tensor([[0, 2, 3, 4]])
        changed = x.clone()
        changed[:, 1] += 10
        naive_delta = float(
            (bridge.phrase_conv(changed.transpose(1, 2)).transpose(1, 2)[:, 0] - plain[:, 0]).abs().max()
        )
        guarded_delta = float(
            (
                gap_aware_conv(changed, mask, gaps, bridge.phrase_conv)[:, 0]
                - gap_aware_conv(x, mask, gaps, bridge.phrase_conv)[:, 0]
            )
            .abs()
            .max()
        )
        assert naive_delta > 0.1
        assert guarded_delta == 0
        padded_x = F.pad(x, (0, 0, 0, 3))
        padded_mask = F.pad(mask, (0, 3))
        padded_pos = F.pad(pos, (0, 3), value=-1)
        padded_out = gap_aware_conv(padded_x, padded_mask, padded_pos, bridge.phrase_conv)
        torch.testing.assert_close(padded_out[:, :4], contiguous)
        assert not padded_out[:, 4:].any()

    # Exercise the actual historical aligner: a discarded special token closes
    # a gap in the compact IDs, proving the hand-built gap can arise upstream.
    root = Path(__file__).resolve().parents[3]
    alignment_source = subprocess.check_output(
        ["git", "show", "efce6f8:src/xov_bridge/eviseq_xov/data/source_alignment.py"],
        cwd=root,
        text=True,
    )
    alignment_namespace = {}
    exec(compile(alignment_source, "historical_alignment.py", "exec"), alignment_namespace)

    class TokenizerFixture:
        all_special_ids = (2,)

        def __call__(self, text, **kwargs):
            assert text == "a#b"
            return {"input_ids": [10, 2, 11], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}

    actual_alignment = alignment_namespace["align_source_tokens"](
        "a#b", 0, [(0, 1), (1, 2), (2, 3)], TokenizerFixture()
    )
    assert actual_alignment["copy_token_ids"] == [10, 11]

    result = {
        "scope": "synthetic research prototype; no pretrained model, tokenizer-corpus, DDP or ROUGE test",
        "identity_at_initialization": True,
        "gap_crossing_naive_max_delta": naive_delta,
        "gap_crossing_corrected_max_delta": guarded_delta,
        "historical_aligner_compacts_gap_fixture": actual_alignment["copy_token_ids"],
        "nonlinear_before_pool_order_swap_max_delta": order_delta,
        "contiguous_conv_max_error": float((contiguous - plain).abs().max()),
        "padding_invariance": True,
        "empty_alignment_identity": True,
        "relative_value_residual_rms_max": cap_ratio,
        "first_backward_gradient_l1": grad_first,
        "second_backward_gradient_l1": grad_second,
        "post_norm_effect_after_one_step": float(
            (F.rms_norm(active.value_memory, (16,)) - F.rms_norm(active.memory, (16,))).detach().abs().max()
        ),
    }
    Path(__file__).with_name("probe_repairs_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
