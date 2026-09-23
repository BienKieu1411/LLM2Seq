"""Falsification probe of historical XOV's local-order claim, synthetic only."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


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
    pass


source = subprocess.check_output(["git", "show", "efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py"], text=True)
source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
namespace = dict(EncoderState=EncoderState, BridgeState=BridgeState, validate_architecture=validate_architecture)
exec(compile(source, "historical_xov.py", "exec"), namespace)
torch.manual_seed(256)
bridge = namespace["CrossTokenizerOrderedValueBridge"](16, 16, {"lexical_rank": 8})
embedding = torch.nn.Embedding(20, 16)
with torch.no_grad():
    bridge.lexical_up.weight.normal_(0, 0.2)
# Research-only alternate ordering; production src remains untouched.
alternative_source = source.replace(
    "            ordered,\n            content,", "            F.silu(ordered),\n            content,"
).replace("self.lexical_up(F.silu(pooled).to(", "self.lexical_up(pooled.to(")
assert alternative_source != source
alternative_namespace = dict(
    EncoderState=EncoderState, BridgeState=BridgeState, validate_architecture=validate_architecture
)
exec(compile(alternative_source, "research_alternative.py", "exec"), alternative_namespace)
alternative = alternative_namespace["CrossTokenizerOrderedValueBridge"](16, 16, {"lexical_rank": 8})
alternative.load_state_dict(bridge.state_dict())
mask = torch.ones(1, 4, dtype=torch.bool)
state = EncoderState(torch.randn(1, 4, 16), mask, mask)


def output(ids, destinations, model=bridge):
    return model(
        state,
        embedding,
        copy_token_ids=torch.tensor([ids]),
        copy_token_mask=mask,
        copy_encoder_indices=torch.tensor([destinations]),
        copy_token_indices=torch.arange(4)[None],
        copy_alignment_weights=torch.ones(1, 4),
    ).value_memory


with torch.no_grad():
    # Different internal order, same endpoints; all four decoder subwords map
    # into one encoder token. Linear conv + averaging loses internal order.
    many_a = output([1, 2, 3, 4], [0, 0, 0, 0])
    many_b = output([1, 3, 2, 4], [0, 0, 0, 0])
    one_a = output([1, 2, 3, 4], [0, 1, 2, 3])
    one_b = output([1, 3, 2, 4], [0, 1, 2, 3])
    alt_a = output([1, 2, 3, 4], [0, 0, 0, 0], alternative)
    alt_b = output([1, 3, 2, 4], [0, 0, 0, 0], alternative)
    # Pure positive row scaling is removed by ideal RMS normalization;
    # PyTorch epsilon leaves only roundoff-scale deviations here.
    radial_a = torch.nn.functional.rms_norm(state.final, (16,))
    radial_b = torch.nn.functional.rms_norm(1.05 * state.final, (16,))
    result = {
        "scope": "synthetic counterexample, not corpus prevalence or score evidence",
        "many_to_one_order_swap_max_delta": float((many_a - many_b).abs().max()),
        "one_to_one_order_swap_max_delta": float((one_a - one_b).abs().max()),
        "pure_radial_norm_delta": float((radial_a - radial_b).abs().max()),
        "activation_before_scatter_order_swap_max_delta": float((alt_a - alt_b).abs().max()),
        "activation_before_scatter_one_to_one_control_max_delta": float(
            (output([1, 2, 3, 4], [0, 1, 2, 3], alternative) - one_a).abs().max()
        ),
    }
assert result["many_to_one_order_swap_max_delta"] < 1e-6
assert result["one_to_one_order_swap_max_delta"] > 1e-5
assert result["activation_before_scatter_order_swap_max_delta"] > 1e-5
assert result["activation_before_scatter_one_to_one_control_max_delta"] < 1e-6
Path(__file__).with_name("probe_order_collision_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
