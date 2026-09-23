"""Historical XOV tensor probe; no pretrained models or end-to-end claims."""

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
    # Fixtures isolate the actual bridge implementation from package loading.
    # This does not test historical config validation.
    pass


source = subprocess.check_output(["git", "show", "efce6f8:src/xov_bridge/eviseq_xov/modeling/xov.py"], text=True)
source = "\n".join(line for line in source.splitlines() if not line.startswith("from ."))
namespace = dict(
    EncoderState=EncoderState,
    BridgeState=BridgeState,
    validate_architecture=validate_architecture,
)
exec(compile(source, "historical_xov.py", "exec"), namespace)
torch.manual_seed(123)
bridge = namespace["CrossTokenizerOrderedValueBridge"](16, 16, {"lexical_rank": 8})
embedding = torch.nn.Embedding(30, 16)
hidden = torch.randn(2, 5, 16, requires_grad=True)
mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
state = EncoderState(hidden, mask, mask)
alignment = dict(
    copy_token_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
    copy_token_mask=torch.ones(2, 4, dtype=torch.bool),
    copy_encoder_indices=torch.arange(4).expand(2, -1),
    copy_token_indices=torch.arange(4).expand(2, -1),
    copy_alignment_weights=torch.ones(2, 4),
)
initial = bridge(state, embedding, **alignment)
result = {
    "scope": "actual historical bridge with fixture state/config; synthetic tensors only",
    "initial_value_identity_max_error": (initial.value_memory - initial.memory).abs().max().item(),
}
optimizer = torch.optim.SGD(bridge.parameters(), lr=0.2)
target = torch.randn_like(hidden)
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    output = bridge(state, embedding, **alignment)
    normalized = torch.nn.functional.rms_norm(output.value_memory, (16,))
    loss = (normalized * target).mean()
    loss.backward()
    result[f"backward_{step + 1}_grad_norms"] = {
        name: float(parameter.grad.norm()) if parameter.grad is not None else None
        for name, parameter in bridge.named_parameters()
    }
    optimizer.step()
output = bridge(state, embedding, **alignment)
result["keys_memory_max_error"] = float((output.memory - initial.memory).abs().max().detach())
result["copy_memory_max_error"] = float((output.copy_memory - initial.copy_memory).abs().max().detach())
result["post_norm_value_delta"] = float(
    (torch.nn.functional.rms_norm(output.value_memory, (16,)) - torch.nn.functional.rms_norm(output.memory, (16,)))
    .norm()
    .detach()
)
result["padding_residual_max_error"] = float((output.value_memory - output.memory)[~mask].abs().max().detach())
empty = bridge(state, embedding)
result["empty_alignment_max_error"] = float((empty.value_memory - empty.memory).abs().max().detach())
assert result["initial_value_identity_max_error"] == 0
assert result["keys_memory_max_error"] == 0
assert result["copy_memory_max_error"] == 0
assert result["padding_residual_max_error"] == 0
assert result["empty_alignment_max_error"] == 0
assert result["post_norm_value_delta"] > 0
for name in ["lexical_up.weight", "lexical_down.weight", "phrase_conv.weight", "value_gate_raw"]:
    assert result["backward_2_grad_norms"][name] > 0
Path(__file__).with_name("probe_xov_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
