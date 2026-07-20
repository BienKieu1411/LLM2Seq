import torch
import torch.nn as nn
from adabimask.mask_policy import LayerMaskPolicy, MaskPolicyConfig
from adabimask.routed_attention import RoutedSelfAttention, make_bidirectional_mask


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.calls += 1
        # The causal mask blocks the final key for query zero; the full mask does not.
        visible = (attention_mask[:, :, :1, -1:] == 0).to(hidden_states.dtype)
        return hidden_states + visible.reshape(hidden_states.shape[0], 1, 1), None


def causal_mask(batch_size=1, sequence_length=4):
    mask = torch.full((sequence_length, sequence_length), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None].expand(batch_size, -1, -1, -1).clone()


def test_bidirectional_mask_removes_triangle_and_keeps_padding():
    hidden = torch.zeros(1, 4, 3)
    mask = causal_mask()
    mask[..., -1] = torch.finfo(mask.dtype).min
    full = make_bidirectional_mask(mask, hidden)
    assert full.shape == (1, 1, 4, 4)
    assert torch.all(full[..., :3] == 0)
    assert torch.all(full[..., -1] < 0)


def test_fixed_routes_compute_only_one_attention_branch():
    hidden = torch.zeros(1, 4, 2)
    mask = causal_mask()
    policy = LayerMaskPolicy(1, MaskPolicyConfig(mode="full", num_groups=1, budget_groups=1))
    base = DummyAttention()
    routed = RoutedSelfAttention(base, 0, policy)
    output, _ = routed(hidden_states=hidden, attention_mask=mask, past_key_value=None)
    assert base.calls == 1
    assert torch.all(output == 1)


def test_soft_route_computes_two_branches_and_updates_gate():
    hidden = torch.zeros(1, 4, 2)
    mask = causal_mask()
    policy = LayerMaskPolicy(
        1,
        MaskPolicyConfig(mode="learnable", num_groups=1, budget_groups=1, init_probability=0.5),
    )
    policy.train()
    base = DummyAttention()
    routed = RoutedSelfAttention(base, 0, policy)
    output, _ = routed(hidden_states=hidden, attention_mask=mask, past_key_value=None)
    assert base.calls == 2
    assert torch.allclose(output, torch.full_like(output, 0.5))
    output.sum().backward()
    assert policy.gate_logits.grad is not None
