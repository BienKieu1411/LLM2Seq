import torch
from adabimask.mask_policy import LayerMaskPolicy, MaskPolicyConfig
from adabimask.routed_attention import RoutedSelfAttention
from transformers import Qwen3Config, Qwen3Model


def test_routed_attention_matches_real_qwen3_layer_signature():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3Model(config)
    policy = LayerMaskPolicy(
        4,
        MaskPolicyConfig(mode="learnable", num_groups=2, budget_groups=1, hard_eval=True),
    )
    model.policy = policy
    for layer_index, layer in enumerate(model.layers):
        layer.self_attn = RoutedSelfAttention(layer.self_attn, layer_index, policy)

    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    attention_mask = torch.tensor([[1] * 12, [1] * 9 + [0] * 3])
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    loss = output.last_hidden_state.float().square().mean() + policy.regularization()["loss_gate"]
    loss.backward()

    assert output.last_hidden_state.shape == (2, 12, config.hidden_size)
    assert policy.gate_logits.grad is not None
    assert torch.isfinite(policy.gate_logits.grad).all()
