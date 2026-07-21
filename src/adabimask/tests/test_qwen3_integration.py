import torch
from adabimask.mask_policy import LayerMaskPolicy, MaskPolicyConfig
from adabimask.pretrained_decoder import CrossAttentionInjectedLayer, normalized_layer_indices
from adabimask.routed_attention import RoutedLinearAttention, RoutedSelfAttention
from transformers import Qwen3Config, Qwen3Model, Qwen3_5TextConfig, Qwen3_5TextModel


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


def test_routing_supports_qwen35_hybrid_attention_and_deltanet():
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config)
    policy = LayerMaskPolicy(
        4,
        MaskPolicyConfig(mode="learnable", num_groups=1, budget_groups=1, hard_eval=True),
    )
    model.policy = policy
    for layer_index, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn = RoutedSelfAttention(layer.self_attn, layer_index, policy)
        else:
            layer.linear_attn = RoutedLinearAttention(layer.linear_attn, layer_index, policy)

    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    attention_mask = torch.tensor([[1] * 12, [1] * 9 + [0] * 3])
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    loss = output.last_hidden_state.float().square().mean() + policy.regularization()["loss_gate"]
    loss.backward()

    assert output.last_hidden_state.shape == (2, 12, config.hidden_size)
    assert policy.gate_logits.grad is not None
    assert torch.isfinite(policy.gate_logits.grad).all()


def test_cross_attention_wrapper_flows_through_qwen35_text_model():
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config)
    source_attention = model.layers[-1].self_attn
    model.layers = torch.nn.ModuleList(
        [CrossAttentionInjectedLayer(layer, config, source_attention, 0.0) for layer in model.layers]
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    memory = torch.randn(2, 9, config.hidden_size)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones(2, 6),
        use_cache=False,
        encoder_hidden_states=memory,
        encoder_attention_mask=torch.tensor([[1] * 9, [1] * 7 + [0] * 2]),
    )
    output.last_hidden_state.float().square().mean().backward()
    assert output.last_hidden_state.shape == (2, 6, config.hidden_size)
    assert all(layer.cross_gate.grad is not None for layer in model.layers)
    assert normalized_layer_indices(24, 16) == (0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23)
