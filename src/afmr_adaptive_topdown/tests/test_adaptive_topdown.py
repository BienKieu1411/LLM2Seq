import copy

import pytest
import torch
from eviseq_afmr.config import adaptive_topdown_settings, validate_config
from eviseq_afmr.modeling.adaptive_topdown import AdaptiveTopDownKeyBridge
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.outputs import EncoderState


def _settings(**overrides):
    values = {
        "enabled": True,
        "dim": 16,
        "num_heads": 4,
        "mixer_layers": 1,
        "region_width": 6,
        "region_stride": 4,
        "topdown_rank": 12,
        "gate_init": 0.05,
        "gate_max": 0.30,
        "max_relative_rms": 0.30,
        "output_init_rms": 1.0e-3,
    }
    values.update(overrides)
    return values


def _architecture(*, direct: bool = False):
    return {
        "name": "afmr_adaptive_topdown",
        "bridge_mode": "direct_projection" if direct else "afmr",
        "controller_dim": 16,
        "depth_taps": 2,
        "depth_rank": 8,
        "depth_gate_init": 0.02,
        "depth_gate_max": 0.15,
        "feature_rank": 8,
        "feature_gate_init": 0.02,
        "feature_gate_max": 0.20,
        "focus_hidden": 16,
        "focus_windows": [4, 8],
        "focus_overlap": 0.5,
        "focus_strength_init": 0.10,
        "focus_strength_max": 1.0,
        "temperature_init": 1.0,
        "temperature_min": 0.5,
        "temperature_max": 2.0,
        "contextual_value": {"enabled": False},
        "adaptive_topdown": _settings(),
    }


def test_shapes_masks_and_all_invalid_rows_are_finite():
    torch.manual_seed(2)
    module = AdaptiveTopDownKeyBridge(24, 16, _settings())
    memory = torch.randn(3, 19, 24)
    content = torch.zeros(3, 19, dtype=torch.bool)
    content[0, 2:17] = True
    content[1, 7:11] = True
    correction = module(memory, content, torch.randn(3, 16))
    assert correction.shape == memory.shape
    assert torch.isfinite(correction).all()
    assert not correction[0, 2:17].eq(0).all()
    assert correction.masked_select(~content.unsqueeze(-1)).eq(0).all()
    assert correction[2].eq(0).all()
    correction_rms = torch.sqrt(correction.float().square().mean(dim=-1))
    memory_rms = torch.sqrt(memory.float().square().mean(dim=-1))
    assert torch.all(correction_rms[content] <= 0.30 * memory_rms[content] + 1.0e-6)


def test_long_document_pooling_stays_finite_with_extreme_pool_scores():
    torch.manual_seed(23)
    module = AdaptiveTopDownKeyBridge(24, 16, _settings())
    with torch.no_grad():
        module.pool_score.weight.mul_(10000.0)
    memory = torch.randn(1, 4096, 24)
    content = torch.ones(1, 4096, dtype=torch.bool)
    correction = module(memory, content, torch.randn(1, 16))
    assert torch.isfinite(correction).all()
    assert correction.abs().sum() > 0
    empty_memory = torch.randn(1, 4096, 24)
    empty_content = torch.zeros(1, 4096, dtype=torch.bool)
    extreme_empty = module(empty_memory, empty_content, torch.full((1, 16), 1.0e6))
    assert torch.isfinite(extreme_empty).all() and extreme_empty.eq(0).all()


def test_regional_path_uses_content_only_despite_prefix_and_padding_changes():
    torch.manual_seed(3)
    module = AdaptiveTopDownKeyBridge(24, 16, _settings()).eval()
    article = torch.randn(1, 13, 24)
    controller = torch.randn(1, 16)

    def run(prefix: int, padding: int):
        memory = torch.randn(1, prefix + article.shape[1] + padding, 24)
        memory[:, prefix : prefix + article.shape[1]] = article
        content = torch.zeros(memory.shape[:2], dtype=torch.bool)
        content[:, prefix : prefix + article.shape[1]] = True
        return module(memory, content, controller)[:, prefix : prefix + article.shape[1]]

    expected = run(0, 0)
    for prefix, padding in ((3, 0), (0, 17), (5, 21)):
        torch.testing.assert_close(run(prefix, padding), expected, atol=2.0e-6, rtol=2.0e-5)


def test_first_backward_updates_pooling_mixer_and_topdown_routes():
    torch.manual_seed(5)
    module = AdaptiveTopDownKeyBridge(24, 16, _settings())
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-2)
    memory = torch.randn(2, 21, 24, requires_grad=True)
    content = torch.ones(2, 21, dtype=torch.bool)
    content[:, :2] = False
    controller = torch.randn(2, 16, requires_grad=True)
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    correction = module(memory, content, controller)
    loss = (correction * torch.randn_like(correction)).sum()
    loss.backward()

    parameters = dict(module.named_parameters())
    for name in (
        "pool_score.weight",
        "region_position.weight",
        "mixers.0.attention.in_proj_weight",
        "topdown_attention.in_proj_weight",
        "topdown_down.weight",
        "topdown_out.weight",
        "gate_raw.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert controller.grad is not None and controller.grad.abs().sum() > 0
    optimizer.step()
    for name in ("pool_score.weight", "mixers.0.attention.in_proj_weight", "topdown_out.weight"):
        assert not torch.equal(parameters[name], before[name]), name


def test_direct_projection_remains_a_true_bridge_free_control():
    torch.manual_seed(7)
    bridge = AdaptiveFullMemoryResidualBridge(12, 16, _architecture(direct=True))
    assert not hasattr(bridge, "controller")
    assert not hasattr(bridge, "adaptive_topdown")
    final = torch.randn(2, 9, 12)
    valid = torch.ones(2, 9, dtype=torch.bool)
    content = valid.clone()
    content[:, :2] = False
    output = bridge(
        EncoderState(final, (final + 0.2, final), valid, content),
        torch.randn(2, 3, 16),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([32.0, 48.0]),
    )
    expected = bridge.direct_projection(final.float())
    torch.testing.assert_close(output.memory, expected)
    assert output.source_bias.eq(0).all()
    assert output.value_memory is None


def test_topdown_changes_keys_but_preserves_h0_value_anchor():
    torch.manual_seed(11)
    bridge = AdaptiveFullMemoryResidualBridge(12, 16, _architecture())
    final = torch.randn(2, 17, 12)
    valid = torch.ones(2, 17, dtype=torch.bool)
    content = valid.clone()
    content[:, :2] = False
    state = EncoderState(final, (final + 0.2, final), valid, content)
    prompt = torch.randn(2, 3, 16)
    prompt_mask = torch.ones(2, 3, dtype=torch.bool)
    output = bridge(state, prompt, prompt_mask, torch.tensor([32.0, 48.0]))
    expected_anchor = bridge.base_projection(final.float())
    expected_anchor = expected_anchor.masked_fill(~valid.unsqueeze(-1), 0.0)
    torch.testing.assert_close(output.value_memory, expected_anchor, rtol=0, atol=0)
    assert not torch.equal(
        output.memory.masked_select(content.unsqueeze(-1)), output.value_memory.masked_select(content.unsqueeze(-1))
    )


def test_nested_config_validation_rejects_invalid_heads_and_stride():
    architecture = _architecture()
    bad_heads = copy.deepcopy(architecture)
    bad_heads["adaptive_topdown"]["num_heads"] = 3
    with pytest.raises(ValueError, match="divisible"):
        adaptive_topdown_settings(bad_heads)
    bad_stride = copy.deepcopy(architecture)
    bad_stride["adaptive_topdown"]["region_stride"] = 7
    with pytest.raises(ValueError, match="must not exceed"):
        adaptive_topdown_settings(bad_stride)


def test_adaptive_architecture_requires_enabled_bridge():
    # Use the real base schema so the assertion covers integration with the
    # top-level validator rather than just the nested helper.
    from pathlib import Path

    from eviseq_afmr.config import load_config

    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_base.yaml")
    config["architecture"]["adaptive_topdown"]["enabled"] = False
    with pytest.raises(ValueError, match="requires"):
        validate_config(config)


def test_tiny_end_to_end_ce_reaches_adaptive_bridge_on_first_backward():
    from pathlib import Path

    from eviseq_afmr.config import load_config
    from eviseq_afmr.modeling.model import EviSeqAFMR
    from eviseq_afmr.runtime import build_loaders

    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["architecture"].update(name="afmr_adaptive_topdown", adaptive_topdown=_settings())
    validate_config(config)
    model = EviSeqAFMR(config)
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    output = model(
        **{key: value for key, value in batch.items() if isinstance(value, torch.Tensor)},
        return_logits=False,
    )
    output.loss.backward()
    topdown = model.bridge.adaptive_topdown
    assert topdown is not None and torch.isfinite(output.loss)
    for parameter in (
        topdown.pool_score.weight,
        topdown.mixers[0].attention.in_proj_weight,
        topdown.topdown_attention.in_proj_weight,
        topdown.topdown_out.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert output.bridge.value_memory is not None
    assert torch.isfinite(output.bridge.value_memory).all()
