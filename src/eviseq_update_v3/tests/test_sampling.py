"""Standalone candidate sampling; benchmark evaluation keeps greedy decoding."""

import pytest
import torch
from eviseq_update_v3.evaluation.generate import _sample_token, generate_greedy, generate_sampled
from eviseq_update_v3.modeling.model import EviSeqAFMR
from eviseq_update_v3.runtime import _TinyTokenizer, build_loaders
from test_planned_semantic_v3 import config_planned
from test_semantic_read import model_config


def test_top_p_retains_highest_probability_and_temperature_changes_concentration():
    scores = torch.tensor([[4.0, 0.0, -1.0]]).expand(5000, -1)
    rng = torch.Generator().manual_seed(3)
    assert _sample_token(scores, 0.01, 1, rng).eq(0).all()
    cold = _sample_token(scores, 1, 0.1, torch.Generator().manual_seed(1))
    hot = _sample_token(scores, 1, 10, torch.Generator().manual_seed(1))
    assert cold.eq(0).float().mean() > 0.99
    assert hot.eq(0).float().mean() < 0.6


@pytest.mark.parametrize("planned", [False, True])
def test_sampling_reproducible_cached_and_independent_of_reference(planned):
    torch.manual_seed(42)
    cfg = model_config("independent_bounded")
    cfg["decoder"]["query_cross_gate"] = True
    cfg["decoder"]["grounded_copy"]["semantic_read"]["rank"] = 8
    cfg["decoder"]["grounded_copy"]["semantic_read"]["num_heads"] = 4
    if planned:
        cfg = config_planned()
    model = EviSeqAFMR(cfg).eval()
    raw_batch = next(iter(build_loaders(cfg, max_train_examples=2)["train"]))
    batch = {key: value for key, value in raw_batch.items() if isinstance(value, torch.Tensor)}
    tokenizer = _TinyTokenizer()
    options = dict(max_new_tokens=4, min_new_tokens=2, top_p=0.8, temperature=0.7)
    _, first = generate_sampled(model, batch, tokenizer, generator=torch.Generator().manual_seed(71), **options)
    batch["labels"].fill_(-100)
    batch["decoder_input_ids"].fill_(0)
    _, second = generate_sampled(model, batch, tokenizer, generator=torch.Generator().manual_seed(71), **options)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    _, third = generate_sampled(model, batch, tokenizer, generator=torch.Generator().manual_seed(72), **options)
    assert not torch.equal(first, third)
    _, greedy = generate_greedy(model, batch, tokenizer, max_new_tokens=4, min_new_tokens=2)
    _, tiny_p = generate_sampled(model, batch, tokenizer, max_new_tokens=4, min_new_tokens=2, top_p=1e-9)
    torch.testing.assert_close(greedy, tiny_p, atol=0, rtol=0)
    assert all(layer.cross._cache is None for layer in model.decoder.backbone.layers)


@pytest.mark.parametrize(
    "options",
    [
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"temperature": 0},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
    ],
)
def test_invalid_sampling_options_fail_before_forward(options):
    with pytest.raises(ValueError):
        generate_sampled(None, None, None, **options)
