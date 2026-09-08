import pytest
import torch
from eviseq_update.evaluation.generate import _sample_token, generate_greedy, generate_sampled
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.runtime import _TinyTokenizer
from test_supervised_training import config, tensors


def test_top_p_retains_highest_probability_and_temperature_changes_concentration():
    scores = torch.tensor([[4.0, 0.0, -1.0]]).expand(5000, -1)
    rng = torch.Generator().manual_seed(3)
    assert _sample_token(scores, 0.01, 1, rng).eq(0).all()
    cold = _sample_token(scores, 1, 0.1, torch.Generator().manual_seed(1))
    hot = _sample_token(scores, 1, 10, torch.Generator().manual_seed(1))
    assert cold.eq(0).float().mean() > 0.99
    assert hot.eq(0).float().mean() < 0.6


def test_sampling_reproducible_cached_and_independent_of_reference():
    cfg = config()
    model = EviSeqAFMR(cfg).eval()
    batch = tensors(cfg)
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


@pytest.mark.parametrize("options", [{"top_p": 0}, {"top_p": 1.1}, {"temperature": 0}, {"temperature": float("inf")}])
def test_invalid_sampling_options_fail_before_forward(options):
    with pytest.raises(ValueError):
        generate_sampled(None, None, None, **options)
