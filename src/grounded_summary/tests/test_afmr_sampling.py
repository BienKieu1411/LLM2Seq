import torch

from grounded_summary.evaluation.generate import _sample_token


def test_top_k_never_samples_outside_k_candidates():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    generator = torch.Generator(device="cpu").manual_seed(11)
    draws = torch.cat([_sample_token(scores, top_k=2, top_p=1.0, generator=generator) for _ in range(128)])
    assert set(draws.tolist()) <= {0, 1}


def test_top_p_keeps_the_smallest_prefix_covering_probability_mass():
    scores = torch.tensor([[8.0, 7.0, -8.0, -9.0]])
    generator = torch.Generator(device="cpu").manual_seed(12)
    draws = torch.cat(
        [_sample_token(scores, top_k=0, top_p=0.70, temperature=1.0, generator=generator) for _ in range(128)]
    )
    assert set(draws.tolist()) <= {0, 1}


def test_sampling_falls_back_when_constraints_mask_everything():
    scores = torch.full((2, 5), -float("inf"))
    generator = torch.Generator(device="cpu").manual_seed(13)
    result = _sample_token(scores, top_k=1, top_p=0.9, generator=generator)
    assert result.shape == (2,)
    assert torch.isfinite(result.float()).all()
