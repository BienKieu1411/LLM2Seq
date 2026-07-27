from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from llm2seq_v5.phrase_pointer import StatefulPhrasePointer
from llm2seq_v5.training import _tokenizers

from llm2seq_v5.data import source_phrase_start_labels


def _pointer(*, generate_probability: float = 0.98, continuation: bool = True) -> StatefulPhrasePointer:
    torch.manual_seed(7)
    return StatefulPhrasePointer(
        hidden_size=4,
        vocabulary_size=10,
        rank=2,
        phrase_hidden_size=4,
        dropout=0.0,
        generate_probability_init=generate_probability,
        use_continuation=continuation,
    )


def _generation_step(
    pointer: StatefulPhrasePointer,
    *,
    source_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    previous: torch.Tensor | None = None,
    lm_token: int = 2,
):
    batch, source_length = source_ids.shape
    memory = torch.randn(batch, source_length, 4)
    logits = torch.full((batch, 10), -6.0)
    logits[:, lm_token] = 6.0
    return pointer.generation_step(
        decoder_state=torch.randn(batch, 4),
        lm_logits=logits,
        source_memory=memory,
        source_token_ids=source_ids,
        source_unit_ids=unit_ids,
        source_copy_mask=unit_ids.gt(0),
        previous_responsibility=previous,
        attention_bias=None,
    )


def test_phrase_labels_ignore_prompt_and_cross_unit_ngrams():
    labels = source_phrase_start_labels(
        source_token_ids=[9, 2, 3, 4, 5],
        unit_ids=[0, 1, 1, 2, 2],
        target_token_ids=[2, 3, 4, 8],
    )
    assert labels.shape == (5, 3)
    assert labels[0].tolist() == [-1.0, -1.0, -1.0]
    assert labels[1, 0].item() == 1.0  # 2,3 is present and stays in unit 1.
    assert labels[1, 1].item() == -1.0  # 2,3,4 crosses into unit 2.
    assert labels[2, 0].item() == -1.0  # 3,4 crosses the boundary.


def test_all_mask_is_exact_pure_lm_and_normalized():
    pointer = _pointer().eval()
    source_ids = torch.tensor([[2, 3, 4]])
    units = torch.zeros_like(source_ids)
    logits = torch.tensor([[0.1, -0.2, 0.3, 0.0, -0.4, 0.2, 0.7, -0.1, 0.4, 0.6]])
    step = pointer.generation_step(
        decoder_state=torch.randn(1, 4),
        lm_logits=logits,
        source_memory=torch.randn(1, 3, 4),
        source_token_ids=source_ids,
        source_unit_ids=units,
        source_copy_mask=units.gt(0),
        previous_responsibility=None,
        attention_bias=None,
    )
    torch.testing.assert_close(step.log_probabilities, torch.log_softmax(logits, dim=-1))
    torch.testing.assert_close(step.mode_probabilities, torch.tensor([[1.0, 0.0, 0.0]]))
    assert step.source_pointer_mass.count_nonzero().item() == 0
    torch.testing.assert_close(torch.logsumexp(step.log_probabilities, dim=-1), torch.zeros(1))


def test_bayes_state_keeps_copy_probability_and_marginalizes_duplicates():
    pointer = _pointer(generate_probability=0.99).eval()
    source = torch.tensor([[2, 5, 2]])
    units = torch.ones_like(source)
    step = _generation_step(pointer, source_ids=source, unit_ids=units, lm_token=2)
    responsibility = pointer.posterior_source_responsibility(step, torch.tensor([2]), source)
    assert 0.0 < responsibility.sum().item() < 0.02
    assert responsibility[0, 0].item() > 0.0
    assert responsibility[0, 2].item() > 0.0
    assert responsibility[0, 1].item() == 0.0
    absent = pointer.posterior_source_responsibility(step, torch.tensor([7]), source)
    assert absent.count_nonzero().item() == 0


def test_continuation_tracks_three_token_chain_without_crossing_units():
    pointer = _pointer(generate_probability=0.5).eval()
    source = torch.tensor([[2, 3, 4]])
    same_unit = torch.ones_like(source)
    first = _generation_step(pointer, source_ids=source, unit_ids=same_unit, lm_token=2)
    state_2 = pointer.posterior_source_responsibility(first, torch.tensor([2]), source)
    second = _generation_step(
        pointer,
        source_ids=source,
        unit_ids=same_unit,
        previous=state_2,
        lm_token=3,
    )
    assert second.continuation_attention.argmax(dim=-1).item() == 1
    state_3 = pointer.posterior_source_responsibility(second, torch.tensor([3]), source)
    third = _generation_step(
        pointer,
        source_ids=source,
        unit_ids=same_unit,
        previous=state_3,
        lm_token=4,
    )
    assert third.continuation_attention.argmax(dim=-1).item() == 2

    split_units = torch.tensor([[1, 1, 2]])
    blocked = _generation_step(
        pointer,
        source_ids=source,
        unit_ids=split_units,
        previous=state_3,
        lm_token=4,
    )
    assert blocked.mode_probabilities[0, 2].item() == 0.0


def test_no_continuation_ablation_removes_third_mode():
    pointer = _pointer(generate_probability=0.5, continuation=False).eval()
    source = torch.tensor([[2, 3, 4]])
    units = torch.ones_like(source)
    previous = torch.tensor([[0.5, 0.0, 0.0]])
    step = _generation_step(
        pointer,
        source_ids=source,
        unit_ids=units,
        previous=previous,
        lm_token=3,
    )
    assert step.mode_probabilities[0, 2].item() == 0.0


def _teacher_loss(pointer: StatefulPhrasePointer, with_prompt: bool = False):
    source_ids = torch.tensor([[2, 3, 4]])
    unit_ids = torch.ones_like(source_ids)
    memory = torch.randn(1, 3, 4, requires_grad=True)
    target_states = torch.randn(1, 2, 4)
    labels = torch.tensor([[2, 3]])
    decoder_inputs = torch.tensor([[1, 2]])
    if with_prompt:
        target_states = torch.cat([torch.randn(1, 1, 4), target_states], dim=1)
        labels = torch.cat([torch.tensor([[-100]]), labels], dim=1)
        decoder_inputs = torch.cat([torch.tensor([[9]]), decoder_inputs], dim=1)
    states = target_states.detach().requires_grad_(True)
    # Make the pretrained LM confidently wrong on both gold tokens. This
    # exercises the log-space mixture rather than a merely uniform softmax.
    logits = torch.full((2, 10), -1000.0)
    logits[:, 9] = 1000.0
    logits.requires_grad_()
    phrase_labels = source_phrase_start_labels([2, 3, 4], [1, 1, 1], [2, 3]).unsqueeze(0)
    losses = pointer.teacher_forced_loss(
        decoder_states=states,
        lm_logits=logits,
        labels=labels,
        decoder_input_ids=decoder_inputs,
        source_memory=memory,
        source_token_ids=source_ids,
        source_unit_ids=unit_ids,
        source_copy_mask=unit_ids.gt(0),
        attention_bias=None,
        phrase_labels=phrase_labels,
    )
    return losses, states, memory


def test_extreme_lm_logits_keep_pointer_loss_and_gradients_finite():
    pointer = _pointer(generate_probability=0.9).train()
    losses, states, memory = _teacher_loss(pointer)
    total = sum(
        losses[name]
        for name in (
            "loss_phrase_mixture",
            "loss_phrase_copy",
            "loss_phrase_continue",
            "loss_phrase_labels",
            "loss_phrase_coverage",
        )
    )
    assert torch.isfinite(total)
    total.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert pointer.mode_gate.bias.grad is not None and torch.isfinite(pointer.mode_gate.bias.grad).all()
    assert losses["phrase_continuation_available_rate"].item() > 0.0


def test_unsupervised_prompt_does_not_change_coverage():
    pointer = _pointer().eval()
    torch.manual_seed(11)
    without_prompt, _, _ = _teacher_loss(pointer, with_prompt=False)
    torch.manual_seed(11)
    with_prompt, _, _ = _teacher_loss(pointer, with_prompt=True)
    torch.testing.assert_close(
        without_prompt["loss_phrase_coverage"],
        with_prompt["loss_phrase_coverage"],
    )


def test_source_cache_matches_uncached_and_clears():
    pointer = _pointer().eval()
    memory = torch.randn(1, 3, 4)
    states = torch.randn(1, 1, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    uncached = pointer._scores(states, memory, mask, None)
    pointer.prepare_source_cache(memory)
    cached = pointer._scores(states, memory, mask, None)
    for left, right in zip(uncached, cached):
        torch.testing.assert_close(left, right)
    pointer.clear_source_cache()
    assert pointer._cached_keys is None
    assert pointer._cached_phrase_logits is None


class _FakeTokenizer:
    def __init__(self, vocab, *, eos=1):
        self._vocab = vocab
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = eos
        self.unk_token_id = None
        self.padding_side = "left"

    def get_vocab(self):
        return dict(self._vocab)


@pytest.mark.parametrize(
    "decoder",
    [
        _FakeTokenizer({"a": 0, "b": 2}),
        _FakeTokenizer({"a": 0, "b": 1}, eos=2),
    ],
)
def test_pointer_rejects_vocab_or_special_id_mismatch(monkeypatch, decoder):
    import transformers

    encoder = _FakeTokenizer({"a": 0, "b": 1})
    tokenizers = iter((encoder, decoder))
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: next(tokenizers)),
    )
    config = {
        "model": {"encoder_name": "encoder", "decoder_name": "decoder"},
        "phrase_pointer": {"enabled": True},
    }
    with pytest.raises(ValueError, match="requires"):
        _tokenizers(config)
