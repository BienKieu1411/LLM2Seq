import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from eviseq_afmr.config import load_config
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS, align_copy_tokens, pad_copy_alignments
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.evaluation.generate import generate_greedy
from eviseq_afmr.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _TinyTokenizer, build_loaders
from eviseq_afmr.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def setup_model():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["ce_chunk_size"] = 8
    model = EviSeqAFMR(config)
    loader = build_loaders(config, max_train_examples=2)["train"]
    return config, model, next(iter(loader)), loader.collate_fn.decoder_tokenizer


def forward(model, batch, logits=False):
    return model(
        **{key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}, return_logits=logits
    )


def bridge(model, batch):
    return model.encode_source(
        *(
            batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "source_content_mask",
                "decoder_prompt_ids",
                "decoder_prompt_mask",
            )
        ),
        torch.full((batch["input_ids"].shape[0],), 32.0),
        **{key: batch[key] for key in COPY_INPUT_KEYS},
    )


def test_alignment_handles_different_segmentations_and_decoder_vocabulary():
    class Tokenizer:
        all_special_ids = [99]

        def __call__(self, text, **kwargs):
            assert text == "Spain 2003"
            return {"input_ids": [17, 18, 19, 99], "offset_mapping": [(0, 5), (6, 8), (8, 10), (0, 0)]}

    row = align_copy_tokens("Spain 2003", 2, [(0, 2), (2, 4), (4, 7), (8, 12), (0, 0)], Tokenizer())
    assert row["copy_token_ids"] == [17, 18, 19]
    assert row["copy_encoder_indices"] == [1, 2, 3, 3]
    assert row["copy_token_indices"] == [0, 0, 1, 2]
    assert row["copy_alignment_weights"] == [0.4, 0.6, 1.0, 1.0]


def test_alignment_excludes_prompt_truncated_boundary_and_special_tokens():
    class Tokenizer:
        all_special_ids = [99]

        def __call__(self, text, **kwargs):
            assert text == "Spain 20"
            return {"input_ids": [99, 17, 18], "offset_mapping": [(0, 0), (0, 5), (6, 8)]}

    row = align_copy_tokens("Spain 2003 hidden", 2, [(0, 2), (2, 7), (8, 10)], Tokenizer())
    assert row["copy_token_ids"] == [17]
    assert row["copy_encoder_indices"] == [1]


def test_alignment_handles_unicode_duplicate_spans_and_missing_content():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [7, 8], "offset_mapping": [(0, 1), (2, 5)]}

    row = align_copy_tokens("é abc", 0, [(0, 1), (0, 1), (2, 3), (4, 5)], Tokenizer())
    assert row["copy_token_ids"] == [7]
    assert row["copy_alignment_weights"] == [0.5, 0.5]


def test_copy_inputs_do_not_depend_on_reference_or_include_decoder_instruction():
    collator = SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {
            "encoder_prefix": "instruction: ",
            "decoder_prompt": "output: ",
            "max_source_length": 6,
        },
        grounded_copy=True,
    )
    a = collator([CanonicalRecord("a", "same source Spain 2003", "one reference")])
    collator.include_targets = False
    b = collator([CanonicalRecord("a", "same source Spain 2003", "another reference")])
    for key in COPY_INPUT_KEYS:
        torch.testing.assert_close(a[key], b[key])
    assert a["copy_encoder_indices"].min() >= 2
    assert a["copy_alignment_weights"].dtype == torch.float32


def test_mixture_is_normalized_sums_duplicate_ids_and_falls_back_to_lm():
    torch.manual_seed(1)
    head = GroundedCopyHead(8, 4, 0.05)
    hidden, logits = torch.randn(2, 3, 8), torch.randn(2, 3, 13)
    state = CopyState(
        torch.randn(2, 4, 4),
        torch.tensor([[4, 4, 5, 6], [7, 8, 9, 10]]),
        torch.tensor([[True, True, True, False], [False, False, False, False]]),
        torch.zeros(2, 4),
    )
    log_attention, log_copy, log_gen = head.distribution(hidden, state)
    torch.testing.assert_close(log_copy[0].exp(), torch.full((3, 1), 0.05))
    expected = torch.softmax(logits, -1) * log_gen.exp()
    expected.scatter_add_(-1, state.token_ids[:, None].expand(-1, 3, -1), log_attention.exp() * log_copy.exp())
    mixed = head.mix_logits(hidden, logits, state)
    torch.testing.assert_close(torch.softmax(mixed, -1), expected)
    torch.testing.assert_close(expected.sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(mixed[1], logits[1], rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_chunked_mixture_ce_matches_dense_loss_and_gradients(chunk_size):
    torch.manual_seed(4)
    head = GroundedCopyHead(8, 4, 0.15)
    lm = torch.nn.Linear(8, 13)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    keys = torch.randn(2, 4, 4, requires_grad=True)
    bias = torch.randn(2, 4, requires_grad=True)
    state = CopyState(
        keys,
        torch.tensor([[4, 4, 5, 6], [7, 8, 9, 10]]),
        torch.tensor([[True, True, True, False], [False, False, False, False]]),
        bias,
    )
    labels = torch.tensor([[-100, -100, 4, 12, 5], [-100, -100, -100, 7, 11]])
    logits = head.mix_logits(hidden, lm(hidden), state)
    expected = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1))
    params = [hidden, keys, bias, *head.query.parameters(), *head.gate.parameters(), *lm.parameters()]
    expected_grads = torch.autograd.grad(expected, params)
    actual = head.loss(hidden, labels, state, lm, chunk_size)
    actual_grads = torch.autograd.grad(actual, params)
    torch.testing.assert_close(actual, expected)
    for left, right in zip(actual_grads, expected_grads):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)


def test_sparse_pooling_values_gradients_and_empty_alignment():
    torch.manual_seed(6)
    head = GroundedCopyHead(8, 4, 0.05)
    embedding = torch.nn.Embedding(13, 8)
    memory, bias = torch.randn(2, 4, 8, requires_grad=True), torch.randn(2, 4, requires_grad=True)
    rows = pad_copy_alignments(
        [
            dict(
                copy_token_ids=[4, 5],
                copy_encoder_indices=[0, 1, 2],
                copy_token_indices=[0, 0, 1],
                copy_alignment_weights=[0.4, 0.6, 1.0],
            ),
            dict(copy_token_ids=[], copy_encoder_indices=[], copy_token_indices=[], copy_alignment_weights=[]),
        ]
    )
    state = head.prepare(memory, bias, torch.ones(2, 4, dtype=torch.bool), embedding, **rows)
    torch.testing.assert_close(state.bias[0], torch.stack((bias[0, 0] * 0.4 + bias[0, 1] * 0.6, bias[0, 2])))
    assert state.mask.tolist() == [[True, True], [False, False]]
    loss = head.loss(torch.randn(2, 3, 8), torch.tensor([[4, 5, 4], [4, 5, 4]]), state, torch.nn.Linear(8, 13), 4)
    loss.backward()
    for parameter in (
        memory,
        bias,
        embedding.weight,
        head.query.weight,
        head.context_key.weight,
        head.lexical_key.weight,
        head.gate.weight,
        head.gate.bias,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert memory.grad[1].eq(0).all()
    assert memory.grad[:, 3].eq(0).all()


def test_model_training_fp32_autocast_and_warmup_optimizer():
    torch.manual_seed(8)
    config, model, batch, _ = setup_model()
    set_stage_trainability(model, "interface_warmup")
    optimizer = build_optimizer(model, config, "interface_warmup")
    head = model.decoder.grounded_copy
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    before = head.gate.bias.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = forward(model, batch)
    output.loss.backward()
    for parameter in (head.query.weight, head.context_key.weight, head.lexical_key.weight, head.gate.bias):
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, head.gate.bias)
    set_stage_trainability(model, "full_finetune")
    optimizer.zero_grad(set_to_none=True)
    forward(model, batch).loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.encoder.parameters()
    )


def test_checkpoint_compatibility_and_mixture_dense_model_parity(tmp_path):
    config, model, batch, _ = setup_model()
    model.eval()
    torch.testing.assert_close(forward(model, batch).loss, forward(model, batch, logits=True).loss)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, None, config, epoch=1, step=1)
    restored = EviSeqAFMR(config).eval()
    load_checkpoint(path, restored, config=config)
    torch.testing.assert_close(forward(model, batch).loss, forward(restored, batch).loss, rtol=0, atol=0)
    old = copy.deepcopy(config)
    old["decoder"].pop("grounded_copy")
    assert "grounded_copy" not in architecture_spec(old)
    with pytest.raises(ValueError):
        load_checkpoint(path, EviSeqAFMR(old), config=old)


def test_copy_initialization_preserves_shared_module_initialization():
    config, _, _, _ = setup_model()
    torch.manual_seed(21)
    enabled = EviSeqAFMR(config)
    config["decoder"]["grounded_copy"]["enabled"] = False
    torch.manual_seed(21)
    disabled = EviSeqAFMR(config)
    for name, tensor in disabled.state_dict().items():
        torch.testing.assert_close(enabled.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("gate_bias", [-1000.0, 1000.0])
def test_extreme_gates_and_all_ignored_labels_are_finite(gate_bias):
    head = GroundedCopyHead(8, 4, 0.05)
    head.gate.bias.data.fill_(gate_bias)
    state = CopyState(
        torch.randn(2, 2, 4),
        torch.tensor([[4, 5], [4, 5]]),
        torch.tensor([[True, True], [False, False]]),
        torch.zeros(2, 2),
    )
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    lm_head = torch.nn.Linear(8, 13)
    labels = torch.full((2, 3), -100)
    loss = head.loss(hidden, labels, state, lm_head, 2)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    labels[0, 1], labels[1, 2] = 12, 4
    loss = head.loss(hidden, labels, state, lm_head, 2)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cached_copy_decoder_matches_full_prefix_after_compaction(dtype):
    config, _, batch, tokenizer = setup_model()
    config["model"]["dtype"] = dtype
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        state = bridge(model, batch)
        decoder = model.decoder
        tokens = torch.tensor([[4, 5], [6, 7]])
        kwargs = dict(value_memory=state.value_memory, copy_state=state.copy_state)
        decoder.prepare_cross_cache(state.memory, state.value_memory)
        _, cache, _ = decoder(tokens, state.memory, state.memory_mask, state.source_bias, use_cache=True, **kwargs)
        rows = torch.tensor([1])
        cache.batch_select_indices(rows)
        decoder.select_cross_cache(rows)
        kwargs = dict(value_memory=state.value_memory[rows], copy_state=state.copy_state.index_select(rows))
        step, _, _ = decoder(
            torch.tensor([[8]]),
            state.memory[rows],
            state.memory_mask[rows],
            state.source_bias[rows],
            use_cache=True,
            past_key_values=cache,
            **kwargs,
        )
        decoder.clear_cross_cache()
        full, _, _ = decoder(
            torch.tensor([[6, 7, 8]]), state.memory[rows], state.memory_mask[rows], state.source_bias[rows], **kwargs
        )
        torch.testing.assert_close(step[:, -1], full[:, -1], atol=0.005 if dtype == "bfloat16" else 1e-6, rtol=1e-4)
        _, compact = generate_greedy(model, batch, tokenizer, 4, compact_finished=True)
        _, uncompact = generate_greedy(model, batch, tokenizer, 4, compact_finished=False)
        torch.testing.assert_close(compact, uncompact)
        assert all(layer.cross._cache is None for layer in decoder.backbone.layers)
