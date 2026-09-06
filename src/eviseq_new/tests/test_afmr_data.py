import json

import pytest
import torch
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.dataset import JsonlSummarizationDataset
from eviseq_afmr.data.normalization import detokenize
from eviseq_afmr.data.prepare import prepare_split
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.runtime import _TinyTokenizer


def test_length_buckets_reduce_padding_and_resume_by_epoch():
    from eviseq_afmr.data.sampling import LengthBucketBatchSampler

    lengths = list(range(1, 104))
    sampler = LengthBucketBatchSampler(lengths, 8, seed=7, multiplier=50)
    sampler.set_epoch(3)
    first = list(sampler)
    assert sorted(i for batch in first for i in batch) == list(range(len(lengths)))
    assert len(first) == len(sampler) == 13
    assert sum(max(lengths[i] for i in batch) * len(batch) for batch in first) < sum(lengths) * 1.15
    recreated = LengthBucketBatchSampler(lengths, 8, seed=7, multiplier=50)
    recreated.set_epoch(3)
    assert list(recreated) == first
    recreated.set_epoch(4)
    assert list(recreated) != first


def test_preparation_preserves_text_and_discards_external_labels(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "x",
                "text": ["first sentence.", "second sentence."],
                "summary": ["first sentence."],
                "label": [999],
                "preparation": "old",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "test.jsonl"
    prepare_split(source, destination)
    row = json.loads(destination.read_text(encoding="utf-8"))
    assert row == {"id": "x", "text": "first sentence.\nsecond sentence.", "summary": "first sentence."}


def test_cli_prepare_requires_no_evidence_options(tmp_path):
    from eviseq_afmr.cli import main

    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": ["a.", "b."], "summary": "a.", "label": [0]}) + "\n")
    output = tmp_path / "train.jsonl"
    main(["prepare", str(source), str(output)])
    assert set(json.loads(output.read_text())) == {"id", "text", "summary"}


def test_collator_keeps_decoder_inputs_and_labels_aligned():
    record = CanonicalRecord("x", "one. two.", "one.")
    collator = SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {"encoder_prefix": "", "decoder_prompt": "sum:", "max_source_length": 16, "max_target_length": 8},
    )
    batch = collator([record])
    assert batch["decoder_input_ids"].shape == batch["labels"].shape
    supervised = batch["labels"].ne(-100)
    assert torch.equal(batch["decoder_input_ids"][supervised], batch["labels"][supervised])
    assert batch["labels"][0, -1] == _TinyTokenizer.eos_token_id
    assert not any(key.startswith("allocation") for key in batch)


def test_chat_prompt_uses_native_template_and_excludes_reference():
    class ChatTokenizer(_TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Summarize faithfully."}]
            assert kwargs == {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return [7, 8, 9]

    tokenizer = ChatTokenizer()
    collator = SummarizationCollator(
        tokenizer,
        tokenizer,
        {
            "decoder_prompt": "Summarize faithfully.",
            "decoder_chat_template": True,
            "decoder_prefix": "Abstract:\n",
        },
    )
    first = collator([CanonicalRecord("x", "same source", "first reference")])
    second = collator([CanonicalRecord("x", "same source", "different reference")])
    expected = [7, 8, 9] + tokenizer("Abstract:\n")["input_ids"]
    assert first["decoder_prompt_ids"].tolist() == [expected]
    torch.testing.assert_close(first["decoder_prompt_ids"], second["decoder_prompt_ids"])
    assert first["labels"][0, : len(expected)].eq(-100).all()


@pytest.mark.parametrize("kind", ["mapping", "batch_encoding", "tensor", "batched_tensor"])
def test_chat_prompt_accepts_tokenizer_return_containers(kind):
    from transformers import BatchEncoding

    class ChatTokenizer(_TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            values = [7, 8, 9]
            return {
                "mapping": {"input_ids": values, "attention_mask": [1, 1, 1]},
                "batch_encoding": BatchEncoding({"input_ids": values}),
                "tensor": torch.tensor(values),
                "batched_tensor": torch.tensor([values]),
            }[kind]

    collator = SummarizationCollator(
        _TinyTokenizer(),
        ChatTokenizer(),
        {
            "decoder_prompt": "Summarize faithfully.",
            "decoder_chat_template": True,
        },
    )
    batch = collator([CanonicalRecord("x", "same source", "reference")])
    assert batch["decoder_prompt_ids"].tolist() == [[7, 8, 9]]
    assert batch["labels"][0, :3].eq(-100).all()


@pytest.mark.parametrize("grounded_copy", [False, True])
def test_real_fast_tokenizer_chat_prompt_batches_in_train_and_eval(grounded_copy):
    from collections.abc import Mapping

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "[UNK]",
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "Summarize",
        "faithfully",
        "same",
        "source",
        "reference",
        "Abstract",
        "assistant",
        ":",
        ".",
    ]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant:{% endif %}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Summarize faithfully."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    expected = list(rendered["input_ids"] if isinstance(rendered, Mapping) else rendered)
    expected += tokenizer("Abstract:\n", add_special_tokens=False)["input_ids"]
    collator = SummarizationCollator(
        tokenizer,
        tokenizer,
        {
            "decoder_prompt": "Summarize faithfully.",
            "decoder_chat_template": True,
            "decoder_prefix": "Abstract:\n",
        },
        grounded_copy=grounded_copy,
    )
    records = [CanonicalRecord("x", "same source", "reference"), CanonicalRecord("y", "source", "reference")]
    train = collator(records)
    collator.include_targets = False
    evaluation = collator(records)
    assert train["decoder_prompt_ids"].tolist() == [expected, expected]
    torch.testing.assert_close(train["decoder_prompt_ids"], evaluation["decoder_prompt_ids"])
    assert train["labels"][:, : len(expected)].eq(-100).all()
    assert evaluation["labels"].eq(-100).all()


@pytest.mark.parametrize("invalid", ["rendered prompt", ["input_ids", "attention_mask"], [[7], [8]]])
def test_invalid_chat_token_ids_fail_before_dataloader_workers(invalid):
    class ChatTokenizer(_TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return invalid

    with pytest.raises(ValueError, match="integer token IDs"):
        SummarizationCollator(
            _TinyTokenizer(),
            ChatTokenizer(),
            {
                "decoder_prompt": "Summarize",
                "decoder_chat_template": True,
            },
        )


def test_old_processed_data_is_normalized_without_repreparing(tmp_path):
    raw = {"id": "x", "text": "parkinson 's disease ( pd ) .\nPatients did n't recover .", "summary": "pd ( 50 % ) ."}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    legacy = JsonlSummarizationDataset(path, {})[0]
    normalized = JsonlSummarizationDataset(path, {"detokenize": True})[0]
    assert legacy.source == raw["text"]
    assert normalized.source == "parkinson's disease (pd).\nPatients didn't recover."
    assert normalized.target == "pd (50%)."
    assert detokenize(normalized.source) == normalized.source


def test_normalization_matches_t5gemma_sentence_preprocessing():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[3] / "src/T5Gemma/scripts/prepare_cnndm_json.py"
    spec = importlib.util.spec_from_file_location("t5_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentences = [
        "parkinson 's disease ( pd ) .",
        "`` It did n't change '' , ５０ % improved .",
        "P [ 0.05 ] ; they 've recovered !",
    ]
    expected = module.join_field(sentences, "\n")
    assert detokenize("\n".join(sentences)) == expected


def test_external_labels_cannot_change_any_model_input():
    raw = {"id": "x", "text": "one. two.", "summary": "one."}
    clean = CanonicalRecord.from_mapping(raw)
    poisoned = CanonicalRecord.from_mapping({**raw, "label": "invalid oracle label"})
    collator = SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {"encoder_prefix": "", "decoder_prompt": "sum:"},
    )
    expected, actual = collator([clean]), collator([poisoned])
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_reference_cannot_change_encoder_or_prompt_inputs():
    raw = {"id": "x", "text": "one. two.", "summary": "one."}
    collator = SummarizationCollator(_TinyTokenizer(), _TinyTokenizer(), {"decoder_prompt": "sum:"})
    clean = collator([CanonicalRecord.from_mapping(raw)])
    changed = collator([CanonicalRecord.from_mapping({**raw, "summary": "different reference"})])
    for key in ("input_ids", "attention_mask", "source_content_mask", "decoder_prompt_ids", "decoder_prompt_mask"):
        torch.testing.assert_close(clean[key], changed[key])


def test_v2_source_target_records_are_accepted_with_default_afmr_fields():
    record = CanonicalRecord.from_mapping({"id": "x", "source": "article", "target": "summary"})
    assert record.source == "article"
    assert record.target == "summary"
