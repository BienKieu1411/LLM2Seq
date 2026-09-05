import json

import torch
from eviseq_afmr.data.collate import SummarizationCollator
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
