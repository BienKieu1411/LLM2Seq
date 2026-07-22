import json
import tempfile
from pathlib import Path

import pytest
import torch

from genbridge.data import (
    DirectSummarizationDataset,
    EvidenceSeq2SeqCollator,
    EvidenceSeq2SeqDataset,
    clean_wikihow_metadata,
    decoder_prompt_ids,
    greedy_evidence_labels,
    prompted_source_features,
    split_evidence_units,
)


def test_dataset_limit_is_applied_before_item_processing():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        path.write_text(
            "".join(
                json.dumps({"source": f"source {index}", "target": f"target {index}"}) + "\n"
                for index in range(5)
            ),
            encoding="utf-8",
        )
        dataset = DirectSummarizationDataset(path, tokenizer=object(), data_config={}, max_examples=2)
        assert len(dataset) == 2


def test_oracle_evidence_selects_summary_bearing_sentences():
    source = [
        "Luộc nước trong nồi.",
        "Cho mì vào nước sôi.",
        "Bầu trời hôm nay nhiều mây.",
        "Vớt mì ra và để ráo.",
    ]
    labels = greedy_evidence_labels(source, "Cho mì vào nước sôi. Vớt mì và để ráo.", 3)
    assert labels[1] == 1.0
    assert labels[3] == 1.0
    assert labels[2] == 0.0


def test_oracle_ignores_salience_when_reference_has_no_lexical_evidence():
    labels = greedy_evidence_labels(
        ["Luộc nước.", "Cho mì vào nồi."],
        "Completely unrelated English target.",
        3,
    )
    assert labels == [-1.0, -1.0]


def test_fixed_budget_can_supervise_multiple_units_for_one_sentence_target():
    source = [
        "Bão đổ bộ vào miền Trung.",
        "Ba nghìn người phải sơ tán.",
        "Trường học đóng cửa vào thứ Hai.",
        "Đội bóng giành chiến thắng.",
    ]
    target = "Bão đổ bộ miền Trung khiến ba nghìn người sơ tán và trường học đóng cửa thứ Hai."
    labels = greedy_evidence_labels(
        source,
        target,
        max_evidence_units=6,
        budget_mode="fixed",
        fixed_budget=3,
        rouge1_weight=0.7,
        rouge2_weight=0.3,
    )
    assert labels == [1.0, 1.0, 1.0, 0.0]


def test_long_document_profile_groups_sentences_into_units():
    units = split_evidence_units(
        "Câu một. Câu hai. Câu ba. Câu bốn. Câu năm.",
        {"evidence_unit": "sentence_group", "sentences_per_unit": 2},
    )
    assert units == ["Câu một. Câu hai.", "Câu ba. Câu bốn.", "Câu năm."]


def test_paragraph_profile_preserves_explicit_boundaries():
    units = split_evidence_units(
        "Đoạn thứ nhất có hai câu. Câu tiếp theo.\n\nĐoạn thứ hai.",
        {"evidence_unit": "paragraph"},
    )
    assert units == ["Đoạn thứ nhất có hai câu. Câu tiếp theo.", "Đoạn thứ hai."]


def test_collator_aligns_units_and_evidence_padding():
    collator = EvidenceSeq2SeqCollator(2, 16, 8)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "unit_ids": torch.tensor([0, 1, 1]),
                "evidence_labels": torch.tensor([1.0]),
                "decoder_input_ids": torch.tensor([2, 8]),
                "labels": torch.tensor([8, 2]),
            },
            {
                "input_ids": torch.tensor([6, 7]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "unit_ids": torch.tensor([0, 1]),
                "evidence_labels": torch.tensor([0.0, 1.0]),
                "decoder_input_ids": torch.tensor([2]),
                "labels": torch.tensor([9]),
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[3, 4, 5], [2, 6, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert batch["unit_ids"].tolist() == [[0, 1, 1], [0, 0, 1]]
    assert batch["evidence_labels"].tolist() == [[1.0, -1.0], [0.0, 1.0]]
    assert batch["labels"].tolist() == [[8, 2], [9, -100]]


def test_wikihow_metadata_cleaning_preserves_surrounding_text():
    text = 'Mở Settings { " smallUrl " : "https://x", "licensing": "<div></div>" } ngay.'
    assert clean_wikihow_metadata(text) == "Mở Settings ngay."


def test_decoder_prompt_uses_task_prefix_after_fallback_start_token():
    class Tokenizer:
        bos_token_id = None
        pad_token_id = 7
        eos_token_id = 8

        def __call__(self, text, add_special_tokens=False):
            assert text == "Các bước:\n"
            assert not add_special_tokens
            return {"input_ids": [10, 11, 12]}

    assert decoder_prompt_ids(Tokenizer(), {"decoder_prefix": "Các bước:\n"}) == [
        7,
        10,
        11,
        12,
    ]


def test_decoder_prompt_uses_non_thinking_chat_template_when_available():
    class Tokenizer:
        bos_token_id = None
        pad_token_id = 7
        eos_token_id = 8
        chat_template = "qwen-template"

        def __init__(self):
            self.messages = None
            self.template_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.template_kwargs = kwargs
            return {"input_ids": [20, 21, 22], "attention_mask": [1, 1, 1]}

        def __call__(self, text, add_special_tokens=False):
            assert text == "Các bước:\n"
            assert not add_special_tokens
            return {"input_ids": [10, 11]}

    tokenizer = Tokenizer()
    seed = decoder_prompt_ids(
        tokenizer,
        {
            "decoder_instruction": "Chỉ sinh bản tóm tắt.",
            "decoder_prefix": "Các bước:\n",
            "enable_thinking": False,
        },
    )
    assert seed == [20, 21, 22, 10, 11]
    assert tokenizer.messages == [{"role": "user", "content": "Chỉ sinh bản tóm tắt."}]
    assert tokenizer.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_truncated_evidence_unit_matches_visible_source_tokens():
    class Tokenizer:
        chat_template = None

        def __call__(self, text, add_special_tokens=False):
            mapping = {
                "P": [90],
                "S": [91],
                "A.": [1, 2],
                "B.": [3, 4, 5],
            }
            return {"input_ids": mapping[text]}

        def decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens
            assert token_ids == [3, 4]
            return "B phần nhìn thấy"

    ids, unit_ids, units = prompted_source_features(
        Tokenizer(),
        "A. B.",
        {
            "source_prefix": "P",
            "target_prefix": "S",
            "sentence_separator": "",
            "max_source_length": 6,
            "use_chat_template": False,
        },
    )
    assert ids == [90, 1, 2, 3, 4, 91]
    assert unit_ids == [0, 1, 1, 2, 2, 0]
    assert units == ["A.", "B phần nhìn thấy"]


def test_precomputed_oracle_is_recomputed_for_partially_visible_last_unit():
    class Tokenizer:
        chat_template = None
        bos_token_id = None
        pad_token_id = 7
        eos_token_id = 8

        def __call__(self, text, add_special_tokens=False, **kwargs):
            mapping = {
                "P": [90],
                "S": [91],
                "A.": [1, 2],
                "B tail.": [3, 4, 5],
                "tail": [6],
            }
            return {"input_ids": mapping[text]}

        def decode(self, token_ids, skip_special_tokens=True):
            assert token_ids == [3, 4]
            return "B"

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        path.write_text(
            json.dumps(
                {"id": "x", "source": "A. B tail.", "target": "tail"},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = EvidenceSeq2SeqDataset(
            path,
            Tokenizer(),
            {
                "source_prefix": "P",
                "target_prefix": "S",
                "decoder_prefix": "",
                "sentence_separator": "",
                "max_source_length": 6,
                "max_target_length": 8,
                "use_chat_template": False,
                "use_decoder_chat_template": False,
                "oracle_max_units": 3,
            },
            precompute_evidence=True,
        )
        # The full-text oracle selects "B tail.", but "tail" is truncated
        # away. The cached full-text label must not leak into supervision.
        assert dataset.evidence_cache == [[0.0, 1.0]]
        assert dataset[0]["evidence_labels"].tolist() == [-1.0, -1.0]


def test_decoder_prompt_does_not_consume_target_budget():
    class Tokenizer:
        chat_template = None
        bos_token_id = None
        pad_token_id = 7
        eos_token_id = 8

        def __call__(self, text, add_special_tokens=False, **kwargs):
            return {"input_ids": [1]}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        path.write_text(
            json.dumps({"source": "Nguồn.", "target": "Đích."}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        dataset = EvidenceSeq2SeqDataset(
            path,
            Tokenizer(),
            {
                "source_prefix": "P",
                "target_prefix": "S",
                "decoder_prefix": "prefix",
                "max_source_length": 8,
                "max_target_length": 2,
                "use_chat_template": False,
                "use_decoder_chat_template": False,
            },
        )
        feature = dataset[0]
        # One ignored prompt position plus two summary positions (token + EOS).
        assert feature["labels"].tolist() == [-100, 1, 8]
        collator = EvidenceSeq2SeqCollator(
            pad_token_id=7,
            max_source_length=8,
            max_target_length=2,
            decoder_prompt_length=2,
        )
        batch = collator([feature])
        assert batch["labels"].shape[1] == 3
        assert batch["labels"].tolist() == [[-100, 1, 8]]
