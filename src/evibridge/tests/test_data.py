import json
import tempfile
from pathlib import Path

import torch

from evibridge.data import (
    EvidenceSeq2SeqCollator,
    DirectSummarizationDataset,
    clean_wikihow_metadata,
    greedy_evidence_labels,
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
    assert batch["unit_ids"].tolist() == [[0, 1, 1], [0, 1, 0]]
    assert batch["evidence_labels"].tolist() == [[1.0, -1.0], [0.0, 1.0]]
    assert batch["labels"].tolist() == [[8, 2], [9, -100]]


def test_wikihow_metadata_cleaning_preserves_surrounding_text():
    text = 'Mở Settings { " smallUrl " : "https://x", "licensing": "<div></div>" } ngay.'
    assert clean_wikihow_metadata(text) == "Mở Settings ngay."
