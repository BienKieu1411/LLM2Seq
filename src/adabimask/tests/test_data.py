import torch

from adabimask.data import Seq2SeqCollator, clean_wikihow_metadata


def test_wikihow_image_metadata_is_removed_without_losing_text():
    text = 'Mở Settings { " smallUrl " : "https://x", "bigWidth": 760, "licensing": "<div></div>" } ngay.'
    cleaned = clean_wikihow_metadata(text)
    assert cleaned == "Mở Settings ngay."
    assert "smallUrl" not in cleaned


def test_standalone_collator_preserves_valid_eos_when_pad_equals_eos():
    collator = Seq2SeqCollator(pad_token_id=2, max_source_length=8, max_target_length=8)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([5, 2]),
                "decoder_input_ids": torch.tensor([2, 7]),
                "labels": torch.tensor([7, 2]),
            },
            {
                "input_ids": torch.tensor([6]),
                "decoder_input_ids": torch.tensor([2]),
                "labels": torch.tensor([8]),
            },
        ]
    )
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["decoder_attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[7, 2], [8, -100]]
