import torch

from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.salience import lexical_source_salience
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.runtime import _TinyTokenizer


def _collator(**kwargs):
    return SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {"encoder_prefix": "Source: ", "max_source_length": 16, "max_target_length": 12},
        salience_supervision=True,
        **kwargs,
    )


def test_bigram_and_trigram_labels_follow_visible_encoder_offsets():
    batch = _collator()(
        [CanonicalRecord("x", "Alpha beta gamma. Delta epsilon zeta.", "beta gamma and delta epsilon zeta")]
    )
    # BOS, encoder prefix, six source tokens, EOS.
    assert batch["source_content_mask"].tolist() == [[False, False, True, True, True, True, True, True, False]]
    assert batch["source_salience_labels"].tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]]
    assert batch["source_salience_mask"].tolist() == [True]
    assert batch["source_salience_labels"].dtype == torch.float32


def test_no_overlap_and_all_positive_rows_are_masked_out():
    batch = _collator()(
        [
            CanonicalRecord("none", "alpha beta gamma", "different words"),
            CanonicalRecord("all", "alpha beta", "alpha beta"),
        ]
    )
    assert batch["source_salience_mask"].tolist() == [False, False]
    assert batch["source_salience_labels"][0].eq(0).all()
    assert batch["source_salience_labels"][1].sum().item() == 2


def test_source_truncation_excludes_unseen_gold_overlap():
    collator = SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {"max_source_length": 3, "max_target_length": 8},
        salience_supervision=True,
    )
    batch = collator([CanonicalRecord("x", "alpha beta gamma delta", "gamma delta")])
    assert batch["source_salience_labels"].eq(0).all()
    assert batch["source_salience_mask"].tolist() == [False]


def test_target_truncation_excludes_untrained_gold_suffix():
    collator = SummarizationCollator(
        _TinyTokenizer(),
        _TinyTokenizer(),
        {"max_source_length": 16, "max_target_length": 3},
        salience_supervision=True,
    )
    batch = collator([CanonicalRecord("x", "alpha beta gamma delta", "unrelated words gamma delta")])
    assert batch["source_salience_labels"].eq(0).all()
    assert batch["source_salience_mask"].tolist() == [False]


def test_matching_words_mark_each_encoder_subtoken():
    labels, valid = lexical_source_salience(
        "hello world unrelated",
        "HELLO, world!",
        [(0, 0), (0, 2), (2, 5), (6, 9), (9, 11), (12, 21), (0, 0)],
        [False, True, True, True, True, True, False],
        prefix_length=0,
    )
    assert labels == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert valid


def _word_labels(source: str, target: str) -> tuple[list[tuple[str, float]], bool]:
    tokenizer = _TinyTokenizer()
    encoded = tokenizer(source, add_special_tokens=True, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    content = [end > start for start, end in offsets]
    labels, valid = lexical_source_salience(source, target, offsets, content, prefix_length=0)
    return ([(source[start:end], labels[index]) for index, (start, end) in enumerate(offsets) if end > start], valid)


def test_source_sentence_boundary_does_not_create_cross_sentence_ngrams():
    tokens, valid = _word_labels(
        "alpha beta. gamma delta. epsilon zeta",
        "beta gamma epsilon zeta",
    )
    assert tokens == [
        ("alpha", 0.0),
        ("beta.", 0.0),
        ("gamma", 0.0),
        ("delta.", 0.0),
        ("epsilon", 1.0),
        ("zeta", 1.0),
    ]
    assert valid


def test_target_sentence_boundary_does_not_create_cross_sentence_ngrams():
    tokens, valid = _word_labels("alpha beta gamma", "alpha beta. gamma omega")
    assert tokens == [("alpha", 1.0), ("beta", 1.0), ("gamma", 0.0)]
    assert valid


def test_newline_is_a_sentence_boundary_for_source_and_target_matching():
    tokens, valid = _word_labels("alpha beta\ngamma delta", "beta gamma")
    assert tokens == [("alpha", 0.0), ("beta", 0.0), ("gamma", 0.0), ("delta", 0.0)]
    assert not valid


def test_inference_and_default_collator_do_not_return_reference_labels():
    record = CanonicalRecord("x", "alpha beta gamma", "alpha beta")
    eval_collator = _collator()
    eval_collator.include_targets = False
    assert "source_salience_labels" not in eval_collator([record])
    assert "source_salience_mask" not in eval_collator([record])

    default_collator = SummarizationCollator(_TinyTokenizer(), _TinyTokenizer(), {})
    assert "source_salience_labels" not in default_collator([record])
