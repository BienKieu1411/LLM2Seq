from eviseq_afmr.evaluation.generate import append_jsonl, existing_ids
from eviseq_afmr.evaluation.metrics import summarization_metrics


def test_jsonl_append_and_resume_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, [{"id": "a", "prediction": "one", "reference": "one"}])
    append_jsonl(path, [{"id": "b", "prediction": "two", "reference": "two"}])
    assert existing_ids(path) == {"a", "b"}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_diagnostic_metrics_are_bounded():
    metrics = summarization_metrics(["a b c", ""], ["a b c", "a"])
    assert 0 <= metrics["rouge1"] <= 100
    assert 0 <= metrics["rouge2"] <= 100
    assert 0 <= metrics["rougeL"] <= 100


def test_diagnostic_backend_matches_installed_rouge():
    from rouge import Rouge

    predictions, references = ["a a b c.", "a b b."], ["a b c.", "a b c."]
    expected = Rouge().get_scores(predictions, references, avg=True)
    actual = summarization_metrics(predictions, references)
    assert abs(actual["rouge2"] - 100 * expected["rouge-2"]["f"]) < 1e-8
    assert "1.0.0" in actual["rouge_backend"]


def test_greedy_constraints_match_transformers_processors():
    import torch
    from eviseq_afmr.evaluation.generate import (
        _apply_repetition_penalty,
        _no_repeat_ngram_tokens,
    )
    from transformers import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor

    token_ids = torch.tensor([[1, 2, 3, 2, 4], [2, 2, 1, 2, 1]])
    scores = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, 6.0], [1.0, -2.0, 3.0, -4.0, 5.0, 6.0]])
    actual = scores.clone()
    _apply_repetition_penalty(actual, token_ids, 1.05)
    expected = RepetitionPenaltyLogitsProcessor(1.05)(token_ids, scores.clone())
    torch.testing.assert_close(actual, expected)

    for ngram_size in (1, 2, 3, 4):
        expected_scores = NoRepeatNGramLogitsProcessor(ngram_size)(token_ids, scores.clone())
        expected_banned = [torch.where(row == -float("inf"))[0].tolist() for row in expected_scores]
        assert _no_repeat_ngram_tokens(token_ids, ngram_size) == expected_banned
