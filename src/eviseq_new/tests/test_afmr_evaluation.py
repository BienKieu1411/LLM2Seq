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
