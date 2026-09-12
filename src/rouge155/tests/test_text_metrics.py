from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from rouge155.evaluate_bertscore import evaluate as evaluate_bertscore
from rouge155.evaluate_alignscore import _sentence_split, _source_chunks, evaluate as evaluate_alignscore
from rouge155.metric_io import load_jsonl


def _write_predictions(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def test_metric_loader_preserves_multiple_references_and_requires_source(tmp_path: Path) -> None:
    path = _write_predictions(
        tmp_path / "predictions.jsonl",
        [{"id": "x", "source": "source", "prediction": "prediction", "reference": ["r1", "r2"]}],
    )
    rows = load_jsonl(path, source_field="source")
    assert rows[0].references == ("r1", "r2")
    assert rows[0].source == "source"

    with pytest.raises(ValueError, match="Missing 'source'"):
        load_jsonl(
            _write_predictions(tmp_path / "missing.jsonl", [{"prediction": "p", "reference": "r"}]),
            source_field="source",
        )


def test_metric_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_predictions(
        tmp_path / "duplicates.jsonl",
        [
            {"id": "same", "prediction": "p", "reference": "r"},
            {"id": "same", "prediction": "p2", "reference": "r2"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate example ID"):
        load_jsonl(path)


def test_bertscore_uses_local_scorer_and_supports_multiple_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    predictions = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"id": "a", "prediction": "p a", "reference": ["r a", "r a alt"]},
            {"id": "b", "prediction": "p b", "reference": "r b"},
        ],
    )

    class FakeScorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def score(self, candidates, references, **kwargs):
            assert references[0] == ["r a", "r a alt"]
            assert kwargs["batch_size"] == 2
            return (torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]), torch.tensor([0.5, 0.6]))

    monkeypatch.setitem(sys.modules, "bert_score", types.SimpleNamespace(BERTScorer=FakeScorer))
    result = evaluate_bertscore(predictions, model_path, tmp_path / "bertscore.json", batch_size=2, num_layers=12)
    assert result["bertscore"]["f1"] == pytest.approx(55.0)
    assert result["num_layers"] == 12


def test_alignscore_uses_paper_direction_and_transforms_to_hallucination_score(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"id": "supported", "source": "source", "prediction": "p1", "reference": "r1"},
            {"id": "unsupported", "source": "source", "prediction": "p2", "reference": "r2"},
        ],
    )

    class FakeAlignScore:
        _splitter = "test"

        def score_document(self, source: str, prediction: str) -> float:
            assert source == "source"
            return 0.9 if prediction == "p1" else 0.2

    result = evaluate_alignscore(
        predictions,
        tmp_path / "unused-local-model",
        tmp_path / "unused-alignscore.ckpt",
        tmp_path / "alignscore.json",
        scorer=FakeAlignScore(),
        details=True,
    )
    assert result["metric"] == "AlignScore-nli_sp"
    assert result["paper"] == "Zha et al., ACL 2023"
    assert result["alignscore_consistency"] == pytest.approx(0.55)
    assert result["hallucination_score"] == pytest.approx(0.45)
    assert result["score_direction"] == "hallucination_score_lower_is_better"
    assert result["rows"][0]["hallucination_score"] == pytest.approx(0.1)


def test_alignscore_reproduces_source_chunk_and_claim_sentence_shape() -> None:
    sentences = _sentence_split("First statement. Second statement. Third statement.")
    assert len(sentences) == 3
    chunks = _source_chunks("First statement. Second statement. Third statement.", chunk_words=100)
    assert chunks == ["First statement. Second statement. Third statement."]
