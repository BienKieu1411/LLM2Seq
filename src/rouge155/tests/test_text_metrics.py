from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from rouge155.evaluate_bertscore import _patch_tokenizer_max_length, evaluate as evaluate_bertscore
from rouge155.evaluate_alignscore import (
    LocalAlignScore,
    _sentence_split,
    _source_chunks,
    evaluate as evaluate_alignscore,
)
from rouge155.evaluate_scale import LocalSCALE, _build_chunks, evaluate as evaluate_scale
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


def test_bertscore_patches_transformers_large_length_sentinel() -> None:
    class FakeTokenizer:
        model_max_length = 10**30
        init_kwargs: dict[str, int] = {}

    class FakeConfig:
        model_type = "roberta"
        max_position_embeddings = 514

    class FakeModel:
        config = FakeConfig()

    class FakeScorer:
        _tokenizer = FakeTokenizer()
        _model = FakeModel()

    scorer = FakeScorer()
    assert _patch_tokenizer_max_length(scorer) == 512
    assert scorer._tokenizer.model_max_length == 512
    assert scorer._tokenizer.init_kwargs["model_max_length"] == 512


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


def test_alignscore_joins_source_from_test_file_when_predictions_omit_it(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"id": "a", "prediction": "p a", "reference": "r a"},
            {"id": "b", "prediction": "p b", "reference": "r b"},
        ],
    )
    source_file = _write_predictions(
        tmp_path / "test.jsonl",
        [
            {"id": "a", "text": "source a"},
            {"id": "b", "text": "source b"},
        ],
    )

    class FakeAlignScore:
        _splitter = "test"

        def score_document(self, source: str, prediction: str) -> float:
            assert source == f"source {prediction[-1]}"
            return 0.8

    result = evaluate_alignscore(
        predictions,
        tmp_path / "unused-local-model",
        tmp_path / "unused-alignscore.ckpt",
        tmp_path / "alignscore.json",
        source_file=source_file,
        scorer=FakeAlignScore(),
    )
    assert result["source_joined_by_id"] is True
    assert result["alignscore_consistency"] == pytest.approx(0.8)


def test_alignscore_reports_source_file_ids_that_are_missing(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.jsonl",
        [{"id": "missing", "prediction": "p", "reference": "r"}],
    )
    source_file = _write_predictions(tmp_path / "test.jsonl", [{"id": "other", "text": "source"}])

    with pytest.raises(ValueError, match="no record for prediction IDs"):
        evaluate_alignscore(
            predictions,
            tmp_path / "unused-local-model",
            tmp_path / "unused-alignscore.ckpt",
            source_file=source_file,
            scorer=object(),
        )


def test_alignscore_reproduces_source_chunk_and_claim_sentence_shape() -> None:
    sentences = _sentence_split("First statement. Second statement. Third statement.")
    assert len(sentences) == 3
    chunks = _source_chunks("First statement. Second statement. Third statement.", chunk_words=100)
    assert chunks == ["First statement. Second statement. Third statement."]


def test_alignscore_retries_pair_truncation_when_claim_exhausts_budget() -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.truncation_modes: list[object] = []

        def __call__(self, contexts, claims, **kwargs):
            self.truncation_modes.append(kwargs["truncation"])
            if kwargs["truncation"] == "only_first":
                raise Exception("Sequence to truncate too short to respect the provided max_length")
            return {
                "input_ids": torch.ones((len(contexts), 4), dtype=torch.long),
                "attention_mask": torch.ones((len(contexts), 4), dtype=torch.long),
            }

    scorer = object.__new__(LocalAlignScore)
    scorer.tokenizer = FakeTokenizer()
    scorer.max_length = 512
    encoded = scorer._encode(["context"], ["very long claim"])

    assert scorer.tokenizer.truncation_modes == ["only_first", True]
    assert encoded["input_ids"].shape == (1, 4)


def test_scale_uses_batched_yes_probability_for_source_support() -> None:
    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            width = max(len(feature["input_ids"]) for feature in features)
            input_ids = torch.zeros((len(features), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, feature in enumerate(features):
                values = torch.tensor(feature["input_ids"], dtype=torch.long)
                input_ids[index, : len(values)] = values
                attention_mask[index, : len(values)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["input_ids"].shape[0] == 2
            return {"scores": [torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 0.0]])]}

    scorer = object.__new__(LocalSCALE)
    scorer.tokenizer = FakeTokenizer()
    scorer.model = FakeModel()
    scorer.device = "cpu"
    scorer.yes_token_id = 0
    scorer.no_token_id = 1
    scores = scorer._forward([torch.tensor([[1, 2]]), torch.tensor([[3]])])

    assert scores[0] == pytest.approx(torch.softmax(torch.tensor([3.0, 1.0]), dim=0)[0].item())
    assert scores[1] == pytest.approx(torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[0].item())


def test_scale_chunking_retains_prompt_and_limits_source_window() -> None:
    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=False):
            # One token per whitespace-delimited item plus EOS.  This is enough
            # to verify the released token-window arithmetic without a model.
            token_count = len(text.split()) + 1
            return {"input_ids": torch.arange(token_count, dtype=torch.long).unsqueeze(0)}

    chunks = _build_chunks(
        FakeTokenizer(),
        "one two three four five six seven eight nine ten eleven twelve",
        "a claim",
        chunk_size=20,
        window_size=0.25,
    )

    assert len(chunks) >= 2
    assert all(chunk.ndim == 2 and chunk.shape[0] == 1 for chunk in chunks)
    assert all(chunk.shape[1] <= 20 for chunk in chunks)


def test_scale_joins_source_and_reports_high_is_better_score(tmp_path: Path) -> None:
    predictions = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"id": "a", "prediction": "supported", "reference": "r a"},
            {"id": "b", "prediction": "unsupported", "reference": "r b"},
        ],
    )
    source_file = _write_predictions(
        tmp_path / "test.jsonl",
        [{"id": "a", "text": "source a"}, {"id": "b", "text": "source b"}],
    )

    class FakeScale:
        _splitter = "test"

        def score_document(self, source: str, prediction: str) -> float:
            assert source in {"source a", "source b"}
            return 0.9 if prediction == "supported" else 0.2

    result = evaluate_scale(
        predictions,
        tmp_path / "unused-model",
        tmp_path / "scale.json",
        source_file=source_file,
        scorer=FakeScale(),
        details=True,
    )

    assert result["metric"] == "SCALE"
    assert result["scale_consistency"] == pytest.approx(0.55)
    assert result["unsupported_content_proxy"] == pytest.approx(0.45)
    assert result["score_direction"] == "scale_consistency_higher_is_better"
    assert result["source_joined_by_id"] is True
    assert result["rows"][0]["scale_consistency"] == pytest.approx(0.9)
