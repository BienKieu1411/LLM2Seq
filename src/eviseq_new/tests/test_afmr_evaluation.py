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


def test_chat_prompt_is_not_penalized_as_generated_text(monkeypatch):
    from pathlib import Path

    from eviseq_afmr.config import load_config
    from eviseq_afmr.evaluation import generate
    from eviseq_afmr.modeling.model import EviSeqAFMR
    from eviseq_afmr.runtime import build_loaders

    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    loader = build_loaders(config)["train"]
    batch = next(iter(loader))
    model = EviSeqAFMR(config).eval()
    model.config["data"]["decoder_chat_template"] = True
    seen = []
    original = generate._apply_repetition_penalty

    def track(scores, ids, penalty):
        seen.append(ids.shape[1])
        return original(scores, ids, penalty)

    monkeypatch.setattr(generate, "_apply_repetition_penalty", track)
    generate.generate_greedy(
        model, batch, loader.collate_fn.decoder_tokenizer, 3, min_new_tokens=3, repetition_penalty=1.05
    )
    assert seen == [0, 1, 2]


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
        _apply_no_repeat_ngram,
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
        actual = scores.clone()
        _apply_no_repeat_ngram(actual, token_ids, ngram_size)
        torch.testing.assert_close(actual, expected_scores)


def test_finished_rows_are_removed_without_changing_greedy_tokens():
    from types import SimpleNamespace

    import torch
    from eviseq_afmr.evaluation.generate import generate_greedy
    from eviseq_afmr.runtime import _TinyTokenizer

    class Cache:
        def batch_select_indices(self, indices):
            pass

    class Decoder:
        def __init__(self):
            self.sizes = []

        def eval(self):
            pass

        def prepare_cross_cache(self, memory):
            self.cached = memory.clone()

        def clear_cross_cache(self):
            self.cached = None

        def select_cross_cache(self, indices):
            self.cached = self.cached.index_select(0, indices)

        def __call__(self, tokens, memory, mask, bias, decode_mask, **kwargs):
            self.sizes.append(tokens.shape[0])
            torch.testing.assert_close(memory, self.cached)
            stop = decode_mask.sum(-1) >= memory[:, 0, 0]
            selected = torch.where(stop, 2, 3)
            scores = torch.zeros(tokens.shape[0], 1, 4)
            scores[:, 0].scatter_(1, selected[:, None], 10.0)
            return scores, Cache(), None

    class Model:
        def __init__(self):
            self.decoder = Decoder()

        def eval(self):
            pass

        def encode_source(self, *args):
            return SimpleNamespace(
                memory=torch.tensor([1.0, 3.0, 5.0]).reshape(3, 1, 1),
                memory_mask=torch.ones(3, 1, dtype=torch.bool),
                source_bias=torch.zeros(3, 1),
            )

    batch = {
        "input_ids": torch.ones(3, 2, dtype=torch.long),
        "attention_mask": torch.ones(3, 2),
        "source_content_mask": torch.ones(3, 2),
        "decoder_prompt_ids": torch.ones(3, 1, dtype=torch.long),
        "decoder_prompt_mask": torch.ones(3, 1),
    }
    baseline, optimized = Model(), Model()
    _, expected = generate_greedy(baseline, batch, _TinyTokenizer(), 8, compact_finished=False)
    _, actual = generate_greedy(optimized, batch, _TinyTokenizer(), 8, compact_finished=True)
    torch.testing.assert_close(actual, expected)
    assert optimized.decoder.sizes == [3, 2, 2, 1, 1]
    assert baseline.decoder.sizes == [3] * 5
    assert optimized.decoder.cached is None
