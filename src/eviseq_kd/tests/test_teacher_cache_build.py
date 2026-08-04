from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import torch


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 5
    pad_token = "<pad>"
    eos_token = "<eos>"
    all_special_ids = [0, 3, 5]
    chat_template = None
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}

    def __len__(self) -> int:
        return 8

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(8)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(self, prompts, **kwargs):
        del kwargs
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = [[1, 2], [6, 7, 1]][: len(prompts)]
        width = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            padding = [self.pad_token_id] * (width - len(row))
            input_ids.append(padding + row)
            attention_mask.append([0] * len(padding) + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return " ".join(f"tok-{int(token_id)}" for token_id in token_ids)


class _FakeTeacher:
    def __init__(self) -> None:
        self.forward_inputs: list[torch.Tensor] = []

    def to(self, device):
        del device
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def generate(self, input_ids, **kwargs):
        del kwargs
        suffix = torch.tensor([[1, 3, 2, 5], [2, 3, 1, 5]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)

    def __call__(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        assert tuple(attention_mask.shape) == tuple(input_ids.shape)
        self.forward_inputs.append(input_ids.detach().cpu())
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 8), -10.0, device=input_ids.device)
        for position in range(input_ids.shape[1] - 1):
            next_token = input_ids[:, position + 1]
            logits[torch.arange(input_ids.shape[0]), position, next_token] = 10.0
        return SimpleNamespace(logits=logits)


def test_build_cache_stores_forward_topk_with_explicit_alignment(tmp_path, monkeypatch) -> None:
    build_cache_module = importlib.import_module("eviseq_kd.build_cache")
    from eviseq_kd.cache import load_cache

    tokenizer = _FakeTokenizer()
    teacher = _FakeTeacher()
    calls = {"tokenizer": 0, "model": 0}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert name == "fake-teacher"
            assert kwargs["trust_remote_code"] is True
            calls["tokenizer"] += 1
            return tokenizer

    class _AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert name == "fake-teacher"
            assert kwargs["trust_remote_code"] is True
            calls["model"] += 1
            return teacher

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer, AutoModelForCausalLM=_AutoModel),
    )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("placeholder\n", encoding="utf-8")
    config = {
        "_meta": {"config_path": str(tmp_path / "config.yaml")},
        "data": {"train_file": str(rows_path), "clean_wikihow_metadata": False},
        "training": {"distillation": {"teacher_model": "fake-teacher"}},
    }
    monkeypatch.setattr(build_cache_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        build_cache_module,
        "read_jsonl",
        lambda *args, **kwargs: [
            {"id": "one", "source": "first", "target": "gold one"},
            {"id": "two", "source": "second", "target": "gold two"},
        ],
    )

    output = build_cache_module.build_cache(
        "ignored.yaml",
        str(tmp_path / "teacher.jsonl"),
        teacher_model_name="fake-teacher",
        device_name="cpu",
        batch_size=2,
        max_new_tokens=4,
        num_beams=1,
        top_k=2,
    )
    cache = load_cache(output)

    assert calls == {"tokenizer": 1, "model": 1}
    assert len(teacher.forward_inputs) == 2
    assert teacher.forward_inputs[0].shape == (2, 7)
    assert teacher.forward_inputs[1].shape == (2, 6)
    assert cache.metadata["top_k"] == 2
    assert cache.metadata["has_gold_topk"] is True
    assert cache.metadata["teacher_tokenizer_vocab_size"] == 8
    assert cache.metadata["tokenizer_vocab_size"] == 8
    assert cache.metadata["teacher_tokenizer_fingerprint"] == cache.metadata["tokenizer_fingerprint"]
    assert cache.metadata["topk_alignment"]["tokenization"] == "teacher"

    first, second = cache.records
    assert first.pseudo_token_ids == [1, 2, 5]
    assert second.pseudo_token_ids == [2, 1, 5]
    assert first.teacher_generated_token_ids == [1, 3, 2, 5]
    assert first.teacher_topk_positions == [0, 2, 3]
    assert second.teacher_topk_positions == [0, 2, 3]
    assert len(first.teacher_topk_ids) == len(first.pseudo_token_ids) == 3
    assert len(first.teacher_topk_logits) == 3
    assert first.teacher_topk_ids[0][0] == 1
    assert first.teacher_topk_ids[1][0] == 2
    assert second.teacher_topk_ids[0][0] == 2
    assert second.teacher_topk_ids[1][0] == 1
    assert first.gold_token_ids == [1, 2, 5]
    assert len(first.gold_topk_ids) == len(first.gold_token_ids) == 3
    assert first.prompt_token_count == 2
    assert second.prompt_token_count == 3
    assert first.prompt_sequence_width == second.prompt_sequence_width == 3


def test_cache_loader_rejects_misaligned_topk_rows(tmp_path) -> None:
    from eviseq_kd.cache import TeacherRecord, write_cache

    record = TeacherRecord(
        "one",
        "hash",
        "text",
        [1, 2],
        [[1, 0]],
        [[2.0, 1.0]],
    )
    try:
        write_cache(
            tmp_path / "teacher.jsonl",
            {
                "has_topk": True,
                "top_k": 2,
                "topk_alignment": {"rows": "one row per pseudo token"},
            },
            [record],
        )
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned top-k rows must be rejected for the explicit schema")
