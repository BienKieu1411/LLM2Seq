from __future__ import annotations

from pathlib import Path

import torch

from eviseq_kd.build_cache import _resolve_path
from eviseq_kd.cache import TeacherRecord, load_cache, write_cache
from eviseq_kd.kd import top1_agreement, topk_distillation_loss
from eviseq_kd.student.configuration import load_config


def test_topk_loss_matches_renormalized_reference() -> None:
    student = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
    ids = torch.tensor([[[0, 2]]])
    teacher = torch.tensor([[[1.0, 0.0]]])
    temperature = 2.0
    actual = topk_distillation_loss(student, ids, teacher, temperature=temperature)
    student_selected = student.detach()[..., [0, 2]] / temperature
    teacher_probs = torch.softmax(teacher / temperature, dim=-1)
    reference = (
        torch.nn.functional.kl_div(torch.log_softmax(student_selected, dim=-1), teacher_probs, reduction="batchmean")
        * temperature**2
    )
    assert torch.allclose(actual, reference)
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_topk_mask_and_agreement_ignore_padding() -> None:
    student = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    ids = torch.tensor([[[0], [0]]])
    teacher = torch.zeros_like(ids, dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    assert top1_agreement(student, ids, mask=mask).item() == 1.0
    assert topk_distillation_loss(student, ids, teacher, mask=mask).item() >= 0.0


def test_cache_roundtrip(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    record = TeacherRecord("a", "hash", "step one", [10, 11], [[1]], [[0.5]])
    write_cache(path, {"teacher_model": "teacher", "has_topk": True}, [record])
    cache = load_cache(path)
    assert cache.metadata["teacher_model"] == "teacher"
    assert cache.get("a", 0, source_hash="hash").pseudo_target == "step one"


def test_cache_rejects_source_mismatch(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    write_cache(path, {}, [TeacherRecord("a", "expected", "text")])
    cache = load_cache(path)
    try:
        cache.get("a", 0, source_hash="different")
    except ValueError as exc:
        assert "source hash mismatch" in str(exc)
    else:
        raise AssertionError("source mismatch must be rejected")


def test_cache_does_not_fallback_when_id_is_wrong(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    write_cache(path, {}, [TeacherRecord("known", "hash", "text")])
    cache = load_cache(path)
    try:
        cache.get("wrong", 0, source_hash="hash")
    except KeyError as exc:
        assert "example ID" in str(exc)
    else:
        raise AssertionError("a mismatched ID must not fall back to the dataset index")


def test_cache_rejects_missing_source_hash(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    write_cache(path, {}, [TeacherRecord("a", "", "text")])
    cache = load_cache(path)
    try:
        cache.get("a", 0, source_hash="hash")
    except ValueError as exc:
        assert "no source hash" in str(exc)
    else:
        raise AssertionError("records without a source hash must be rejected")


def test_cache_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "teacher.jsonl"
    write_cache(
        path,
        {},
        [TeacherRecord("a", "hash-1", "first"), TeacherRecord("a", "hash-2", "second")],
    )
    try:
        load_cache(path)
    except ValueError as exc:
        assert "duplicate example IDs" in str(exc)
    else:
        raise AssertionError("duplicate example IDs must be rejected")


def test_import_isolation() -> None:
    import eviseq_kd
    import sys

    assert "eviseq" not in sys.modules
    assert eviseq_kd.__file__ is not None


def test_config_resolves_bundled_dataset_relative_to_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke_a100.yaml"
    config = load_config(config_path)
    dataset_path = _resolve_path(config["data"]["train_file"], config)
    assert dataset_path.is_file()
    assert dataset_path.name == "train.jsonl"

    stale_layout_path = _resolve_path("src/eviseq_kd/datasets/wikilingua/train.jsonl", config)
    assert stale_layout_path == dataset_path


def test_trainer_builds_missing_teacher_cache_on_demand(tmp_path, monkeypatch) -> None:
    from eviseq_kd import trainer

    cache_path = tmp_path / "teacher.jsonl"
    config = {
        "_meta": {"config_path": str(tmp_path / "config.yaml")},
        "training": {
            "distillation": {
                "enabled": True,
                "cache_path": str(cache_path),
                "teacher_model": "teacher",
            }
        },
        "limits": {"max_train_examples": 3},
        "generation": {"max_new_tokens": 7},
        "data": {"max_source_length": 11},
    }
    called = {}

    monkeypatch.setattr(trainer, "load_config", lambda _: config)

    def fake_build(config_path, output_path, **kwargs):
        called.update(kwargs)
        cache_path.write_text("cache", encoding="utf-8")
        return cache_path

    monkeypatch.setattr(trainer, "build_teacher_cache", fake_build)
    assert trainer._ensure_teacher_cache("ignored.yaml", auto_build=True) == cache_path
    assert called["max_examples"] == 3
    assert called["max_new_tokens"] == 7

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("existing teacher cache must be reused")

    monkeypatch.setattr(trainer, "build_teacher_cache", unexpected_rebuild)
    assert trainer._ensure_teacher_cache("ignored.yaml", auto_build=True) == cache_path


def test_kd_collator_keeps_gold_and_pseudo_branches_separate() -> None:
    from eviseq_kd.dataset import KDCollator

    class BaseCollator:
        def __call__(self, features):
            assert all(not any(key.startswith("pseudo_") for key in item) for item in features)
            return {"gold_rows": torch.tensor([item["input_ids"].numel() for item in features])}

    collator = KDCollator(BaseCollator(), decoder_pad_id=0, max_decoder_length=8)
    features = [
        {
            "input_ids": torch.ones(2, dtype=torch.long),
            "pseudo_decoder_input_ids": torch.tensor([1, 2, 3]),
            "pseudo_decoder_attention_mask": torch.ones(3, dtype=torch.long),
            "pseudo_labels": torch.tensor([-100, 2, 3]),
            "teacher_topk_ids": torch.tensor([[0, 1], [1, 2], [2, 3]]),
            "teacher_topk_logits": torch.zeros(3, 2),
            "teacher_kd_mask": torch.tensor([False, True, True]),
        },
        {
            "input_ids": torch.ones(3, dtype=torch.long),
            "pseudo_decoder_input_ids": torch.tensor([1, 2]),
            "pseudo_decoder_attention_mask": torch.ones(2, dtype=torch.long),
            "pseudo_labels": torch.tensor([-100, 2]),
            "teacher_topk_ids": torch.tensor([[0, 1], [1, 2]]),
            "teacher_topk_logits": torch.zeros(2, 2),
            "teacher_kd_mask": torch.tensor([False, True]),
        },
    ]
    batch = collator(features)
    assert tuple(batch["pseudo_decoder_input_ids"].shape) == (2, 3)
    assert tuple(batch["teacher_topk_ids"].shape) == (2, 3, 2)
    assert batch["teacher_kd_mask"].tolist() == [[False, True, True], [False, True, False]]
