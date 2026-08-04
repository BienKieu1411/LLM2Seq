from __future__ import annotations

from pathlib import Path

import torch
from eviseq_kd.cache import TeacherRecord, load_cache, write_cache
from eviseq_kd.kd import logits_kl_loss, sequence_kd_loss, top1_agreement, topk_distillation_loss
from eviseq_kd.paths import resolve_artifact_path, resolve_input_path
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


def test_topk_loss_with_log_normalizer_matches_collapsed_full_vocab_kl() -> None:
    student = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
    teacher_full = torch.tensor([[[1.5, 0.5, -0.5, -1.5]]])
    ids = torch.tensor([[[0, 1]]])
    temperature = 2.0
    teacher_topk = teacher_full.gather(-1, ids)
    teacher_log_normalizer = torch.logsumexp(teacher_full / temperature, dim=-1)
    actual = topk_distillation_loss(
        student,
        ids,
        teacher_topk,
        temperature=temperature,
        teacher_log_normalizers=teacher_log_normalizer,
    )

    teacher_probs = torch.softmax(teacher_full / temperature, dim=-1)
    student_probs = torch.softmax(student / temperature, dim=-1)
    teacher_collapsed = torch.cat(
        [teacher_probs.gather(-1, ids), teacher_probs[..., 2:].sum(dim=-1, keepdim=True)],
        dim=-1,
    )
    student_collapsed = torch.cat(
        [student_probs.gather(-1, ids), student_probs[..., 2:].sum(dim=-1, keepdim=True)],
        dim=-1,
    )
    reference = (teacher_collapsed * (teacher_collapsed.log() - student_collapsed.log())).sum() * temperature**2
    assert torch.allclose(actual, reference, atol=1.0e-6)


def test_topk_mask_and_agreement_ignore_padding() -> None:
    student = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    ids = torch.tensor([[[0], [0]]])
    teacher = torch.zeros_like(ids, dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    assert top1_agreement(student, ids, mask=mask).item() == 1.0
    assert topk_distillation_loss(student, ids, teacher, mask=mask).item() >= 0.0


def test_sequence_kd_and_full_kl_ignore_padding_and_handle_all_masked() -> None:
    student = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2]])
    sequence = sequence_kd_loss(student, labels)
    sequence.backward()
    assert torch.isfinite(sequence)
    assert student.grad is not None and student.grad[:, 0].abs().sum().item() == 0.0

    student.grad = None
    teacher = torch.randn(1, 3, 4, requires_grad=True)
    full = logits_kl_loss(student, teacher, labels=labels, temperature=2.0)
    full.backward()
    assert torch.isfinite(full)
    assert teacher.grad is None
    assert student.grad is not None and student.grad[:, 0].abs().sum().item() == 0.0

    all_masked = logits_kl_loss(student, teacher, labels=torch.full((1, 3), -100))
    assert all_masked.item() == 0.0


def test_topk_rejects_unaligned_vocab_and_width() -> None:
    student = torch.zeros(1, 1, 2)
    teacher = torch.zeros(1, 1, 2)
    try:
        topk_distillation_loss(student, torch.tensor([[[0, 1, 2]]]), torch.zeros(1, 1, 3))
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("K > V must be rejected")
    try:
        topk_distillation_loss(student, torch.tensor([[[0, 3]]]), teacher)
    except ValueError as exc:
        assert "student vocabulary" in str(exc)
    else:
        raise AssertionError("teacher IDs outside the student vocabulary must be rejected")


def test_kd_wrapper_reuses_each_supervised_forward_for_ce_and_logits(monkeypatch) -> None:
    import eviseq_kd.model as model_module

    class DummyBase(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            del config
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self._contrastive_scale = 1.0
            self.calls = 0

        def set_contrastive_scale(self, value):
            self._contrastive_scale = float(value)

        def forward(self, *, decoder_input_ids, labels, return_full_logits=False, **kwargs):
            del kwargs
            self.calls += 1
            logits = self.weight * torch.ones(
                (*decoder_input_ids.shape, 4),
                device=decoder_input_ids.device,
            )
            loss_ce = (self.weight - 1.0).square()
            result = {"loss": loss_ce, "loss_ce": loss_ce}
            if return_full_logits:
                result["logits"] = logits
            return result

    monkeypatch.setattr(model_module, "EviSeq", DummyBase)
    model = model_module.EviSeqKD(
        {
            "training": {
                "distillation": {
                    "enabled": True,
                    "sequence_enabled": True,
                    "sequence_weight": 0.3,
                    "logit_enabled": True,
                    "logit_weight": 0.3,
                    "temperature": 2.0,
                    "logit_path_mix": 0.5,
                }
            }
        }
    )
    labels = torch.tensor([[1, 2]])
    ids = torch.tensor([[[0], [0]]])
    teacher_logits = torch.zeros(1, 2, 1)
    normalizers = torch.full((1, 2), torch.log(torch.tensor(4.0)))
    output = model(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        decoder_input_ids=torch.ones(1, 2, dtype=torch.long),
        decoder_attention_mask=torch.ones(1, 2, dtype=torch.long),
        labels=labels,
        pseudo_decoder_input_ids=torch.ones(1, 2, dtype=torch.long),
        pseudo_decoder_attention_mask=torch.ones(1, 2, dtype=torch.long),
        pseudo_labels=labels,
        teacher_topk_ids=ids,
        teacher_topk_logits=teacher_logits,
        teacher_topk_log_normalizers=normalizers,
        teacher_gold_topk_ids=ids,
        teacher_gold_topk_logits=teacher_logits,
        teacher_gold_topk_log_normalizers=normalizers,
    )
    assert model.base.calls == 2
    output["loss"].backward()
    assert model.base.weight.grad is not None


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


def test_epoch_best_and_last_checkpoints_are_complete(tmp_path) -> None:
    from eviseq_kd.student.training.checkpoint import (
        load_checkpoint,
        save_configured_epoch_checkpoints,
        save_last_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    config = {
        "checkpoint": {
            "save_each_epoch": True,
            "save_best": True,
            "best_metric": "eval_loss_ce",
            "best_mode": "min",
        }
    }
    first = save_configured_epoch_checkpoints(
        model,
        tmp_path,
        config,
        epoch=1,
        global_step=10,
        validation_metrics={"eval_loss_ce": 2.0},
    )
    assert Path(first["epoch_path"]).is_file()
    assert Path(first["best_path"]).is_file()

    save_configured_epoch_checkpoints(
        model,
        tmp_path,
        config,
        epoch=2,
        global_step=20,
        validation_metrics={"eval_loss_ce": 3.0},
    )
    assert load_checkpoint(torch.nn.Linear(2, 2), tmp_path / "best.pt")["epoch"] == 1

    save_last_checkpoint(model, tmp_path / "last.pt", config, epoch=2, global_step=20)
    payload = load_checkpoint(torch.nn.Linear(2, 2), tmp_path / "last.pt")
    assert payload["checkpoint_role"] == "last"
    assert payload["epoch"] == 2
    assert set(payload["model_state_dict"]) == set(model.state_dict())


def test_import_isolation() -> None:
    import sys

    import eviseq_kd

    assert "eviseq" not in sys.modules
    assert eviseq_kd.__file__ is not None


def test_config_resolves_bundled_dataset_relative_to_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke_a100.yaml"
    config = load_config(config_path)
    dataset_path = resolve_input_path(config["data"]["train_file"], config)
    assert dataset_path.is_file()
    assert dataset_path.name == "train.jsonl"

    stale_layout_path = resolve_input_path("src/eviseq_kd/datasets/wikilingua/train.jsonl", config)
    assert stale_layout_path == dataset_path


def test_artifacts_resolve_from_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_artifact_path("runs/cache.jsonl") == tmp_path / "runs" / "cache.jsonl"


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
            "teacher_topk_log_normalizers": torch.zeros(3),
            "teacher_kd_mask": torch.tensor([False, True, True]),
        },
        {
            "input_ids": torch.ones(3, dtype=torch.long),
            "pseudo_decoder_input_ids": torch.tensor([1, 2]),
            "pseudo_decoder_attention_mask": torch.ones(2, dtype=torch.long),
            "pseudo_labels": torch.tensor([-100, 2]),
            "teacher_topk_ids": torch.tensor([[0, 1], [1, 2]]),
            "teacher_topk_logits": torch.zeros(2, 2),
            "teacher_topk_log_normalizers": torch.zeros(2),
            "teacher_kd_mask": torch.tensor([False, True]),
        },
    ]
    batch = collator(features)
    assert tuple(batch["pseudo_decoder_input_ids"].shape) == (2, 3)
    assert tuple(batch["teacher_topk_ids"].shape) == (2, 3, 2)
    assert batch["teacher_kd_mask"].tolist() == [[False, True, True], [False, True, False]]


def test_topk_dataset_features_require_alignment_metadata() -> None:
    from eviseq_kd.dataset import KDText2TextDataset

    class Tokenizer:
        eos_token_id = 4

        def __len__(self):
            return 4

    dataset = object.__new__(KDText2TextDataset)
    dataset.decoder_tokenizer = Tokenizer()
    dataset.teacher_cache = type("Cache", (), {"metadata": {}})()
    record = TeacherRecord(
        "a",
        "hash",
        "text",
        teacher_topk_ids=[[1, 2]],
        teacher_topk_logits=[[0.0, 1.0]],
        teacher_topk_log_normalizers=[1.5],
    )
    try:
        dataset._topk_features(record, target_count=1, prompt_prefix_count=1)
    except ValueError as exc:
        assert "teacher_vocab_size" in str(exc)
    else:
        raise AssertionError("top-k cache without vocab metadata must fail loudly")


def test_topk_dataset_features_map_ids_and_prefix_prompt_mask() -> None:
    from eviseq_kd.dataset import KDText2TextDataset

    class Tokenizer:
        eos_token_id = 4

        def __len__(self):
            return 4

    dataset = object.__new__(KDText2TextDataset)
    dataset.decoder_tokenizer = Tokenizer()
    dataset.teacher_cache = type(
        "Cache",
        (),
        {
            "metadata": {
                "teacher_vocab_size": 3,
                "vocab_alignment": {"type": "explicit", "mapping": [2, 0, 1]},
            }
        },
    )()
    dataset.student_vocab_size = 4
    dataset._teacher_vocab_mapping = None
    record = TeacherRecord(
        "a",
        "hash",
        "text",
        teacher_topk_ids=[[0, 1], [2, 1]],
        teacher_topk_logits=[[1.0, 0.0], [0.0, 1.0]],
        teacher_topk_log_normalizers=[1.5, 1.5],
    )
    result = dataset._topk_features(record, target_count=2, prompt_prefix_count=1)
    assert result["teacher_topk_ids"].tolist() == [[0, 0], [2, 0], [1, 0]]
    assert result["teacher_kd_mask"].tolist() == [False, True, True]


def test_truncated_pseudo_target_masks_soft_kd_on_synthetic_eos() -> None:
    from eviseq_kd.dataset import KDText2TextDataset

    class Tokenizer:
        eos_token_id = 4

        def __len__(self):
            return 5

    dataset = object.__new__(KDText2TextDataset)
    dataset.decoder_tokenizer = Tokenizer()
    dataset.teacher_cache = type(
        "Cache",
        (),
        {
            "metadata": {
                "teacher_vocab_size": 5,
                "vocab_alignment": "identity",
                "topk_includes_eos": True,
            }
        },
    )()
    dataset.student_vocab_size = 5
    dataset._teacher_vocab_mapping = None
    record = TeacherRecord(
        "a",
        "hash",
        "text",
        pseudo_token_ids=[1, 2, 3, 4],
        teacher_topk_ids=[[1], [2], [3], [4]],
        teacher_topk_logits=[[2.0], [2.0], [2.0], [2.0]],
        teacher_topk_log_normalizers=[2.5, 2.5, 2.5, 2.5],
        generated_eos_observed=True,
    )
    result = dataset._topk_features(
        record,
        target_count=3,
        prompt_prefix_count=1,
        student_target_ids=torch.tensor([1, 2, 4]),
    )
    assert result["teacher_kd_mask"].tolist() == [False, True, True, False]
