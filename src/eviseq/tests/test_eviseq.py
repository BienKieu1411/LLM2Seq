from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from eviseq.configuration import load_config, resolve_data_path, validate_config
from eviseq.data.dataset import LengthBucketBatchSampler, read_jsonl
from eviseq.evaluation.metrics import exact_match_score, token_f1_score
from eviseq.modeling.attention import mix_attention_outputs
from eviseq.modeling.bridge import balanced_salience_loss
from eviseq.training.checkpoint import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_configured_epoch_checkpoints,
    save_last_checkpoint,
)
from eviseq.training.objectives import evidence_info_nce_loss
from eviseq.training.trainer import _capture_optimizer_moments, _restore_optimizer_moments

ROOT = Path(__file__).resolve().parents[1]


def test_project_layout_has_clear_responsibility_boundaries() -> None:
    assert not (ROOT / "eviseq").exists()
    for directory in ("data", "evaluation", "modeling", "training"):
        assert (ROOT / "core" / directory / "__init__.py").is_file()
    for directory in ("ablations", "models", "tasks", "templates"):
        assert (ROOT / "configs" / directory).is_dir()
    assert (ROOT / "scripts" / "run.sh").is_file()


def _architecture(config: dict) -> dict:
    return {key: copy.deepcopy(config[key]) for key in ("model", "native_attention", "bridge", "decoder", "objectives")}


def test_all_configs_load_without_model_access() -> None:
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        config = load_config(path)
        assert config["_meta"]["config_path"] == str(path.resolve())


def test_dataset_recipes_share_the_same_model_graph() -> None:
    configs = [
        load_config(ROOT / "configs" / "tasks" / name)
        for name in ("wikilingua.yaml", "cnndm.yaml", "pubmed.yaml", "arxiv.yaml")
    ]
    assert all(_architecture(config) == _architecture(configs[0]) for config in configs[1:])


def test_configured_dataset_paths_resolve_from_repo_or_flattened_package() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "wikilingua.yaml")
    path = resolve_data_path(config["data"]["train_file"], config)
    assert path == ROOT / "datasets" / "wikilingua" / "train.jsonl"
    stale_layout_path = resolve_data_path("src/eviseq/datasets/wikilingua/train.jsonl", config)
    assert stale_layout_path == path


def test_generic_task_template_disables_summary_only_losses() -> None:
    config = load_config(ROOT / "configs" / "templates" / "custom_text2text.yaml")
    assert config["task"]["metrics"] == ["exact_match", "token_f1"]
    assert config["data"]["supervise_evidence"] is False
    assert config["objectives"]["salience_weight"] == 0.0
    assert config["objectives"]["use_evidence_contrastive"] is False


def test_task_recipe_templates_load_without_a_test_split() -> None:
    for name in ("translation", "question_answering", "classification"):
        config = load_config(ROOT / "configs" / "templates" / f"{name}.yaml")
        assert config["task"]["format"] == "text_to_text"
        assert config["data"]["supervise_evidence"] is False
    classification = load_config(ROOT / "configs" / "templates" / "classification.yaml")
    assert classification["data"]["test_file"] == ""


def test_arbitrary_json_fields_and_templates_are_mapped(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text(
        json.dumps(
            {
                "uid": "x1",
                "question": "Where?",
                "context": ["First sentence.", "Second sentence."],
                "answer": "Here.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_jsonl(
        path,
        data_config={
            "id_field": "uid",
            "source_template": "Question: {question}\nContext: {context}",
            "target_field": "answer",
            "list_separator": " ",
        },
    )
    assert rows == [
        {
            "id": "x1",
            "source": "Question: Where?\nContext: First sentence. Second sentence.",
            "target": "Here.",
        }
    ]


def test_nested_field_mapping_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text('{"payload":{"input":"abc","output":"xyz"}}\n', encoding="utf-8")
    rows = read_jsonl(
        path,
        data_config={"source_field": "payload.input", "target_field": "payload.output"},
    )
    assert rows[0]["source"] == "abc"
    assert rows[0]["target"] == "xyz"


def test_missing_task_field_has_an_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text('{"input":"abc"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="target"):
        read_jsonl(path, data_config={"source_field": "input", "target_field": "target"})


def test_non_summary_task_cannot_accidentally_enable_evidence_loss() -> None:
    config = load_config(ROOT / "configs" / "templates" / "custom_text2text.yaml")
    config["objectives"]["use_evidence_contrastive"] = True
    config["objectives"]["evidence_contrastive_weight"] = 0.1
    with pytest.raises(ValueError, match="supervise_evidence"):
        validate_config(config)


def test_warmup_can_be_disabled() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "wikilingua.yaml")
    config["training"]["interface_warmup_epochs"] = 0
    config["objectives"]["evidence_contrastive_warmup_epochs"] = 0
    validate_config(config)


def test_builtin_general_metrics() -> None:
    predictions = ["Paris", "red green"]
    references = ["paris", "red blue"]
    assert exact_match_score(predictions, references)["exact_match"] == 50.0
    assert token_f1_score(predictions, references)["token_f1"] == 75.0


def test_checkpoint_can_initialize_a_new_task_strictly_or_partially(tmp_path: Path) -> None:
    source = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    path = tmp_path / "last.pt"
    save_last_checkpoint(source, path, {"experiment": {"name": "old"}}, epoch=4, global_step=12)

    exact = copy.deepcopy(source)
    report = initialize_from_checkpoint(exact, path)
    assert report["loaded_tensors"] == len(source.state_dict())
    assert report["skipped_tensors"] == []

    changed = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))
    with pytest.raises(RuntimeError):
        initialize_from_checkpoint(changed, path)
    report = initialize_from_checkpoint(changed, path, strict=False)
    assert report["loaded_tensors"] == 2
    assert report["skipped_tensors"]


def test_epoch_and_best_checkpoints_are_complete_and_loadable(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
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
    assert (tmp_path / "epoch_001.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert first["best_value"] == 2.0

    second = save_configured_epoch_checkpoints(
        model,
        tmp_path,
        config,
        epoch=2,
        global_step=20,
        validation_metrics={"eval_loss_ce": 3.0},
    )
    assert (tmp_path / "epoch_002.pt").is_file()
    assert "best_path" not in second

    restored = nn.Linear(3, 2)
    assert load_checkpoint(restored, tmp_path / "epoch_002.pt")["checkpoint_role"] == "epoch"
    assert load_checkpoint(restored, tmp_path / "best.pt")["epoch"] == 1


def test_adam_moments_survive_the_stage_boundary() -> None:
    layer = nn.Linear(3, 2)
    warmup_optimizer = torch.optim.AdamW(layer.parameters(), lr=1.0e-4)
    layer(torch.randn(4, 3)).square().mean().backward()
    warmup_optimizer.step()
    carried = _capture_optimizer_moments(layer, warmup_optimizer)
    carried_count = len(carried)

    full_optimizer = torch.optim.AdamW(layer.parameters(), lr=1.0e-5)
    assert _restore_optimizer_moments(layer, full_optimizer, carried) == carried_count


def test_zero_bidirectional_gate_is_exactly_causal() -> None:
    torch.manual_seed(3)
    causal = torch.randn(2, 5, 4, 8)
    bidirectional = torch.randn_like(causal)
    actual = mix_attention_outputs(causal, bidirectional, "evidence", torch.zeros(4))
    torch.testing.assert_close(actual, causal, rtol=0, atol=0)


def test_salience_pairwise_term_rewards_correct_order() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    correct = balanced_salience_loss(torch.tensor([[2.0, -1.0, -2.0]]), labels, valid, ranking_weight=0.25)
    reversed_order = balanced_salience_loss(
        torch.tensor([[-2.0, 1.0, 2.0]]),
        labels,
        valid,
        ranking_weight=0.25,
    )
    assert correct < reversed_order


def test_evidence_contrastive_backpropagates() -> None:
    query = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_()
    keys = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1).requires_grad_()
    labels = torch.tensor([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0]])
    valid = labels.ge(0)
    result = evidence_info_nce_loss(query, keys, labels, valid, num_hard_negatives=2)
    result["evidence_contrastive_loss"].backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert keys.grad is not None and torch.isfinite(keys.grad).all()


def test_length_bucket_sampler_covers_each_example_once() -> None:
    lengths = list(range(1, 17))
    batches = list(LengthBucketBatchSampler(lengths, batch_size=4, seed=7, bucket_size_multiplier=4))
    assert sorted(index for batch in batches for index in batch) == list(range(16))


def test_run_script_has_no_upload_or_push_operation() -> None:
    script = (ROOT / "scripts" / "run.sh").read_text(encoding="utf-8").lower()
    assert "push_to_hub" not in script
    assert "huggingface-cli upload" not in script
    assert "git push" not in script
    assert "last.pt resolved_config.yaml" in script
