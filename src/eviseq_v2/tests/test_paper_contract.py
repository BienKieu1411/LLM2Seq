from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from eviseq_v2.config import load_config
from eviseq_v2.dev_table import build_table
from eviseq_v2.evaluate import (
    _complete_paper_test,
    _load_verified_checkpoint,
    _reserve_paper_test,
)
from eviseq_v2.paper_compare import compare
from eviseq_v2.parameter_manifest import build_parameter_manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m"


class _ToyEviSeq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.model = nn.Linear(4, 4, bias=False)
        self.encoder.evidence_norm = nn.LayerNorm(4)
        self.encoder.evidence_head = nn.Linear(4, 1)
        self.encoder.evidence_view_gate = nn.Parameter(torch.zeros(2, 2))
        self.encoder.generic_token_gate = nn.Linear(4, 2)
        self.adapter = nn.Linear(4, 4, bias=False)
        self.alignment_head = nn.Linear(4, 3, bias=False)
        self.evidence_contrastive_head = nn.Linear(4, 2, bias=False)
        self.decoder = nn.Module()
        self.decoder.embed_tokens = nn.Embedding(8, 4)
        self.decoder.cross_attn = nn.Linear(4, 4, bias=False)
        self.decoder.cross_attn_norm = nn.LayerNorm(4)
        self.decoder.cross_gate = nn.Parameter(torch.zeros(()))
        self.decoder.lm_head = nn.Linear(4, 8, bias=False)
        self.decoder.lm_head.weight = self.decoder.embed_tokens.weight


def _parameter_config(target: int, *, eligible: bool = True) -> dict:
    return {
        "native_attention": {"backend": "qwen_native", "variant": "evidence"},
        "reporting": {
            "parameter_efficiency_claim_eligible": eligible,
            "target_total_footprint_approx": target,
        },
        "_meta": {
            "architecture_sha256": "a" * 64,
            "inference_protocol_sha256": "b" * 64,
            "evaluation_contract_sha256": "c" * 64,
        },
    }


def test_parameter_manifest_counts_tied_weights_once_and_uses_resident_budget() -> None:
    model = _ToyEviSeq()
    unique = sum(parameter.numel() for parameter in model.parameters())
    state_elements = sum(tensor.numel() for tensor in model.state_dict().values())
    manifest = build_parameter_manifest(model, _parameter_config(unique + 1))
    assert state_elements > unique
    assert manifest["resident_training_total_unique"] == unique
    assert sum(manifest["by_component"].values()) == unique
    assert manifest["deployable_resident_without_train_aux"] + manifest["training_only_total_unique"] == unique
    assert manifest["strictly_under_budget"] is True
    assert manifest["by_component"]["decoder_cross_attention"] == 25
    expected_training_only = model.alignment_head.weight.numel() + model.evidence_contrastive_head.weight.numel()
    assert manifest["by_component"]["training_only_contrastive"] == expected_training_only
    assert manifest["training_only_total_unique"] == expected_training_only


def test_parameter_budget_fails_closed_but_ineligible_control_is_reported() -> None:
    model = _ToyEviSeq()
    unique = sum(parameter.numel() for parameter in model.parameters())
    with pytest.raises(RuntimeError, match="not strictly below"):
        build_parameter_manifest(model, _parameter_config(unique))
    manifest = build_parameter_manifest(model, _parameter_config(unique - 1, eligible=False))
    assert manifest["strictly_under_budget"] is False
    assert manifest["parameter_efficiency_claim_eligible"] is False


def test_evaluation_recomputes_and_binds_parameter_and_data_manifests(tmp_path: Path) -> None:
    model = _ToyEviSeq()
    unique = sum(parameter.numel() for parameter in model.parameters())
    config = _parameter_config(unique + 1)
    manifest = build_parameter_manifest(model, config)
    config["_runtime"] = {"parameter_manifest": manifest}
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    _json(tmp_path / "parameter_manifest.json", manifest)
    data_manifest = {
        name: {"num_examples": index, "sha256": str(index) * 64}
        for index, name in enumerate(("train", "validation", "test"), start=1)
    }
    _json(tmp_path / "data_manifest.json", data_manifest)
    payload = {"config": copy.deepcopy(config), "data_manifest": copy.deepcopy(data_manifest)}
    actual: dict = {}
    returned = _load_verified_checkpoint(
        model,
        checkpoint,
        config=config,
        checkpoint=checkpoint,
        original_loader=lambda _model, _path: payload,
        actual_manifest=actual,
    )
    assert returned is payload
    assert actual == manifest

    broken = copy.deepcopy(payload)
    broken["config"]["_runtime"]["parameter_manifest"]["resident_training_total_unique"] += 1
    with pytest.raises(RuntimeError, match="Checkpoint-embedded parameter manifest"):
        _load_verified_checkpoint(
            model,
            checkpoint,
            config=config,
            checkpoint=checkpoint,
            original_loader=lambda _model, _path: broken,
            actual_manifest={},
        )

    drifted = copy.deepcopy(payload)
    drifted["config"]["_meta"]["evaluation_contract_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="prompt/generation contract"):
        _load_verified_checkpoint(
            model,
            checkpoint,
            config=config,
            checkpoint=checkpoint,
            original_loader=lambda _model, _path: drifted,
            actual_manifest={},
        )


def test_paper_test_reservation_is_one_shot_and_records_completion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    output = tmp_path / "last_test_predictions.jsonl"
    config = _parameter_config(1_000)
    fingerprint = {"num_examples": 2, "sha256": "f" * 64}
    marker, reservation = _reserve_paper_test(checkpoint, output, config, fingerprint)
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "reserved"
    with pytest.raises(FileExistsError, match="already reserved or completed"):
        _reserve_paper_test(checkpoint, output, config, fingerprint)
    metrics = {
        "predictions_sha256": "e" * 64,
        "num_examples": 2,
        "predictions_file": str(output),
    }
    _complete_paper_test(marker, reservation, metrics)
    completed = json.loads(marker.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["evaluation_contract_sha256"] == "c" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dev_artifacts(tmp_path: Path) -> dict[str, dict[str, Path]]:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(
        json.dumps({"id": "train_1", "source": "train", "target": "train"}) + "\n",
        encoding="utf-8",
    )
    validation_rows = [
        {"id": "val_1", "source": "source one", "target": "Reference one."},
        {"id": "val_2", "source": "source two", "target": "Reference two."},
    ]
    validation.write_text("".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8")
    test.write_text(
        json.dumps({"id": "test_1", "source": "test", "target": "test"}) + "\n",
        encoding="utf-8",
    )
    bases = {
        "c0": ROOT / "configs" / "ablations" / "c0_causal.yaml",
        "c2": ROOT / "configs" / "ablations" / "c2_dec2enc.yaml",
        "c3-no-cl": ROOT / "configs" / "ablations" / "c3_no_contrastive.yaml",
        "c3": ROOT / "configs" / "wikilingua.yaml",
    }
    artifacts: dict[str, dict[str, Path]] = {}
    fingerprint = None
    for index, (name, base) in enumerate(bases.items()):
        directory = tmp_path / name
        directory.mkdir()
        config_path = directory / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_base_": str(base),
                    "experiment": {"name": f"test_{name}", "output_dir": str(directory)},
                    "data": {
                        "train_file": str(train),
                        "validation_file": str(validation),
                        "test_file": str(test),
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        config = load_config(config_path)
        if fingerprint is None:
            from eviseq_v2.data import dataset_fingerprint

            fingerprint = dataset_fingerprint(str(validation), 0)
        predictions = directory / "last_validation_predictions.jsonl"
        prediction_rows = [
            {
                "id": row["id"],
                "reference": row["target"],
                "prediction": f"prediction {index} {row['id']}",
            }
            for row in validation_rows
        ]
        predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8")
        predictions_hash = _sha(predictions)
        metrics = {
            "num_examples": 2,
            "evaluation_split": "validation",
            "paper_test": False,
            "checkpoint_data_matches_current": True,
            "evaluation_data_fingerprint": fingerprint,
            "checkpoint_data_fingerprint": fingerprint,
            "architecture_sha256": config["_meta"]["architecture_sha256"],
            "inference_protocol_sha256": config["_meta"]["inference_protocol_sha256"],
            "evaluation_contract_sha256": config["_meta"]["evaluation_contract_sha256"],
            "native_attention_variant": config["native_attention"]["variant"],
            "predictions_sha256": predictions_hash,
            "generation": {
                "max_new_tokens": 256,
                "min_new_tokens": 16,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 3,
            },
            "source_prefix": config["data"]["source_prefix"],
            "max_source_length": config["data"]["max_source_length"],
            "max_target_length": config["data"]["max_target_length"],
            "training_parameters": 1_369_121_774,
            "deployable_parameters": 1_368_594_414,
        }
        rouge = {
            "rouge1": 50.0 + index,
            "rouge2": 20.0 + index,
            "rougeL": 45.0 + index,
            "num_examples": 2,
            "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
            "stemming": True,
            "pyrouge_default_args": PROTOCOL,
            "prediction_field": "prediction",
            "reference_field": "reference",
            "predictions_file": str(predictions),
            "predictions_sha256": predictions_hash,
        }
        metrics_path = predictions.with_suffix(".metrics.json")
        rouge_path = predictions.with_suffix(".rouge155.json")
        _json(metrics_path, metrics)
        _json(rouge_path, rouge)
        artifacts[name] = {
            "config": config_path,
            "metrics": metrics_path,
            "rouge": rouge_path,
        }
    return artifacts


def test_dev_table_accepts_exactly_three_validation_ablations_and_main(tmp_path: Path) -> None:
    artifacts = _dev_artifacts(tmp_path)
    output = tmp_path / "dev_table.json"
    result = build_table(artifacts, output)
    assert result["table_role"] == "DEV_ONLY_MODEL_SELECTION"
    assert result["num_core_ablations"] == 3
    assert [row["name"] for row in result["rows"]] == ["c0", "c2", "c3-no-cl", "c3"]
    assert all("evidence_conditioned_encoder_view" in row for row in result["rows"])
    assert all("deployable_parameters" in row for row in result["rows"])
    assert all("training_only_parameters" in row for row in result["rows"])
    assert result["rows"][-1]["delta_rouge2_vs_c3"] == 0.0
    assert output.is_file() and output.with_suffix(".md").is_file()


def test_dev_table_rejects_test_or_sampled_artifacts(tmp_path: Path) -> None:
    artifacts = _dev_artifacts(tmp_path)
    metrics_path = artifacts["c0"]["metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["evaluation_split"] = "test"
    _json(metrics_path, metrics)
    with pytest.raises(ValueError, match="not DEV"):
        build_table(artifacts, tmp_path / "invalid.json")


def _paper_artifacts(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "paper.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "_base_": str(ROOT / "configs" / "wikilingua.yaml"),
                "benchmark": {
                    "paper": {
                        "num_examples": 2,
                        "test_sha256": "f" * 64,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    loaded_config = load_config(config)
    architecture_sha256 = loaded_config["_meta"]["architecture_sha256"]
    candidate_predictions = tmp_path / "candidate.jsonl"
    baseline_predictions = tmp_path / "baseline.jsonl"
    rows = [
        {"id": "test_1", "reference": "Một.", "prediction": "Một."},
        {"id": "test_2", "reference": "Hai.", "prediction": "Hai."},
    ]
    candidate_predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    baseline_rows = [dict(row, prediction=f"baseline {index}") for index, row in enumerate(rows)]
    baseline_predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in baseline_rows), encoding="utf-8"
    )
    fingerprint = {"num_examples": 2, "sha256": "f" * 64}
    generation = {
        "max_new_tokens": 256,
        "min_new_tokens": 16,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
    }
    common_metrics = {
        "num_examples": 2,
        "evaluation_split": "test",
        "checkpoint_test_matches_current": True,
        "test_data_fingerprint": fingerprint,
        "generation": generation,
        "source_prefix": "Tóm tắt thành các bước hành động ngắn, đúng thứ tự; không thêm thông tin.\nVăn bản:\n",
        "max_source_length": 3072,
        "max_target_length": 384,
        "checkpoint_parameters_match_model": True,
    }
    candidate_metrics = {
        **common_metrics,
        "paper_test": True,
        "architecture_sha256": architecture_sha256,
        "inference_protocol_sha256": loaded_config["_meta"]["inference_protocol_sha256"],
        "evaluation_contract_sha256": loaded_config["_meta"]["evaluation_contract_sha256"],
        "predictions_sha256": _sha(candidate_predictions),
        "training_parameters": 1_369_121_774,
        "deployable_parameters": 1_368_594_414,
        "parameter_manifest": {
            "architecture_sha256": architecture_sha256,
            "resident_training_total_unique": 1_369_121_774,
            "deployable_resident_without_train_aux": 1_368_594_414,
        },
    }
    baseline_metrics = {
        **common_metrics,
        "predictions_sha256": _sha(baseline_predictions),
        "base_model": "google/t5gemma-2-1b-1b",
        "checkpoint_base_model": "google/t5gemma-2-1b-1b",
        "total_parameters": 1_700_000_000,
    }
    rouge_common = {
        "num_examples": 2,
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "stemming": True,
        "pyrouge_default_args": PROTOCOL,
        "prediction_field": "prediction",
        "reference_field": "reference",
    }
    candidate_rouge = {
        **rouge_common,
        "rouge1": 63.0,
        "rouge2": 33.0,
        "rougeL": 59.0,
        "predictions_file": str(candidate_predictions),
        "predictions_sha256": _sha(candidate_predictions),
    }
    baseline_rouge = {
        **rouge_common,
        "rouge1": 62.013,
        "rouge2": 32.654,
        "rougeL": 58.143,
        "predictions_file": str(baseline_predictions),
        "predictions_sha256": _sha(baseline_predictions),
    }
    paths = {
        "config": config,
        "candidate_predictions": candidate_predictions,
        "baseline_predictions": baseline_predictions,
        "candidate_metrics": tmp_path / "candidate.metrics.json",
        "baseline_metrics": tmp_path / "baseline.metrics.json",
        "candidate_rouge": tmp_path / "candidate.rouge155.json",
        "baseline_rouge": tmp_path / "baseline.rouge155.json",
    }
    _json(paths["candidate_metrics"], candidate_metrics)
    _json(paths["baseline_metrics"], baseline_metrics)
    _json(paths["candidate_rouge"], candidate_rouge)
    _json(paths["baseline_rouge"], baseline_rouge)
    return paths


def _compare(paths: dict[str, Path]) -> dict:
    return compare(
        str(paths["config"]),
        str(paths["candidate_rouge"]),
        str(paths["candidate_metrics"]),
        str(paths["baseline_rouge"]),
        str(paths["baseline_metrics"]),
    )


def test_compare_accepts_bound_full_test_artifacts_and_uses_actual_baseline(tmp_path: Path) -> None:
    paths = _paper_artifacts(tmp_path)
    result = _compare(paths)
    assert result["comparison_valid"] is True
    assert result["candidate_minus_baseline"]["rouge2"] == pytest.approx(0.346)
    assert result["paper_target_pass"] is True
    assert result["candidate_parameters_resident"] < result["baseline_parameters"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "rouge-score"),
        ("stemming", False),
        ("pyrouge_default_args", "different"),
        ("prediction_field", "generated"),
    ),
)
def test_compare_rejects_rouge_protocol_mismatch(tmp_path: Path, field: str, value: object) -> None:
    paths = _paper_artifacts(tmp_path)
    artifact = json.loads(paths["baseline_rouge"].read_text(encoding="utf-8"))
    artifact[field] = value
    _json(paths["baseline_rouge"], artifact)
    with pytest.raises(ValueError):
        _compare(paths)


@pytest.mark.parametrize("failure", ("split", "paper_gate", "count", "fingerprint", "locked_split"))
def test_compare_rejects_split_count_or_fingerprint_mismatch(tmp_path: Path, failure: str) -> None:
    paths = _paper_artifacts(tmp_path)
    if failure == "locked_split":
        for key in ("candidate_metrics", "baseline_metrics"):
            artifact = json.loads(paths[key].read_text(encoding="utf-8"))
            artifact["test_data_fingerprint"] = {"num_examples": 2, "sha256": "e" * 64}
            _json(paths[key], artifact)
        with pytest.raises(ValueError, match="locked official paper test split"):
            _compare(paths)
        return
    target = paths["candidate_metrics"] if failure != "fingerprint" else paths["baseline_metrics"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    if failure == "split":
        artifact["evaluation_split"] = "validation"
    elif failure == "paper_gate":
        artifact["paper_test"] = False
    elif failure == "count":
        artifact["num_examples"] = 3
    else:
        artifact["test_data_fingerprint"] = {"num_examples": 2, "sha256": "e" * 64}
    _json(target, artifact)
    with pytest.raises(ValueError):
        _compare(paths)


@pytest.mark.parametrize("failure", ("hash", "identity", "score_lock", "references"))
def test_compare_rejects_unbound_or_wrong_baseline(tmp_path: Path, failure: str) -> None:
    paths = _paper_artifacts(tmp_path)
    if failure == "hash":
        artifact = json.loads(paths["candidate_metrics"].read_text(encoding="utf-8"))
        artifact["predictions_sha256"] = "0" * 64
        _json(paths["candidate_metrics"], artifact)
    elif failure == "identity":
        artifact = json.loads(paths["baseline_metrics"].read_text(encoding="utf-8"))
        artifact["base_model"] = "some/other-model"
        _json(paths["baseline_metrics"], artifact)
    elif failure == "score_lock":
        artifact = json.loads(paths["baseline_rouge"].read_text(encoding="utf-8"))
        artifact["rouge2"] = 31.0
        _json(paths["baseline_rouge"], artifact)
    else:
        rows = [
            {"id": "test_1", "reference": "Khác.", "prediction": "baseline"},
            {"id": "test_2", "reference": "Hai.", "prediction": "baseline"},
        ]
        paths["baseline_predictions"].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        new_hash = _sha(paths["baseline_predictions"])
        for key in ("baseline_metrics", "baseline_rouge"):
            artifact = json.loads(paths[key].read_text(encoding="utf-8"))
            artifact["predictions_sha256"] = new_hash
            _json(paths[key], artifact)
    with pytest.raises(ValueError):
        _compare(paths)
