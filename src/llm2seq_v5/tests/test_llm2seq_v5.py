from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from llm2seq_v5.adapter import StableTokenLayerFusion, SummaryAdapterV2
from llm2seq_v5.architecture_check import probe_future_token_influence
from llm2seq_v5.checkpoint import load_last_checkpoint, save_last_checkpoint
from llm2seq_v5.config import load_config, validate_config
from llm2seq_v5.contrastive import (
    SourceAlignmentHead,
    hard_negative_indices,
    info_nce_loss,
    last_prompt_states,
    masked_last_pool,
    masked_mean_pool,
    per_example_nll,
    source_memory_for_mining,
    source_swap_contrastive_loss,
)
from llm2seq_v5.decoder import (
    QwenCopiedCrossAttention,
)
from llm2seq_v5.encoder import EmbeddingTokenEncoder, _config_is_bidirectional
from llm2seq_v5.final_audit import audit_final_claim
from llm2seq_v5.metrics import rouge_scores
from llm2seq_v5.smoke_gate import evaluate_smoke_run
from llm2seq_v5.training import (
    _contrastive_scale,
    _parameter_component,
    _tokenizers,
    validation_loss,
    verify_declared_parameter_budget,
    verify_locked_data_manifest,
)

from llm2seq_v5.data import SummarizationDataset, decoder_seed_ids, encode_source, greedy_evidence_labels


def test_main_v5_profile_is_phrase_continuation_single_memory():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"
    assert config["adapter"]["num_bidirectional_layers"] == 6
    assert config["adapter"]["use_summary_planner"] is True
    assert config["adapter"]["num_summary_slots"] == 16
    assert config["adapter"]["summary_planner_layers"] == 2
    assert config["adapter"]["layer_fusion_gate_init"] == 0.0
    assert config["decoder"]["cross_gate_init"] == 0.05
    assert config["objectives"]["use_contrastive"] is True
    assert config["objectives"]["use_prompt_alignment"] is False
    assert config["objectives"]["contrastive_weight"] == 0.0
    assert config["objectives"]["contrastive_temperature"] == 0.07
    assert config["objectives"]["contrastive_pooling"] == "mean_last"
    assert config["objectives"]["source_swap_weight"] == 0.05
    assert config["objectives"]["source_swap_margin"] == 0.20
    assert config["objectives"]["source_swap_strategy"] == "hard_in_batch"
    assert config["benchmark"]["diagnostic"]["rouge2"] == 19.5308
    assert config["benchmark"]["paper"]["rouge2"] == 32.654
    assert config["objectives"]["response_alignment_weight"] == 0.0
    assert config["phrase_pointer"]["enabled"] is True
    assert config["phrase_pointer"]["use_continuation"] is True
    assert config["objectives"]["phrase_mixture_weight"] == 1.0
    assert config["objectives"]["label_smoothing"] == 0.0
    # Preserved from v2
    assert config["decoder"]["cross_attention_every"] == 1
    assert config["decoder"]["initialize_cross_from_self"] is True
    assert config["checkpoint"] == {
        "save_best": False,
        "save_each_epoch": False,
        "save_last": True,
    }


def _final_audit_artifacts(config):
    locked = config["benchmark"]["data"]["test"]
    target = config["benchmark"]["paper"]
    protocol = config["benchmark"]["paper_protocol"]
    rouge_protocol = "-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m"
    candidate_predictions_sha256 = "a" * 64
    baseline_predictions_sha256 = "b" * 64
    candidate_scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "stemming": True,
        "pyrouge_default_args": rouge_protocol,
        "prediction_field": "prediction",
        "reference_field": "reference",
        "predictions_sha256": candidate_predictions_sha256,
        "num_examples": 3901,
        "rouge1": 63.0,
        "rouge2": 33.0,
        "rougeL": 59.0,
    }
    candidate_metrics = {
        "evaluation_split": "test",
        "num_examples": 3901,
        "predictions_sha256": candidate_predictions_sha256,
        "deployable_parameters": 1_527_949_729,
        "training_parameters": 1_528_477_089,
        "checkpoint_test_matches_current": True,
        "checkpoint_parameters_match_model": True,
        "test_data_fingerprint": dict(locked),
        "generation": dict(protocol["generation"]),
        "max_source_length": protocol["max_source_length"],
        "max_target_length": protocol["max_target_length"],
        "source_prefix": protocol["source_prefix"],
    }
    baseline_scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "stemming": True,
        "pyrouge_default_args": rouge_protocol,
        "prediction_field": "prediction",
        "reference_field": "reference",
        "predictions_sha256": baseline_predictions_sha256,
        "num_examples": 3901,
        "rouge1": target["rouge1"],
        "rouge2": target["rouge2"],
        "rougeL": target["rougeL"],
    }
    baseline_metrics = {
        "evaluation_split": "test",
        "num_examples": 3901,
        "predictions_sha256": baseline_predictions_sha256,
        "base_model": "/models/t5gemma-2-1b-1b",
        "checkpoint_base_model": "google/t5gemma-2-1b-1b",
        "unique_parameter_elements": 2_012_345_678,
        "checkpoint_test_matches_current": True,
        "checkpoint_parameters_match_model": True,
        "test_data_fingerprint": dict(locked),
        "generation": dict(protocol["generation"]),
        "max_source_length": protocol["max_source_length"],
        "max_target_length": protocol["max_target_length"],
        "source_prefix": protocol["source_prefix"],
    }
    return candidate_scores, candidate_metrics, baseline_scores, baseline_metrics


def test_final_audit_uses_actual_artifacts_and_requires_strict_superiority():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    artifacts = _final_audit_artifacts(config)
    result = audit_final_claim(config, *artifacts)
    assert result["passed"] is True
    assert result["comparable"] is True
    assert result["smaller_than_baseline"] is True
    assert result["all_rouge_strictly_surpassed"] is True
    assert result["rounded_2b_count_used_for_final_claim"] is False
    assert result["candidate_total_minus_baseline_parameters"] == -483_868_589
    assert result["candidate_deployable_minus_baseline_parameters"] == -484_395_949
    assert result["total_smaller_than_baseline"] is True
    assert result["predictions_sha256_bound"] is True
    assert result["evaluation_split_verified_as_test"] is True
    assert result["perl_rouge_protocol"]["stemming"] is True

    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = artifacts
    candidate_scores = dict(candidate_scores)
    candidate_scores["rouge2"] = baseline_scores["rouge2"]
    tied = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )
    assert tied["comparable"] is True
    assert tied["passed"] is False
    assert tied["strictly_surpassed"]["rouge2"] is False


def test_final_audit_rejects_wrong_baseline_or_split():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = _final_audit_artifacts(config)
    baseline_scores = dict(baseline_scores)
    baseline_scores["rouge2"] -= 1.0
    baseline_metrics = dict(baseline_metrics)
    baseline_metrics["test_data_fingerprint"] = {
        "num_examples": 3901,
        "sha256": "0" * 64,
    }
    result = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )
    assert result["passed"] is False
    assert result["comparable"] is False
    assert any("fingerprint" in reason for reason in result["comparability_reasons"])
    assert any("does not reproduce" in reason for reason in result["comparability_reasons"])


def test_final_audit_rejects_decoding_or_preprocessing_mismatch():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = _final_audit_artifacts(config)
    candidate_metrics = copy.deepcopy(candidate_metrics)
    baseline_metrics = copy.deepcopy(baseline_metrics)
    candidate_metrics["generation"]["repetition_penalty"] = 1.15
    baseline_metrics["max_source_length"] = 2048
    result = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )
    assert result["passed"] is False
    assert result["protocol_parity_checked"] is True
    assert any("repetition_penalty" in reason for reason in result["comparability_reasons"])
    assert any("max_source_length" in reason for reason in result["comparability_reasons"])


def test_final_audit_rejects_unbound_predictions_or_non_test_evaluation():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = _final_audit_artifacts(config)
    candidate_metrics = copy.deepcopy(candidate_metrics)
    baseline_metrics = copy.deepcopy(baseline_metrics)
    candidate_metrics["predictions_sha256"] = "c" * 64
    baseline_metrics["evaluation_split"] = "validation"

    result = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )

    assert result["passed"] is False
    assert result["comparable"] is False
    assert result["predictions_sha256_bound"] is False
    assert result["evaluation_split_verified_as_test"] is False
    assert any("not computed from" in reason for reason in result["comparability_reasons"])
    assert any("evaluation_split" in reason for reason in result["comparability_reasons"])


def test_final_audit_rejects_different_or_unstemmed_perl_protocol():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = _final_audit_artifacts(config)
    candidate_scores = copy.deepcopy(candidate_scores)
    baseline_scores = copy.deepcopy(baseline_scores)
    candidate_scores["stemming"] = False
    baseline_scores["pyrouge_default_args"] = "-a"
    baseline_scores["backend"] = "Perl ROUGE-1.5.5 via pyrouge==0.1.4"

    result = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )

    assert result["passed"] is False
    assert result["comparable"] is False
    assert any("stemming=true" in reason for reason in result["comparability_reasons"])
    assert any("arguments" in reason for reason in result["comparability_reasons"])
    assert any("same pyrouge" in reason for reason in result["comparability_reasons"])


def test_full_training_requires_exact_locked_data_but_pilot_allows_subsets():
    root = Path(__file__).parents[1]
    full = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    locked = full["benchmark"]["data"]
    manifest = {split: dict(locked[split]) for split in ("train", "validation", "test")}
    assert verify_locked_data_manifest(full, manifest) == {
        "train": "exact_match",
        "validation": "exact_match",
        "test": "exact_match",
    }
    broken = copy.deepcopy(manifest)
    broken["test"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="test data differs"):
        verify_locked_data_manifest(full, broken)

    pilot = load_config(root / "configs/pilot_phrase_continuation_2000.yaml")
    assert verify_locked_data_manifest(pilot, manifest) == {
        "train": "partial_not_comparable",
        "validation": "partial_not_comparable",
        "test": "exact_match",
    }


def test_preflight_requires_conservative_total_below_declared_parameter_target():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    summary = {
        "total": 1_528_477_089,
        "alignment_head": 527_360,
    }
    report = verify_declared_parameter_budget(config, summary)
    assert report["candidate_total_parameters"] == 1_528_477_089
    assert report["candidate_deployable_parameters"] == 1_527_949_729
    assert report["total_below_declared_target"] is True

    oversized = {"total": 2_000_000_000, "alignment_head": 1}
    with pytest.raises(RuntimeError, match="not below"):
        verify_declared_parameter_budget(config, oversized)


def test_config_rejects_encoder_layer_indices_outside_checkpoint_depth():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    config["adapter"]["fuse_layers"] = [-100]
    with pytest.raises(ValueError, match="outside the encoder"):
        validate_config(config)


def test_embedding_attention_mode_detection_uses_explicit_checkpoint_flags():
    assert _config_is_bidirectional(SimpleNamespace(use_bidirectional_attention=True)) is True
    assert _config_is_bidirectional(SimpleNamespace(is_causal=False)) is True
    assert _config_is_bidirectional(SimpleNamespace(is_causal=True)) is False


def test_future_context_probe_distinguishes_causal_and_bidirectional_behavior():
    class ToyCausalEncoder(nn.Module):
        def forward(self, input_ids, attention_mask):
            states = input_ids.float().unsqueeze(-1) * attention_mask.unsqueeze(-1)
            return (states.cumsum(dim=1),)

    class ToyBidirectionalEncoder(nn.Module):
        def forward(self, input_ids, attention_mask):
            states = input_ids.float().unsqueeze(-1) * attention_mask.unsqueeze(-1)
            global_state = states.sum(dim=1, keepdim=True).expand_as(states)
            return (global_state,)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    causal = probe_future_token_influence(
        ToyCausalEncoder(),
        input_ids,
        attention_mask,
        vocab_size=32,
        pad_token_id=0,
    )
    bidirectional = probe_future_token_influence(
        ToyBidirectionalEncoder(),
        input_ids,
        attention_mask,
        vocab_size=32,
        pad_token_id=0,
    )
    assert causal["relative_l2_change"] == 0.0
    assert bidirectional["relative_l2_change"] > 1e-6


def test_config_rejects_invalid_encoder_revision_controls():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    config["model"]["encoder_revision"] = ""
    with pytest.raises(ValueError, match="encoder_revision"):
        validate_config(config)
    config["model"]["encoder_revision"] = "fixed-revision"
    config["model"]["encoder_trust_remote_code"] = "true"
    with pytest.raises(ValueError, match="encoder_trust_remote_code"):
        validate_config(config)

    config = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    config["data"]["source_add_special_tokens"] = True
    with pytest.raises(ValueError, match="cannot both be enabled"):
        validate_config(config)


def test_source_tokenizer_default_special_wrapper_preserves_unit_alignment():
    class WrappedTokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            ids = [10 + (ord(char) % 17) for char in str(text) if not char.isspace()]
            if add_special_tokens:
                ids = [1, *ids, 2]
            return {"input_ids": ids}

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(value) for value in ids)

    ids, unit_ids, units = encode_source(
        WrappedTokenizer(),
        "Câu thứ nhất. Câu thứ hai.",
        {
            "source_prefix": "passage: ",
            "source_add_special_tokens": True,
            "append_source_eos": False,
            "sentence_separator": "\n",
            "max_source_length": 64,
        },
    )
    assert ids[0] == 1
    assert ids[-1] == 2
    assert unit_ids[0] == 0
    assert unit_ids[-1] == 0
    assert len(ids) == len(unit_ids)
    assert len(units) == 2
    assert any(value == 1 for value in unit_ids)
    assert any(value == 2 for value in unit_ids)


def test_encoder_revision_is_forwarded_to_config_weights_and_tokenizer(monkeypatch):
    import transformers

    calls = []
    fake_config = SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=2,
        vocab_size=32,
        use_bidirectional_attention=True,
        use_cache=True,
    )

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = fake_config

        def forward(self, input_ids, attention_mask, **kwargs):
            del kwargs
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
            hidden = hidden * attention_mask.unsqueeze(-1)
            return SimpleNamespace(hidden_states=(hidden, hidden, hidden))

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 0
        unk_token_id = None
        padding_side = "left"

        def get_vocab(self):
            return {"<pad>": 0, "<eos>": 1, "token": 2}

    def fake_config_load(name, **kwargs):
        calls.append(("config", name, kwargs))
        return fake_config

    def fake_model_load(name, **kwargs):
        calls.append(("model", name, kwargs))
        return FakeBackbone()

    def fake_tokenizer_load(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return FakeTokenizer()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fake_config_load))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(fake_model_load))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer_load))

    encoder = EmbeddingTokenEncoder(
        "encoder-id",
        torch.float32,
        False,
        expected_hidden_layers=2,
        expected_attention_mode="bidirectional",
        trust_remote_code=True,
        revision="fixed-revision",
    )
    assert encoder.revision == "fixed-revision"
    config = {
        "model": {
            "encoder_name": "encoder-id",
            "encoder_trust_remote_code": True,
            "encoder_revision": "fixed-revision",
            "decoder_name": "decoder-id",
        }
    }
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    assert encoder_tokenizer.padding_side == "right"
    assert decoder_tokenizer.padding_side == "right"

    config_call = next(call for call in calls if call[0] == "config")
    model_call = next(call for call in calls if call[0] == "model")
    encoder_tokenizer_call = next(call for call in calls if call[0] == "tokenizer" and call[1] == "encoder-id")
    for call in (config_call, model_call, encoder_tokenizer_call):
        assert call[2]["revision"] == "fixed-revision"
        assert call[2]["trust_remote_code"] is True


def _write_smoke_artifacts(
    run_dir: Path,
    *,
    cross_ratio: float = 0.1,
    prefix_rate: float = 5.0,
    source_swap_accuracy: float = 0.88,
    source_swap_nll_gap: float = 0.1,
    training_parameters: int = 1_528_477_089,
    memory_bank_count: int = 3,
    prefix_to_embedding_rms_ratio: float = 1.0,
    routing_entropy: float | None = None,
    prefix_drift_ratio: float = 0.70,
    prefix_swap_nll_gap: float = 0.10,
    prefix_swap_accuracy: float = 0.80,
    num_examples: int = 20,
) -> None:
    run_dir.mkdir()
    (run_dir / "last.pt").touch()
    (run_dir / "COMPLETE").touch()
    (run_dir / "last_validation_predictions.metrics.json").write_text(
        json.dumps(
            {
                "num_examples": num_examples,
                "evaluation_split": "validation",
                "rouge1": 10.0,
                "rouge2": 2.0,
                "rougeL": 9.0,
                "prediction_words_mean": 30.0,
                "empty_prediction_rate": 0.0,
                "unique_prediction_rate": 100.0,
                "dominant_prefix_5gram_rate": prefix_rate,
                "repeated_trigram_rate_mean": 0.0,
                "checkpoint_parameters_match_model": True,
                "parameter_budget_reached": True,
                "source_swap": True,
                "query_adaptive_routing": memory_bank_count > 1,
                "memory_bank_count": memory_bank_count,
                "deployable_parameters": 1_527_949_729,
                "training_parameters": training_parameters,
                "parameter_target_declared": 2_000_000_000,
                "phrase_pointer_enabled": True,
                "phrase_generation_modes": {
                    "generate": 0.90,
                    "new_span": 0.07,
                    "continue_span": 0.03,
                },
                "phrase_generation_mode_observations": num_examples * 10,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "validation_history.jsonl").write_text(
        json.dumps(
            {
                "eval_loss": 3.0,
                "eval_loss_ce": 2.5,
                "eval_loss_response_alignment": 0.7,
                "eval_response_alignment_cosine": 0.4,
                "eval_response_alignment_accuracy": 0.3,
                "eval_loss_phrase_mixture": 2.4,
                "eval_loss_phrase_copy": 1.1,
                "eval_loss_phrase_continue": 1.2,
                "eval_loss_phrase_labels": 0.5,
                "eval_loss_phrase_coverage": 0.1,
                "eval_phrase_mode_generate": 0.9,
                "eval_phrase_mode_new": 0.07,
                "eval_phrase_mode_continue": 0.03,
                "eval_summary_prefix_rms": 0.8,
                "eval_prefix_to_embedding_rms_ratio": prefix_to_embedding_rms_ratio,
                "eval_prefix_drift_ratio": prefix_drift_ratio,
                "eval_prefix_swap_nll_gap": prefix_swap_nll_gap,
                "eval_prefix_swap_accuracy": prefix_swap_accuracy,
                "eval_loss_contrastive": 1.0,
                "eval_loss_source_swap": 0.8,
                "eval_cross_residual_ratio": cross_ratio,
                "eval_prompt_retrieval_accuracy": 0.25,
                "eval_source_swap_accuracy": source_swap_accuracy,
                "eval_source_swap_nll_gap": source_swap_nll_gap,
                "eval_memory_routing_entropy": (
                    routing_entropy if routing_entropy is not None else (0.9 if memory_bank_count > 1 else 0.0)
                ),
                "eval_adaptive_routing_delta": 0.01 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_lexical": 0.30 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_semantic": 0.35 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_summary": 0.35 if memory_bank_count > 1 else 1.0,
                "eval_examples": float(num_examples),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_smoke_gate_accepts_flow_but_makes_no_generalization_claim(tmp_path: Path):
    run_dir = tmp_path / "smoke"
    _write_smoke_artifacts(run_dir)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert "not a generalization" in result["scope"]


def test_smoke_gate_accepts_single_bank_without_router_collapse_gate(tmp_path: Path):
    run_dir = tmp_path / "single_bank"
    _write_smoke_artifacts(run_dir, memory_bank_count=1)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert result["memory_bank_count"] == 1
    assert "router_not_collapsed" not in result["gates"]
    assert result["mean_routes"] == {"summary": 1.0}


def test_smoke_gate_rejects_disconnected_cross_attention_and_prefix_collapse(tmp_path: Path):
    run_dir = tmp_path / "collapsed"
    _write_smoke_artifacts(run_dir, cross_ratio=0.0, prefix_rate=80.0)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is False
    assert "cross_attention_connected" in result["failed_gates"]
    assert "no_dominant_fixed_prefix" in result["failed_gates"]


def test_smoke_gate_rejects_source_ignorance_and_oversized_total(tmp_path: Path):
    run_dir = tmp_path / "source_ignored"
    _write_smoke_artifacts(
        run_dir,
        source_swap_accuracy=0.40,
        source_swap_nll_gap=-0.1,
        training_parameters=2_100_000_000,
    )
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is False
    assert any("correct-source preference is weak" in message for message in result["warnings"])
    assert "total_below_declared_t5gemma_budget" in result["failed_gates"]


def test_smoke_gate_warns_on_near_chance_source_swap_accuracy(tmp_path: Path):
    run_dir = tmp_path / "chance_swap"
    _write_smoke_artifacts(run_dir, source_swap_accuracy=0.56)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert any("correct-source preference is weak" in message for message in result["warnings"])


def test_smoke_gate_does_not_infer_source_use_from_cross_residual_magnitude(tmp_path: Path):
    run_dir = tmp_path / "token_cross"
    _write_smoke_artifacts(run_dir, cross_ratio=1e-3)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert "cross_attention_connected" not in result["failed_gates"]


def test_smoke_gate_warns_when_summary_prefix_dominates_embedding_scale(tmp_path: Path):
    run_dir = tmp_path / "prefix_scale"
    _write_smoke_artifacts(run_dir, prefix_to_embedding_rms_ratio=30.0)
    result = evaluate_smoke_run(run_dir)
    # Non-blocking: the healthy band is unmeasured, so this reports, not fails.
    assert result["passed"] is True
    assert any("summary prefix RMS" in message for message in result["warnings"])


def test_smoke_gate_is_quiet_when_prefix_scale_is_healthy(tmp_path: Path):
    run_dir = tmp_path / "prefix_ok"
    _write_smoke_artifacts(run_dir, prefix_to_embedding_rms_ratio=1.0)
    result = evaluate_smoke_run(run_dir)
    assert result["warnings"] == []


def test_smoke_gate_labels_low_prefix_self_drift_as_descriptive_only(tmp_path: Path):
    run_dir = tmp_path / "prefix_inert"
    _write_smoke_artifacts(run_dir, prefix_to_embedding_rms_ratio=1.0, prefix_drift_ratio=0.002)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert any("descriptive only" in message for message in result["warnings"])
    assert not any("prefix may be inert" in message for message in result["warnings"])


def test_smoke_gate_reports_drift_independently_of_rms_ratio(tmp_path: Path):
    # Cosine distance is scale-free, so a badly-scaled prefix that the backbone
    # nonetheless rewrites is a physically reachable state -- unlike a relative-L2
    # drift, which would be pinned to ~1/RMS and could not distinguish the two.
    run_dir = tmp_path / "prefix_scaled_but_moving"
    _write_smoke_artifacts(run_dir, prefix_to_embedding_rms_ratio=50.0, prefix_drift_ratio=0.70)
    result = evaluate_smoke_run(run_dir)
    assert not any("descriptive only" in message for message in result["warnings"])
    assert any("summary prefix RMS" in message for message in result["warnings"])


def test_smoke_gate_does_not_report_unmeasured_prefix_as_inert(tmp_path: Path):
    # A config with no summary planner never forwards a prefix. Reporting 0.0
    # would read as "the prefix is inert" for a run that has no prefix at all.
    run_dir = tmp_path / "no_planner"
    _write_smoke_artifacts(run_dir, prefix_drift_ratio=float("nan"))
    result = evaluate_smoke_run(run_dir)
    assert not any("descriptive only" in message for message in result["warnings"])
    assert "finite_source_diagnostics" not in result["failed_gates"]


def test_smoke_gate_does_not_hardcode_source_swap_learning_threshold(tmp_path: Path):
    run_dir = tmp_path / "smoke_swap"
    _write_smoke_artifacts(run_dir, source_swap_accuracy=0.50, source_swap_nll_gap=0.0)
    result = evaluate_smoke_run(run_dir, expected_examples=20)
    assert result["passed"] is True
    assert any("20 validation examples" in message for message in result["warnings"])


def test_smoke_gate_uses_validation_count_in_source_warning(tmp_path: Path):
    run_dir = tmp_path / "pilot_swap"
    _write_smoke_artifacts(
        run_dir,
        source_swap_accuracy=0.50,
        source_swap_nll_gap=0.0,
        num_examples=2000,
    )
    result = evaluate_smoke_run(run_dir, expected_examples=2000)
    assert result["passed"] is True
    assert any("2000 validation examples" in message for message in result["warnings"])


def test_smoke_gate_warns_when_prefix_swap_does_not_hurt_nll(tmp_path: Path):
    run_dir = tmp_path / "prefix_unused"
    _write_smoke_artifacts(run_dir, prefix_swap_nll_gap=-0.01, prefix_swap_accuracy=0.40)
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is True
    assert any("dense-memory path may make the prefix redundant" in message for message in result["warnings"])


def test_smoke_gate_labels_python_rouge_as_off_scale_not_wrong(tmp_path: Path):
    # A pilot reading rouge2 ~= 8-14 has not collapsed. The warning must say the
    # Python backend is off-SCALE from the Perl targets, not interchangeable.
    run_dir = tmp_path / "backend_label"
    _write_smoke_artifacts(run_dir)
    warning = evaluate_smoke_run(run_dir)["rouge_backend_warning"]
    assert "NOT on the same scale" in warning
    assert "Unicode-preserving whitespace diagnostic" in warning
    assert "do not mix the two scales" in warning


def test_smoke_gate_requires_prefix_rms_ratio_to_be_recorded(tmp_path: Path):
    run_dir = tmp_path / "missing_ratio"
    _write_smoke_artifacts(run_dir)
    history = run_dir / "validation_history.jsonl"
    payload = json.loads(history.read_text(encoding="utf-8"))
    payload.pop("eval_prefix_to_embedding_rms_ratio")
    history.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_smoke_run(run_dir)
    assert result["passed"] is False
    assert "finite_source_diagnostics" in result["failed_gates"]


def test_training_surfaces_previously_discarded_diagnostics():
    # These are computed in LLM2SeqV5.forward but were dropped by the metric
    # allow-list, so they never reached validation_history.jsonl.
    from llm2seq_v5 import training as training_module

    source = Path(training_module.__file__).read_text(encoding="utf-8")
    block = source.split("metric_names = (", 1)[1].split("\n    )", 1)[0]
    for name in (
        "salience_precision",
        "salience_recall",
        "projection_gate",
        "salience_attention_gate",
        "prefix_to_embedding_rms_ratio",
        "prefix_swap_nll_gap",
        "prefix_swap_accuracy",
    ):
        assert f'"{name}"' in block, f"{name} is computed but not surfaced"


def test_run_script_defaults_to_prospective_summary_bridge():
    root = Path(__file__).parents[1]
    script = (root / "run.sh").read_text(encoding="utf-8")
    assert 'MAIN_CONFIG="${CONFIG:-configs/qwen3_embedding_0_6b_phrase_continuation.yaml}"' in script
    assert 'SMOKE_CONFIG="configs/smoke_phrase_continuation_100.yaml"' in script
    assert 'PILOT_CONFIG="configs/pilot_phrase_continuation_2000.yaml"' in script


def test_architecture_check_verifies_copy_before_loading_trained_weights():
    # Exact Q/K/V/O equality is an installation invariant. A trained checkpoint
    # uses different LRs for native self-attention and copied cross-attention, so
    # checking equality after load would falsely reject every healthy run.
    from llm2seq_v5 import architecture_check as architecture_module

    source = Path(architecture_module.__file__).read_text(encoding="utf-8")
    copy_check = source.index('if bool(config["decoder"].get("initialize_cross_from_self", True)):')
    checkpoint_load = source.index("if checkpoint_path is not None:")
    assert copy_check < checkpoint_load


def test_unit_test_launcher_is_offline_and_has_no_publish_commands():
    root = Path(__file__).parents[1]
    script = (root / "run.sh").read_text(encoding="utf-8")
    assert 'if [[ "$MODE" == "test" ]]' in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export TRANSFORMERS_OFFLINE=1" in script
    forbidden = ("push_to_hub", "huggingface-cli upload", "hf upload", "upload_file(")
    assert all(command not in script for command in forbidden)


def test_phrase_continuation_experiment_surface_has_required_profiles():
    root = Path(__file__).parents[1]
    expected = {
        "base.yaml",
        "qwen3_embedding_0_6b_phrase_continuation.yaml",
        "smoke_phrase_continuation_100.yaml",
        "pilot_phrase_continuation_2000.yaml",
        "ablations/no_continuation.yaml",
        "ablations/no_coverage.yaml",
        "ablations/no_phrase_prior.yaml",
        "ablations/v4_psb_control.yaml",
        "pilot_ablations/no_continuation_2000.yaml",
        "pilot_ablations/no_coverage_2000.yaml",
        "pilot_ablations/no_phrase_prior_2000.yaml",
        "pilot_ablations/v4_psb_control_2000.yaml",
    }
    actual = {str(path.relative_to(root / "configs")) for path in (root / "configs").rglob("*.yaml")}
    assert expected <= actual
    assert not any(any(token in name.lower() for token in ("hiroute", "pplx", "nemotron")) for name in actual)


def test_phrase_continuation_ablations_change_the_claimed_components():
    root = Path(__file__).parents[1]
    main = load_config(root / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    v4 = load_config(root / "configs/ablations/v4_psb_control.yaml")
    no_prior = load_config(root / "configs/ablations/no_phrase_prior.yaml")
    no_continuation = load_config(root / "configs/ablations/no_continuation.yaml")
    no_coverage = load_config(root / "configs/ablations/no_coverage.yaml")
    v2_core = load_config(root / "configs/pilot_ablations/v2_core_single_bank_2000.yaml")

    assert main["phrase_pointer"]["enabled"] is True
    assert main["adapter"]["num_bidirectional_layers"] == 6
    assert v2_core["adapter"]["num_bidirectional_layers"] == 4
    assert v4["phrase_pointer"]["enabled"] is False
    assert v4["objectives"]["response_alignment_weight"] == 0.15
    assert v4["objectives"]["phrase_mixture_weight"] == 0.0
    assert no_prior["phrase_pointer"]["phrase_bias_scale"] == 0.0
    assert no_prior["objectives"]["phrase_label_weight"] == 0.0
    assert no_continuation["phrase_pointer"]["use_continuation"] is False
    assert no_continuation["objectives"]["phrase_continue_weight"] == 0.0
    assert no_coverage["objectives"]["phrase_coverage_weight"] == 0.0

    pilot_paths = [
        root / "configs/pilot_ablations/v4_psb_control_2000.yaml",
        root / "configs/pilot_ablations/no_phrase_prior_2000.yaml",
        root / "configs/pilot_ablations/no_continuation_2000.yaml",
        root / "configs/pilot_ablations/no_coverage_2000.yaml",
    ]
    for path in pilot_paths:
        pilot = load_config(path)
        assert pilot["limits"] == {
            "max_train_examples": 2000,
            "max_validation_examples": 512,
            "max_test_examples": 0,
        }
        assert pilot["data"]["subset_strategy"] == "hashed"
        assert pilot["data"]["subset_seed"] == 42
        assert pilot["training"]["interface_warmup_epochs"] == 2
        assert pilot["training"]["full_finetune_epochs"] == 4

    assert _parameter_component("phrase_pointer.query_projection.weight") == "adapter"


def test_config_rejects_best_checkpoint(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {encoder_name: e, decoder_name: d, hidden_size: 8}
adapter: {layer_fusion: true, fuse_layers: [-1], num_heads: 1}
decoder: {cross_attention_every: 1, cross_gate_init: 0.1}
training: {interface_warmup_epochs: 0, full_finetune_epochs: 1, batch_size: 1, gradient_accumulation_steps: 1}
data: {train_file: a, validation_file: b, test_file: c, max_source_length: 8, max_target_length: 4}
checkpoint: {save_best: true}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="last.pt only"):
        load_config(path)


def test_config_validates_contrastive_temperature(tmp_path: Path):
    path = tmp_path / "bad_temp.yaml"
    path.write_text(
        """
model: {encoder_name: e, decoder_name: d, hidden_size: 8}
adapter: {layer_fusion: true, fuse_layers: [-1], num_heads: 1}
decoder: {cross_attention_every: 1, cross_gate_init: 0.1}
training: {interface_warmup_epochs: 0, full_finetune_epochs: 1, batch_size: 1, gradient_accumulation_steps: 1}
data: {train_file: a, validation_file: b, test_file: c, max_source_length: 8, max_target_length: 4}
objectives: {use_contrastive: true, contrastive_temperature: -0.1}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="temperature"):
        load_config(path)


# --- Contrastive module tests ---


def test_masked_mean_pool():
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    mask = torch.tensor([[1, 1, 0]])  # Only first two tokens valid
    pooled = masked_mean_pool(states, mask)
    expected = torch.tensor([[2.0, 3.0]])  # mean of [1,2] and [3,4]
    assert torch.allclose(pooled, expected)


def test_masked_mean_pool_all_valid():
    states = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    pooled = masked_mean_pool(states, mask)
    expected = states.mean(dim=1)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_masked_last_pool_matches_final_valid_source_token():
    states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled = masked_last_pool(states, mask)
    assert torch.equal(pooled[0], states[0, 1])
    assert torch.equal(pooled[1], states[1, 2])


def test_source_alignment_head_shapes():
    head = SourceAlignmentHead(hidden_size=16, projection_size=8)
    memory = torch.randn(4, 10, 16)  # [B=4, S=10, D=16]
    memory_mask = torch.ones(4, 10, dtype=torch.long)
    prompt_states = torch.randn(4, 16)
    result = head(memory, memory_mask, prompt_states)
    assert result["memory_repr"].shape == (4, 8)
    assert result["decoder_repr"].shape == (4, 8)
    # Check L2 normalization
    norms = result["memory_repr"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_info_nce_loss_perfect_alignment():
    """When decoder and memory representations match perfectly, loss should be low."""
    torch.manual_seed(42)
    # Create aligned representations
    repr_base = torch.randn(8, 16)
    repr_base = torch.nn.functional.normalize(repr_base, dim=-1)
    loss = info_nce_loss(repr_base, repr_base, temperature=0.07)
    assert loss.item() < 0.5  # Should be very low for perfect alignment


def test_info_nce_loss_random_is_higher():
    """Random decoder representations should have higher loss than aligned ones."""
    torch.manual_seed(42)
    memory = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    # Aligned
    aligned_loss = info_nce_loss(memory, memory, temperature=0.07)
    # Random
    decoder_random = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    random_loss = info_nce_loss(memory, decoder_random, temperature=0.07)
    assert random_loss.item() > aligned_loss.item()


def test_info_nce_loss_single_example():
    """With batch_size=1, contrastive loss should return zero."""
    memory = torch.randn(1, 16)
    decoder = torch.randn(1, 16)
    loss = info_nce_loss(memory, decoder, temperature=0.07)
    assert loss.item() == 0.0


def test_info_nce_loss_gradients_flow():
    """Verify gradients flow through the contrastive loss."""
    memory_leaf = torch.randn(4, 8, requires_grad=True)
    decoder_leaf = torch.randn(4, 8, requires_grad=True)
    memory = torch.nn.functional.normalize(memory_leaf, dim=-1)
    decoder = torch.nn.functional.normalize(decoder_leaf, dim=-1)
    loss = info_nce_loss(memory, decoder, temperature=0.1)
    loss.backward()
    assert memory_leaf.grad is not None
    assert decoder_leaf.grad is not None
    assert not torch.all(memory_leaf.grad == 0)


def test_hard_source_negatives_are_non_self_and_choose_nearest_document():
    memory = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.9, 0.1], [0.9, 0.1]],
            [[-1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    mask = torch.ones(3, 2, dtype=torch.long)
    source_repr = source_memory_for_mining(memory, mask)
    indices, similarities = hard_negative_indices(source_repr)
    assert indices[0].item() == 1
    assert indices[1].item() == 0
    assert torch.all(indices != torch.arange(3))
    assert torch.isfinite(similarities).all()


def test_source_mining_respects_embedding_encoder_pooling_recipe():
    memory = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 6.0]]])
    mask = torch.ones(1, 3, dtype=torch.long)
    mean = source_memory_for_mining(memory, mask, pooling="mean")
    mean_last = source_memory_for_mining(memory, mask, pooling="mean_last")
    assert not torch.allclose(mean, mean_last)
    with pytest.raises(ValueError, match="pooling"):
        source_memory_for_mining(memory, mask, pooling="unsupported")


def test_last_prompt_states_exclude_teacher_forced_target_content():
    states = torch.zeros(2, 5, 3)
    states[:, 1] = torch.tensor([1.0, 2.0, 3.0])
    states[0, 2:] = 100.0
    states[1, 2:] = -100.0
    labels = torch.tensor(
        [
            [-100, 10, 11, 12, 13],
            [-100, 20, 21, 22, 23],
        ]
    )
    prompt = last_prompt_states(states, labels)
    assert torch.equal(prompt[0], prompt[1])
    assert torch.equal(prompt[0], torch.tensor([1.0, 2.0, 3.0]))


def test_teacher_forcing_seed_matches_the_autoregressive_generation_boundary(tmp_path: Path):
    class ToyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = None

        def __call__(
            self,
            text,
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        ):
            del add_special_tokens
            ids = [3 + (ord(char) % 23) for char in str(text) if not char.isspace()]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids}

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(value) for value in ids)

    path = tmp_path / "one.jsonl"
    path.write_text(
        json.dumps({"id": "x", "source": "Nguồn thứ nhất.", "target": "Tóm tắt."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer = ToyTokenizer()
    config = {
        "source_prefix": "SRC: ",
        "sentence_separator": "\n",
        "decoder_instruction": "summarize",
        "decoder_prefix": "OUT: ",
        "use_decoder_chat_template": False,
        "max_source_length": 64,
        "max_target_length": 32,
        "oracle_max_units": 4,
        "clean_wikihow_metadata": True,
    }
    dataset = SummarizationDataset(path, tokenizer, tokenizer, config)
    item = dataset[0]
    seed = decoder_seed_ids(tokenizer, config)
    target_ids = tokenizer("Tóm tắt.", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    boundary = len(seed) - 1

    assert item["decoder_input_ids"].tolist() == seed + target_ids[:-1]
    assert item["labels"].tolist() == [-100] * boundary + target_ids
    # Training predicts the first summary token at the final fixed prompt
    # position, exactly where greedy generation takes its first-token logits.
    assert item["decoder_input_ids"][boundary].item() == seed[-1]
    assert item["labels"][boundary].item() == target_ids[0]


def test_per_example_nll_and_source_swap_preference():
    labels = torch.tensor([[0, 1], [1, -100]])
    supervised = labels.ne(-100)
    positive_logits = torch.tensor(
        [
            [8.0, -8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
        ],
        requires_grad=True,
    )
    negative_logits = (-positive_logits.detach().clone()).requires_grad_(True)
    positive_nll = per_example_nll(positive_logits, labels, supervised)
    negative_nll = per_example_nll(negative_logits, labels, supervised)
    good_loss = source_swap_contrastive_loss(positive_nll, negative_nll, margin=0.2)
    bad_loss = source_swap_contrastive_loss(negative_nll, positive_nll, margin=0.2)
    assert good_loss < bad_loss
    good_loss.backward()
    assert positive_logits.grad is not None
    assert negative_logits.grad is not None


def test_contrastive_warmup_does_not_reset_during_full_finetune():
    first = _contrastive_scale("interface_warmup", 1, 1, 100, 2)
    end = _contrastive_scale("interface_warmup", 2, 100, 100, 2)
    full = _contrastive_scale("full_finetune", 1, 1, 100, 2)
    assert 0.0 < first < end
    assert end == 1.0
    assert full == 1.0


def test_validation_measures_source_use_on_an_exact_heldout_limit():
    class DiagnosticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, compute_source_diagnostics=False, **kwargs):
            self.calls.append((int(input_ids.shape[0]), bool(compute_source_diagnostics)))
            value = input_ids.float().mean()
            return {
                "loss": value,
                "prompt_retrieval_accuracy": value.new_tensor(0.5),
                "source_swap_accuracy": value.new_tensor(0.75),
            }

    model = DiagnosticModel()
    loader = [
        {"input_ids": torch.ones(3, 2, dtype=torch.long)},
        {"input_ids": torch.full((3, 2), 4, dtype=torch.long)},
    ]
    metrics = validation_loss(model, loader, torch.device("cpu"), {}, max_examples=4)
    assert metrics["eval_examples"] == 4.0
    assert metrics["eval_loss"] == pytest.approx((3.0 + 4.0) / 4.0)
    assert metrics["eval_prompt_retrieval_accuracy"] == 0.5
    assert model.calls == [(3, True), (1, True)]


# --- Adapter tests (preserved from v2) ---


def test_layer_fusion_has_stable_last_layer_prior():
    fusion = StableTokenLayerFusion(8, [-1, -2, -3, -4], dropout=0.0)
    hidden = tuple(torch.randn(2, 5, 8) for _ in range(5))
    output = fusion(hidden, torch.ones(2, 5, dtype=torch.long))
    assert output.shape == (2, 5, 8)
    assert fusion.last_mean_weights is not None
    assert int(fusion.last_mean_weights.argmax()) == 0
    assert torch.allclose(fusion.last_mean_weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_adapter_is_bidirectional_and_preserves_full_memory():
    torch.manual_seed(3)
    config = {
        "layer_fusion": True,
        "fuse_layers": [-1, -2],
        "projection_ffn_size": 16,
        "projection_gate_init": 0.1,
        "num_bidirectional_layers": 2,
        "num_heads": 2,
        "ffn_size": 32,
        "dropout": 0.0,
        "bidirectional_gate_init": 0.5,
        "use_salience": True,
        "salience_hidden_size": 8,
        "salience_gate_init": 0.1,
    }
    adapter = SummaryAdapterV2(8, 8, config).eval()
    base = [torch.randn(1, 4, 8) for _ in range(3)]
    mask = torch.ones(1, 4, dtype=torch.long)
    units = torch.tensor([[1, 1, 2, 2]])
    labels = torch.tensor([[1.0, 0.0]])
    first = adapter(tuple(base), mask, units, labels)
    changed = copy.deepcopy(base)
    changed[-1][:, -1] += 5.0
    second = adapter(tuple(changed), mask, units, labels)
    assert first.memory.shape == (1, 4, 8)
    assert first.attention_bias.shape == (1, 4)
    assert torch.isfinite(first.loss_salience)
    assert not torch.allclose(first.memory[:, 0], second.memory[:, 0])


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = nn.RMSNorm(4)
        self.k_norm = nn.RMSNorm(4)


def test_cross_attention_copies_all_native_projections():
    source = _Attention()
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        initializer_range=0.02,
    )
    cross = QwenCopiedCrossAttention(source, nn.RMSNorm(8), config, 0.0, True)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert torch.equal(getattr(cross, name).weight, getattr(source, name).weight)
        assert getattr(cross, name).weight.data_ptr() != getattr(source, name).weight.data_ptr()
    output = cross(
        torch.randn(2, 3, 8),
        torch.randn(2, 5, 8),
        torch.ones(2, 5, dtype=torch.long),
        None,
    )
    assert output.shape == (2, 3, 8)


def test_checkpoint_is_complete_last_only(tmp_path: Path):
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    path = tmp_path / "last.pt"
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_last_checkpoint(model, path, {"x": 1}, 15, 123, {"train": {"sha256": "x"}})
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_last_checkpoint(model, path)
    assert payload["checkpoint_role"] == "last"
    assert payload["stores_full_model_state"]
    assert payload["parameter_element_count"] == sum(parameter.numel() for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    with pytest.raises(ValueError):
        save_last_checkpoint(model, tmp_path / "best.pt", {}, 1, 1, {})


def test_oracle_and_locked_rouge():
    labels = greedy_evidence_labels(
        ["Mở cài đặt.", "Khởi động lại thiết bị.", "Mua một chiếc bàn."],
        "Khởi động lại thiết bị.",
    )
    assert labels == [0.0, 1.0, 0.0]
    scores = rouge_scores(["khởi động lại thiết bị"], ["Khởi động lại thiết bị"])
    assert scores["rouge1"] > 99.9
    assert scores["rouge2"] > 99.9
