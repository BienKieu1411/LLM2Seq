from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from llm2seq_v3.adapter import StableTokenLayerFusion, SummaryAdapterV2
from llm2seq_v3.architecture_check import probe_future_token_influence
from llm2seq_v3.checkpoint import load_last_checkpoint, save_last_checkpoint
from llm2seq_v3.config import load_config, validate_config
from llm2seq_v3.contrastive import (
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
from llm2seq_v3.decoder import (
    PretrainedQwenDecoder,
    QwenCopiedCrossAttention,
    QwenDecoderLayerWithCrossAttention,
)
from llm2seq_v3.encoder import EmbeddingTokenEncoder, _config_is_bidirectional
from llm2seq_v3.encoder_compare import compare_encoder_pilots
from llm2seq_v3.final_audit import audit_final_claim
from llm2seq_v3.metrics import rouge_scores
from llm2seq_v3.model import LLM2SeqV3
from llm2seq_v3.paper_compare import compare_paper_scores
from llm2seq_v3.pilot_compare import compare_pilots
from llm2seq_v3.smoke_gate import evaluate_smoke_run
from llm2seq_v3.training import (
    _contrastive_scale,
    _parameter_component,
    _tokenizers,
    build_optimizer,
    validation_loss,
    verify_declared_parameter_budget,
    verify_locked_data_manifest,
)

from llm2seq_v3.data import SummarizationDataset, decoder_seed_ids, encode_source, greedy_evidence_labels


def test_single_bank_control_is_v3_with_contrastive():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b.yaml")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"
    # v3 changes
    assert config["adapter"]["num_bidirectional_layers"] == 6
    assert config["decoder"]["cross_gate_init"] == 0.30
    assert config["objectives"]["use_contrastive"] is True
    assert config["objectives"]["contrastive_weight"] == 0.05
    assert config["objectives"]["contrastive_temperature"] == 0.07
    assert config["objectives"]["contrastive_pooling"] == "mean_last"
    assert config["objectives"]["source_swap_weight"] == 0.10
    assert config["objectives"]["source_swap_margin"] == 0.20
    assert config["objectives"]["source_swap_strategy"] == "hard_in_batch"
    assert config["benchmark"]["diagnostic"]["rouge2"] == 19.5308
    assert config["benchmark"]["paper"]["rouge2"] == 32.654
    assert config["objectives"]["label_smoothing"] == 0.10
    # Preserved from v2
    assert config["decoder"]["cross_attention_every"] == 1
    assert config["decoder"]["initialize_cross_from_self"] is True
    assert config["checkpoint"] == {
        "save_best": False,
        "save_each_epoch": False,
        "save_last": True,
    }


def test_paper_comparison_requires_same_backend_and_full_test_split():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "num_examples": 3901,
        "rouge1": 63.0,
        "rouge2": 33.0,
        "rougeL": 59.0,
    }
    candidate_metrics = {
        "deployable_parameters": 1_527_949_729,
        "training_parameters": 1_528_477_089,
        "checkpoint_test_matches_current": True,
        "checkpoint_parameters_match_model": True,
        "test_data_record": config["benchmark"]["data"]["test"],
    }
    result = compare_paper_scores(config, scores, candidate_metrics)
    assert result["comparable"] is True
    assert result["parameter_budget_reached"] is True
    assert result["rouge2_target_reached"] is True
    assert result["all_rouge_targets_reached"] is True
    assert result["paper_target_reached"] is True

    scores["num_examples"] = 20
    smoke = compare_paper_scores(config, scores, candidate_metrics)
    assert smoke["comparable"] is False
    assert "candidate_minus_target" not in smoke


def test_paper_comparison_rejects_wrong_split_or_oversized_candidate():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "num_examples": 3901,
        "rouge1": 63.0,
        "rouge2": 33.0,
        "rougeL": 59.0,
    }
    bad_metrics = {
        "deployable_parameters": 2_100_000_000,
        "training_parameters": 2_100_000_000,
        "checkpoint_test_matches_current": True,
        "checkpoint_parameters_match_model": True,
        "test_data_record": {"num_examples": 3901},
    }
    result = compare_paper_scores(config, scores, bad_metrics)
    assert result["comparable"] is False
    assert result["parameter_budget_reached"] is False
    assert any("parameter" in reason for reason in result["reasons"])


def _final_audit_artifacts(config):
    locked = config["benchmark"]["data"]["test"]
    target = config["benchmark"]["paper"]
    candidate_scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "num_examples": 3901,
        "rouge1": 63.0,
        "rouge2": 33.0,
        "rougeL": 59.0,
    }
    candidate_metrics = {
        "deployable_parameters": 1_527_949_729,
        "training_parameters": 1_528_477_089,
        "checkpoint_test_matches_current": True,
        "checkpoint_parameters_match_model": True,
        "test_data_record": {"num_examples": int(locked["num_examples"])},
    }
    baseline_scores = {
        "backend": "Perl ROUGE-1.5.5 via pyrouge==0.1.3",
        "num_examples": 3901,
        "rouge1": target["rouge1"],
        "rouge2": target["rouge2"],
        "rougeL": target["rougeL"],
    }
    baseline_metrics = {
        "base_model": "/models/t5gemma-2-1b-1b",
        "checkpoint_base_model": "google/t5gemma-2-1b-1b",
        "unique_parameter_elements": 2_012_345_678,
        "checkpoint_test_matches_current": True,
        "test_data_record": {"num_examples": int(locked["num_examples"])},
    }
    return candidate_scores, candidate_metrics, baseline_scores, baseline_metrics


def test_final_audit_uses_actual_artifacts_and_requires_strict_superiority():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
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
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    candidate_scores, candidate_metrics, baseline_scores, baseline_metrics = _final_audit_artifacts(config)
    baseline_scores = dict(baseline_scores)
    baseline_scores["rouge2"] -= 1.0
    baseline_metrics = dict(baseline_metrics)
    baseline_metrics["test_data_record"] = {"num_examples": 1}
    result = audit_final_claim(
        config,
        candidate_scores,
        candidate_metrics,
        baseline_scores,
        baseline_metrics,
    )
    assert result["passed"] is False
    assert result["comparable"] is False
    assert any("test data" in reason for reason in result["comparability_reasons"])
    assert any("does not reproduce" in reason for reason in result["comparability_reasons"])


def test_full_training_requires_exact_locked_data_but_pilot_allows_subsets():
    root = Path(__file__).parents[1]
    full = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    locked = full["benchmark"]["data"]
    manifest = {split: dict(locked[split]) for split in ("train", "validation", "test")}
    assert verify_locked_data_manifest(full, manifest) == {
        "train": "exact_match",
        "validation": "exact_match",
        "test": "exact_match",
    }
    broken = copy.deepcopy(manifest)
    broken["test"]["num_examples"] = 1
    with pytest.raises(RuntimeError, match="test data differs"):
        verify_locked_data_manifest(full, broken)

    pilot = load_config(root / "configs/pilot_hiroute_2000.yaml")
    assert verify_locked_data_manifest(pilot, broken) == {
        "train": "partial_not_comparable",
        "validation": "partial_not_comparable",
        "test": "partial_not_comparable",
    }


def test_preflight_requires_conservative_total_below_declared_parameter_target():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
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


def test_hiroute_config_preserves_depth_routing():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/qwen3_0_6b_hiroute.yaml")
    assert config["adapter"]["hierarchical_sentence_context"] is True
    assert config["adapter"]["depth_routed_memory"] is True
    assert config["adapter"]["num_bidirectional_layers"] == 8
    assert config["adapter"]["lexical_layers"] == [-21, -17, -13, -9]
    assert config["adapter"]["semantic_layers"] == [-1, -5, -9, -13]
    assert config["decoder"]["memory_bank_count"] == 3
    assert config["decoder"]["memory_routing_mode"] == "attention_output"
    assert config["decoder"]["query_adaptive_routing"] is True
    assert config["decoder"]["query_router_max_delta"] == 2.0
    assert config["objectives"]["routing_balance_weight"] == 0.01
    assert config["adapter"]["branch_global_context"] is True
    assert config["adapter"]["branch_context_gate_init"] == 0.10
    # v3 contrastive should also be active
    assert config["objectives"]["use_contrastive"] is True
    assert config["objectives"]["label_smoothing"] == 0.10
    assert config["checkpoint"]["save_last"] is True
    assert config["checkpoint"]["save_best"] is False


def test_embedding_encoder_profiles_have_valid_depths_and_budgeted_shapes():
    root = Path(__file__).parents[1]
    pplx = load_config(root / "configs/pplx_embed_v1_0_6b_hiroute.yaml")
    assert pplx["model"]["encoder_name"] == "perplexity-ai/pplx-embed-v1-0.6b"
    assert pplx["model"]["encoder_hidden_size"] == 1024
    assert pplx["model"]["encoder_num_hidden_layers"] == 28
    assert pplx["model"]["encoder_attention_mode"] == "bidirectional"
    assert pplx["model"]["encoder_trust_remote_code"] is True
    assert pplx["model"]["encoder_revision"] == "2c4d510dd4a732063c31a0f70193e35067b51fd8"
    assert pplx["objectives"]["contrastive_pooling"] == "mean"
    assert pplx["data"]["source_prefix"] == ""
    assert pplx["data"]["append_source_eos"] is False
    assert pplx["data"]["source_add_special_tokens"] is True
    assert pplx["adapter"]["lexical_layers"] == [-21, -17, -13, -9]
    assert pplx["decoder"]["cross_attention_every"] == 1

    nemotron = load_config(root / "configs/nemotron3_embed_1b_hiroute.yaml")
    assert nemotron["model"]["encoder_name"] == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert nemotron["model"]["encoder_hidden_size"] == 2048
    assert nemotron["model"]["encoder_num_hidden_layers"] == 16
    assert nemotron["model"]["encoder_attention_mode"] == "bidirectional"
    assert nemotron["model"]["encoder_trust_remote_code"] is False
    assert nemotron["model"]["encoder_revision"] == "a5e0f804b9e90a1ca6784ecbf6e41595774fc834"
    assert nemotron["objectives"]["contrastive_pooling"] == "mean"
    assert nemotron["adapter"]["fuse_layers"] == [-1, -3, -6, -8]
    assert nemotron["adapter"]["lexical_layers"] == [-12, -10, -8, -6]
    assert nemotron["adapter"]["semantic_layers"] == [-1, -3, -6, -8]
    assert nemotron["adapter"]["num_bidirectional_layers"] == 8
    assert nemotron["decoder"]["cross_attention_every"] == 2
    assert nemotron["data"]["source_prefix"] == "passage: "
    assert nemotron["data"]["append_source_eos"] is False
    assert nemotron["data"]["source_add_special_tokens"] is True

    pplx_light = load_config(root / "configs/pplx_embed_v1_0_6b_native_light.yaml")
    assert pplx_light["adapter"]["num_bidirectional_layers"] == 2
    assert pplx_light["adapter"]["depth_routed_memory"] is False
    assert pplx_light["adapter"]["hierarchical_sentence_context"] is True
    assert pplx_light["decoder"]["memory_bank_count"] == 1
    assert pplx_light["decoder"]["query_adaptive_routing"] is False

    nemotron_light = load_config(root / "configs/nemotron3_embed_1b_native_light.yaml")
    assert nemotron_light["adapter"]["num_bidirectional_layers"] == 2
    assert nemotron_light["adapter"]["depth_routed_memory"] is False
    assert nemotron_light["decoder"]["memory_bank_count"] == 1
    assert nemotron_light["decoder"]["cross_attention_every"] == 2


def test_config_rejects_encoder_layer_indices_outside_checkpoint_depth():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/nemotron3_embed_1b_hiroute.yaml")
    config["adapter"]["lexical_layers"] = [-21]
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
    config = load_config(root / "configs/pplx_embed_v1_0_6b_hiroute.yaml")
    config["model"]["encoder_revision"] = ""
    with pytest.raises(ValueError, match="encoder_revision"):
        validate_config(config)
    config["model"]["encoder_revision"] = "2c4d510dd4a732063c31a0f70193e35067b51fd8"
    config["model"]["encoder_trust_remote_code"] = "true"
    with pytest.raises(ValueError, match="encoder_trust_remote_code"):
        validate_config(config)

    config = load_config(root / "configs/pplx_embed_v1_0_6b_hiroute.yaml")
    config["data"]["append_source_eos"] = True
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
        padding_side = "left"

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


def test_pilot_control_changes_only_hiroute_memory_components():
    root = Path(__file__).parents[1]
    main = load_config(root / "configs/pilot_hiroute_2000.yaml")
    control = load_config(root / "configs/pilot_single_bank_2000.yaml")
    assert main["limits"] == control["limits"]
    assert main["training"] == control["training"]
    assert main["adapter"]["num_bidirectional_layers"] == 8
    assert control["adapter"]["num_bidirectional_layers"] == 8
    assert main["adapter"]["hierarchical_sentence_context"] is True
    assert control["adapter"]["hierarchical_sentence_context"] is True
    assert main["adapter"]["depth_routed_memory"] is True
    assert control["adapter"].get("depth_routed_memory", False) is False
    assert main["decoder"]["memory_bank_count"] == 3
    assert control["decoder"].get("memory_bank_count", 1) == 1


def test_pilot_comparison_requires_hiroute_quality_and_utilization(tmp_path: Path):
    main = tmp_path / "main"
    control = tmp_path / "control"
    main.mkdir()
    control.mkdir()
    data_record = {"num_examples": 512}
    common = {
        "num_examples": 512,
        "checkpoint_parameters_match_model": True,
        "test_data_record": data_record,
        "encoder_name": "Qwen/Qwen3-Embedding-0.6B",
        "decoder_name": "Qwen/Qwen3-0.6B",
        "rouge_backend": "rouge==1.0.0 (diagnostic)",
    }
    (main / "last_test_predictions.metrics.json").write_text(
        json.dumps({**common, "rouge1": 42.0, "rouge2": 21.0, "rougeL": 40.0}),
        encoding="utf-8",
    )
    (control / "last_test_predictions.metrics.json").write_text(
        json.dumps({**common, "rouge1": 41.0, "rouge2": 20.0, "rougeL": 39.0}),
        encoding="utf-8",
    )
    validation = {
        "epoch": 6,
        "eval_examples": 256,
        "eval_prompt_retrieval_accuracy": 0.20,
        "eval_source_swap_accuracy": 0.70,
        "eval_source_swap_nll_gap": 0.30,
        "eval_source_swap_negative_similarity": 0.80,
        "eval_memory_routing_entropy": 0.90,
        "eval_adaptive_routing_delta": 0.01,
        "eval_memory_route_lexical": 0.30,
        "eval_memory_route_semantic": 0.35,
        "eval_memory_route_summary": 0.35,
    }
    (main / "validation_history.jsonl").write_text(json.dumps(validation) + "\n", encoding="utf-8")
    (control / "validation_history.jsonl").write_text(
        json.dumps({"epoch": 6, "eval_examples": 256}) + "\n",
        encoding="utf-8",
    )
    result = compare_pilots(main, control)
    assert result["hiroute_minus_control"]["rouge2"] == 1.0
    assert result["main_source_utilization"]["source_swap_negative_similarity"] == 0.80
    assert result["recommend_full_hiroute_run"] is True


def test_pilot_comparison_rejects_different_test_rows(tmp_path: Path):
    main = tmp_path / "main"
    control = tmp_path / "control"
    main.mkdir()
    control.mkdir()
    common = {
        "num_examples": 512,
        "rouge1": 40.0,
        "rouge2": 20.0,
        "rougeL": 39.0,
        "checkpoint_parameters_match_model": True,
        "encoder_name": "encoder",
        "decoder_name": "decoder",
        "rouge_backend": "rouge==1.0.0 (diagnostic)",
    }
    (main / "last_test_predictions.metrics.json").write_text(
        json.dumps(
            {
                **common,
                "test_data_record": {"num_examples": 512},
            }
        ),
        encoding="utf-8",
    )
    (control / "last_test_predictions.metrics.json").write_text(
        json.dumps(
            {
                **common,
                "test_data_record": {"num_examples": 511},
            }
        ),
        encoding="utf-8",
    )
    validation = json.dumps({"epoch": 6, "eval_examples": 256}) + "\n"
    (main / "validation_history.jsonl").write_text(validation, encoding="utf-8")
    (control / "validation_history.jsonl").write_text(validation, encoding="utf-8")
    result = compare_pilots(main, control)
    assert result["comparable"] is False
    assert result["comparability_gates"]["same_test_data"] is False
    assert result["recommend_full_hiroute_run"] is False


def test_encoder_pilot_comparison_ranks_only_healthy_comparable_runs(tmp_path: Path):
    data_record = {"num_examples": 512}
    runs = {}
    for label, encoder, rouge2 in (
        ("qwen3", "Qwen/Qwen3-Embedding-0.6B", 19.0),
        ("pplx", "perplexity-ai/pplx-embed-v1-0.6b", 21.0),
        ("nemotron", "nvidia/Nemotron-3-Embed-1B-BF16", 20.0),
    ):
        run_dir = tmp_path / label
        run_dir.mkdir()
        runs[label] = run_dir
        (run_dir / "last_test_predictions.metrics.json").write_text(
            json.dumps(
                {
                    "num_examples": 512,
                    "rouge1": rouge2 + 20.0,
                    "rouge2": rouge2,
                    "rougeL": rouge2 + 19.0,
                    "empty_prediction_rate": 0.0,
                    "checkpoint_parameters_match_model": True,
                    "parameter_budget_reached": True,
                    "test_data_record": data_record,
                    "encoder_name": encoder,
                    "decoder_name": "Qwen/Qwen3-0.6B",
                    "rouge_backend": "rouge==1.0.0 (diagnostic)",
                    "deployable_parameters": 1_500_000_000,
                    "training_parameters": 1_501_000_000,
                    "parameter_target_declared": 2_000_000_000,
                    "memory_bank_count": 3,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "validation_history.jsonl").write_text(
            json.dumps(
                {
                    "epoch": 6,
                    "eval_examples": 256,
                    "eval_cross_residual_ratio": 0.03,
                    "eval_prompt_retrieval_accuracy": 0.2,
                    "eval_source_swap_accuracy": 0.8,
                    "eval_source_swap_nll_gap": 0.2,
                    "eval_memory_routing_entropy": 0.9,
                    "eval_memory_route_lexical": 0.3,
                    "eval_memory_route_semantic": 0.35,
                    "eval_memory_route_summary": 0.35,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    result = compare_encoder_pilots(runs)
    assert result["comparable"] is True
    assert result["ranking_by_rouge2_then_rouge1_rougeL"] == ["pplx", "nemotron", "qwen3"]
    assert result["recommended_for_full_run"] == "pplx"

    # Exact-zero collapsed routes must not make a declared three-bank system
    # look like a healthy single-bank model.
    (runs["qwen3"] / "validation_history.jsonl").write_text(
        json.dumps(
            {
                "epoch": 6,
                "eval_examples": 256,
                "eval_cross_residual_ratio": 0.03,
                "eval_prompt_retrieval_accuracy": 0.2,
                "eval_source_swap_accuracy": 0.8,
                "eval_source_swap_nll_gap": 0.2,
                "eval_memory_routing_entropy": 0.0,
                "eval_memory_route_lexical": 0.0,
                "eval_memory_route_semantic": 0.0,
                "eval_memory_route_summary": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    collapsed = compare_encoder_pilots(runs)
    assert collapsed["runs"]["qwen3"]["health_gates"]["router_not_collapsed"] is False
    assert "qwen3" not in collapsed["ranking_by_rouge2_then_rouge1_rougeL"]


def _write_smoke_artifacts(
    run_dir: Path,
    *,
    cross_ratio: float = 0.1,
    prefix_rate: float = 5.0,
    source_swap_accuracy: float = 0.60,
    source_swap_nll_gap: float = 0.1,
    training_parameters: int = 1_528_477_089,
    memory_bank_count: int = 3,
) -> None:
    run_dir.mkdir()
    (run_dir / "last.pt").touch()
    (run_dir / "COMPLETE").touch()
    (run_dir / "last_test_predictions.metrics.json").write_text(
        json.dumps(
            {
                "num_examples": 20,
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
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "validation_history.jsonl").write_text(
        json.dumps(
            {
                "eval_loss": 3.0,
                "eval_loss_ce": 2.5,
                "eval_loss_contrastive": 1.0,
                "eval_loss_source_swap": 0.8,
                "eval_cross_residual_ratio": cross_ratio,
                "eval_prompt_retrieval_accuracy": 0.25,
                "eval_source_swap_accuracy": source_swap_accuracy,
                "eval_source_swap_nll_gap": source_swap_nll_gap,
                "eval_memory_routing_entropy": 0.9 if memory_bank_count > 1 else 0.0,
                "eval_adaptive_routing_delta": 0.01 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_lexical": 0.30 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_semantic": 0.35 if memory_bank_count > 1 else 0.0,
                "eval_memory_route_summary": 0.35 if memory_bank_count > 1 else 1.0,
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
    assert "cross_attention_active" in result["failed_gates"]
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
    assert "correct_source_preferred" in result["failed_gates"]
    assert "total_below_declared_t5gemma_budget" in result["failed_gates"]


def test_run_script_defaults_to_output_routed_hiroute():
    root = Path(__file__).parents[1]
    script = (root / "run.sh").read_text(encoding="utf-8")
    assert 'MAIN_CONFIG="${CONFIG:-configs/qwen3_0_6b_hiroute.yaml}"' in script
    assert 'SMOKE_CONFIG="configs/smoke_hiroute_100.yaml"' in script


def test_unit_test_launcher_is_offline_and_has_no_publish_commands():
    root = Path(__file__).parents[1]
    script = (root / "run.sh").read_text(encoding="utf-8")
    assert 'if [[ "$MODE" == "test" ]]' in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export TRANSFORMERS_OFFLINE=1" in script
    forbidden = ("push_to_hub", "huggingface-cli upload", "hf upload", "upload_file(")
    assert all(command not in script for command in forbidden)


def test_main_ablations_change_only_the_claimed_hiroute_components():
    root = Path(__file__).parents[1]
    no_contrastive = load_config(root / "configs/ablations/no_contrastive.yaml")
    assert no_contrastive["adapter"]["depth_routed_memory"] is True
    assert no_contrastive["decoder"]["memory_routing_mode"] == "attention_output"
    assert no_contrastive["objectives"]["use_contrastive"] is False

    no_hiroute = load_config(root / "configs/ablations/no_hiroute_memory.yaml")
    assert no_hiroute["adapter"]["num_bidirectional_layers"] == 8
    assert no_hiroute["adapter"]["hierarchical_sentence_context"] is True
    assert no_hiroute["adapter"]["depth_routed_memory"] is False
    assert no_hiroute["decoder"]["memory_bank_count"] == 1
    assert no_hiroute["objectives"]["use_contrastive"] is True

    no_branch_context = load_config(root / "configs/ablations/no_branch_global_context.yaml")
    assert no_branch_context["adapter"]["depth_routed_memory"] is True
    assert no_branch_context["adapter"]["branch_global_context"] is False
    assert no_branch_context["decoder"]["query_adaptive_routing"] is True

    pre_attention = load_config(root / "configs/ablations/pre_attention_memory_routing.yaml")
    assert pre_attention["decoder"]["memory_routing_mode"] == "memory"
    assert pre_attention["decoder"]["query_adaptive_routing"] is False
    assert pre_attention["adapter"]["depth_routed_memory"] is True

    static_output = load_config(root / "configs/ablations/static_output_routing.yaml")
    assert static_output["decoder"]["memory_routing_mode"] == "attention_output"
    assert static_output["decoder"]["query_adaptive_routing"] is False

    cyclic_swap = load_config(root / "configs/ablations/cyclic_source_swap.yaml")
    assert cyclic_swap["adapter"]["depth_routed_memory"] is True
    assert cyclic_swap["objectives"]["use_source_swap"] is True
    assert cyclic_swap["objectives"]["source_swap_strategy"] == "cyclic"


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


def test_source_alignment_head_handles_4d_memory():
    """HiRoute depth-routed memory is [B, K, S, D]."""
    head = SourceAlignmentHead(hidden_size=16, projection_size=8)
    memory = torch.randn(3, 3, 10, 16)  # [B=3, K=3, S=10, D=16]
    memory_mask = torch.ones(3, 10, dtype=torch.long)
    prompt_states = torch.randn(3, 16)
    result = head(memory, memory_mask, prompt_states)
    assert result["memory_repr"].shape == (3, 8)
    assert result["decoder_repr"].shape == (3, 8)


def test_source_alignment_head_uses_decoder_bank_routing():
    torch.manual_seed(9)
    head = SourceAlignmentHead(hidden_size=16, projection_size=8)
    memory = torch.randn(3, 3, 10, 16)
    memory_mask = torch.ones(3, 10, dtype=torch.long)
    prompt_states = torch.randn(3, 16)
    routing = torch.tensor([0.1, 0.2, 0.7], requires_grad=True)
    routed = torch.einsum("k,bksd->bsd", routing, memory)
    expected = head(routed, memory_mask, prompt_states)
    actual = head(memory, memory_mask, prompt_states, bank_weights=routing)
    assert torch.allclose(actual["memory_repr"], expected["memory_repr"], atol=1e-6)
    actual["memory_repr"].sum().backward()
    assert routing.grad is not None


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


class _TinyEncoder(nn.Module):
    """Four differentiable hidden-state depths without any model loading."""

    def __init__(self, vocabulary_size: int = 32, hidden_size: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        states = self.embedding(input_ids)
        outputs = [states]
        for layer in self.layers:
            states = states + 0.2 * torch.tanh(layer(states))
            outputs.append(states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0))
        return tuple(outputs)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)


class _TinyOutputRoutedDecoder(nn.Module):
    """Small decoder that uses the production bank-wise cross-attention."""

    def __init__(self, vocabulary_size: int = 32, hidden_size: int = 8):
        super().__init__()
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            initializer_range=0.02,
        )
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.cross_attn = QwenCopiedCrossAttention(
            _Attention(),
            nn.RMSNorm(hidden_size),
            config,
            dropout=0.0,
            initialize_from_self=True,
        )
        self.memory_router_logits = nn.Parameter(torch.tensor([0.4, -0.2, 0.1]))
        self.cross_gate = nn.Parameter(torch.tensor(0.4))
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.last_cross_residual_ratio = torch.tensor(0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_bias: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        states = self.embedding(input_ids)
        bank_outputs = self.cross_attn(
            states,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_attention_bias,
        )
        routing = self.memory_routing_mean().to(bank_outputs.dtype)
        routed = torch.einsum("k,bktd->btd", routing, bank_outputs)
        cross_residual = torch.tanh(self.cross_gate) * routed
        with torch.no_grad():
            self.last_cross_residual_ratio = (
                cross_residual.float().square().mean().sqrt() / states.float().square().mean().sqrt().clamp_min(1e-8)
            )
        return states + cross_residual, None

    def memory_routing_mean(self) -> torch.Tensor:
        return F.softmax(self.memory_router_logits.float(), dim=0)

    def memory_routing_mean_for_loss(self) -> torch.Tensor:
        return self.memory_routing_mean()

    def memory_routing_entropy(self) -> torch.Tensor:
        routing = self.memory_routing_mean()
        return -(routing * routing.clamp_min(1e-8).log()).sum() / torch.log(routing.new_tensor(3.0))

    def routing_balance_loss(self) -> torch.Tensor:
        routing = self.memory_routing_mean()
        return (routing * (routing.clamp_min(1e-8).log() + torch.log(routing.new_tensor(3.0)))).sum()

    def adaptive_routing_delta_mean(self) -> torch.Tensor:
        return self.cross_gate.new_zeros(())

    def cross_residual_ratio_mean(self) -> torch.Tensor:
        return self.last_cross_residual_ratio.to(self.cross_gate.device)

    def cross_gate_mean(self) -> torch.Tensor:
        return torch.tanh(self.cross_gate)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)


def _tiny_full_objective_model() -> LLM2SeqV3:
    model = LLM2SeqV3.__new__(LLM2SeqV3)
    nn.Module.__init__(model)
    model.encoder = _TinyEncoder()
    model.adapter = SummaryAdapterV2(
        8,
        8,
        {
            "hidden_size": 8,
            "layer_fusion": True,
            "fuse_layers": [-1, -2],
            "projection_ffn_size": 16,
            "projection_gate_init": 0.1,
            "num_bidirectional_layers": 1,
            "num_heads": 2,
            "ffn_size": 24,
            "dropout": 0.0,
            "bidirectional_gate_init": 0.2,
            "use_salience": True,
            "salience_hidden_size": 8,
            "salience_gate_init": 0.1,
            "depth_routed_memory": True,
            "lexical_layers": [-4, -3],
            "semantic_layers": [-2, -1],
            "branch_projection_ffn_size": 16,
        },
    )
    model.decoder = _TinyOutputRoutedDecoder()
    model.alignment_head = SourceAlignmentHead(8, projection_size=4, pooling="mean_last")
    model.salience_weight = 0.1
    model.use_contrastive = True
    model.use_prompt_alignment = True
    model.use_source_swap = True
    model.contrastive_weight = 0.05
    model.contrastive_temperature = 0.07
    model.source_swap_weight = 0.1
    model.source_swap_margin = 0.2
    model.source_swap_temperature = 1.0
    model.source_swap_strategy = "hard_in_batch"
    model.contrastive_pooling = "mean_last"
    model.routing_balance_weight = 0.01
    model.label_smoothing = 0.1
    model._stage = "full_finetune"
    model._contrastive_scale = 1.0
    return model


def _tiny_objective_batch() -> dict[str, torch.Tensor]:
    batch_size = 4
    decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5]]).expand(batch_size, -1).clone()
    return {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 7],
                [8, 9, 10, 11, 12, 13],
                [8, 9, 10, 11, 12, 14],
            ]
        ),
        "attention_mask": torch.ones(batch_size, 6, dtype=torch.long),
        "unit_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]).expand(batch_size, -1),
        "evidence_labels": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": torch.ones_like(decoder_input_ids),
        "labels": torch.tensor(
            [
                [-100, 3, 4, 5, 6],
                [-100, 4, 5, 6, 7],
                [-100, 5, 6, 7, 8],
                [-100, 6, 7, 8, 9],
            ]
        ),
    }


def test_full_hiroute_contrastive_objective_backpropagates_end_to_end_offline():
    """Production loss composition reaches every major v3 component."""
    torch.manual_seed(23)
    model = _tiny_full_objective_model().train()
    outputs = model(**_tiny_objective_batch())
    assert torch.isfinite(outputs["loss"])
    assert outputs["loss_contrastive"].item() > 0.0
    assert outputs["loss_source_swap"].item() > 0.0
    assert torch.isfinite(outputs["source_swap_negative_similarity"])
    outputs["loss"].backward()

    parameters = {
        "encoder": model.encoder.embedding.weight,
        "summary_adapter": model.adapter.bidirectional_layers[0].attention.q_proj.weight,
        "lexical_bank": model.adapter.lexical_projection.base.weight,
        "semantic_bank": model.adapter.semantic_projection.base.weight,
        "cross_q": model.decoder.cross_attn.q_proj.weight,
        "cross_k": model.decoder.cross_attn.k_proj.weight,
        "cross_v": model.decoder.cross_attn.v_proj.weight,
        "cross_o": model.decoder.cross_attn.o_proj.weight,
        "router": model.decoder.memory_router_logits,
        "decoder": model.decoder.embedding.weight,
        "lm_head": model.decoder.lm_head.weight,
        "alignment_memory": model.alignment_head.memory_proj[-1].weight,
        "alignment_decoder": model.alignment_head.decoder_proj[-1].weight,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert parameter.grad.float().norm() > 0.0, f"zero gradient for {name}"

    model.eval()
    with torch.no_grad():
        diagnostic = model(**_tiny_objective_batch(), compute_source_diagnostics=True)
    expected_eval_loss = diagnostic["loss_ce"] + model.salience_weight * diagnostic["loss_salience"]
    assert torch.allclose(diagnostic["loss"], expected_eval_loss)
    assert diagnostic["loss_contrastive"].item() > 0.0
    assert diagnostic["loss_source_swap"].item() > 0.0


def test_full_finetune_optimizer_contains_every_trainable_parameter_once():
    model = _tiny_full_objective_model()
    model.set_training_stage("full_finetune")
    optimizer, _ = build_optimizer(
        model,
        {
            "full_encoder_lr": 1.2e-5,
            "full_decoder_lr": 1.0e-5,
            "full_adapter_lr": 3.0e-5,
            "full_cross_attention_lr": 5.0e-5,
            "fused_optimizer": False,
        },
        "full_finetune",
        total_steps=10,
    )
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual = {id(parameter) for parameter in grouped}
    assert actual == expected
    assert len(grouped) == len(actual)
    assert {group["component"] for group in optimizer.param_groups} == {
        "adapter",
        "cross_attention",
        "decoder",
        "encoder",
    }


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


def test_hiroute_adapter_preserves_three_full_length_memories():
    torch.manual_seed(7)
    adapter = SummaryAdapterV2(
        8,
        8,
        {
            "hidden_size": 8,
            "layer_fusion": True,
            "fuse_layers": [-1, -2],
            "projection_ffn_size": 16,
            "num_bidirectional_layers": 1,
            "num_heads": 2,
            "ffn_size": 24,
            "dropout": 0.0,
            "use_salience": True,
            "salience_hidden_size": 8,
            "depth_routed_memory": True,
            "lexical_layers": [-3, -2],
            "semantic_layers": [-2, -1],
            "branch_projection_ffn_size": 16,
            "branch_global_context": True,
            "branch_context_gate_init": 0.1,
            "hierarchical_sentence_context": True,
            "sentence_context_size": 8,
            "sentence_context_heads": 2,
            "sentence_context_ffn_size": 16,
            "sentence_context_layers": 1,
            "sentence_broadcast_gate_init": 0.1,
        },
    ).eval()
    output = adapter(
        tuple(torch.randn(2, 6, 8) for _ in range(4)),
        torch.ones(2, 6, dtype=torch.long),
        torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]]),
        torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    )
    assert output.memory.shape == (2, 3, 6, 8)
    assert output.attention_bias.shape == (2, 6)
    assert torch.isfinite(output.memory).all()
    assert torch.isfinite(output.loss_salience)
    assert adapter.lexical_context_gate is not None
    assert adapter.semantic_context_gate is not None
    assert torch.allclose(adapter.branch_context_gate_mean(), torch.tensor(0.1), atol=1e-6)
    (output.memory.float().square().mean() + output.loss_salience).backward()
    assert adapter.lexical_context_gate.grad is not None
    assert adapter.semantic_context_gate.grad is not None
    assert adapter.lexical_context_gate.grad.abs() > 0.0
    assert adapter.semantic_context_gate.grad.abs() > 0.0


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = nn.RMSNorm(4)
        self.k_norm = nn.RMSNorm(4)


class _BaseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(8)
        self.post_attention_layernorm = nn.RMSNorm(8)
        self.self_attn = _Attention()
        self.mlp = nn.Identity()


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


def test_cross_attention_normalizes_each_memory_bank_independently_and_caches_all_banks():
    torch.manual_seed(13)
    source = _Attention()
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        initializer_range=0.02,
    )
    cross = QwenCopiedCrossAttention(source, nn.RMSNorm(8), config, 0.0, True).eval()
    query = torch.randn(2, 3, 8)
    banks = torch.randn(2, 3, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    bank_outputs = cross(query, banks, mask, None)
    assert bank_outputs.shape == (2, 3, 3, 8)

    # Independent softmaxes are not equivalent to attending once over a
    # pre-averaged source memory.
    pre_mixed = cross(query, banks.mean(dim=1), mask, None)
    assert not torch.allclose(bank_outputs.mean(dim=1), pre_mixed, atol=1e-6)

    cross.prepare_memory_cache(banks)
    cached = cross(query, banks, mask, None)
    assert torch.allclose(cached, bank_outputs, atol=1e-6)
    assert cross._memory_cache is not None
    assert cross._memory_cache[0].shape[:2] == (2, 3)
    cross.clear_memory_cache()


def _routed_layer(
    index: int,
    total: int,
    query_adaptive: bool = False,
) -> QwenDecoderLayerWithCrossAttention:
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        initializer_range=0.02,
    )
    return QwenDecoderLayerWithCrossAttention(
        _BaseLayer(),
        config,
        dropout=0.0,
        gate_init=0.3,
        initialize_from_self=True,
        memory_bank_count=3,
        memory_routing_mode="attention_output",
        layer_index=index,
        total_layers=total,
        query_adaptive_routing=query_adaptive,
        query_router_max_delta=2.0,
    )


def _pre_attention_routed_layer(index: int, total: int) -> QwenDecoderLayerWithCrossAttention:
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        initializer_range=0.02,
    )
    return QwenDecoderLayerWithCrossAttention(
        _BaseLayer(),
        config,
        dropout=0.0,
        gate_init=0.3,
        initialize_from_self=True,
        memory_bank_count=3,
        memory_routing_mode="memory",
        layer_index=index,
        total_layers=total,
    )


def test_hiroute_routes_attention_outputs_after_independent_softmaxes():
    layer = _routed_layer(1, 3)
    layer.memory_router_logits.data.copy_(torch.tensor([0.0, 0.0, 0.0]))
    outputs = torch.stack(
        [
            torch.full((2, 4, 8), 1.0),
            torch.full((2, 4, 8), 2.0),
            torch.full((2, 4, 8), 6.0),
        ],
        dim=1,
    )
    routed = layer.route_attention_outputs(outputs)
    assert torch.allclose(routed, torch.full_like(routed, 3.0))


def test_output_routing_backpropagates_to_every_bank_and_router():
    """The generation objective must train all banks, not only summary memory."""
    torch.manual_seed(31)
    layer = _routed_layer(1, 3)
    query = torch.randn(2, 4, 8, requires_grad=True)
    banks = torch.randn(2, 3, 5, 8, requires_grad=True)
    bank_outputs = layer.cross_attn(
        query,
        banks,
        torch.ones(2, 5, dtype=torch.long),
        None,
    )
    routed = layer.route_attention_outputs(bank_outputs)
    # A non-symmetric target ensures router gradients cannot vanish merely
    # because a plain sum happens to cancel across banks.
    target = torch.linspace(-1.0, 1.0, routed.numel()).reshape_as(routed)
    F.mse_loss(routed, target).backward()

    assert banks.grad is not None
    bank_gradient_norms = banks.grad.float().square().sum(dim=(0, 2, 3)).sqrt()
    assert torch.all(bank_gradient_norms > 0)
    assert layer.memory_router_logits.grad is not None
    assert layer.memory_router_logits.grad.float().norm() > 0
    assert layer.cross_attn.q_proj.weight.grad is not None
    assert layer.cross_attn.k_proj.weight.grad is not None
    assert layer.cross_attn.v_proj.weight.grad is not None
    assert layer.cross_attn.o_proj.weight.grad is not None


def test_query_adaptive_router_starts_at_depth_prior_then_changes_by_token():
    torch.manual_seed(37)
    layer = _routed_layer(1, 3, query_adaptive=True)
    states = torch.stack(
        [
            torch.ones(3, 8),
            -torch.ones(3, 8),
        ]
    )
    initial = layer.routing_weights(states)
    static = layer.static_routing_weights().view(1, 1, 3).expand_as(initial)
    assert torch.allclose(initial, static, atol=1e-7)

    assert layer.memory_router_proj is not None
    with torch.no_grad():
        layer.memory_router_proj.weight[0].fill_(0.25)
        layer.memory_router_proj.weight[2].fill_(-0.25)
    adapted = layer.routing_weights(states)
    assert not torch.allclose(adapted[0], adapted[1])
    assert torch.allclose(adapted.sum(dim=-1), torch.ones_like(adapted[..., 0]))
    assert torch.all(adapted > 0.0)
    assert layer.last_adaptive_routing_delta is not None
    assert layer.last_adaptive_routing_delta > 0.0

    # With identical queries, different post-attention bank evidence can also
    # change the route. This verifies output-aware rather than query-only gating.
    with torch.no_grad():
        layer.memory_router_proj.weight.zero_()
        assert layer.memory_router_bank_weight is not None
        layer.memory_router_bank_weight[0].fill_(0.5)
        layer.memory_router_bank_weight[2].fill_(-0.5)
    bank_outputs = torch.randn(2, 3, 3, 8)
    identical_queries = states[:1].expand(2, -1, -1)
    output_aware = layer.routing_weights(identical_queries, bank_outputs)
    assert not torch.allclose(output_aware[0], output_aware[1])


def test_query_adaptive_output_router_receives_generation_gradient():
    torch.manual_seed(41)
    layer = _routed_layer(1, 3, query_adaptive=True)
    assert layer.memory_router_proj is not None
    states = torch.randn(2, 4, 8, requires_grad=True)
    bank_outputs = torch.randn(2, 3, 4, 8, requires_grad=True)
    target = torch.randn(2, 4, 8)
    routed = layer.route_attention_outputs(bank_outputs, router_states=states)
    F.mse_loss(routed, target).backward()
    assert layer.memory_router_proj.weight.grad is not None
    assert layer.memory_router_proj.weight.grad.float().norm() > 0.0
    assert layer.memory_router_bank_weight is not None
    assert layer.memory_router_bank_weight.grad is not None
    assert layer.memory_router_bank_weight.grad.float().norm() > 0.0
    assert states.grad is not None
    assert bank_outputs.grad is not None


def test_query_router_parameters_use_cross_attention_learning_rate():
    assert _parameter_component("decoder.backbone.layers.0.memory_router_logits") == "cross_attention"
    assert _parameter_component("decoder.backbone.layers.0.memory_router_norm.weight") == "cross_attention"
    assert _parameter_component("decoder.backbone.layers.0.memory_router_proj.weight") == "cross_attention"
    assert _parameter_component("decoder.backbone.layers.0.memory_router_bank_weight") == "cross_attention"


def test_legacy_pre_attention_routing_accepts_its_single_context_and_cache():
    layer = _pre_attention_routed_layer(1, 3).eval()
    memory = torch.randn(2, 3, 5, 8)
    routed_memory = layer.route_memory(memory)
    context = layer.cross_attn(
        torch.randn(2, 4, 8),
        routed_memory,
        torch.ones(2, 5, dtype=torch.long),
        None,
    )
    assert context.shape == (2, 4, 8)
    assert layer.route_attention_outputs(context).shape == (2, 4, 8)
    layer.prepare_cross_attention_cache(memory)
    assert layer.cross_attn._memory_cache is not None
    assert layer.cross_attn._memory_cache[0].shape[1] == 1


def test_global_routing_balance_prevents_collapse_without_uniform_per_layer():
    decoder = PretrainedQwenDecoder.__new__(PretrainedQwenDecoder)
    nn.Module.__init__(decoder)
    decoder.backbone = nn.Module()
    layers = [_routed_layer(index, 3) for index in range(3)]
    decoder.backbone.layers = nn.ModuleList(layers)

    # Strongly specialized layers can still be globally balanced.
    layers[0].memory_router_logits.data.copy_(torch.tensor([8.0, -8.0, -8.0]))
    layers[1].memory_router_logits.data.copy_(torch.tensor([-8.0, 8.0, -8.0]))
    layers[2].memory_router_logits.data.copy_(torch.tensor([-8.0, -8.0, 8.0]))
    assert decoder.routing_balance_loss().item() < 1e-5
    assert decoder.memory_routing_entropy().item() > 0.999

    # Collapse of every depth onto summary receives a clear penalty.
    for layer in layers:
        layer.memory_router_logits.data.copy_(torch.tensor([-8.0, -8.0, 8.0]))
    assert decoder.routing_balance_loss().item() > 1.0
    assert decoder.memory_routing_entropy().item() < 0.01


def test_routing_balance_regularizes_actual_query_adaptive_routes():
    decoder = PretrainedQwenDecoder.__new__(PretrainedQwenDecoder)
    nn.Module.__init__(decoder)
    decoder.backbone = nn.Module()
    layers = [_routed_layer(index, 3, query_adaptive=True) for index in range(3)]
    decoder.backbone.layers = nn.ModuleList(layers)

    # The static depth priors are globally balanced.
    static_loss = decoder.routing_balance_loss().detach()
    assert static_loss.item() < 1e-5

    # Make every token dynamically prefer the summary bank while preserving
    # the balanced static priors. The loss must now see this real collapse.
    states = torch.ones(2, 4, 8)
    outputs = torch.randn(2, 3, 4, 8)
    for layer in layers:
        assert layer.memory_router_proj is not None
        with torch.no_grad():
            layer.memory_router_proj.weight[0].fill_(-0.25)
            layer.memory_router_proj.weight[1].fill_(-0.25)
            layer.memory_router_proj.weight[2].fill_(0.25)
        layer.route_attention_outputs(outputs, router_states=states)
    dynamic_loss = decoder.routing_balance_loss()
    assert dynamic_loss > static_loss + 1e-4
    dynamic_loss.backward()
    for layer in layers:
        assert layer.memory_router_proj is not None
        assert layer.memory_router_proj.weight.grad is not None
        assert layer.memory_router_proj.weight.grad.float().norm() > 0.0


def test_interface_warmup_freezes_native_decoder_but_trains_cross_attention_and_router():
    """Exercise production freeze logic without constructing a HF model."""
    decoder = PretrainedQwenDecoder.__new__(PretrainedQwenDecoder)
    nn.Module.__init__(decoder)
    decoder.backbone = nn.Module()
    layer = _routed_layer(0, 1, query_adaptive=True)
    decoder.backbone.layers = nn.ModuleList([layer])
    decoder.lm_head = nn.Linear(8, 32, bias=False)

    decoder.set_backbone_trainable(False)
    assert all(not parameter.requires_grad for parameter in layer.base_layer.parameters())
    assert all(parameter.requires_grad for parameter in layer.cross_attn_norm.parameters())
    assert all(parameter.requires_grad for parameter in layer.cross_attn.parameters())
    assert layer.cross_gate.requires_grad
    assert layer.memory_router_logits.requires_grad
    assert layer.memory_router_norm is not None
    assert layer.memory_router_proj is not None
    assert layer.memory_router_bank_weight is not None
    assert all(parameter.requires_grad for parameter in layer.memory_router_norm.parameters())
    assert all(parameter.requires_grad for parameter in layer.memory_router_proj.parameters())
    assert layer.memory_router_bank_weight.requires_grad
    assert all(not parameter.requires_grad for parameter in decoder.lm_head.parameters())

    decoder.set_backbone_trainable(True)
    assert all(parameter.requires_grad for parameter in decoder.parameters())


def test_checkpoint_is_complete_last_only(tmp_path: Path):
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    path = tmp_path / "last.pt"
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_last_checkpoint(model, path, {"x": 1}, 15, 123, {"train": {"num_examples": 1}})
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
