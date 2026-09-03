from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import eviseq.modeling.architecture as architecture_module
import eviseq.training.trainer as trainer_module
import pytest
import torch
import torch.nn as nn
from eviseq.config import load_config
from eviseq.data.dataset import (
    LengthBucketBatchSampler,
    Text2TextDataset,
    aligned_external_evidence_labels,
    encode_source,
    greedy_evidence_labels,
    mentor_greedy_evidence_labels,
    read_jsonl,
    split_units_with_spans,
    target_sentence_ids,
    visible_target_sentences,
)
from eviseq.evaluation.engine import _load_resume_records, _verify_or_write_resume_manifest
from eviseq.evaluation.generation import _apply_no_repeat_ngram, _apply_repetition_penalty, _blocked_tokens
from eviseq.evaluation.metrics import exact_match_score, token_f1_score
from eviseq.modeling.architecture import EviSeq
from eviseq.modeling.attention import (
    evidence_key_attention_bias,
    mix_attention_outputs,
    pool_units,
    unit_evidence_token_bias,
)
from eviseq.modeling.bridge import EvidenceBridge, balanced_salience_loss
from eviseq.modeling.decoder import QwenCopiedCrossAttention
from eviseq.modeling.encoder import NativeDualMaskQwenEncoder
from eviseq.training.checkpoint import (
    assert_evaluation_config_matches_checkpoint,
    initialize_from_checkpoint,
    load_checkpoint,
    save_configured_epoch_checkpoints,
    save_last_checkpoint,
)
from eviseq.training.engine import DistributedLengthBucketBatchSampler
from eviseq.training.engine import _parameter_component as engine_parameter_component
from eviseq.training.objectives import (
    EvidenceContrastiveHead,
    PromptConditionedEvidenceHead,
    SourcePromptAlignmentHead,
    _evidence_masks_and_hard_negatives,
    evidence_info_nce_loss,
    hard_negative_indices,
    last_prompt_states,
    pairwise_geometry_preservation_loss,
    per_example_nll,
    sentence_evidence_info_nce_loss,
    source_memory_for_mining,
    source_swap_contrastive_loss,
)
from eviseq.training.online_kd import GoldPrefixTeacher, topk_kl_loss
from eviseq.training.trainer import _capture_optimizer_moments, _parameter_component, _restore_optimizer_moments


def test_external_evidence_labels_are_zero_based_and_truncation_safe() -> None:
    assert aligned_external_evidence_labels([0, 2], 3, 5) == [1.0, 0.0, 1.0]
    assert aligned_external_evidence_labels([4], 3, 5) is None
    assert aligned_external_evidence_labels([1], 3, 5, index_base=1) == [1.0, 0.0, 0.0]


def test_mentor_oracle_uses_fixed_three_sentence_budget_and_ascii_cleaning() -> None:
    """The optional fallback must stay byte-for-byte compatible with preparation."""

    units = [
        "αβ is metadata, not evidence.",
        "Treatment reduced symptom severity after twelve weeks.",
        "No serious adverse events were reported.",
        "Follow-up continued for one year.",
        "An unrelated protocol was used elsewhere.",
    ]
    target = "Treatment reduced symptom severity and no serious adverse events were reported."
    expected = mentor_greedy_evidence_labels(units, target, summary_size=3)
    actual = greedy_evidence_labels(
        units,
        target,
        max_units=12,
        oracle_style="mentor",
        mentor_summary_size=3,
    )
    assert actual == expected
    assert sum(value > 0.5 for value in actual) <= 3


def test_pairwise_geometry_preservation_is_zero_for_identical_geometry() -> None:
    source = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    projected = source * 3.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    loss = pairwise_geometry_preservation_loss(source, projected, valid)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_source_swap_loss_prefers_the_correct_source() -> None:
    positive = torch.tensor([0.5, 0.8])
    negative = torch.tensor([1.2, 1.4])
    assert (
        source_swap_contrastive_loss(positive, negative, margin=0.2).item()
        < source_swap_contrastive_loss(negative, positive, margin=0.2).item()
    )


def test_source_mining_is_target_free_and_selects_the_closest_wrong_source() -> None:
    memory = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    mask = torch.ones(3, 2, dtype=torch.long)
    representations = source_memory_for_mining(memory, mask, pooling="mean")
    indices, similarities = hard_negative_indices(representations)
    assert indices.tolist() == [1, 0, 1]
    assert similarities[0].item() > 0.9


def test_per_example_nll_reduces_flattened_logits_by_row() -> None:
    labels = torch.tensor([[1, -100, 0], [2, 1, -100]])
    supervised = labels.ne(-100)
    logits = torch.tensor(
        [
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ]
    )
    values = per_example_nll(logits, labels, supervised)
    assert values.shape == (2,)
    assert values[0].item() == pytest.approx(values[1].item(), rel=1e-6)


def test_pairwise_geometry_preservation_backpropagates_to_projected_units() -> None:
    source = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    projected = torch.tensor([[[1.0, 0.0], [0.8, 0.6], [1.0, 1.0]]], requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    loss = pairwise_geometry_preservation_loss(source, projected, valid)
    assert loss.item() > 0.0
    loss.backward()
    assert projected.grad is not None
    assert projected.grad.abs().sum() > 0


def test_pool_units_ignores_ids_outside_the_requested_geometry_budget() -> None:
    token_states = torch.tensor([[[10.0], [20.0], [30.0]]])
    unit_ids = torch.tensor([[1, 2, 99]])
    pooled, valid = pool_units(token_states, unit_ids, unit_count=2)
    torch.testing.assert_close(pooled, torch.tensor([[[10.0], [20.0]]]))
    assert valid.tolist() == [[True, True]]


def test_unit_attention_bias_does_not_route_unknown_ids_to_the_last_unit() -> None:
    bias, source_keys = unit_evidence_token_bias(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[True, True]]),
        torch.tensor([[1, 2, 99]]),
        torch.ones(1, 3, dtype=torch.long),
    )
    assert source_keys.tolist() == [[True, True, False]]
    assert bias[0, 2].item() == pytest.approx(0.0)


ROOT = Path(__file__).resolve().parents[1]
PUBMED_CONFIG = ROOT / "configs" / "models" / "pplx_pubmed_pceb_corrected.yaml"


def test_project_layout_has_clear_responsibility_boundaries() -> None:
    assert not (ROOT / "eviseq").exists()
    for directory in ("data", "evaluation", "modeling", "training"):
        assert (ROOT / "core" / directory / "__init__.py").is_file()
    assert (ROOT / "core" / "config.py").is_file()
    assert (ROOT / "configs" / "base.yaml").is_file()
    assert (ROOT / "scripts" / "run.sh").is_file()


def test_pubmed_yaml_config_is_fresh_and_uses_the_pplx_route() -> None:
    first = load_config(PUBMED_CONFIG)
    second = load_config(PUBMED_CONFIG)
    first["training"]["batch_size"] = 1
    assert second["training"]["batch_size"] != 1
    assert first["model"]["encoder_name"] == "perplexity-ai/pplx-embed-v1-0.6b"
    assert first["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"
    assert first["data"]["train_file"] == "datasets/pubmed/train.jsonl"


def test_config_inheritance_records_the_recipe_path() -> None:
    config = load_config(PUBMED_CONFIG)
    assert config["_meta"]["config_path"].endswith("pplx_pubmed_pceb_corrected.yaml")
    assert config["objectives"]["evidence_contrastive_attention_aligned"] is True


def test_identity_initialized_bridge_projection_preserves_memory_and_learns() -> None:
    bridge = EvidenceBridge(
        8,
        8,
        {
            "trainable_identity_projection": True,
            "salience_gate_init": 0.1,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    memory = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.long)
    output = bridge(memory, mask, None, None, None, None)
    torch.testing.assert_close(output.memory, memory)
    assert output.projection_residual_ratio is not None
    assert output.projection_residual_ratio.item() == pytest.approx(0.0, abs=1e-7)
    output.memory.square().mean().backward()
    assert bridge.projection.update.weight.grad is not None
    assert bridge.projection.update.weight.grad.abs().sum() > 0


def test_identity_initialized_projection_is_rng_neutral() -> None:
    """Identity initialization must not consume random numbers."""

    torch.manual_seed(17)
    expected = torch.randn(11)
    torch.manual_seed(17)
    EvidenceBridge(
        8,
        8,
        {
            "trainable_identity_projection": True,
            "salience_gate_init": 0.1,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    actual = torch.randn(11)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_identity_initialized_projection_accepts_bfloat16_without_autocast() -> None:
    bridge = EvidenceBridge(
        8,
        8,
        {
            "trainable_identity_projection": True,
            "salience_gate_init": 0.1,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    memory = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    result = bridge(memory, torch.ones(2, 5, dtype=torch.long), None, None, None, None)
    assert result.memory.dtype == torch.bfloat16
    result.memory.float().square().mean().backward()
    assert bridge.projection.update.weight.grad is not None


def test_unequal_width_bridge_accepts_bfloat16_without_autocast() -> None:
    """General encoder->decoder bridges must not depend on CUDA autocast."""

    bridge = EvidenceBridge(
        8,
        4,
        {
            "salience_gate_init": 0.1,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    memory = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    result = bridge(memory, torch.ones(2, 5, dtype=torch.long), None, None, None, None)
    assert result.memory.dtype == torch.bfloat16
    result.memory.float().square().mean().backward()
    assert bridge.projection[1].weight.grad is not None


def _tiny_native_encoder_for_trainability(variant: str) -> NativeDualMaskQwenEncoder:
    """Construct the control surface without downloading a Qwen checkpoint."""

    encoder = NativeDualMaskQwenEncoder.__new__(NativeDualMaskQwenEncoder)
    nn.Module.__init__(encoder)
    encoder.model = nn.Linear(4, 4)
    encoder.evidence_norm = nn.RMSNorm(4)
    encoder.evidence_head = nn.Linear(4, 1)
    encoder.generic_token_gate = nn.Linear(4, 2)
    encoder.evidence_view_gate = nn.Parameter(torch.zeros(3, 2))
    encoder.variant = variant
    return encoder


@pytest.mark.parametrize(
    ("variant", "expect_evidence_gate", "expect_generic_gate"),
    [
        ("evidence", True, False),
        ("dec2enc", True, True),
        ("full", False, False),
        ("causal", False, False),
    ],
)
def test_native_encoder_only_trains_attention_controls_used_by_variant(
    variant: str,
    expect_evidence_gate: bool,
    expect_generic_gate: bool,
) -> None:
    encoder = _tiny_native_encoder_for_trainability(variant)
    encoder.set_trainable(False)

    assert all(parameter.requires_grad for parameter in encoder.evidence_norm.parameters())
    assert all(parameter.requires_grad for parameter in encoder.evidence_head.parameters())
    assert all(parameter.requires_grad is expect_generic_gate for parameter in encoder.generic_token_gate.parameters())
    assert encoder.evidence_view_gate.requires_grad is expect_evidence_gate


@pytest.mark.parametrize("variant", ["evidence", "full", "causal", "dec2enc"])
def test_full_finetune_allows_unused_native_attention_controls_to_stay_frozen(variant: str) -> None:
    class TinyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def set_backbone_trainable(self, trainable: bool) -> None:
            self.weight.requires_grad = trainable

    model = EviSeq.__new__(EviSeq)
    nn.Module.__init__(model)
    model.encoder = _tiny_native_encoder_for_trainability(variant)
    model.adapter = nn.Linear(4, 4)
    model.decoder = TinyDecoder()
    model.alignment_head = None
    model.evidence_contrastive_head = None
    model.prompt_conditioned_evidence_head = None
    model.register_parameter("prompt_bridge_fusion_logit", None)
    model.prompt_conditioned_inference_bridge = False
    model.evidence_contrastive_attention_aligned = False
    model.evidence_hard_negatives_warmup = 1
    model.evidence_hard_negatives_full = 2

    model.set_training_stage("full_finetune")

    assert all(parameter.requires_grad for parameter in model.encoder.model.parameters())
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(
        parameter.requires_grad is (variant == "dec2enc") for parameter in model.encoder.generic_token_gate.parameters()
    )
    assert model.encoder.evidence_view_gate.requires_grad is (variant in {"evidence", "dec2enc"})


@pytest.mark.parametrize(
    "name",
    [
        "encoder.evidence_norm.weight",
        "encoder.evidence_head.0.weight",
        "encoder.evidence_view_gate",
        "encoder.generic_token_gate.weight",
    ],
)
def test_engine_classifies_native_evidence_controls_as_adapter(name: str) -> None:
    """Interface warmup must give native bridge controls an adapter LR.

    These parameters are nested under ``encoder`` for checkpoint ownership,
    but they are the live evidence interface.  Without the explicit mapping
    the engine would reject them because warmup intentionally has no encoder
    learning rate.
    """

    assert engine_parameter_component(name) == "adapter"


def test_evidence_bias_prefers_positive_units_in_actual_attention_scores() -> None:
    """Gold-positive salience must become a larger decoder cross-attention prior.

    Query/key content scores are deliberately tied, so the only possible
    preference comes from the bridge's token evidence bias.
    """

    logits = torch.tensor([[2.0, -2.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    mask = torch.ones_like(unit_ids)
    bias, source_keys = unit_evidence_token_bias(
        logits,
        valid,
        unit_ids,
        mask,
        scale=1.0,
        evidence_gate=torch.tensor(0.25),
    )
    assert source_keys.all()
    attention = torch.softmax(bias, dim=-1)
    positive_mass = attention[:, :2].sum(dim=-1)
    negative_mass = attention[:, 2:].sum(dim=-1)
    assert bool((positive_mass > negative_mass).all())
    equal_bias, _ = unit_evidence_token_bias(
        torch.zeros_like(logits),
        valid,
        torch.tensor([[1, 2, 2, 2]]),
        torch.ones(1, 4, dtype=torch.long),
        evidence_gate=torch.tensor(0.25),
    )
    equal_attention = torch.softmax(equal_bias, dim=-1)
    assert equal_attention[0, 0].item() == pytest.approx(equal_attention[0, 1:].sum().item(), abs=1e-6)


def test_evidence_attention_falls_back_to_valid_keys_when_no_unit_is_visible() -> None:
    bias = evidence_key_attention_bias(
        torch.zeros(1, 2),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[1, 1, 0]], dtype=torch.long),
        dtype=torch.float32,
    )
    assert torch.isfinite(bias).all()
    assert torch.isfinite(bias[0, 0, 0, :2]).all()
    assert bias[0, 0, 0, 2].item() < -1.0e30


def test_corrected_bridge_prefers_positive_units_in_the_exact_bias_passed_to_decoder() -> None:
    """Verify the actual bridge output, not only the standalone helper.

    The decoder receives this vector as its additive cross-attention key
    prior in every copied cross-attention layer.  With tied content scores,
    the positive source unit must therefore receive more attention mass.
    """

    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.25,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    output = bridge(
        torch.zeros(1, 4, 4),
        torch.ones(1, 4, dtype=torch.long),
        torch.tensor([[1, 1, 2, 2]]),
        torch.tensor([[2.0, -2.0]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[1.0, 0.0]]),
    )
    assert output.attention_bias is not None
    attention = torch.softmax(output.attention_bias.float(), dim=-1)
    assert attention[:, :2].sum() > attention[:, 2:].sum()
    assert output.positive_attention_prior_gap is not None
    assert output.positive_attention_prior_gap.item() > 0.0


def test_legacy_gated_bridge_preserves_legacy_neutral_bias_semantics() -> None:
    """Legacy checkpoints must retain their original gated neutral group."""

    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "signed_tanh",
            "salience_length_normalization": "legacy_gated",
            "salience_gate_init": 0.10,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    output = bridge(
        torch.zeros(1, 4, 4),
        torch.ones(1, 4, dtype=torch.long),
        torch.tensor([[0, 0, 1, 1]]),
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[0.0]]),
    )
    assert output.attention_bias is not None
    torch.testing.assert_close(output.attention_bias[:, :2], output.attention_bias[:, 2:])


def test_attention_prior_diagnostic_uses_the_same_clipped_logits_as_sdpa_bias() -> None:
    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.10,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    output = bridge(
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
        torch.tensor([[1, 2]]),
        torch.tensor([[100.0, -100.0]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[1.0, 0.0]]),
    )
    gate = output.salience_attention_gate
    assert gate is not None and output.positive_attention_prior_gap is not None
    assert output.positive_attention_prior_gap.item() == pytest.approx(10.0 * gate.item(), abs=1e-6)


def test_evidence_projection_keys_accept_decoder_width_after_unequal_width_bridge() -> None:
    """EviCL keys are pooled from bridge memory, not raw encoder memory."""

    head = EvidenceContrastiveHead(key_hidden_size=8, decoder_hidden_size=8, projection_size=4)
    query, keys = head(torch.zeros(2, 8), torch.zeros(2, 3, 8))
    assert query.shape == (2, 4)
    assert keys.shape == (2, 3, 4)


def test_sigmoid_evidence_gate_cannot_reverse_positive_attention_preference() -> None:
    bridge = EvidenceBridge(
        8,
        8,
        {
            "trainable_identity_projection": False,
            "salience_gate_parameterization": "sigmoid",
            "salience_gate_init": 0.1,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    with torch.no_grad():
        bridge.salience_attention_gate.fill_(-100.0)
    assert bridge.attention_gate().item() >= 0.0


def test_corrected_static_pceb_freezes_only_its_unused_fusion_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled inference route must not expose an unused trainable scalar."""

    class TinyEncoder(nn.Module):
        hidden_size = 4

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def set_trainable(self, trainable: bool) -> None:
            self.weight.requires_grad = trainable

    class TinyDecoder(nn.Module):
        def __init__(self, *_: object) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.cross_attention_indices = (0, 1)
            self.weight = nn.Parameter(torch.zeros(()))

        def set_backbone_trainable(self, trainable: bool) -> None:
            self.weight.requires_grad = trainable

    monkeypatch.setattr(architecture_module, "build_encoder", lambda *_: TinyEncoder())
    monkeypatch.setattr(architecture_module, "PretrainedQwenDecoder", TinyDecoder)

    settings = load_config(PUBMED_CONFIG)
    settings["model"]["encoder_hidden_size"] = 4
    settings["model"]["decoder_hidden_size"] = 4
    model = EviSeq(settings)
    assert model.prompt_bridge_fusion_logit is not None
    assert not model.prompt_bridge_fusion_logit.requires_grad
    torch.testing.assert_close(model.prompt_bridge_fusion_gate(), torch.zeros(()))

    model.set_training_stage("full_finetune")
    assert not model.prompt_bridge_fusion_logit.requires_grad


def test_build_experiment_honors_the_validation_evidence_cache_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-checkpoint validation must not recompute PubMed labels per epoch."""

    calls: list[bool] = []

    class RecordedDataset:
        def __init__(self, *_: object, precompute_evidence: bool, **__: object) -> None:
            calls.append(precompute_evidence)

    config = load_config(PUBMED_CONFIG)
    monkeypatch.setattr(trainer_module.stable, "_tokenizers", lambda _: (object(), object()))
    monkeypatch.setattr(trainer_module, "EviSeq", lambda _: object())
    monkeypatch.setattr(trainer_module, "Text2TextDataset", RecordedDataset)

    trainer_module.build_experiment(config)
    assert calls == [True, True]


def test_bridge_reroute_reuses_memory_and_updates_only_attention_prior() -> None:
    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.25,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    memory = torch.randn(1, 5, 4)
    mask = torch.ones(1, 5, dtype=torch.long)
    unit_ids = torch.tensor([[0, 1, 1, 2, 2]])
    valid = torch.tensor([[True, True]])
    static = bridge(memory, mask, unit_ids, torch.tensor([[0.0, 0.0]]), valid, None)
    routed = bridge.reroute(static, torch.tensor([[2.0, -2.0]]))
    assert routed.memory.data_ptr() == static.memory.data_ptr()
    torch.testing.assert_close(routed.memory, static.memory)
    assert routed.attention_bias is not None and static.attention_bias is not None
    assert not torch.allclose(routed.attention_bias, static.attention_bias)
    assert routed.attention_bias[0, 1:3].mean() > routed.attention_bias[0, 3:].mean()


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


def test_evaluation_refuses_changed_resolved_configuration() -> None:
    settings = load_config(PUBMED_CONFIG)
    payload = {"config": copy.deepcopy(settings)}
    assert_evaluation_config_matches_checkpoint(payload, settings)
    changed = copy.deepcopy(settings)
    changed["data"]["source_prefix"] = "A different article prompt"
    with pytest.raises(RuntimeError, match="configuration differs from the checkpoint"):
        assert_evaluation_config_matches_checkpoint(payload, changed)


def test_evaluation_refuses_missing_resolved_configuration() -> None:
    with pytest.raises(RuntimeError, match="resolved configuration"):
        assert_evaluation_config_matches_checkpoint({}, load_config(PUBMED_CONFIG))


def test_evaluation_resume_is_bound_to_resolved_configuration_and_dataset(tmp_path: Path) -> None:
    settings = load_config(PUBMED_CONFIG)
    payload = {"checkpoint_role": "last", "epoch": 4, "global_step": 123, "config": copy.deepcopy(settings)}
    rows = [{"id": "doc-1", "source": "article", "target": "abstract"}]
    output = tmp_path / "predictions.jsonl"
    _verify_or_write_resume_manifest(
        output,
        resume=False,
        checkpoint=tmp_path / "last.pt",
        payload=payload,
        config=settings,
        rows=rows,
    )
    output.write_text('{"id":"doc-1","prediction":"abstract","reference":"abstract"}\n', encoding="utf-8")
    _verify_or_write_resume_manifest(
        output,
        resume=True,
        checkpoint=tmp_path / "last.pt",
        payload=payload,
        config=settings,
        rows=rows,
    )
    changed_payload = {**payload, "global_step": 124}
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        _verify_or_write_resume_manifest(
            output,
            resume=True,
            checkpoint=tmp_path / "last.pt",
            payload=changed_payload,
            config=settings,
            rows=rows,
        )


def test_evaluation_resume_refuses_reordered_or_changed_middle_rows(tmp_path: Path) -> None:
    settings = load_config(PUBMED_CONFIG)
    payload = {"checkpoint_role": "last", "epoch": 4, "global_step": 123, "config": copy.deepcopy(settings)}
    rows = [
        {"id": "doc-1", "source": "article 1", "target": "abstract 1"},
        {"id": "doc-2", "source": "article 2", "target": "abstract 2"},
        {"id": "doc-3", "source": "article 3", "target": "abstract 3"},
    ]
    output = tmp_path / "predictions.jsonl"
    _verify_or_write_resume_manifest(
        output,
        resume=False,
        checkpoint=tmp_path / "last.pt",
        payload=payload,
        config=settings,
        rows=rows,
    )
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        _verify_or_write_resume_manifest(
            output,
            resume=True,
            checkpoint=tmp_path / "last.pt",
            payload=payload,
            config=settings,
            rows=[rows[0], rows[2], rows[1]],
        )
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        _verify_or_write_resume_manifest(
            output,
            resume=True,
            checkpoint=tmp_path / "last.pt",
            payload=payload,
            config=settings,
            rows=[rows[0], {**rows[1], "target": "changed"}, rows[2]],
        )
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        _verify_or_write_resume_manifest(
            output,
            resume=True,
            checkpoint=tmp_path / "last.pt",
            payload=payload,
            config=settings,
            rows=[rows[0], {**rows[1], "source": "changed"}, rows[2]],
        )


def test_evaluation_resume_accepts_cleaned_wikihow_reference(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        '{"id":"doc-1","prediction":"summary","reference":"Use now."}\n',
        encoding="utf-8",
    )
    records, processed = _load_resume_records(
        output,
        [{"id": "doc-1", "source": "article", "target": 'Use {"smallUrl":"x"} now.'}],
        clean_metadata=True,
    )
    assert records[0]["reference"] == "Use now."
    assert processed == {"doc-1"}


def test_prompt_conditioned_head_is_optimized_with_the_bridge() -> None:
    """PCEB is a training-only head, but it must still receive adapter LR."""

    assert _parameter_component("prompt_conditioned_evidence_head.query_projection.1.weight") == "adapter"
    assert _parameter_component("prompt_conditioned_evidence_head.context_gate_logit") == "adapter"
    assert engine_parameter_component("prompt_conditioned_evidence_head.query_projection.1.weight") == "adapter"


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


def test_cross_attention_cache_rejects_a_different_source_layout() -> None:
    """A same-sized batch must not reuse K/V from a different source length."""

    source_attention = SimpleNamespace(
        q_proj=nn.Linear(4, 4, bias=False),
        k_proj=nn.Linear(4, 2, bias=False),
        v_proj=nn.Linear(4, 2, bias=False),
        o_proj=nn.Linear(4, 4, bias=False),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
    )
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        initializer_range=0.02,
    )
    cross = QwenCopiedCrossAttention(
        source_attention,
        nn.LayerNorm(4),
        config,
        dropout=0.0,
        initialize_from_self=True,
    ).eval()
    cached_memory = torch.randn(1, 3, 4)
    cross.prepare_memory_cache(cached_memory)
    with pytest.raises(RuntimeError, match="Stale cross-attention cache"):
        cross(
            torch.randn(1, 1, 4),
            torch.randn(1, 4, 4),
            torch.ones(1, 4, dtype=torch.long),
            None,
        )


def test_cross_attention_rejects_multihead_four_dimensional_bias() -> None:
    source_attention = SimpleNamespace(
        q_proj=nn.Linear(4, 4, bias=False),
        k_proj=nn.Linear(4, 2, bias=False),
        v_proj=nn.Linear(4, 2, bias=False),
        o_proj=nn.Linear(4, 4, bias=False),
        q_norm=nn.Identity(),
        k_norm=nn.Identity(),
    )
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        initializer_range=0.02,
    )
    cross = QwenCopiedCrossAttention(
        source_attention,
        nn.LayerNorm(4),
        config,
        dropout=0.0,
        initialize_from_self=True,
    ).eval()
    with pytest.raises(ValueError, match="shape \\[batch, 1, 1, source_length\\]"):
        cross(
            torch.randn(1, 1, 4),
            torch.randn(1, 4, 4),
            torch.ones(1, 4, dtype=torch.long),
            torch.zeros(1, 2, 1, 4),
        )


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


def test_prompt_conditioned_evidence_query_is_source_sensitive_and_target_free() -> None:
    """PCEB must not collapse a shared prompt to one document-agnostic query."""

    torch.manual_seed(7)
    head = PromptConditionedEvidenceHead(hidden_size=8, projection_size=4, context_gate_init=0.5)
    prompt = torch.randn(1, 8).repeat(2, 1).requires_grad_()
    source_context = torch.randn(2, 8, requires_grad=True)
    sentence_reprs = torch.randn(2, 4, 8, requires_grad=True)
    query, keys = head(prompt, source_context, sentence_reprs)
    assert not torch.allclose(query[0], query[1])

    labels = torch.tensor([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0]])
    salience = torch.randn(2, 4, requires_grad=True)
    result = evidence_info_nce_loss(
        query,
        keys,
        labels,
        labels.ge(0),
        num_hard_negatives=2,
        salience_logits=salience,
        salience_logit_bias=0.10,
    )
    result["evidence_contrastive_loss"].backward()
    assert prompt.grad is not None and prompt.grad.abs().sum() > 0
    assert source_context.grad is not None and source_context.grad.abs().sum() > 0
    assert sentence_reprs.grad is not None and sentence_reprs.grad.abs().sum() > 0
    assert salience.grad is not None and salience.grad.abs().sum() > 0


class _DualBridgeTestEncoder(nn.Module):
    """Small differentiable encoder used to test the full DualBridge route."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.evidence = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, unit_ids: torch.Tensor):
        memory = self.embedding(input_ids)
        units, valid = pool_units(memory, unit_ids, int(unit_ids.max().item()))
        return SimpleNamespace(
            memory=memory,
            unit_logits=self.evidence(units).squeeze(-1),
            valid_units=valid,
            native_gate_mean=memory.new_zeros(()),
        )


class _DualBridgeTestDecoder(nn.Module):
    """Records source priors while keeping CE differentiable through them."""

    def __init__(self, hidden_size: int, vocabulary_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.source_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.cross_attention_indices = (0, 1, 2, 3)
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_bias: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        encoder_cross_attention_start_layer: int | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        del encoder_attention_mask, attention_mask, use_cache
        if encoder_hidden_states is None:
            recorded_bias = self.embedding.weight.new_zeros((input_ids.shape[0], 0))
            source = self.embedding.weight.new_zeros((input_ids.shape[0], self.embedding.embedding_dim))
        else:
            recorded_bias = (
                encoder_hidden_states.new_zeros(encoder_hidden_states.shape[:2])
                if encoder_attention_bias is None
                else encoder_attention_bias.detach().clone()
            )
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_bias": recorded_bias,
                "has_source": input_ids.new_tensor(encoder_hidden_states is not None),
                "cross_attention_start_layer": input_ids.new_tensor(
                    -1 if encoder_cross_attention_start_layer is None else encoder_cross_attention_start_layer
                ),
            }
        )
        if encoder_hidden_states is None:
            pass
        elif encoder_attention_bias is None:
            weights = torch.full_like(encoder_hidden_states[..., 0], 1.0 / encoder_hidden_states.shape[1])
            source = (weights.unsqueeze(-1) * encoder_hidden_states.float()).sum(dim=1)
        else:
            weights = torch.softmax(encoder_attention_bias.float(), dim=-1)
            source = (weights.unsqueeze(-1) * encoder_hidden_states.float()).sum(dim=1)
        states = self.embedding(input_ids).float() + self.source_projection(source).unsqueeze(1)
        return states, None

    def cross_attention_probe_start_layer(self, probe_layers: int) -> int:
        if not 0 <= int(probe_layers) <= len(self.cross_attention_indices):
            raise ValueError("invalid probe layer count")
        return len(self.cross_attention_indices) - int(probe_layers)

    def clear_cross_attention_cache(self) -> None:
        """Match the real decoder's public cache-reset contract."""

        return None

    def cross_gate_mean(self) -> torch.Tensor:
        return self.embedding.weight.new_zeros(())

    def cross_residual_ratio_mean(self) -> torch.Tensor:
        return self.embedding.weight.new_zeros(())


def _tiny_dualbridge_model() -> EviSeq:
    """Instantiate EviSeq's real forward path without loading checkpoints."""

    model = EviSeq.__new__(EviSeq)
    nn.Module.__init__(model)
    hidden_size = 8
    model.encoder = _DualBridgeTestEncoder(hidden_size)
    model.adapter = EvidenceBridge(
        hidden_size,
        hidden_size,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.25,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    model.decoder = _DualBridgeTestDecoder(hidden_size)
    model.salience_weight = 0.0
    model.bridge_geometry_weight = 0.0
    model.bridge_geometry_max_units = 12
    model.use_contrastive = False
    model.contrastive_weight = 0.0
    model.contrastive_temperature = 0.07
    model.contrastive_pooling = "mean_last"
    model.use_source_swap = False
    model.source_swap_weight = 0.0
    model.source_swap_margin = 0.2
    model.source_swap_temperature = 1.0
    model.source_swap_strategy = "hard_in_batch"
    model.label_smoothing = 0.0
    model.alignment_head = None
    model.use_evidence_contrastive = True
    model.evidence_contrastive_weight = 0.10
    model.evidence_contrastive_temperature = 0.07
    model.evidence_hard_negatives = 2
    model.evidence_hard_negative_salience_boost = 0.0
    model.evidence_hard_negative_attention_boost = 0.0
    model.evidence_contrastive_salience_bias = 0.15
    model.evidence_contrastive_head = None
    model.evidence_contrastive_mode = "prompt_conditioned"
    model.prompt_conditioned_inference_bridge = True
    model.prompt_conditioned_static_neutral_probe = False
    model.prompt_bridge_dynamic_salience_mix = 0.5
    model.prompt_bridge_dynamic_logit_scale = 8.0
    model.prompt_bridge_dynamic_logit_clip = 2.0
    model.prompt_bridge_source_probe_layers = 2
    model.evidence_contrastive_attention_aligned = True
    model.prompt_conditioned_evidence_head = PromptConditionedEvidenceHead(
        hidden_size=hidden_size,
        projection_size=4,
        context_gate_init=0.5,
    )
    model.prompt_bridge_fusion_logit = nn.Parameter(torch.logit(torch.tensor(0.20)))
    model._contrastive_scale = 1.0
    model._evidence_contrastive_scale = 1.0
    return model


def test_source_swap_path_backpropagates_through_the_decoder_source_route() -> None:
    torch.manual_seed(37)
    model = _tiny_dualbridge_model().train()
    model.use_contrastive = True
    model.contrastive_weight = 0.05
    model.contrastive_across_accumulation = False
    model.alignment_head = SourcePromptAlignmentHead(8, projection_size=4, pooling="mean")
    model.use_source_swap = True
    model.source_swap_weight = 0.10
    model.source_swap_strategy = "hard_in_batch"
    model.prompt_conditioned_inference_bridge = False
    model.use_evidence_contrastive = False
    model.evidence_contrastive_weight = 0.0
    model.evidence_contrastive_head = None
    model.prompt_conditioned_evidence_head = None

    source_ids = torch.tensor([[1, 2, 3, 4], [8, 9, 10, 11]])
    source_mask = torch.ones_like(source_ids)
    unit_ids = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 2]])
    decoder_inputs = torch.tensor([[3, 4, 5, 6, 7], [3, 4, 5, 8, 9]])
    labels = torch.tensor([[-100, -100, 6, 7, 8], [-100, -100, 9, 10, 11]])

    result = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=decoder_inputs,
        decoder_attention_mask=torch.ones_like(decoder_inputs),
        unit_ids=unit_ids,
        evidence_labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=labels,
    )
    assert result["loss_source_swap"].item() > 0.0
    assert torch.isfinite(result["source_swap_nll_gap"])
    result["loss"].backward()
    assert model.decoder.source_projection.weight.grad is not None
    assert torch.isfinite(model.decoder.source_projection.weight.grad).all()


def test_dualbridge_uses_the_same_fused_prior_for_ce_and_greedy_prefill() -> None:
    """Guard the critical train--inference contract of the new architecture.

    The static prompt prefill may see only the fixed decoder seed.  Full CE
    teacher forcing and inference then have to consume the *same* rerouted
    bridge bias, while gradients from CE must still reach the dynamic query,
    source keys, fusion scalar, and bridge gate.
    """

    torch.manual_seed(29)
    model = _tiny_dualbridge_model().train()
    source_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    source_mask = torch.ones_like(source_ids)
    unit_ids = torch.tensor([[0, 1, 1, 2, 2], [0, 1, 1, 2, 2]])
    evidence_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    decoder_inputs = torch.tensor([[3, 4, 5, 6, 7, 8], [3, 4, 5, 9, 10, 11]])
    labels = torch.tensor([[-100, -100, 6, 7, 8, 9], [-100, -100, 9, 10, 11, 12]])

    result = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=decoder_inputs,
        decoder_attention_mask=torch.ones_like(decoder_inputs),
        unit_ids=unit_ids,
        evidence_labels=evidence_labels,
        labels=labels,
    )
    assert len(model.decoder.calls) == 2
    static_prefill, fused_teacher_forcing = model.decoder.calls
    torch.testing.assert_close(static_prefill["input_ids"], decoder_inputs[:, :3])
    torch.testing.assert_close(fused_teacher_forcing["input_ids"], decoder_inputs)
    assert static_prefill["has_source"].item() == 1
    assert static_prefill["cross_attention_start_layer"].item() == 2
    assert fused_teacher_forcing["cross_attention_start_layer"].item() == -1
    with torch.no_grad():
        static_bridge = model.encode(source_ids, source_mask, unit_ids=unit_ids)
        neutral_bridge = model.adapter.reroute(static_bridge, torch.zeros_like(static_bridge.salience_logits))
    torch.testing.assert_close(static_prefill["attention_bias"], neutral_bridge.attention_bias)
    assert not torch.allclose(static_prefill["attention_bias"], fused_teacher_forcing["attention_bias"])
    assert result["prompt_bridge_effective_delta_rms"].item() > 0.0
    assert result["prompt_bridge_effective_delta_nonzero_fraction"].item() > 0.0

    result["loss"].backward()
    assert model.prompt_bridge_fusion_logit.grad is not None
    assert model.prompt_bridge_fusion_logit.grad.abs().sum() > 0
    assert model.prompt_conditioned_evidence_head.query_projection[-1].weight.grad is not None
    assert model.prompt_conditioned_evidence_head.query_projection[-1].weight.grad.abs().sum() > 0
    assert model.prompt_conditioned_evidence_head.key_projection[-1].weight.grad is not None
    assert model.prompt_conditioned_evidence_head.key_projection[-1].weight.grad.abs().sum() > 0
    assert model.adapter.salience_attention_gate.grad is not None
    assert model.adapter.salience_attention_gate.grad.abs().sum() > 0

    with torch.no_grad():
        static_bridge = model.encode(source_ids, source_mask, unit_ids=unit_ids)
        inference_bridge = model.prompt_condition_bridge_for_generation(static_bridge, decoder_inputs[:, :3])
    assert inference_bridge.memory.data_ptr() == static_bridge.memory.data_ptr()
    assert inference_bridge.attention_bias is not None
    torch.testing.assert_close(inference_bridge.attention_bias, fused_teacher_forcing["attention_bias"])


def test_static_pceb_neutral_probe_does_not_reuse_static_teacher_forced_route() -> None:
    """The opt-in static query must come from a target-free neutral probe."""

    torch.manual_seed(31)
    model = _tiny_dualbridge_model().train()
    model.prompt_conditioned_inference_bridge = False
    model.prompt_conditioned_static_neutral_probe = True
    model.prompt_bridge_fusion_logit.requires_grad_(False)
    model.decoder.calls.clear()
    source_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    source_mask = torch.ones_like(source_ids)
    unit_ids = torch.tensor([[0, 1, 1, 2, 2], [0, 1, 1, 2, 2]])
    evidence_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    decoder_inputs = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 9]])
    labels = torch.tensor([[-100, -100, 6, 7], [-100, -100, 9, 10]])

    result = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=decoder_inputs,
        decoder_attention_mask=torch.ones_like(decoder_inputs),
        unit_ids=unit_ids,
        evidence_labels=evidence_labels,
        labels=labels,
        contrastive_mode="local",
    )

    assert result["loss"].isfinite()
    assert len(model.decoder.calls) == 2
    teacher_forcing, neutral_probe = model.decoder.calls
    assert teacher_forcing["input_ids"].shape[1] == decoder_inputs.shape[1]
    assert neutral_probe["input_ids"].shape[1] == 3
    assert bool(neutral_probe["has_source"].all())
    assert neutral_probe["cross_attention_start_layer"].item() == 2


def test_attention_aligned_evidence_loss_updates_the_live_bridge_gate() -> None:
    """The contrastive coupling must reach the energy used by SDPA, not a copy."""

    torch.manual_seed(31)
    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.25,
            "salience_bias_scale": 1.5,
            "salience_ranking_weight": 0.0,
        },
    )
    raw_logits = torch.tensor([[8.0, -7.0, 0.5]], requires_grad=True)
    expected = bridge.attention_gate().detach() * 1.5 * torch.tensor([[5.0, -5.0, 0.5]])
    torch.testing.assert_close(bridge.unit_attention_energy(raw_logits).detach(), expected)
    query = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1)
    result = evidence_info_nce_loss(
        query,
        keys,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.ones(1, 3, dtype=torch.bool),
        num_hard_negatives=2,
        salience_logits=raw_logits,
        salience_logit_bias=0.15,
        attention_prior_energy=bridge.unit_attention_energy(raw_logits),
    )
    result["evidence_contrastive_loss"].backward()
    assert raw_logits.grad is not None and raw_logits.grad.abs().sum() > 0
    assert bridge.salience_attention_gate.grad is not None
    assert bridge.salience_attention_gate.grad.abs().sum() > 0


def test_hard_negative_mining_can_follow_deployed_attention_energy() -> None:
    """Detached mining should surface a high-attention false positive."""

    query = torch.tensor([[1.0, 0.0]])
    keys = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]]]), dim=-1)
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    _, hard_without, _ = _evidence_masks_and_hard_negatives(
        query, keys, labels, valid, num_hard_negatives=1, attention_prior_energy=torch.tensor([[0.0, 0.0, 2.0]])
    )
    _, hard_with, _ = _evidence_masks_and_hard_negatives(
        query,
        keys,
        labels,
        valid,
        num_hard_negatives=1,
        attention_prior_energy=torch.tensor([[0.0, 0.0, 2.0]]),
        attention_mining_boost=1.0,
    )
    assert bool(hard_without[0, 1])
    assert bool(hard_with[0, 2])


def test_dualbridge_evidence_loss_reaches_the_deployed_fused_route() -> None:
    """Isolate EviCL from CE and verify every dynamic routing parameter learns.

    A full-model loss test alone could pass because teacher-forced CE reaches
    the fusion route.  This test makes the evidence InfoNCE term the *only*
    loss, proving that the claimed attention-aligned contrastive objective
    really updates the query/key/fusion/gate path used by greedy decoding.
    """

    torch.manual_seed(37)
    model = _tiny_dualbridge_model().train()
    source_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    source_mask = torch.ones_like(source_ids)
    unit_ids = torch.tensor([[0, 1, 1, 2, 2], [0, 1, 1, 2, 2]])
    evidence_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    decoder_inputs = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 9]])
    labels = torch.tensor([[-100, -100, 6, 7], [-100, -100, 9, 10]])

    static_bridge = model.encode(source_ids, source_mask, unit_ids=unit_ids, evidence_labels=evidence_labels)
    fused_bridge, diagnostics = model._prompt_condition_bridge_for_training(
        static_bridge,
        decoder_inputs,
        torch.ones_like(decoder_inputs),
        labels,
        evidence_labels,
    )
    result = evidence_info_nce_loss(
        query=diagnostics["prompt_bridge_query"],
        keys=diagnostics["prompt_bridge_keys"],
        evidence_labels=evidence_labels,
        valid_units=diagnostics["prompt_bridge_valid_units"],
        temperature=model.evidence_contrastive_temperature,
        num_hard_negatives=model.evidence_hard_negatives,
        salience_logits=diagnostics["prompt_bridge_fused_logits"],
        salience_boost=model.evidence_hard_negative_salience_boost,
        salience_logit_bias=model.evidence_contrastive_salience_bias,
        attention_prior_energy=model.adapter.unit_attention_energy(fused_bridge.salience_logits),
    )
    result["evidence_contrastive_loss"].backward()

    for parameter in (
        model.prompt_bridge_fusion_logit,
        model.prompt_conditioned_evidence_head.query_projection[-1].weight,
        model.prompt_conditioned_evidence_head.key_projection[-1].weight,
        model.adapter.salience_attention_gate,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_dualbridge_calibration_survives_a_bfloat16_attention_bias_cast() -> None:
    """A calibrated dynamic score must remain visible to BF16 SDPA.

    The configured initial dynamic score corresponds to a typical raw cosine
    of 0.06, times scale 8 and fusion 0.5: +/-0.24 in fused unit logits.
    At the default bridge gate this should alter essentially every source-key
    bias after its final BF16 cast; without calibration it rounds to zero.
    """

    bridge = EvidenceBridge(
        4,
        4,
        {
            "salience_gate_parameterization": "sigmoid",
            "salience_length_normalization": "unit_invariant",
            "salience_gate_init": 0.10,
            "salience_bias_scale": 1.0,
            "salience_ranking_weight": 0.0,
        },
    )
    memory = torch.randn(1, 65, 4, dtype=torch.bfloat16)
    mask = torch.ones(1, 65, dtype=torch.long)
    unit_ids = torch.tensor([[0] + [1] * 32 + [2] * 32])
    valid_units = torch.tensor([[True, True]])
    static = bridge(memory, mask, unit_ids, torch.zeros(1, 2), valid_units, None)
    fused = bridge.reroute(static, torch.tensor([[0.24, -0.24]]))
    measured = EviSeq._effective_prompt_bridge_delta(static, fused)
    assert measured["prompt_bridge_effective_delta_rms"].item() > 0.0
    assert measured["prompt_bridge_effective_delta_nonzero_fraction"].item() > 0.90


def test_last_prompt_state_ignores_reference_suffix_states() -> None:
    """Only the state that predicts the first target token may form the query."""

    states = torch.randn(2, 6, 4)
    labels = torch.tensor([[-100, -100, 4, 5, 6, -100], [-100, 8, 9, 10, -100, -100]])
    expected = last_prompt_states(states, labels)
    altered = states.clone()
    altered[0, 3:] += 1000.0
    altered[1, 2:] -= 1000.0
    torch.testing.assert_close(last_prompt_states(altered, labels), expected)


def test_sentence_aligned_evidence_loss_reaches_salience_logits() -> None:
    query = torch.nn.functional.normalize(torch.randn(2, 2, 8), dim=-1).requires_grad_()
    keys = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1).requires_grad_()
    salience = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor(
        [
            [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ]
    )
    result = sentence_evidence_info_nce_loss(
        query,
        torch.ones(2, 2, dtype=torch.bool),
        keys,
        labels,
        labels[:, 0].ge(0),
        num_hard_negatives=2,
        salience_logits=salience,
        salience_logit_bias=0.15,
    )
    result["evidence_contrastive_loss"].backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert keys.grad is not None and torch.isfinite(keys.grad).all()
    assert salience.grad is not None and salience.grad.abs().sum() > 0


def test_sentence_aligned_salience_does_not_contradict_other_sentence_evidence() -> None:
    query = torch.zeros(1, 2, 2, requires_grad=True)
    keys = torch.zeros(1, 3, 2, requires_grad=True)
    salience = torch.zeros(1, 3, requires_grad=True)
    labels = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = sentence_evidence_info_nce_loss(
        query,
        torch.ones(1, 2, dtype=torch.bool),
        keys,
        labels,
        torch.ones(1, 3, dtype=torch.bool),
        num_hard_negatives=2,
        salience_logits=salience,
        salience_logit_bias=0.15,
        global_evidence_labels=torch.tensor([[1.0, 1.0, 0.0]]),
    )
    result["evidence_contrastive_loss"].backward()
    assert salience.grad is not None
    assert salience.grad[0, 0] < 0.0
    assert salience.grad[0, 1] < 0.0
    assert salience.grad[0, 2] > 0.0


def test_sentence_evidence_positive_selection_is_bounded_and_supports_its_target() -> None:
    """The PubMed recipe selects at most three lexical evidence units/case."""

    units = [
        "The trial enrolled adults with chronic disease.",
        "Treatment reduced symptom severity after twelve weeks.",
        "No serious adverse events were reported.",
        "A separate imaging cohort used an unrelated protocol.",
        "Follow-up continued for one year.",
    ]
    labels = greedy_evidence_labels(
        units,
        "Treatment reduced symptom severity and no serious adverse events were reported.",
        max_units=3,
        budget=3,
    )
    selected = [index for index, label in enumerate(labels) if label > 0.5]
    assert 1 <= len(selected) <= 3
    assert 1 in selected
    assert 2 in selected


def test_sentence_hard_negatives_exclude_positive_and_unknown_evidence() -> None:
    query = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0]]]), dim=-1)
    keys = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]]), dim=-1)
    labels = torch.tensor([[[1.0, -1.0, 0.0, 0.0]]])
    result = sentence_evidence_info_nce_loss(
        query,
        torch.ones(1, 1, dtype=torch.bool),
        keys,
        labels,
        torch.ones(1, 4, dtype=torch.bool),
        num_hard_negatives=2,
    )
    assert torch.isfinite(result["evidence_contrastive_loss"])
    assert result["evidence_valid_examples"].item() == pytest.approx(1.0)


def test_online_kd_topk_other_bucket_is_finite_and_backpropagates() -> None:
    """KD trains the student's full logits while teacher targets stay detached."""

    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher_ids = torch.tensor([[[0, 1], [2, 3], [1, 4]], [[0, 6], [3, 5], [2, 4]]])
    teacher_logits = torch.tensor([[[4.0, 3.0], [2.0, 1.0], [3.0, 2.0]], [[5.0, 1.0], [1.0, 0.5], [2.0, 1.0]]])
    normalizers = torch.logsumexp(teacher_logits / 2.0, dim=-1) + 0.2
    mask = torch.tensor([[True, True, False], [True, False, True]])
    loss = topk_kl_loss(student, teacher_ids, teacher_logits, normalizers, mask, temperature=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0


def test_online_kd_prompt_width_ignores_right_padded_target_labels() -> None:
    class FakeTeacherModel:
        config = SimpleNamespace(vocab_size=8)

        def __call__(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.config.vocab_size))

    teacher = GoldPrefixTeacher.__new__(GoldPrefixTeacher)
    teacher.model = FakeTeacherModel()
    teacher.pad_id = 0
    teacher.topk = 2
    teacher.temperature = 2.0
    teacher.batch_size = 2
    teacher.device = torch.device("cpu")

    labels = torch.tensor(
        [
            [-100, -100, 1, 2, -100],
            [-100, -100, 3, -100, -100],
        ]
    )
    targets = teacher.soft_targets(
        torch.tensor([[10, 11], [10, 11]]),
        torch.ones(2, 2, dtype=torch.long),
        labels,
        output_device=torch.device("cpu"),
    )
    assert targets["teacher_mask"].tolist() == [[False, False, True, True, False], [False, False, True, False, False]]


def test_target_sentence_ids_follow_sentence_spans() -> None:
    class OffsetTokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["return_offsets_mapping"] is True
            return {"input_ids": [5, 6, 7, 8], "offset_mapping": [(0, 5), (6, 10), (12, 18), (19, 23)]}

    text = "First line. Second line."
    assert [value[0] for value in split_units_with_spans(text)] == ["First line.", "Second line."]
    assert target_sentence_ids(OffsetTokenizer(), text, [5, 6, 7, 8]) == [1, 1, 2, 2]


def test_sentence_aligned_targets_fail_closed_on_tokenization_mismatch() -> None:
    class MismatchedTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [9], "offset_mapping": [(0, 5)]}

    with pytest.raises(ValueError, match="do not match"):
        target_sentence_ids(MismatchedTokenizer(), "First. Second.", [5, 6], require_offsets=True)


def test_sentence_evidence_uses_only_visible_truncated_target_tokens() -> None:
    class DecodeTokenizer:
        def decode(self, token_ids, **kwargs):
            values = {10: "Objective.", 11: "Result", 12: " continued."}
            return " ".join(values[token_id] for token_id in token_ids)

    assert visible_target_sentences(DecodeTokenizer(), [10, 11], [1, 2]) == ["Objective.", "Result"]


def test_visible_target_sentences_preserve_sentence_id_rows_when_a_span_is_empty() -> None:
    class DecodeTokenizer:
        def decode(self, token_ids, **kwargs):
            return "" if token_ids == [11] else "First sentence."

    assert visible_target_sentences(DecodeTokenizer(), [10, 11, 12], [1, 2, 3]) == [
        "First sentence.",
        "",
        "First sentence.",
    ]


def test_source_prefix_must_leave_article_token_budget() -> None:
    class SourceTokenizer:
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return {"input_ids": [3, 4, 5]}

    with pytest.raises(ValueError, match="leaves no room"):
        encode_source(
            SourceTokenizer(),
            "article",
            {"source_prefix": "too long", "max_source_length": 4},
        )


def test_source_requires_a_visible_tokenizable_unit() -> None:
    class EmptyTokenizer:
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return {"input_ids": []}

    with pytest.raises(ValueError, match="no visible tokenizable units"):
        encode_source(EmptyTokenizer(), "article", {"max_source_length": 16})


def test_external_evidence_labels_fail_closed_when_a_row_is_missing_labels(tmp_path: Path) -> None:
    class TinyTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [3]}

    path = tmp_path / "missing_labels.jsonl"
    path.write_text('{"id":"doc-1","source":"article","target":"summary"}\n', encoding="utf-8")
    config = {
        "source_field": "source",
        "target_field": "target",
        "id_field": "id",
        "use_external_evidence_labels": True,
        "evidence_label_field": "label",
        "max_target_length": 8,
    }
    with pytest.raises(ValueError, match="evidence labels are missing"):
        Text2TextDataset(path, TinyTokenizer(), TinyTokenizer(), config, precompute_evidence=False)


def test_gpu_generation_constraints_match_scalar_reference() -> None:
    torch.manual_seed(0)
    base = torch.randn(2, 11)
    tokens = torch.tensor([[2, 5, 2, 5], [1, 3, 4, 3]])
    actual = base.clone()
    _apply_repetition_penalty(actual, tokens, 1.05)
    _apply_no_repeat_ngram(actual, tokens, 3)

    expected = base.clone()
    for row in range(tokens.shape[0]):
        previous = torch.unique(tokens[row])
        scores = expected[row, previous]
        expected[row, previous] = torch.where(scores < 0, scores * 1.05, scores / 1.05)
        blocked = _blocked_tokens(tokens[row].tolist(), 3)
        if blocked:
            expected[row, blocked] = float("-inf")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_length_bucket_sampler_covers_each_example_once() -> None:
    lengths = list(range(1, 17))
    batches = list(LengthBucketBatchSampler(lengths, batch_size=4, seed=7, bucket_size_multiplier=4))
    assert sorted(index for batch in batches for index in batch) == list(range(16))


def test_distributed_length_bucket_sampler_shards_whole_batches() -> None:
    """Ranks must get disjoint batches and execute an equal number of steps."""

    base = LengthBucketBatchSampler(list(range(1, 17)), batch_size=4, seed=7, bucket_size_multiplier=4)
    rank0 = DistributedLengthBucketBatchSampler(base, rank=0, world_size=2)
    rank1 = DistributedLengthBucketBatchSampler(
        LengthBucketBatchSampler(list(range(1, 17)), batch_size=4, seed=7, bucket_size_multiplier=4),
        rank=1,
        world_size=2,
    )
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    batches0, batches1 = list(rank0), list(rank1)
    assert len(batches0) == len(batches1) == 2
    assert not set(map(tuple, batches0)).intersection(map(tuple, batches1))
    assert sorted(index for batch in (*batches0, *batches1) for index in batch) == list(range(16))


def test_run_script_has_no_upload_or_push_operation() -> None:
    script = (ROOT / "scripts" / "run.sh").read_text(encoding="utf-8").lower()
    assert "push_to_hub" not in script
    assert "huggingface-cli upload" not in script
    assert "git push" not in script
    assert 'role="${eviseq_checkpoint_role:-last}"' not in script
    assert "eviseq_checkpoint_role" not in script
    assert "resolved_config.yaml" in script
    assert "train-ddp" in script
    assert "torchrun" in script
