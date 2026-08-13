from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from eviseq.configuration import load_config, resolve_data_path, validate_config
from eviseq.data.dataset import (
    LengthBucketBatchSampler,
    encode_source,
    greedy_evidence_labels,
    read_jsonl,
    split_units_with_spans,
    target_sentence_ids,
    visible_target_sentences,
)
from eviseq.evaluation.engine import _verify_or_write_resume_manifest
from eviseq.evaluation.generation import _apply_no_repeat_ngram, _apply_repetition_penalty, _blocked_tokens
from eviseq.evaluation.metrics import exact_match_score, token_f1_score
from eviseq.modeling.attention import mix_attention_outputs, unit_evidence_token_bias
from eviseq.modeling.bridge import EvidenceBridge, balanced_salience_loss
from eviseq.training.checkpoint import (
    assert_evaluation_config_matches_checkpoint,
    initialize_from_checkpoint,
    load_checkpoint,
    save_configured_epoch_checkpoints,
    save_last_checkpoint,
)
from eviseq.training.objectives import EvidenceContrastiveHead, evidence_info_nce_loss, sentence_evidence_info_nce_loss
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
    """A fixed-seed ablation must not randomize later objective heads."""

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
    # The fixed -log(unit_length) term makes equal-logit units receive equal
    # total mass even when their numbers of source subwords differ.
    equal_bias, _ = unit_evidence_token_bias(
        torch.zeros_like(logits),
        valid,
        torch.tensor([[1, 2, 2, 2]]),
        torch.ones(1, 4, dtype=torch.long),
        evidence_gate=torch.tensor(0.25),
    )
    equal_attention = torch.softmax(equal_bias, dim=-1)
    assert equal_attention[0, 0].item() == pytest.approx(equal_attention[0, 1:].sum().item(), abs=1e-6)


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
    # With zero source logit and two neutral/two source tokens, the old
    # implementation applied gate * -log(2) to both groups.
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


def test_bridge_projection_recipe_enables_only_the_identity_initialized_variant() -> None:
    base = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned_corrected.yaml")
    variant = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned_bridgeproj.yaml")
    assert base["bridge"]["trainable_identity_projection"] is False
    assert variant["bridge"]["trainable_identity_projection"] is True
    assert variant["model"] == base["model"]
    assert variant["objectives"] == base["objectives"]


def test_bridge_projection_gate20_recipe_changes_only_cross_gate() -> None:
    projection = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned_bridgeproj.yaml")
    gate20 = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned_bridgeproj_gate20.yaml")
    assert gate20["bridge"] == projection["bridge"]
    assert gate20["objectives"] == projection["objectives"]
    assert gate20["decoder"]["cross_gate_init"] == pytest.approx(0.20)
    projection_decoder = dict(projection["decoder"])
    gate20_decoder = dict(gate20["decoder"])
    gate20_decoder.pop("cross_gate_init")
    projection_decoder.pop("cross_gate_init")
    assert gate20_decoder == projection_decoder


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


def test_evaluation_refuses_a_same_shape_checkpoint_with_changed_source_protocol() -> None:
    config = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned.yaml")
    payload = {"config": copy.deepcopy(config)}
    assert_evaluation_config_matches_checkpoint(payload, config)
    changed = copy.deepcopy(config)
    changed["data"]["source_prefix"] = "A different article prompt"
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        assert_evaluation_config_matches_checkpoint(payload, changed)


def test_legacy_identity_checkpoint_config_remains_evaluable() -> None:
    # A checkpoint predating the *identity-projection flag* remains valid
    # when its effective bridge semantics were otherwise unchanged.  The
    # corrected aligned recipe intentionally has a different attention-bias
    # contract and must therefore still fail closed against an old run.
    current = load_config(ROOT / "configs" / "tasks" / "pubmed.yaml")
    legacy = copy.deepcopy(current)
    legacy["bridge"].pop("trainable_identity_projection")
    legacy["bridge"].pop("salience_gate_parameterization")
    legacy["bridge"].pop("salience_length_normalization")
    assert_evaluation_config_matches_checkpoint({"config": legacy}, current)


def test_evaluation_resume_is_bound_to_checkpoint_config_and_dataset(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned.yaml")
    payload = {"checkpoint_role": "last", "epoch": 4, "global_step": 123, "config": copy.deepcopy(config)}
    rows = [{"id": "doc-1", "source": "article", "target": "abstract"}]
    output = tmp_path / "predictions.jsonl"
    _verify_or_write_resume_manifest(
        output,
        resume=False,
        checkpoint=tmp_path / "last.pt",
        payload=payload,
        config=config,
        rows=rows,
    )
    output.write_text('{"id":"doc-1","prediction":"abstract","reference":"abstract"}\n', encoding="utf-8")
    _verify_or_write_resume_manifest(
        output,
        resume=True,
        checkpoint=tmp_path / "last.pt",
        payload=payload,
        config=config,
        rows=rows,
    )
    changed_payload = {**payload, "global_step": 124}
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        _verify_or_write_resume_manifest(
            output,
            resume=True,
            checkpoint=tmp_path / "last.pt",
            payload=changed_payload,
            config=config,
            rows=rows,
        )


def test_sentence_aligned_salience_coupling_requires_union_labels() -> None:
    config = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned.yaml")
    config["data"]["sentence_evidence_use_union_as_salience"] = False
    with pytest.raises(ValueError, match="requires.*union"):
        validate_config(config)


def test_pubmed_evidence_ablations_hold_prompt_and_salience_labels_fixed() -> None:
    main = load_config(ROOT / "configs" / "models" / "pplx_pubmed_aligned_corrected.yaml")
    for name in ("pplx_pubmed_global_matched.yaml", "pplx_pubmed_no_evidence_cl.yaml"):
        ablation = load_config(ROOT / "configs" / "models" / name)
        for field in (
            "source_prefix",
            "sentence_evidence_supervision",
            "sentence_evidence_use_union_as_salience",
        ):
            assert ablation["data"][field] == main["data"][field]
        assert ablation["bridge"] == main["bridge"]


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
    # Unit 0 supports sentence 0 and unit 1 supports sentence 1.  They remain
    # useful q/k negatives for the other sentence, but neither may receive a
    # negative update to the *single* global bridge salience score.
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
    # Unit 1 is deliberately most confusable but unknown; it must never be
    # mined. Unit 0 is the positive. Only units 2/3 are legal negatives.
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

    # The final reference sentence is intentionally cut after token 11.  Its
    # oracle evidence must see "Result", never the absent "continued" token.
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


def test_run_script_has_no_upload_or_push_operation() -> None:
    script = (ROOT / "scripts" / "run.sh").read_text(encoding="utf-8").lower()
    assert "push_to_hub" not in script
    assert "huggingface-cli upload" not in script
    assert "git push" not in script
    assert "last.pt resolved_config.yaml" in script
