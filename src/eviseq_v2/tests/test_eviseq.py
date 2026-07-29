from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from eviseq_v2.bridge import BridgeOutput, EvidenceBridge, balanced_salience_loss
from eviseq_v2.config import architecture_contract, load_config, validate_config
from eviseq_v2.contrastive import (
    SourcePromptAlignmentHead,
    evidence_info_nce_loss,
    exact_duplicate_mask,
    info_nce_loss,
    last_prompt_states,
    masked_mean_pool,
    mine_hard_negatives,
    pairwise_ranking_loss,
)
from eviseq_v2.data_integrity import audit
from eviseq_v2.encoder import EncoderOutput, NativeDualMaskQwenEncoder
from eviseq_v2.generation import generate_sampled_from_memory
from eviseq_v2.model import EviSeq
from eviseq_v2.native_attention import (
    align_trainable_sdpa_bias_heads,
    ensure_sdpa_lse_for_bias_backward,
    evidence_key_attention_bias,
    mix_attention_outputs,
    sdpa_mask,
    unit_evidence_token_bias,
)
from eviseq_v2.training import (
    _build_virtual_contrastive_cache,
    _capture_optimizer_moments,
    _parameter_component,
    _restore_optimizer_moments,
    _restore_rng_state,
    _salience_ranking_accuracy,
    _salience_scores,
    _uses_virtual_gradcache,
    _virtual_duplicate_mask,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from eviseq_v2.data import LengthBucketBatchSampler

ROOT = Path(__file__).resolve().parents[1]


def _tiny_native_encoder(variant: str) -> NativeDualMaskQwenEncoder:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    encoder = NativeDualMaskQwenEncoder.__new__(NativeDualMaskQwenEncoder)
    nn.Module.__init__(encoder)
    encoder.model = Qwen3Model(config)
    encoder.config = config
    encoder.model_name = "offline-tiny-qwen3"
    encoder.hidden_size = 32
    encoder.num_hidden_layers = 2
    encoder.num_heads = 4
    encoder.bidirectional_layer_count = 1
    encoder.first_bidirectional_layer = 1
    encoder.variant = variant
    encoder.evidence_key_bias_scale = 1.0
    encoder.attn_implementation = "sdpa"
    encoder.gradient_checkpointing = False
    encoder.evidence_norm = nn.RMSNorm(32)
    encoder.evidence_head = nn.Sequential(nn.Linear(32, 16, bias=False), nn.SiLU(), nn.Linear(16, 1, bias=True))
    nn.init.zeros_(encoder.evidence_head[-1].weight)
    nn.init.zeros_(encoder.evidence_head[-1].bias)
    encoder.evidence_view_gate = nn.Parameter(torch.zeros(2, 4))
    encoder.generic_token_gate = nn.Linear(32, 4)
    nn.init.zeros_(encoder.generic_token_gate.weight)
    nn.init.zeros_(encoder.generic_token_gate.bias)
    return encoder


class _ToyObjectiveEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.evidence_head = nn.Linear(4, 1)
        self.native_gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: torch.Tensor | None = None,
    ) -> EncoderOutput:
        assert unit_ids is not None
        memory = self.embedding(input_ids)
        memory = memory + self.native_gate * memory.flip(1)
        units = []
        valid = []
        for unit in (1, 2):
            selected = unit_ids.eq(unit) & attention_mask.bool()
            count = selected.sum(dim=1, keepdim=True)
            pooled = (memory * selected.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
            units.append(self.evidence_head(pooled).squeeze(-1))
            valid.append(count.squeeze(-1).gt(0))
        logits = torch.stack(units, dim=1)
        valid_units = torch.stack(valid, dim=1)
        return EncoderOutput(memory, logits, valid_units, self.native_gate.abs())


class _ToyObjectiveDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.memory_projection = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 32, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_attention_bias: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, use_cache
        mask = encoder_attention_mask.unsqueeze(-1).float()
        source = (encoder_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        states = self.embedding(input_ids) + self.memory_projection(source).unsqueeze(1)
        if encoder_attention_bias is not None:
            states = states + encoder_attention_bias.float().mean(dim=1)[:, None, None]
        return states, None

    def cross_gate_mean(self) -> torch.Tensor:
        return self.memory_projection.weight.float().abs().mean()

    def cross_residual_ratio_mean(self) -> torch.Tensor:
        return self.memory_projection.weight.float().square().mean()


def _toy_objective_model() -> EviSeq:
    model = EviSeq.__new__(EviSeq)
    nn.Module.__init__(model)
    model.encoder = _ToyObjectiveEncoder()
    model.adapter = EvidenceBridge(
        4,
        4,
        {"salience_gate_init": 0.1, "salience_bias_scale": 1.0},
    )
    model.decoder = _ToyObjectiveDecoder()
    model.salience_weight = 0.1
    model.use_contrastive = True
    model.contrastive_weight = 0.05
    model.contrastive_temperature = 0.07
    model.contrastive_warmup_epochs = 2
    model.contrastive_across_accumulation = True
    model.alignment_head = SourcePromptAlignmentHead(4, projection_size=4, pooling="mean")
    model._contrastive_scale = 1.0

    model.use_evidence_contrastive = False
    model.evidence_contrastive_weight = 0.05
    model.evidence_contrastive_temperature = 0.07
    model.evidence_hard_negatives = 2
    model.evidence_contrastive_warmup_epochs = 2
    model.evidence_contrastive_head = None
    model._evidence_contrastive_scale = 1.0

    model._stage = "interface_warmup"
    return model


def test_all_configs_load_offline() -> None:
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    assert len(paths) == 12
    for path in paths:
        config = load_config(path)
        assert len(config["_meta"]["architecture_sha256"]) == 64
        assert len(config["_meta"]["inference_protocol_sha256"]) == 64
        assert len(config["_meta"]["evaluation_contract_sha256"]) == 64


def test_one_architecture_and_objective_across_datasets() -> None:
    hashes = {
        load_config(ROOT / "configs" / name)["_meta"]["architecture_sha256"]
        for name in ("wikilingua.yaml", "cnndm.yaml", "pubmed.yaml")
    }
    assert len(hashes) == 1


def test_every_dataset_completes_contrastive_ramp_inside_warmup() -> None:
    for name in ("wikilingua.yaml", "cnndm.yaml", "pubmed.yaml"):
        config = load_config(ROOT / "configs" / name)
        assert (
            config["objectives"]["evidence_contrastive_warmup_epochs"] <= config["training"]["interface_warmup_epochs"]
        )


def test_adam_moments_survive_the_stage_boundary() -> None:
    layer = nn.Linear(3, 2)
    warmup_optimizer = torch.optim.AdamW(layer.parameters(), lr=1.0e-4)
    layer(torch.randn(4, 3)).square().mean().backward()
    warmup_optimizer.step()
    carried = _capture_optimizer_moments(layer, warmup_optimizer)
    expected = {name: state["exp_avg"].clone() for name, state in carried.items()}

    full_optimizer = torch.optim.AdamW(layer.parameters(), lr=1.0e-5)
    assert _restore_optimizer_moments(layer, full_optimizer, carried) == len(expected)
    for name, parameter in layer.named_parameters():
        torch.testing.assert_close(full_optimizer.state[parameter]["exp_avg"], expected[name])


def test_native_attention_controls_change_only_variant_and_output() -> None:
    contracts = []
    paths = (
        ROOT / "configs" / "ablations" / "c0_causal.yaml",
        ROOT / "configs" / "appendix" / "c1_hard_full.yaml",
        ROOT / "configs" / "ablations" / "c2_dec2enc.yaml",
        ROOT / "configs" / "wikilingua.yaml",
    )
    for path in paths:
        config = load_config(path)
        contract = architecture_contract(config)
        variant = contract["native_attention"].pop("variant")
        contracts.append((variant, contract))
    assert [value for value, _ in contracts] == ["causal", "full", "dec2enc", "evidence"]
    assert all(contract == contracts[0][1] for _, contract in contracts[1:])


def test_core_ablation_runner_has_three_unique_runs_and_reuses_main() -> None:
    script = (ROOT / "run.sh").read_text(encoding="utf-8")
    block = script.split("  ablation-all)", 1)[1].split("    ;;", 1)[0]
    assert block.count("train_and_validate") == 3
    assert "c0_causal.yaml" in block
    assert "c2_dec2enc.yaml" in block
    assert "c3_no_contrastive.yaml" in block
    assert "c1_hard_full.yaml" not in block
    assert "c3_evidence.yaml" not in block
    assert not (ROOT / "configs" / "ablations" / "c3_evidence.yaml").exists()


def test_core_ablation_directory_is_exactly_three_and_hard_full_is_appendix() -> None:
    names = sorted(path.name for path in (ROOT / "configs" / "ablations").glob("*.yaml"))
    assert names == ["c0_causal.yaml", "c2_dec2enc.yaml", "c3_no_contrastive.yaml"]
    assert (ROOT / "configs" / "appendix" / "c1_hard_full.yaml").is_file()


def test_core_ablations_share_one_inference_protocol() -> None:
    paths = (
        ROOT / "configs" / "ablations" / "c0_causal.yaml",
        ROOT / "configs" / "ablations" / "c2_dec2enc.yaml",
        ROOT / "configs" / "ablations" / "c3_no_contrastive.yaml",
        ROOT / "configs" / "wikilingua.yaml",
    )
    hashes = {load_config(path)["_meta"]["inference_protocol_sha256"] for path in paths}
    assert len(hashes) == 1


@pytest.mark.parametrize(
    ("eviseq_name", "t5gemma_name"),
    (
        ("wikilingua.yaml", "wikilingua_full_3072.yaml"),
        ("cnndm.yaml", "cnndm_full_1b_1b_4096.yaml"),
        ("pubmed.yaml", "pubmed_full_1b_1b_4096.yaml"),
    ),
)
def test_t5gemma_uses_the_same_dataset_and_generation_protocol(
    eviseq_name: str,
    t5gemma_name: str,
) -> None:
    candidate = load_config(ROOT / "configs" / eviseq_name)
    baseline_path = ROOT.parent / "T5Gemma" / "configs" / t5gemma_name
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    for key in ("source_prefix", "max_source_length", "max_target_length"):
        assert candidate["data"][key] == baseline["data"][key]
    for key in ("min_new_tokens", "max_new_tokens", "repetition_penalty", "no_repeat_ngram_size"):
        assert candidate["generation"][key] == baseline["generation"][key]
    assert baseline["generation"]["num_beams"] == 1
    assert baseline["generation"]["do_sample"] is False


def test_prompt_or_generation_change_updates_evaluation_contract(tmp_path: Path) -> None:
    original = load_config(ROOT / "configs" / "wikilingua.yaml")
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        "_base_: " + str(ROOT / "configs" / "wikilingua.yaml") + "\ngeneration:\n  max_new_tokens: 255\n",
        encoding="utf-8",
    )
    modified = load_config(changed)
    assert modified["_meta"]["architecture_sha256"] == original["_meta"]["architecture_sha256"]
    assert modified["_meta"]["evaluation_contract_sha256"] != original["_meta"]["evaluation_contract_sha256"]


def test_zero_initialized_dual_mask_is_exactly_causal() -> None:
    torch.manual_seed(3)
    causal = torch.randn(2, 5, 4, 8)
    full = torch.randn_like(causal)
    zero = torch.zeros(4)
    generic = torch.randn(2, 5, 4)
    actual_evidence = mix_attention_outputs(causal, full, "evidence", zero)
    actual_generic = mix_attention_outputs(causal, full, "dec2enc", zero, generic_logits=generic)
    torch.testing.assert_close(actual_evidence, causal, rtol=0, atol=0)
    torch.testing.assert_close(actual_generic, causal, rtol=0, atol=0)


def test_selective_encoder_ignores_lower_gates_and_uses_top_gate() -> None:
    torch.manual_seed(5)
    causal = _tiny_native_encoder("causal")
    selective = copy.deepcopy(causal)
    selective.variant = "evidence"
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    unit_ids = torch.tensor([[1, 1, 2, 2]])

    with torch.no_grad():
        selective.evidence_view_gate.zero_()
        selective.evidence_view_gate[0].fill_(0.8)
        causal_memory = causal(input_ids, attention_mask, unit_ids).memory
        inactive_lower_memory = selective(input_ids, attention_mask, unit_ids).memory
        torch.testing.assert_close(inactive_lower_memory, causal_memory, rtol=0, atol=0)

        selective.evidence_view_gate[1].fill_(0.8)
        active_top_memory = selective(input_ids, attention_mask, unit_ids).memory
        assert not torch.equal(active_top_memory, causal_memory)


def test_main_uses_conservative_nonzero_evidence_gate_initialization() -> None:
    config = load_config(ROOT / "configs" / "wikilingua.yaml")
    assert config["native_attention"]["evidence_view_gate_init"] == 0.01


def test_main_uses_only_six_top_bidirectional_layers() -> None:
    config = load_config(ROOT / "configs" / "wikilingua.yaml")
    attention = config["native_attention"]
    assert attention["implementation_revision"] == "selective_evidence_v2"
    assert attention["bidirectional_layer_count"] == 6


def test_pairwise_salience_term_rewards_positive_unit_ranking() -> None:
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    correct = torch.tensor([[2.0, -1.0, -2.0]], requires_grad=True)
    reversed_order = torch.tensor([[-2.0, 1.0, 2.0]])
    correct_loss = balanced_salience_loss(correct, labels, valid, ranking_weight=0.25)
    reversed_loss = balanced_salience_loss(reversed_order, labels, valid, ranking_weight=0.25)
    assert correct_loss < reversed_loss
    correct_loss.backward()
    assert correct.grad is not None and torch.isfinite(correct.grad).all()


def test_salience_ranking_accuracy_uses_pair_counts() -> None:
    totals = {"salience_correct_pairs": 7.5, "salience_pair_count": 10.0}
    assert _salience_ranking_accuracy(totals) == 0.75


def test_manual_native_causal_loop_exactly_matches_transformers_qwen() -> None:
    torch.manual_seed(7)
    encoder = _tiny_native_encoder("causal").eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    with torch.no_grad():
        expected = encoder.model(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = encoder(ids, mask, unit_ids).memory
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_evidence_zero_init_matches_causal_and_receives_gradient() -> None:
    torch.manual_seed(11)
    encoder = _tiny_native_encoder("evidence")
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    encoder.eval()
    with torch.no_grad():
        expected = encoder.model(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = encoder(ids, mask, unit_ids).memory
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    encoder.train()
    encoder.gradient_checkpointing = True
    output = encoder(ids, mask, unit_ids)
    loss = output.memory.float().square().mean() + output.unit_logits.float().square().mean()
    loss.backward()
    assert encoder.evidence_view_gate.grad is not None
    assert torch.isfinite(encoder.evidence_view_gate.grad).all()
    assert encoder.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_frozen_native_encoder_supports_bias_only_sdpa_backward() -> None:
    """Warm-up must train evidence routing while pretrained Q/K/V stay frozen."""

    torch.manual_seed(13)
    encoder = _tiny_native_encoder("evidence")
    encoder.set_trainable(False)
    encoder.evidence_view_gate.data.fill_(0.1)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    output = encoder(ids, mask, unit_ids)
    output.memory.float().square().mean().backward()
    assert encoder.model.layers[0].self_attn.q_proj.weight.grad is None
    assert encoder.evidence_head[-1].weight.grad is not None
    assert torch.isfinite(encoder.evidence_head[-1].weight.grad).all()


def test_sdpa_lse_guard_is_numerically_exact_and_narrow() -> None:
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 4, 3, 8)
    value = torch.randn(1, 4, 3, 8)
    bias = torch.randn(1, 1, 1, 3, requires_grad=True)
    guarded = ensure_sdpa_lse_for_bias_backward(query, key, value, bias)
    assert guarded.requires_grad
    torch.testing.assert_close(guarded, query, rtol=0, atol=0)
    already_differentiable = query.detach().requires_grad_(True)
    assert ensure_sdpa_lse_for_bias_backward(already_differentiable, key, value, bias) is already_differentiable


def test_trainable_sdpa_bias_is_materialized_per_query_head() -> None:
    compact = torch.randn(2, 1, 1, 5, requires_grad=True)
    aligned = align_trainable_sdpa_bias_heads(compact, 4)
    assert aligned.shape == (2, 4, 1, 5)
    assert aligned.is_contiguous()
    aligned.sum().backward()
    torch.testing.assert_close(compact.grad, torch.full_like(compact, 4.0))


def test_hard_controls_are_exact() -> None:
    causal = torch.randn(1, 3, 2, 4)
    full = torch.randn_like(causal)
    gate = torch.randn(2)
    torch.testing.assert_close(mix_attention_outputs(causal, full, "causal", gate), causal, rtol=0, atol=0)
    torch.testing.assert_close(mix_attention_outputs(causal, full, "full", gate), full, rtol=0, atol=0)


def test_evidence_routing_selects_key_location_not_only_total_mass() -> None:
    first_logits = torch.tensor([[4.0, -4.0]])
    second_logits = torch.tensor([[-4.0, 4.0]])
    valid = torch.tensor([[True, True, True]])
    valid = valid[:, :2]
    unit_ids = torch.tensor([[0, 1, 2, 0]])
    attention = torch.tensor([[1, 1, 1, 1]])
    first = evidence_key_attention_bias(first_logits, valid, unit_ids, attention, dtype=torch.float32)
    second = evidence_key_attention_bias(second_logits, valid, unit_ids, attention, dtype=torch.float32)
    assert first.shape == (1, 1, 1, 4)
    assert first[0, 0, 0, 1] > first[0, 0, 0, 2]
    assert second[0, 0, 0, 2] > second[0, 0, 0, 1]
    assert not torch.equal(first, second)
    assert first[0, 0, 0, 0] == torch.finfo(torch.float32).min
    assert first[0, 0, 0, 3] == torch.finfo(torch.float32).min


def test_evidence_key_bias_changes_attention_context_by_selected_unit() -> None:
    logits_left = torch.tensor([[4.0, -4.0]], requires_grad=True)
    logits_right = -logits_left.detach()
    valid = torch.tensor([[True, True]])
    unit_ids = torch.tensor([[0, 1, 2]])
    attention = torch.ones_like(unit_ids)
    left_bias = evidence_key_attention_bias(logits_left, valid, unit_ids, attention, dtype=torch.float32)
    right_bias = evidence_key_attention_bias(logits_right, valid, unit_ids, attention, dtype=torch.float32)
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros_like(query)
    value = torch.tensor([[[[0.0], [1.0], [10.0]]]])
    left = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=left_bias)
    right = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=right_bias)
    assert left[0, 0, 0, 0] < right[0, 0, 0, 0]
    left[0, 0, 0, 0].backward()
    assert logits_left.grad is not None and torch.isfinite(logits_left.grad).all()


def test_evidence_allocation_has_no_sentence_length_prior() -> None:
    logits = torch.zeros(1, 2)
    valid = torch.tensor([[True, True]])
    unit_ids = torch.tensor([[1, 1, 2]])
    attention = torch.ones_like(unit_ids)
    bias, source_keys = unit_evidence_token_bias(logits, valid, unit_ids, attention)
    assert bool(source_keys.all())
    unit_one_mass = bias[0, :2].exp().sum()
    unit_two_mass = bias[0, 2:].exp().sum()
    torch.testing.assert_close(unit_one_mass, unit_two_mass)


def test_decoder_salience_bias_uses_same_length_normalized_allocation() -> None:
    bridge = EvidenceBridge(4, 4, {"salience_gate_init": 0.5, "salience_bias_scale": 1.0})
    output = bridge(
        torch.randn(1, 3, 4),
        torch.ones(1, 3, dtype=torch.long),
        torch.tensor([[1, 1, 2]]),
        torch.zeros(1, 2),
        torch.tensor([[True, True]]),
        None,
    )
    assert output.attention_bias is not None
    assert output.attention_bias[0, 0] == output.attention_bias[0, 1]
    assert output.attention_bias[0, 0] < output.attention_bias[0, 2]


def test_decoder_bias_normalizes_prefix_and_eos_as_one_neutral_unit() -> None:
    bridge = EvidenceBridge(4, 4, {"salience_gate_init": 0.5, "salience_bias_scale": 1.0})
    output = bridge(
        torch.randn(1, 6, 4),
        torch.tensor([[1, 1, 1, 1, 1, 0]]),
        torch.tensor([[0, 0, 1, 1, 2, 0]]),
        torch.zeros(1, 2),
        torch.tensor([[True, True]]),
        None,
    )
    assert output.attention_bias is not None
    gate = torch.tensor(0.5)
    ungated = output.attention_bias.float() / gate
    neutral_mass = ungated[0, :2].exp().sum()
    unit_one_mass = ungated[0, 2:4].exp().sum()
    unit_two_mass = ungated[0, 4:5].exp().sum()
    torch.testing.assert_close(neutral_mass, unit_one_mass)
    torch.testing.assert_close(neutral_mass, unit_two_mass)
    assert output.attention_bias[0, 5] == 0


def test_sdpa_masks_keep_padding_out_of_both_views() -> None:
    padding = torch.tensor([[1, 1, 1, 0]])
    causal = sdpa_mask(padding, True, 4)
    full = sdpa_mask(padding, False, 4)
    assert causal.shape == (1, 1, 4, 4)
    assert full.shape == (1, 1, 1, 4)
    assert not bool(causal[..., 3].any())
    assert not bool(full[..., 3].any())
    assert bool(causal[0, 0, 2, 0]) and not bool(causal[0, 0, 0, 2])


def test_gold_labels_change_only_auxiliary_loss_not_memory_or_bias() -> None:
    bridge = EvidenceBridge(8, 8, {"salience_gate_init": 0.1, "salience_bias_scale": 1.0})
    memory = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.long)
    unit_ids = torch.tensor([[1, 1, 2, 2]])
    logits = torch.tensor([[2.0, -1.0]])
    valid = torch.tensor([[True, True]])
    first = bridge(memory, mask, unit_ids, logits, valid, torch.tensor([[1.0, 0.0]]))
    second = bridge(memory, mask, unit_ids, logits, valid, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(first.memory, second.memory, rtol=0, atol=0)
    torch.testing.assert_close(first.attention_bias, second.attention_bias, rtol=0, atol=0)
    assert float(first.loss_salience) != float(second.loss_salience)


def test_encoder_interface_parameters_get_warmup_lr_group() -> None:
    for name in (
        "encoder.evidence_norm.weight",
        "encoder.evidence_head.3.weight",
        "encoder.evidence_view_gate",
        "encoder.generic_token_gate.weight",
        "alignment_head.source_projection.1.weight",
    ):
        assert _parameter_component(name) == "adapter"
    assert _parameter_component("encoder.model.layers.0.mlp.up_proj.weight") == "encoder"
    assert _parameter_component("decoder.model.layers.0.cross_attn.q_proj.weight") == "cross_attention"


def test_forbidden_objectives_fail_closed() -> None:
    config = load_config(ROOT / "configs" / "wikilingua.yaml")
    for name in (
        "source_swap_weight",
        "response_alignment_weight",
        "phrase_mixture_weight",
        "label_smoothing",
    ):
        broken = copy.deepcopy(config)
        broken["objectives"][name] = 0.1
        with pytest.raises(
            ValueError, match="EviSeq V2 permits CE, salience, evidence contrastive, and optional document InfoNCE only"
        ):
            validate_config(broken)


def test_main_v2_uses_hard_evidence_instead_of_saturated_document_retrieval() -> None:
    config = load_config(ROOT / "configs" / "wikilingua.yaml")
    assert config["bridge"]["salience_ranking_weight"] == 0.25
    objectives = config["objectives"]
    assert objectives["use_contrastive"] is False
    assert objectives["contrastive_weight"] == 0.0
    assert objectives["contrastive_across_accumulation"] is False
    assert objectives["use_evidence_contrastive"] is True
    assert objectives["evidence_contrastive_weight"] == 0.10
    assert objectives["evidence_contrastive_temperature"] == 0.07
    assert objectives["evidence_contrastive_projection_size"] == 256
    assert objectives["evidence_hard_negatives"] == 4
    assert objectives["evidence_hard_negative_salience_boost"] == 0.10
    assert objectives["evidence_contrastive_warmup_epochs"] == 1
    assert "source_swap_weight" not in objectives
    assert "response_alignment_weight" not in objectives
    assert "phrase_mixture_weight" not in objectives
    no_contrastive = load_config(ROOT / "configs" / "ablations" / "c3_no_contrastive.yaml")
    assert no_contrastive["native_attention"]["variant"] == "evidence"
    assert no_contrastive["objectives"]["use_evidence_contrastive"] is False
    assert no_contrastive["objectives"]["evidence_contrastive_weight"] == 0.0


def test_prompt_state_precedes_all_teacher_forced_summary_tokens() -> None:
    states = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    labels = torch.tensor([[-100, -100, 7, 8, 9], [-100, 4, 5, 6, -100]])
    actual = last_prompt_states(states, labels)
    torch.testing.assert_close(actual, torch.stack((states[0, 2], states[1, 1])))


def test_prompt_source_info_nce_prefers_matching_pairs_and_backpropagates() -> None:
    source = torch.eye(4, requires_grad=True)
    matching = source.clone()
    permuted = matching.roll(1, dims=0)
    matched_loss, accuracy = info_nce_loss(source, matching, temperature=0.07)
    mismatched_loss, _ = info_nce_loss(source, permuted, temperature=0.07)
    assert matched_loss < mismatched_loss
    assert float(accuracy) == 1.0
    matched_loss.backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_evidence_contrastive_keeps_multiple_positives_and_backpropagates() -> None:
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    keys = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    good = evidence_info_nce_loss(query, torch.nn.functional.normalize(keys, dim=-1), labels, valid)
    bad_keys = keys.detach().clone()
    bad_keys[:, :2] = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    bad = evidence_info_nce_loss(
        query.detach(),
        torch.nn.functional.normalize(bad_keys, dim=-1),
        labels,
        valid,
    )
    assert good["evidence_contrastive_loss"] < bad["evidence_contrastive_loss"]
    good["evidence_contrastive_loss"].backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert keys.grad is not None and torch.isfinite(keys.grad).all()
    assert torch.count_nonzero(keys.grad[:, :2]).item() > 0


def test_salience_boost_prioritises_false_positive_hard_negative() -> None:
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[[1.0, 0.0], [0.9, 0.0], [0.85, 0.0]]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    salience = torch.tensor([[0.0, -10.0, 10.0]])
    negatives, positives = mine_hard_negatives(
        query,
        keys,
        labels,
        valid,
        num_hard_negatives=1,
        salience_logits=salience,
        salience_boost=0.1,
    )
    assert positives[0].tolist() == [0]
    assert negatives[0].tolist() == [2]


def test_total_objective_backpropagates_ce_salience_and_contrastive_once() -> None:
    torch.manual_seed(17)
    model = _toy_objective_model()
    model.train()

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        unit_ids=torch.tensor([[1, 1, 2], [1, 2, 2]]),
        evidence_labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        decoder_input_ids=torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]]),
        decoder_attention_mask=torch.ones(2, 4, dtype=torch.long),
        labels=torch.tensor([[-100, 15, 16, 17], [-100, 18, 19, 20]]),
    )
    expected = outputs["loss_ce"] + outputs["weighted_salience"] + outputs["weighted_contrastive"]
    torch.testing.assert_close(outputs["loss"], expected)
    outputs["loss"].backward()

    gradients = (
        model.decoder.lm_head.weight.grad,
        model.encoder.embedding.weight.grad,
        model.encoder.evidence_head.weight.grad,
        model.adapter.salience_attention_gate.grad,
        model.alignment_head.source_projection[-1].weight.grad,
        model.alignment_head.prompt_projection[-1].weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_gradcache_matches_full_virtual_batch_contrastive_gradient() -> None:
    torch.manual_seed(29)
    first = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "unit_ids": torch.tensor([[1, 1, 2], [1, 2, 2]]),
        "evidence_labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "decoder_input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
        "decoder_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 13, 14], [-100, 15, 16]]),
    }
    second = {
        "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "unit_ids": torch.tensor([[1, 2, 2], [1, 1, 2]]),
        "evidence_labels": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "decoder_input_ids": torch.tensor([[17, 18, 19], [20, 21, 22]]),
        "decoder_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 23, 24], [-100, 25, 26]]),
    }
    batches = [first, second]
    full_model = _toy_objective_model().train()
    cached_model = copy.deepcopy(full_model).train()
    full_batch = {name: torch.cat([batch[name] for batch in batches], dim=0) for name in first}

    full_representations = full_model(
        **full_batch,
        contrastive_mode="representations_only",
    )
    full_loss, _ = info_nce_loss(
        full_representations["source_repr"],
        full_representations["prompt_repr"],
        full_model.contrastive_temperature,
        duplicate_mask=exact_duplicate_mask(full_batch["input_ids"], full_batch["attention_mask"]),
    )
    (full_model.contrastive_weight * full_loss).backward()

    cache = _build_virtual_contrastive_cache(
        cached_model,
        batches,
        torch.device("cpu"),
        {"bf16": False, "fp16": False},
        scale=1.0,
    )
    assert cache["effective_batch_size"] == 4
    assert all(value.dtype == torch.float32 and not value.requires_grad for value in cache["source_gradients"])
    assert all(value.dtype == torch.float32 and not value.requires_grad for value in cache["prompt_gradients"])
    assert _virtual_duplicate_mask(batches).shape == (4, 4)
    for index, batch in enumerate(batches):
        _restore_rng_state(cache["rng_states"][index], torch.device("cpu"))
        representations = cached_model(**batch, contrastive_mode="deferred")
        surrogate = (representations["source_repr"] * cache["source_gradients"][index]).sum() + (
            representations["prompt_repr"] * cache["prompt_gradients"][index]
        ).sum()
        surrogate.backward()

    cached_parameters = dict(cached_model.named_parameters())
    compared = 0
    for name, parameter in full_model.named_parameters():
        if parameter.grad is None:
            continue
        assert cached_parameters[name].grad is not None
        torch.testing.assert_close(cached_parameters[name].grad, parameter.grad, rtol=2.0e-5, atol=2.0e-6)
        compared += 1
    assert compared > 4


def test_virtual_duplicate_mask_handles_cross_microbatch_dynamic_padding() -> None:
    first = {
        "input_ids": torch.tensor([[4, 5, 0], [8, 9, 7]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    second = {
        "input_ids": torch.tensor([[4, 5, 99, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0, 0]]),
    }
    duplicates = _virtual_duplicate_mask([first, second])
    assert duplicates.tolist() == [
        [False, False, True],
        [False, False, False],
        [True, False, False],
    ]


def test_gradcache_handles_a_one_example_final_window_and_replays_rng() -> None:
    model = _toy_objective_model().train()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "unit_ids": torch.tensor([[1, 1, 2]]),
        "evidence_labels": torch.tensor([[1.0, 0.0]]),
        "decoder_input_ids": torch.tensor([[4, 5, 6]]),
        "decoder_attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, 7, 8]]),
    }
    torch.manual_seed(101)
    cache = _build_virtual_contrastive_cache(
        model,
        [batch],
        torch.device("cpu"),
        {"bf16": False, "fp16": False},
        scale=1.0,
    )
    assert cache["effective_batch_size"] == 1
    assert cache["loss"] == 0
    assert not bool(cache["source_gradients"][0].bool().any())
    assert not bool(cache["prompt_gradients"][0].bool().any())

    state = cache["rng_states"][0]
    _restore_rng_state(state, torch.device("cpu"))
    first_mask = torch.nn.functional.dropout(torch.ones(32), p=0.5, training=True)
    _restore_rng_state(state, torch.device("cpu"))
    replay_mask = torch.nn.functional.dropout(torch.ones(32), p=0.5, training=True)
    torch.testing.assert_close(first_mask, replay_mask, rtol=0, atol=0)


def test_salience_log_uses_micro_aggregated_counts() -> None:
    precision, recall, f1 = _salience_scores(
        {
            "salience_tp": 3.0,
            "salience_predicted_count": 4.0,
            "salience_gold_count": 6.0,
        }
    )
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.6)


def test_long_source_mean_pool_accumulates_in_fp32() -> None:
    states = torch.full((2, 4096, 4), 0.125, dtype=torch.bfloat16)
    mask = torch.ones(2, 4096, dtype=torch.long)
    pooled = masked_mean_pool(states, mask)
    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, torch.full((2, 4), 0.125))


def test_duplicate_sources_are_not_false_negatives() -> None:
    ids = torch.tensor([[1, 2, 0], [1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]])
    duplicates = exact_duplicate_mask(ids, mask)
    assert duplicates.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    head = SourcePromptAlignmentHead(4, projection_size=4, pooling="mean")
    memory = torch.randn(3, 3, 4)
    prompt = torch.randn(3, 4)
    representations = head(memory, mask, prompt)
    loss, _ = info_nce_loss(
        representations["source_repr"],
        representations["prompt_repr"],
        temperature=0.1,
        duplicate_mask=duplicates,
    )
    assert torch.isfinite(loss)


def test_pretrained_encoder_controls_keep_one_decoder_and_one_memory() -> None:
    pplx = load_config(ROOT / "configs" / "encoders" / "pplx_0_6b.yaml")
    nemo = load_config(ROOT / "configs" / "encoders" / "nemotron_1b.yaml")
    for config in (pplx, nemo):
        assert config["native_attention"]["backend"] == "pretrained_native"
        assert config["native_attention"]["variant"] == "pretrained"
    assert pplx["decoder"]["cross_attention_every"] == nemo["decoder"]["cross_attention_every"] == 1
    assert pplx["decoder"]["memory_bank_count"] == nemo["decoder"]["memory_bank_count"] == 1
    assert pplx["model"]["decoder_name"] == nemo["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"


def test_pplx_pubmed_changes_only_the_encoder_specific_contract() -> None:
    config = load_config(ROOT / "configs" / "encoders" / "pplx_0_6b_pubmed.yaml")
    assert config["model"]["encoder_name"] == "perplexity-ai/pplx-embed-v1-0.6b"
    assert config["model"]["decoder_name"] == "Qwen/Qwen3-0.6B"
    assert config["native_attention"] == {
        "backend": "pretrained_native",
        "variant": "pretrained",
    }
    assert config["training"]["interface_warmup_epochs"] == 1
    assert config["training"]["full_finetune_epochs"] == 3
    assert config["training"]["batch_size"] == 96
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["length_bucketing"] is True
    assert config["data"]["max_source_length"] == 4096
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["source_prefix"] == ""


def test_pubmed_uses_one_plus_three_plus_two_and_five_candidates() -> None:
    config = load_config(ROOT / "configs" / "pubmed.yaml")
    training = config["training"]
    ranking = config["ranking"]
    assert training["interface_warmup_epochs"] == 1
    assert training["full_finetune_epochs"] == 3
    assert training["ranking_finetune_epochs"] == 2
    assert training["ranking_batch_size"] == 8
    assert training["ranking_gradient_accumulation_steps"] == 4
    assert ranking["enabled"] is True
    assert ranking["num_candidates"] == 5
    assert ranking["max_examples"] == 40_000
    assert ranking["generation_batch_size"] == 8
    assert ranking["candidate_min_new_tokens"] == 32
    assert ranking["candidate_max_new_tokens"] == 384
    assert ranking["candidate_scoring_chunk_size"] == 40
    assert ranking["sampling_top_k"] == 64
    assert ranking["quality_weights"] == {"r1": 0.0, "r2": 0.6, "rL": 0.4}


def test_regular_and_ranking_stages_use_their_own_accumulation_settings() -> None:
    source = (ROOT / "eviseq_v2" / "training.py").read_text(encoding="utf-8")
    regular_stage = source.split("def _run_stage(", 1)[1].split("def _run_ranking_stage(", 1)[0]
    ranking_stage = source.split("def _run_ranking_stage(", 1)[1].split("def _parameter_component(", 1)[0]
    assert 'training.get("gradient_accumulation_steps", 1)' in regular_stage
    assert "ranking_gradient_accumulation_steps" not in regular_stage
    assert 'training.get("ranking_gradient_accumulation_steps", 1)' in ranking_stage


def test_ranked_checkpoint_reuses_phase2_resolved_manifest() -> None:
    source = (ROOT / "eviseq_v2" / "training.py").read_text(encoding="utf-8")
    phase3 = source.split("last_path = stable.train", 1)[1]
    reload_position = phase3.index("config = load_config(phase2_config_path)")
    save_position = phase3.index("save_last_checkpoint(model, ranked_last_path, config")
    assert reload_position < save_position
    assert "generate_candidates(\n                str(phase2_config_path)," in phase3


def test_ranking_training_fields_do_not_change_deployed_architecture_contract() -> None:
    config = load_config(ROOT / "configs" / "wikilingua.yaml")
    changed = copy.deepcopy(config)
    changed["ranking"]["enabled"] = True
    changed["ranking"]["candidates_file"] = "/different/training/artifact.jsonl"
    assert architecture_contract(changed) == architecture_contract(config)


def test_vectorized_pairwise_ranking_masks_padding_and_backpropagates() -> None:
    scores = torch.tensor(
        [[0.9, 0.4, 99.0], [0.1, 0.2, 0.3]],
        requires_grad=True,
    )
    quality = torch.tensor([[30.0, 20.0, 999.0], [30.0, 20.0, 10.0]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    loss, accuracy, pair_count = pairwise_ranking_loss(
        scores,
        quality,
        margin=0.01,
        minimum_quality_gap=0.5,
        valid_mask=valid,
    )
    assert pair_count.item() == 4
    assert accuracy.item() == pytest.approx(0.5)
    assert loss.item() > 0.0
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 2].item() == 0.0
    assert torch.isfinite(scores.grad).all()


def test_vectorized_pairwise_ranking_excludes_documents_without_quality_gap() -> None:
    scores = torch.tensor([[0.1, 0.2], [0.8, 0.1]], requires_grad=True)
    quality = torch.tensor([[10.0, 10.0], [20.0, 10.0]])
    loss, accuracy, pair_count = pairwise_ranking_loss(
        scores,
        quality,
        minimum_quality_gap=0.5,
    )
    assert pair_count.item() == 1
    assert accuracy.item() == 1.0
    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None


def test_phase3_forward_scores_candidate_chunks_and_reaches_encoder_gradient() -> None:
    torch.manual_seed(29)
    model = _toy_objective_model()
    model.ranking_weight = 0.15
    model.ranking_margin = 0.01
    model.ranking_length_penalty_alpha = 1.0
    model.ranking_minimum_quality_gap = 0.5
    model.ranking_candidate_chunk_size = 2
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention = torch.ones_like(input_ids)
    unit_ids = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
    evidence = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference_input = torch.tensor([[1, 5, 6, 7], [1, 7, 6, 5]])
    reference_labels = torch.tensor([[-100, 5, 6, 7], [-100, 7, 6, 5]])
    candidates = torch.tensor(
        [
            [[1, 5, 6, 7], [1, 5, 7, 6], [1, 7, 6, 5]],
            [[1, 7, 6, 5], [1, 7, 5, 6], [0, 0, 0, 0]],
        ]
    )
    candidate_labels = candidates.clone()
    candidate_labels[:, :, 0] = -100
    candidate_labels[1, 2] = -100
    candidate_mask = candidates.ne(0).long()
    outputs = model.forward_ranking(
        input_ids=input_ids,
        attention_mask=attention,
        unit_ids=unit_ids,
        evidence_labels=evidence,
        reference_decoder_input_ids=reference_input,
        reference_decoder_attention_mask=torch.ones_like(reference_input),
        reference_labels=reference_labels,
        candidate_decoder_input_ids=candidates,
        candidate_decoder_attention_mask=candidate_mask,
        candidate_labels=candidate_labels,
        candidate_quality_scores=torch.tensor([[30.0, 20.0, 10.0], [25.0, 15.0, 0.0]]),
    )
    assert outputs["candidate_pair_count"].item() == 4
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert model.encoder.embedding.weight.grad is not None
    assert model.decoder.memory_projection.weight.grad is not None
    assert torch.isfinite(model.encoder.embedding.weight.grad).all()


def test_candidate_sampling_reuses_memory_and_returns_requested_count() -> None:
    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prepared = 0
            self.cleared = 0

        def prepare_cross_attention_cache(self, memory: torch.Tensor) -> None:
            self.prepared += 1
            self.cached_rows = memory.shape[0]

        def clear_cross_attention_cache(self) -> None:
            self.cleared += 1

        def forward(self, input_ids: torch.Tensor, **kwargs):
            del kwargs
            return torch.nn.functional.one_hot(input_ids.remainder(4), num_classes=4).float(), ()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = Decoder()
            self.output = nn.Linear(4, 8, bias=False)

        @property
        def lm_head(self):
            return self.output

    model = Model()
    bridge = BridgeOutput(
        memory=torch.randn(2, 3, 4),
        memory_mask=torch.ones(2, 3, dtype=torch.long),
        attention_bias=None,
        salience_logits=None,
        loss_salience=torch.tensor(0.0),
    )
    sampled = generate_sampled_from_memory(
        model,
        bridge,
        [1],
        max_new_tokens=2,
        min_new_tokens=0,
        pad_token_id=0,
        temperature=0.9,
        top_k=2,
        top_p=0.95,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=3,
    )
    assert sampled.shape == (2, 3, 2)
    assert model.decoder.cached_rows == 6
    assert model.decoder.prepared == model.decoder.cleared == 1


def test_offline_candidate_generation_disables_autograd_and_retries_cuda_oom() -> None:
    source = (ROOT / "eviseq_v2" / "generate_candidates.py").read_text(encoding="utf-8")
    assert "@torch.inference_mode()\ndef generate_candidates(" in source
    assert "except torch.cuda.OutOfMemoryError:" in source
    assert "active_batch_size = max(1, active_batch_size // 2)" in source


def test_length_bucket_sampler_and_single_batch_cache_shortcut() -> None:
    lengths = list(range(1, 17))
    batches = list(
        LengthBucketBatchSampler(
            lengths,
            batch_size=4,
            seed=7,
            bucket_size_multiplier=4,
        )
    )
    assert sorted(index for batch in batches for index in batch) == list(range(16))
    assert all(
        max(lengths[index] for index in batch) - min(lengths[index] for index in batch) <= 3 for batch in batches
    )

    class Objective:
        alignment_head = object()
        contrastive_across_accumulation = True

    model = Objective()
    assert _uses_virtual_gradcache(model, 1) is False
    assert _uses_virtual_gradcache(model, 2) is True


def test_run_script_has_no_upload_or_push_operation() -> None:
    script = (ROOT / "run.sh").read_text(encoding="utf-8").lower()
    assert "push_to_hub" not in script
    assert "huggingface-cli upload" not in script
    assert "git push" not in script
    assert "--paper-test" in script
    assert '[[ ! -f "$output_dir/complete" ]]' in script
    assert '[[ -f "$output_dir/ranking/complete" ]]' in script
    assert "for required in last.pt resolved_config.yaml data_manifest.json parameter_manifest.json" in script


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_data_audit_detects_content_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write(train, [{"id": "1", "source": "same source", "target": "a"}])
    _write(validation, [{"id": "2", "source": " SAME   SOURCE ", "target": "b"}])
    _write(test, [{"id": "3", "source": "other", "target": "c"}])
    config = {
        "data": {
            "train_file": str(train),
            "validation_file": str(validation),
            "test_file": str(test),
        },
        "limits": {},
    }
    with pytest.raises(ValueError, match="cross-split"):
        audit(config)

    config["data_integrity"] = {"fail_on_cross_split": False}
    report = audit(config)
    assert report["passed"] is False
    assert report["fail_on_cross_split"] is False
    assert report["cross_split"]["train__validation"]["source"] == 1


def test_validation_is_default_and_test_requires_explicit_gate() -> None:
    source = (ROOT / "eviseq_v2" / "evaluate.py").read_text(encoding="utf-8")
    assert 'split: str = "validation"' in source
    assert "pass --paper-test" in source
