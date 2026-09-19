"""Train-only evidence supervision must reach the bridge and preserve inference."""

import copy
from pathlib import Path

import pytest
import torch

from eviseq_afmr.config import load_config, validate_config
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import architecture_spec
from eviseq_afmr.training.engine import AFMRTrainer, _salience_batch_stats


def _config():
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["training"]["salience_loss_weight"] = 0.2
    config["model"]["gradient_checkpointing"] = False
    return config


def _forward(model, batch):
    return model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        batch["decoder_input_ids"],
        batch["decoder_attention_mask"],
        batch["labels"],
        return_logits=False,
        source_salience_labels=batch["source_salience_labels"],
        source_salience_mask=batch["source_salience_mask"],
    )


def test_auxiliary_loss_directly_updates_focus_and_never_uses_gold_at_inference():
    torch.manual_seed(19)
    config = _config()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    assert batch["source_salience_mask"].all()
    model = EviSeqAFMR(config)
    output = _forward(model, batch)
    assert output.loss_salience is not None and output.loss_salience.item() > 0
    torch.testing.assert_close(output.loss, output.loss_ce + 0.2 * output.loss_salience)
    salience_grad = torch.autograd.grad(output.loss_salience, model.bridge.focus_output.weight, retain_graph=True)[0]
    assert salience_grad.abs().sum() > 0
    before = model.bridge.focus_output.weight.detach().clone()
    output.loss.backward()
    assert model.bridge.focus_output.weight.grad is not None
    assert model.bridge.focus_output.weight.grad.abs().sum() > 0
    torch.optim.SGD(model.bridge.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.bridge.focus_output.weight)

    with torch.no_grad():
        no_reference = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            batch["decoder_input_ids"],
            batch["decoder_attention_mask"],
            labels=None,
            return_logits=False,
        )
    assert no_reference.loss is None and no_reference.loss_salience is None


def test_salience_loss_excludes_invalid_rows_from_its_denominator():
    source_content = torch.ones(2, 4, dtype=torch.bool)
    salience_labels = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    supplied_mask = torch.tensor([True, False])
    source_bias = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    decoder_labels = torch.ones(2, 5, dtype=torch.long)
    mixed = EviSeqAFMR._salience_ranking_loss(
        source_bias, source_content, salience_labels, supplied_mask, decoder_labels, margin=0.5
    )
    valid_only = EviSeqAFMR._salience_ranking_loss(
        source_bias[:1], source_content[:1], salience_labels[:1], supplied_mask[:1], decoder_labels[:1], margin=0.5
    )
    torch.testing.assert_close(mixed, valid_only)
    assert _salience_batch_stats(
        {
            "source_content_mask": source_content,
            "source_salience_labels": salience_labels,
            "source_salience_mask": supplied_mask,
            "labels": decoder_labels,
        }
    ) == (1, 4)


def test_salience_objective_changes_checkpoint_contract():
    config = _config()
    without = copy.deepcopy(config)
    without["training"]["salience_loss_weight"] = 0.0
    evidence_spec = architecture_spec(config)["evidence_prior_training"]
    assert evidence_spec["loss"] == "valid_token_weighted_pairwise_rank"
    assert evidence_spec["labels"] == "visible_source_sentence_bounded_phrase_length_weight"
    assert "evidence_prior_training" not in architecture_spec(without)
    direct = copy.deepcopy(config)
    direct["architecture"]["bridge_mode"] = "direct_projection"
    with pytest.raises(ValueError, match="salience_loss_weight requires"):
        validate_config(direct)


def test_width_adapter_is_identical_in_matched_full_and_direct_initialization():
    config = _config()["architecture"]
    torch.manual_seed(31)
    full = AdaptiveFullMemoryResidualBridge(24, 32, config)
    full_rng = torch.get_rng_state()
    direct_config = copy.deepcopy(config)
    direct_config["bridge_mode"] = "direct_projection"
    torch.manual_seed(31)
    direct = AdaptiveFullMemoryResidualBridge(24, 32, direct_config)
    torch.testing.assert_close(full.base_projection.weight, direct.direct_projection.weight)
    torch.testing.assert_close(full_rng, torch.get_rng_state())


def test_full_and_direct_start_with_identical_logits_for_matched_seed():
    config = _config()
    config["training"]["salience_loss_weight"] = 0.0
    # This test isolates the legacy zero-initialized AFMR paths. The evidence
    # slots intentionally use a tiny nonzero output so their complete route
    # receives gradients on the first optimizer step.
    config["architecture"]["evidence_slots"]["enabled"] = False
    direct_config = copy.deepcopy(config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    torch.manual_seed(321)
    full = EviSeqAFMR(config).eval()
    full_rng = torch.get_rng_state()
    torch.manual_seed(321)
    direct = EviSeqAFMR(direct_config).eval()
    torch.testing.assert_close(full_rng, torch.get_rng_state())
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    tensor_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        full_output = full(**tensor_inputs)
        direct_output = direct(**tensor_inputs)
    torch.testing.assert_close(full_output.logits, direct_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(full_output.loss, direct_output.loss, rtol=0, atol=0)


def test_trained_source_prior_changes_decoder_logits(monkeypatch):
    torch.manual_seed(47)
    config = _config()
    model = EviSeqAFMR(config).eval()
    batch = next(iter(build_loaders(config, max_train_examples=2)["train"]))
    with torch.no_grad():
        model.bridge.focus_output.weight.normal_(std=0.5)

        def logits():
            return model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["source_content_mask"],
                batch["decoder_prompt_ids"],
                batch["decoder_prompt_mask"],
                batch["decoder_input_ids"],
                batch["decoder_attention_mask"],
                labels=None,
            ).logits

        with_prior = logits()
        monkeypatch.setattr(
            model.bridge,
            "_focus_prior",
            lambda memory, content, controller: memory.new_zeros(content.shape),
        )
        without_prior = logits()
    assert not torch.equal(with_prior, without_prior)


def test_salience_accumulation_matches_large_batch(tmp_path):
    config = _config()
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation_steps"] = 2
    config["training"]["log_every_steps"] = 100
    config["experiment"]["output_dir"] = str(tmp_path / "micro")
    torch.manual_seed(21)
    micro_model = EviSeqAFMR(config)
    large_model = copy.deepcopy(micro_model)
    micro_loader = [copy.deepcopy(batch) for batch in build_loaders(config, max_train_examples=2)["train"]]
    assert len(micro_loader) == 2
    # Keep one row supervised and make the other row an explicitly invalid
    # silver-label example. The accumulated and large-batch paths must still
    # produce the same valid-token-normalized objective.
    for batch in micro_loader:
        for row, example_id in enumerate(batch["ids"]):
            if example_id == "train-0":
                batch["source_salience_mask"][row] = False
    micro_trainer = AFMRTrainer(micro_model, config, "cpu")
    micro_metrics = micro_trainer._run_epoch(
        micro_loader, torch.optim.SGD(micro_model.parameters(), lr=0.01), "full_finetune", True
    )

    large_config = copy.deepcopy(config)
    large_config["training"]["batch_size"] = 2
    large_config["training"]["gradient_accumulation_steps"] = 1
    large_config["experiment"]["output_dir"] = str(tmp_path / "large")
    large_loader = [copy.deepcopy(batch) for batch in build_loaders(large_config, max_train_examples=2)["train"]]
    assert len(large_loader) == 1
    for row, example_id in enumerate(large_loader[0]["ids"]):
        if example_id == "train-0":
            large_loader[0]["source_salience_mask"][row] = False
    large_trainer = AFMRTrainer(large_model, large_config, "cpu")
    large_metrics = large_trainer._run_epoch(
        large_loader, torch.optim.SGD(large_model.parameters(), lr=0.01), "full_finetune", True
    )
    assert micro_metrics["salience_coverage"] == large_metrics["salience_coverage"] == 0.5
    assert micro_metrics["salience_token_coverage"] == large_metrics["salience_token_coverage"]
    for (_, a), (_, b) in zip(micro_model.named_parameters(), large_model.named_parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
