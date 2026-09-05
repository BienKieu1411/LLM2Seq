import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from eviseq_afmr.config import load_config
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.dataset import JsonlSummarizationDataset
from eviseq_afmr.data.prepare import prepare_split
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.evaluation.generate import generate_greedy
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.modeling.outputs import EncoderState
from eviseq_afmr.runtime import _TinyTokenizer, build_loaders, evaluate
from eviseq_afmr.training.engine import AFMRTrainer, seed_everything
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability


def config():
    return load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")


def test_content_windows_invariant_to_prefix_padding_and_batch():
    torch.manual_seed(71)
    cfg = config()["architecture"]
    bridge = AdaptiveFullMemoryResidualBridge(24, 24, cfg).eval()
    with torch.no_grad():
        bridge.focus_output.weight.normal_()
        bridge.feature_up.weight.normal_()
        bridge.depth_out.weight.normal_()
        bridge.depth_content_score.weight.normal_()
        bridge.depth_router.weight.normal_()
    article = torch.randn(1, 13, 24)
    taps = article + torch.randn_like(article)
    prompt = torch.randn(1, 3, 24)

    def run(prefix, padding, batch_size):
        length = prefix + 13 + padding
        states = torch.randn(batch_size, length, 24)
        lower = states.clone()
        states[0, prefix : prefix + 13] = article[0]
        lower[0, prefix : prefix + 13] = taps[0]
        mask = torch.ones(batch_size, length, dtype=torch.bool)
        content = mask.clone()
        content[0] = False
        content[0, prefix : prefix + 13] = True
        if padding:
            mask[0, -padding:] = False
        out = bridge(
            EncoderState(states, (lower, states), mask, content),
            prompt.expand(batch_size, -1, -1),
            torch.ones(batch_size, 3, dtype=torch.bool),
            torch.full((batch_size,), 32.0),
        )
        return out.source_bias[0, prefix : prefix + 13], out.memory[0, prefix : prefix + 13]

    expected_bias, expected_memory = run(0, 0, 1)
    assert expected_bias.abs().max() > 1e-4
    for prefix, padding, batch in ((0, 27, 2), (4, 2, 1), (7, 30, 2)):
        bias, memory = run(prefix, padding, batch)
        torch.testing.assert_close(bias, expected_bias, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(memory, expected_memory, atol=2e-6, rtol=1e-5)


def test_short_documents_have_neutral_focus_prior():
    bridge = AdaptiveFullMemoryResidualBridge(24, 24, config()["architecture"])
    memory = torch.randn(3, 12, 24)
    content = torch.zeros(3, 12, dtype=torch.bool)
    content[1, 2:6] = True
    content[2, :9] = True
    bias = bridge._focus_prior(memory, content, torch.randn(3, 16))
    assert not bias[:2].any()
    assert torch.isfinite(bias).all()


def test_depth_rank_is_actual_bottleneck():
    bridge = AdaptiveFullMemoryResidualBridge(24, 32, config()["architecture"])
    assert bridge.depth_down.weight.shape == (8, 24)
    assert bridge.depth_out.weight.shape == (24, 8)
    assert bridge.feature_down.weight.shape == (8, 24)
    assert bridge.feature_up.weight.shape == (32, 8)


def test_depth_readout_is_tokenwise_and_reads_each_candidate():
    bridge = AdaptiveFullMemoryResidualBridge(24, 24, config()["architecture"])
    controller = torch.zeros(1, bridge.controller_dim)
    first = torch.zeros(1, 2, 24)
    last = torch.zeros_like(first)
    first[0, 0, 0] = 1.0
    last[0, 1, 0] = 1.0
    normalized = [norm(value) for norm, value in zip(bridge.depth_norms, (first, last))]
    weights = bridge._depth_weights(normalized, controller)
    assert weights.shape == (1, 2, 2)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    with torch.no_grad():
        bridge.depth_content_score.weight[0, 0] = 1.0
    weights = bridge._depth_weights(normalized, controller)
    assert weights[0, 0, 0] > weights[0, 0, 1]
    assert weights[0, 1, 1] > weights[0, 1, 0]
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 2))
    changed = [normalized[0].clone(), normalized[1]]
    changed[0][:, 0] = 0
    changed_weights = bridge._depth_weights(changed, controller)
    assert not torch.equal(changed_weights[:, 0], weights[:, 0])
    torch.testing.assert_close(changed_weights[:, 1], weights[:, 1])


def test_main_recipes_are_ce_only_without_allocation_metadata():
    root = Path(__file__).parents[1]
    for name in ("base", "pubmed", "arxiv", "cnndm", "wikilingua", "smoke"):
        cfg = load_config(root / f"configs/afmr_{name}.yaml")
        assert "objective" not in cfg
    batch = next(iter(build_loaders(config())["train"]))
    assert not any(key.startswith("allocation") for key in batch)


def test_ce_only_updates_focus_without_any_positive_labels():
    torch.manual_seed(5)
    cfg = config()
    model = EviSeqAFMR(cfg)
    set_stage_trainability(model, "interface_warmup")
    batch = next(iter(build_loaders(cfg)["train"]))
    assert "allocation_target" not in batch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["source_content_mask"],
            batch["decoder_prompt_ids"],
            batch["decoder_prompt_mask"],
            batch["decoder_input_ids"],
            batch["decoder_attention_mask"],
            batch["labels"],
            return_logits=False,
        )
        out.loss.backward()
        torch.testing.assert_close(out.loss, out.loss_ce)
        assert model.bridge.focus_output.weight.grad.abs().sum() > 0
        if step:
            for p in (
                model.bridge.focus_query.weight,
                model.bridge.focus_key.weight,
                model.bridge.depth_down.weight,
                model.bridge.depth_content_score.weight,
                model.bridge.depth_router.weight,
                model.bridge.feature_down.weight,
            ):
                assert p.grad is not None and p.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.encoder.parameters())
        opt.step()


def test_full_checkpointed_encoder_taps_receive_gradient_without_hidden_dump():
    cfg = config()
    model = EviSeqAFMR(cfg).train()
    batch = next(iter(build_loaders(cfg)["train"]))
    state = model.encoder(batch["input_ids"], batch["attention_mask"], batch["source_content_mask"])
    assert model.encoder.model.config.output_hidden_states is False
    assert model.encoder.model.config.use_cache is False
    assert all(t.requires_grad for t in state.taps)
    assert all(not layer._forward_hooks for layer in model.encoder._tap_layers)
    loss = sum(t.square().mean() for t in state.taps)
    loss.backward()
    assert model.encoder.model.embed_tokens.weight.grad.abs().sum() > 0


def test_chunked_ce_matches_full_logits_gradient():
    torch.manual_seed(24)
    model = EviSeqAFMR(config()).eval()
    model.decoder.lm_head.weight.requires_grad_(True)
    model.decoder.ce_chunk_size = 3
    memory = torch.randn(2, 10, 24)
    mask = torch.ones(2, 10, dtype=torch.bool)
    bias = torch.randn(2, 10, requires_grad=True)
    tokens = torch.randint(3, 128, (2, 6))
    labels = tokens.clone()
    labels[:, :2] = -100
    labels[0, -2:] = -100
    _, _, full = model.decoder(tokens, memory, mask, bias, labels=labels)
    full.backward()
    expected = model.decoder.lm_head.weight.grad.clone()
    expected_bias = bias.grad.clone()
    model.zero_grad(set_to_none=True)
    bias.grad = None
    logits, _, chunked = model.decoder(tokens, memory, mask, bias, labels=labels, return_logits=False)
    chunked.backward()
    assert logits is None
    torch.testing.assert_close(chunked, full)
    torch.testing.assert_close(model.decoder.lm_head.weight.grad, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(bias.grad, expected_bias, atol=1e-6, rtol=1e-5)


def test_decoder_cached_matches_uncached_with_left_padding():
    torch.manual_seed(31)
    decoder = EviSeqAFMR(config()).decoder.eval()
    memory = torch.randn(2, 9, 24)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, -3:] = False
    bias = torch.randn(2, 9)
    tokens = torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]])
    attention = tokens.ne(0)
    with torch.no_grad():
        decoder.prepare_cross_cache(memory)
        _, cache, _ = decoder(tokens, memory, mask, bias, attention, use_cache=True)
        extended = torch.cat((tokens, torch.tensor([[10], [11]])), -1)
        full_mask = torch.cat((attention, torch.ones(2, 1, dtype=torch.bool)), -1)
        cached, _, _ = decoder(extended[:, -1:], memory, mask, bias, full_mask, past_key_values=cache, use_cache=True)
        decoder.clear_cross_cache()
        full, _, _ = decoder(extended, memory, mask, bias, full_mask)
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)


def test_min_length_masks_eos_instead_of_inserting_pad():
    class Decoder:
        def eval(self):
            pass

        def prepare_cross_cache(self, memory):
            pass

        def clear_cross_cache(self):
            pass

        def __call__(self, tokens, *args, **kwargs):
            scores = torch.tensor([0.0, 1.0, 10.0, 9.0]).expand(tokens.shape[0], 1, -1).clone()
            return scores, object(), None

    class Model:
        decoder = Decoder()

        def eval(self):
            pass

        def encode_source(self, *args):
            return SimpleNamespace(memory=None, memory_mask=None, source_bias=None)

    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2),
        "source_content_mask": torch.ones(1, 2),
        "decoder_prompt_ids": torch.ones(1, 1, dtype=torch.long),
        "decoder_prompt_mask": torch.ones(1, 1),
    }
    _, ids = generate_greedy(Model(), batch, _TinyTokenizer(), 4, 2)
    assert ids.tolist() == [[1, 3, 3, 2]]


def test_unicode_and_list_content_roundtrip(tmp_path):
    row = {
        "id": "x",
        "text": ["Dr. Jones measured blood. Twice.", "Result was normal."],
        "summary": "Result was normal.",
    }
    raw, prepared = tmp_path / "raw.jsonl", tmp_path / "train.jsonl"
    raw.write_text(json.dumps(row))
    prepare_split(raw, prepared)
    data = JsonlSummarizationDataset(prepared, {})
    assert data[0].source == "\n".join(row["text"])
    assert not hasattr(data, "records")
    assert JsonlSummarizationDataset(raw, {})[0] == data[0]


def test_full_source_offsets_and_truncation():
    tokenizer = _TinyTokenizer()
    data = {"encoder_prefix": "prefix ", "decoder_prompt": "sum", "max_source_length": 6, "max_target_length": 8}
    collator = SummarizationCollator(tokenizer, tokenizer, data)
    record = CanonicalRecord("x", "one two. three four five.", "one")
    batch = collator([record])
    expected = tokenizer("prefix " + record.source, add_special_tokens=True, truncation=True, max_length=6)
    assert batch["input_ids"][0].tolist() == expected["input_ids"]
    assert not batch["source_content_mask"][0, :2].any()
    assert batch["source_content_mask"][0, 2:].any()


def test_partial_accumulation_matches_large_batch():
    cfg = config()
    cfg["training"]["batch_size"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 4
    cfg["model"]["gradient_checkpointing"] = False
    torch.manual_seed(20)
    small = EviSeqAFMR(cfg)
    large = copy.deepcopy(small)
    small_loader = build_loaders(cfg, max_train_examples=3)["train"]
    big_loader = build_loaders(cfg, batch_size_override=3, max_train_examples=3)["train"]
    trainer = AFMRTrainer(small, cfg, "cpu")
    trainer._run_epoch(small_loader, torch.optim.SGD(small.parameters(), lr=0.01), "full_finetune", True)
    big_config = copy.deepcopy(cfg)
    big_config["training"]["gradient_accumulation_steps"] = 1
    big_trainer = AFMRTrainer(large, big_config, "cpu")
    big_trainer._run_epoch(big_loader, torch.optim.SGD(large.parameters(), lr=0.01), "full_finetune", True)
    for (_, a), (_, b) in zip(small.named_parameters(), large.named_parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert trainer.global_step == big_trainer.global_step == 1


def test_optimizer_exhaustive_groups_and_no_decay_norms():
    model = EviSeqAFMR(config())
    optimizer = build_optimizer(model, config(), "full_finetune")
    assigned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(assigned) == len(set(map(id, assigned))) == len(list(model.parameters()))
    for group in optimizer.param_groups:
        if any(p.ndim < 2 for p in group["params"]):
            assert group["weight_decay"] == 0


def test_training_log_is_compact_and_uses_accumulated_ce(caplog):
    cfg = config()
    cfg["training"]["log_every_steps"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 4
    cfg["model"]["gradient_checkpointing"] = False
    model = EviSeqAFMR(cfg)
    set_stage_trainability(model, "interface_warmup")
    trainer = AFMRTrainer(model, cfg, "cpu")
    trainer.epoch = 1
    optimizer = build_optimizer(model, cfg, "interface_warmup")
    loader = build_loaders(cfg, max_train_examples=3)["train"]
    with caplog.at_level("INFO", logger="eviseq_afmr.train"):
        metrics = trainer._run_epoch(loader, optimizer, "interface_warmup", True)
    messages = [record.getMessage() for record in caplog.records if record.name == "eviseq_afmr.train"]
    assert len(messages) == 1
    fields = dict(field.strip("|").split("=", 1) for field in messages[0].split() if "=" in field)
    assert {"stage", "epoch", "epoch_progress", "step", "total_step", "CE", "grad", "lr", "elapsed"} <= set(fields)
    assert fields["epoch"] == "1/1" and fields["step"] == "1/1"
    assert fields["total_eta"] == "00:00:00"
    assert float(fields["CE"]) == pytest.approx(metrics["ce"], abs=1e-5)
    assert fields["lr"].count("bridge:") == 1
    assert fields["lr"].count("cross_attention:") == 1


def test_resume_rejects_wrong_reference_before_model_load(tmp_path, monkeypatch):
    import eviseq_afmr.runtime as runtime

    monkeypatch.setattr(runtime, "EviSeqAFMR", lambda cfg: pytest.fail("must validate before model load"))
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"id": "test-0", "reference": "wrong", "prediction": "x"}) + "\n")
    with pytest.raises(ValueError, match="prefix"):
        evaluate(config()["_meta"]["config_path"], "missing.pt", path, device="cpu")


def test_resume_after_warmup_matches_continuous_training(tmp_path):
    cfg = config()
    cfg["experiment"]["output_dir"] = str(tmp_path / "original")
    seed_everything(90)
    model = EviSeqAFMR(cfg)
    loaders = build_loaders(cfg)
    trainer = AFMRTrainer(model, cfg, "cpu")
    trainer.fit(loaders["train"], loaders["validation"])
    warm_checkpoint = tmp_path / "original/epoch_001.pt"
    cfg["experiment"]["output_dir"] = str(tmp_path / "resumed")
    resumed = EviSeqAFMR(cfg)
    resumed_trainer = AFMRTrainer(resumed, cfg, "cpu")
    resumed_loaders = build_loaders(cfg)
    resumed_trainer.fit(resumed_loaders["train"], resumed_loaders["validation"], str(warm_checkpoint))
    assert trainer.global_step == resumed_trainer.global_step
    for (name, expected), (_, actual) in zip(model.named_parameters(), resumed.named_parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6, msg=name)
