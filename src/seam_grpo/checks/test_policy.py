import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from eviseq_afmr.config import load_config
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _TinyTokenizer
from eviseq_afmr.training.checkpoint import load_checkpoint, save_checkpoint
from seam_grpo.actor import SEAMActor
from seam_grpo.config import DEFAULTS
from seam_grpo.objective import clipped_terms
from seam_grpo.policy import Policy
from seam_grpo.rollout import repeat_batch, rollout
from seam_grpo.train import train, trajectory


def tiny(copy_enabled=True):
    torch.manual_seed(73)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2] / "eviseq_new"
    cfg = load_config(root / "configs/afmr_smoke.yaml")
    cfg["decoder"]["grounded_copy"]["enabled"] = copy_enabled
    cfg["generation"].update(max_new_tokens=4, min_new_tokens=0, repetition_penalty=1.0, no_repeat_ngram_size=0)
    cfg["data"].update(max_source_length=32, max_target_length=8)
    tokenizer = _TinyTokenizer()
    records = [
        CanonicalRecord.from_mapping(
            {"id": str(i), "text": "alpha beta gamma delta", "summary": "alpha beta"},
            source_field="text",
            target_field="summary",
            id_field="id",
        )
        for i in range(2)
    ]
    batch = SummarizationCollator(tokenizer, tokenizer, cfg["data"], grounded_copy=copy_enabled)(records)
    return cfg, tokenizer, batch


@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("copy_enabled", [True, False])
def test_chunked_logps_match_full_mixture_and_gradients(temperature, copy_enabled):
    cfg, tokenizer, batch = tiny(copy_enabled)
    model = EviSeqAFMR(cfg).eval()
    dense_model = copy.deepcopy(model)
    actions = batch["decoder_input_ids"][:, -3:].clone()
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[-1, -1] = False
    packed = trajectory(batch, actions, mask, 4)
    actual = Policy(model, chunk_size=1)(packed, temperature=temperature)
    keys = {
        "input_ids",
        "attention_mask",
        "source_content_mask",
        "decoder_prompt_ids",
        "decoder_prompt_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "output_budget",
        *COPY_INPUT_KEYS,
    }
    tensors = {key: value for key, value in packed.items() if key in keys}
    dense = dense_model(**tensors, return_logits=True).logits
    width = batch["decoder_prompt_ids"].shape[1]
    expected = (
        torch.log_softmax(dense[:, width - 1 : width + 2].float() / temperature, dim=-1)
        .gather(-1, actions[..., None])
        .squeeze(-1)
    )
    expected = expected.masked_fill(~mask, 0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weights = torch.tensor([[1.0, 0.7, 1.0], [-1.0, -0.5, 0.0]])
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for (name, p), (other, q) in zip(model.named_parameters(), dense_model.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)
    for prefix in ("encoder.", "bridge.", "decoder.grounded_copy.", "decoder.backbone."):
        if prefix == "decoder.grounded_copy." and not copy_enabled:
            continue
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0, prefix


def test_rollout_likelihood_parity_and_rl_weight_update():
    cfg, tokenizer, batch = tiny()
    seam = EviSeqAFMR(cfg).eval()
    expanded = repeat_batch(batch, 2)
    generated = rollout(seam, expanded, tokenizer, 4, 0.8)
    packed = trajectory(expanded, generated["action_ids"], generated["action_mask"], 4)
    policy = Policy(seam, chunk_size=2).eval()
    current = policy(packed, temperature=0.8)
    torch.testing.assert_close(current, generated["old_logps"], atol=3e-6, rtol=3e-6)
    before = seam.encoder.model.embed_tokens.weight.detach().clone()
    optimizer = torch.optim.SGD(seam.parameters(), lr=0.01)
    loss, _ = clipped_terms(
        current, generated["old_logps"], torch.tensor([1.0, -1.0, 0.5, -0.5]), generated["action_mask"]
    )
    (loss / 16).backward()
    optimizer.step()
    assert not torch.equal(before, seam.encoder.model.embed_tokens.weight)
    assert all(layer.cross._cache is None for layer in seam.decoder.decoder_layers)


def test_eos_is_action_even_when_pad_equals_eos(monkeypatch):
    class Decoder(torch.nn.Module):
        def clear_cross_cache(self):
            pass

        def prepare_cross_cache(self, *args, **kwargs):
            pass

        def forward(self, current, *args, past_key_values=None, **kwargs):
            step = 0 if past_key_values is None else past_key_values
            logits = torch.zeros(2, 1, 5)
            logits[0, 0, 2] = 10
            logits[1, 0, 3 if step == 0 else 2] = 10
            return logits, step + 1, None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def encode_source(self, *args, **kwargs):
            return SimpleNamespace(
                memory=torch.zeros(2, 1, 2),
                memory_mask=torch.ones(2, 1),
                source_bias=torch.zeros(2, 1),
                value_memory=None,
                copy_state=None,
            )

    monkeypatch.setattr(
        torch, "multinomial", lambda probabilities, *args, **kwargs: probabilities.argmax(-1, keepdim=True)
    )
    tokenizer = SimpleNamespace(
        eos_token_id=2, pad_token_id=2, batch_decode=lambda rows, **kwargs: [str(row) for row in rows]
    )
    batch = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "source_content_mask": torch.ones(2, 1, dtype=torch.bool),
        "decoder_prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "decoder_prompt_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    result = rollout(Model(), batch, tokenizer, 5, 1.0)
    assert result["action_ids"].tolist() == [[2, 2], [3, 2]]
    assert result["action_mask"].tolist() == [[True, False], [True, True]]
    assert result["old_logps"][0, 1] == 0
    assert not torch.is_inference(result["action_ids"])


@pytest.mark.parametrize("beta", [0.0, 0.01])
def test_post_sft_train_and_standard_checkpoint_reload(tmp_path, beta):
    cfg, tokenizer, batch = tiny()
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"id": "1", "text": "alpha beta gamma", "summary": "alpha beta"}) + "\n")
        cfg["data"][f"{split}_file"] = str(path)
    cfg.pop("_meta", None)
    base = tmp_path / "sft.yaml"
    base.write_text(yaml.safe_dump(cfg))
    model = EviSeqAFMR(cfg)
    model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, return_logits=False).loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    checkpoint = tmp_path / "sft.pt"
    save_checkpoint(checkpoint, model, None, cfg, epoch=4, step=100)
    initial = model.encoder.model.embed_tokens.weight.detach().clone()
    settings = {
        "base_config": str(base),
        "sft_checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "grpo"),
        "training": {
            **DEFAULTS,
            "max_steps": 1,
            "group_size": 2,
            "max_new_tokens": 3,
            "gradient_accumulation_steps": 1,
            "validation_batch_size": 1,
            "beta": beta,
        },
        "reward": {},
    }
    if beta == 0:
        output = Path(settings["output_dir"])
        output.mkdir()
        (output / "stale.txt").write_text("previous run")
        with pytest.raises(FileExistsError, match="overwrite-output-dir"):
            train(settings, "cpu")
    result = train(settings, "cpu", overwrite_output_dir=beta == 0)
    assert not (tmp_path / "grpo/stale.txt").exists()
    assert checkpoint.is_file() and base.is_file()
    assert result["steps"] == 1
    restored = EviSeqAFMR(cfg)
    info = load_checkpoint(tmp_path / "grpo/last.pt", restored, config=cfg, restore_rng=False)
    assert info["step"] == 1 and info["stage"] == "grpo"
    assert (tmp_path / "grpo/best.pt").is_file()
    assert not torch.equal(initial, restored.encoder.model.embed_tokens.weight)
    record = json.loads((tmp_path / "grpo/training_metrics.jsonl").read_text().splitlines()[0])
    assert record["ce"] > 0 and record["behavior_logp_mae"] < 1e-4
    assert record["step"] != 100


def test_bridge_and_copy_gradient_routes(monkeypatch):
    cfg, tokenizer, batch = tiny()
    model = EviSeqAFMR(cfg).eval()
    captured = []
    original = model.encode_source

    def encode(*args, **kwargs):
        state = original(*args, **kwargs)
        for tensor in (state.memory, state.value_memory, state.source_bias):
            if tensor is not None:
                tensor.retain_grad()
                captured.append(tensor)
        return state

    monkeypatch.setattr(model, "encode_source", encode)
    actions = batch["decoder_input_ids"][:, -3:]
    packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    Policy(model, chunk_size=1)(packed, temperature=1.0).sum().backward()
    assert len(captured) >= 2
    for tensor in captured:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    for name, parameter in model.decoder.grounded_copy.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_context_guard_and_cache_cleanup_on_encoder_failure(monkeypatch):
    cfg, tokenizer, batch = tiny()
    model = EviSeqAFMR(cfg).eval()
    with pytest.raises(ValueError, match="exceeds"):
        rollout(model, batch, tokenizer, 257, 1.0)
    calls = []
    original_clear = model.decoder.clear_cross_cache

    def clear():
        calls.append(1)
        original_clear()

    def broken_encoder(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(model.decoder, "clear_cross_cache", clear)
    monkeypatch.setattr(model, "encode_source", broken_encoder)
    with pytest.raises(RuntimeError, match="encoder failed"):
        rollout(model, batch, tokenizer, 4, 1.0)
    assert len(calls) == 2


@pytest.mark.parametrize("legacy_state", [False, True])
@pytest.mark.parametrize("copy_enabled", [False, True])
def test_unique_source_encoding_call_count_and_grad_equivalence(monkeypatch, legacy_state, copy_enabled):
    if legacy_state:
        from eviseq_afmr.modeling.grounded_copy import CopyState
        from eviseq_afmr.modeling.outputs import BridgeState

        monkeypatch.delattr(BridgeState, "index_select")
        monkeypatch.delattr(CopyState, "index_select")
    cfg, _, batch = tiny(copy_enabled=copy_enabled)
    model = EviSeqAFMR(cfg).eval()
    expected_model = copy.deepcopy(model)

    # Build a group of 4 candidates for prompt 0
    first = {k: (v[:1] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    first["output_budget"] = torch.tensor([4.0])
    repeated = repeat_batch(first, 4)
    actions = repeated["decoder_input_ids"][:, -3:]
    packed = trajectory(repeated, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    packed["gold_decoder_input_ids"] = repeated["decoder_input_ids"]
    packed["gold_decoder_attention_mask"] = repeated["decoder_attention_mask"]

    encode_calls = []
    original_encode = model.encode_source

    def counting_encode(*args, **kwargs):
        encode_calls.append(1)
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(model, "encode_source", counting_encode)

    actor = SEAMActor(model, chunk_size=1, score_batch_size=1).train()
    gold_selector = torch.tensor([True, False, False, False])
    logps, entropy, ce_sum = actor(packed, temperature=1.0, gold_selector=gold_selector)

    # encode_source must be called exactly 1 time across all 4 candidates and the gold CE!
    assert len(encode_calls) == 1, f"Expected 1 encode_source call, got {len(encode_calls)}"

    # Compare with reference policy
    ref_policy = Policy(expected_model, chunk_size=1)
    expected_logps = ref_policy(packed, temperature=1.0)
    expected_ce = ref_policy(first, temperature=1.0, ce=True) * first["labels"][:, 1:].ne(-100).sum()

    torch.testing.assert_close(logps, expected_logps, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ce_sum, expected_ce, rtol=1e-6, atol=1e-6)

    (logps.sum() + 0.05 * ce_sum).backward()
    (expected_logps.sum() + 0.05 * expected_ce).backward()

    for (name, p), (other, q) in zip(model.named_parameters(), expected_model.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-5, msg=name)


def test_differentiable_entropy_when_enabled():
    cfg, _, batch = tiny()
    model = EviSeqAFMR(cfg).eval()
    actions = batch["decoder_input_ids"][:, -3:]
    packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)

    policy = Policy(model, chunk_size=1)
    # Default: entropy is detached
    logps, detached_entropy = policy(packed, temperature=1.0, return_entropy=True, compute_entropy_with_grad=False)
    assert not detached_entropy.requires_grad

    # Enabled: entropy has autograd graph
    logps, diff_entropy = policy(packed, temperature=1.0, return_entropy=True, compute_entropy_with_grad=True)
    assert diff_entropy.requires_grad
    diff_entropy.sum().backward()
    grads = [p.grad for p in model.decoder.lm_head.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("copy_enabled", [False, True])
def test_candidate_scoring_only_recomputes_backbone_once(monkeypatch, copy_enabled):
    cfg, _, batch = tiny(copy_enabled)
    seam = EviSeqAFMR(cfg).eval()
    actor = SEAMActor(seam, chunk_size=2, score_batch_size=2).train()
    # One source with two candidates forms one decoder scoring microbatch.
    first = {key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    repeated = repeat_batch(first, 2)
    actions = repeated["decoder_input_ids"][:, -3:]
    packed = trajectory(repeated, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    calls = []
    forward = seam.decoder.backbone.forward

    def counted_forward(*args, **kwargs):
        calls.append(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(seam.decoder.backbone, "forward", counted_forward)
    logps, _, _ = actor(packed, temperature=0.8)
    assert len(calls) == 1
    logps.sum().backward()
    # Policy already checkpoints the backbone. An outer checkpoint around
    # the whole policy used to make it execute three times instead of two.
    assert len(calls) == 2


@pytest.mark.parametrize("copy_enabled", [False, True])
@pytest.mark.parametrize("score_batch_size", [2, 8])
def test_ragged_sources_and_actions_preserve_scores_ce_and_gradients(copy_enabled, score_batch_size):
    cfg, tokenizer, _ = tiny(copy_enabled)
    records = [
        CanonicalRecord.from_mapping(
            {"id": str(i), "text": text, "summary": summary},
            source_field="text",
            target_field="summary",
            id_field="id",
        )
        for i, (text, summary) in enumerate(
            [("alpha beta", "alpha"), ("alpha beta gamma delta epsilon zeta", "beta gamma delta")]
        )
    ]
    batch = SummarizationCollator(tokenizer, tokenizer, cfg["data"], grounded_copy=copy_enabled)(records)
    repeated = repeat_batch(batch, 3)
    actions = repeated["decoder_input_ids"][:, -3:]
    lengths = torch.tensor([1, 3, 3, 2, 1, 3])
    mask = torch.arange(3)[None, :] < lengths[:, None]
    packed = trajectory(repeated, actions, mask, 4)
    packed["gold_decoder_input_ids"] = repeated["decoder_input_ids"]
    packed["gold_decoder_attention_mask"] = repeated["decoder_attention_mask"]
    seam = EviSeqAFMR(cfg).eval()
    reference = copy.deepcopy(seam)
    actor = SEAMActor(seam, chunk_size=2, score_batch_size=score_batch_size).train()
    logps, entropies, ce_sum = actor(
        packed,
        temperature=0.8,
        gold_selector=torch.tensor([True, False, False, True, False, False]),
        compute_entropy_with_grad=True,
    )
    policy = Policy(reference, chunk_size=2)
    expected_logps, expected_entropy = policy(
        packed, temperature=0.8, return_entropy=True, compute_entropy_with_grad=True
    )
    expected_ce = policy(batch, temperature=1.0, ce=True) * batch["labels"][:, 1:].ne(-100).sum()
    torch.testing.assert_close(logps, expected_logps, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(entropies, expected_entropy, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(ce_sum, expected_ce, rtol=2e-5, atol=2e-6)
    weights = torch.tensor([1.0, -0.5, 0.3, -1.0, 0.8, -0.2])[:, None]
    (logps.mul(weights).sum() + 0.05 * ce_sum + 0.01 * entropies.sum()).backward()
    (expected_logps.mul(weights).sum() + 0.05 * expected_ce + 0.01 * expected_entropy.sum()).backward()
    for (name, actual), (_, expected) in zip(seam.named_parameters(), reference.named_parameters(), strict=True):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=4e-4, atol=3e-5, msg=name)
