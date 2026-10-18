"""Checks of the real, version-pinned library integration, not a trainer stub."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.training.checkpoint import save_checkpoint
from seam_grpo.actor import SEAMActor
from seam_grpo.config import DEFAULTS
from seam_grpo.data import trajectory
from seam_grpo.policy import Policy
from seam_grpo.train import train
from seam_grpo.trainer import SEAMGRPOTrainer
from test_policy import tiny
from trl import GRPOTrainer


def test_actor_single_forward_ce_and_entropy():
    cfg, _, batch = tiny()
    seam = EviSeqAFMR(cfg).eval()
    expected_model = copy.deepcopy(seam)
    actions = batch["decoder_input_ids"][:, -3:]
    packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    packed["gold_decoder_input_ids"] = batch["decoder_input_ids"]
    packed["gold_decoder_attention_mask"] = batch["decoder_attention_mask"]
    actor = SEAMActor(seam, chunk_size=1, score_batch_size=1).train()
    logps, entropy, ce_sum = actor(packed, temperature=0.8, gold_selector=torch.tensor([True, False]))
    policy = Policy(expected_model, chunk_size=1)
    expected = policy(packed, temperature=0.8)
    first = {k: (v[:1] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    first["output_budget"] = torch.tensor([4.0])
    ce = policy(first, temperature=1.0, ce=True) * first["labels"][:, 1:].ne(-100).sum()
    torch.testing.assert_close(logps, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ce_sum, ce, rtol=1e-6, atol=1e-6)
    assert actor.training and not seam.training
    assert entropy.shape == logps.shape and not entropy.requires_grad
    (logps.sum() + 0.05 * ce_sum).backward()
    (expected.sum() + 0.05 * ce).backward()
    for (name, p), (other, q) in zip(seam.named_parameters(), expected_model.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-5, msg=name)


def test_stale_rollout_is_rejected_before_policy_forward():
    trainer = object.__new__(SEAMGRPOTrainer)
    trainer.state = SimpleNamespace(global_step=2)
    with pytest.raises(RuntimeError, match="stale"):
        trainer.compute_loss(None, {"seam_policy_version": torch.tensor([1, 1])})


@pytest.mark.parametrize(
    "rollout_backend", ["native", "seam_vllm", "seam_vllm_bad", "seam_vllm_outlier", "seam_vllm_b200"]
)
def test_real_trl_objective_new_versions_and_frozen_full_reference(tmp_path, monkeypatch, rollout_backend):
    bad_logps = rollout_backend == "seam_vllm_bad"
    if rollout_backend.startswith("seam_vllm"):
        import sys

        from fake_vllm import NativeFacade
        from seam_grpo.vllm_rollout import VLLMRollout

        from seam_grpo import train as runtime

        monkeypatch.setattr(runtime, "check_rollout_device", lambda *a: None)
        monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
        monkeypatch.setattr(VLLMRollout, "initialize", lambda self: setattr(self, "engine", NativeFacade(self.trainer)))
    cfg, _, _ = tiny()
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": str(i), "text": "alpha beta gamma", "summary": "alpha beta"}) + "\n"
                for i in range(3 if split == "train" else 1)
            )
        )
        cfg["data"][f"{split}_file"] = str(path)
    cfg.pop("_meta", None)
    base = tmp_path / "sft.yaml"
    base.write_text(yaml.safe_dump(cfg))
    sft = tmp_path / "sft.pt"
    save_checkpoint(sft, EviSeqAFMR(cfg), None, cfg, epoch=4, step=100)
    counters = {"loss": 0}
    original_loss = GRPOTrainer._compute_loss
    original_init = SEAMGRPOTrainer.__init__
    holder = {}

    def counted_loss(self, *args, **kwargs):
        counters["loss"] += 1
        return original_loss(self, *args, **kwargs)

    def captured_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        holder["trainer"] = self
        holder["ref"] = {name: p.detach().clone() for name, p in self.ref_model.named_parameters()}
        self._test_bad_vllm_logps = bad_logps
        self._test_vllm_outlier = rollout_backend == "seam_vllm_outlier"

    monkeypatch.setattr(GRPOTrainer, "_compute_loss", counted_loss)
    monkeypatch.setattr(SEAMGRPOTrainer, "__init__", captured_init)
    settings = {
        "base_config": str(base),
        "sft_checkpoint": str(sft),
        "output_dir": str(tmp_path / "out"),
        "training": {
            **DEFAULTS,
            "rollout_backend": "seam_vllm" if rollout_backend.startswith("seam_vllm") else rollout_backend,
            "epochs": 2,
            "max_steps": 2,
            "group_size": 2,
            "max_new_tokens": 32 if rollout_backend == "seam_vllm_outlier" else 3,
            "gradient_accumulation_steps": 2,
            "beta": 0.01,
            "validation_batch_size": 1,
        },
        "reward": {},
    }
    b200 = rollout_backend == "seam_vllm_b200"
    if b200:
        settings["training"].update(
            prompt_batch_size=8,
            group_size=8,
            gradient_accumulation_steps=4,
            score_micro_batch_size=8,
            logprob_chunk_size=64,
            vllm_group_sampling=True,
        )
    if bad_logps:
        with pytest.raises(RuntimeError, match="likelihood mismatch"):
            train(settings, "cpu")
        assert holder["trainer"].state.global_step == 0
        return
    result = train(settings, "cpu")
    trainer = holder["trainer"]
    assert isinstance(trainer, GRPOTrainer)
    assert counters["loss"] == (8 if b200 else 4)
    assert [row["policy_version"] for row in result["rollouts"]] == [0, 1]
    assert [row["candidates"] for row in result["rollouts"]] == ([256, 256] if b200 else [4, 4])
    assert trainer._buffered_inputs is None
    assert trainer._likelihood_source is None
    assert trainer._active_loss_inputs is None
    metrics = [json.loads(line) for line in (tmp_path / "out/training_metrics.jsonl").read_text().splitlines()]
    assert any("eta" in row and "rollout_reward_seconds" in row and "train_seconds" in row for row in metrics)
    if rollout_backend.startswith("seam_vllm"):
        assert any(
            row.get("rollout_generate_seconds", -1) >= 0 and row.get("rollout_sync_seconds", -1) >= 0 for row in metrics
        )
    if rollout_backend == "seam_vllm_outlier":
        assert any(row.get("behavior_logp_max_error", 0) > 1.0 for row in metrics)
        assert any(row.get("sampling/importance_sampling_clipped_ratio", 0) > 0 for row in metrics)
    for name, p in trainer.ref_model.named_parameters():
        assert not p.requires_grad and p.grad is None
        torch.testing.assert_close(p, holder["ref"][name], rtol=0, atol=0)
    assert len(trainer.train_dataset) == (32 if b200 else 4)  # retain original three, pad the last update
    implementation = json.loads((tmp_path / "out/implementation.json").read_text())
    assert implementation["trl"] == "1.16.0"
    versions = [json.loads(line) for line in (tmp_path / "out/rollout_versions.jsonl").read_text().splitlines()]
    assert [row["updated_policy_version"] for row in versions] == [1, 2]
