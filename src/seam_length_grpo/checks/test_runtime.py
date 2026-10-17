import json

import pytest
import torch
from fixtures import fixture, settings_file
from seam_length_grpo.trainer import LengthTrainer
from seam_length_grpo.warmup import GoldActor
from trl import GRPOTrainer

from seam_length_grpo import train as runtime


@pytest.mark.parametrize("source_weight", [0.0, 0.05])
def test_real_trl_updates_ragged_prompts_and_frozen_teacher(tmp_path, monkeypatch, source_weight):
    settings, tokenizer, path = settings_file(tmp_path, source_weight=source_weight)
    if source_weight == 0:
        settings["length_control"]["source_content_tokens"] = None
    monkeypatch.setattr(runtime, "_tokenizers", lambda config: (tokenizer, tokenizer))
    original_init = LengthTrainer.__init__
    original_step = torch.optim.AdamW.step
    original_gold = GoldActor.forward
    captured = {}

    def check_lengths(self, batch):
        captured.setdefault("word_requests", set()).update(batch["target_words"])
        for words, instruction in zip(batch["target_words"], batch["decoder_instructions"], strict=True):
            assert str(words) in instruction
        return original_gold(self, batch)

    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["trainer"] = self
        assert not self.optimizer.state, "GRPO inherited warm-up optimizer moments"
        captured["actor"] = {
            n: p.detach().clone() for n, p in self.accelerator.unwrap_model(self.model).named_parameters()
        }

    def check_teacher(optimizer, *args, **kwargs):
        if "trainer" not in captured:
            captured["warmup_updates"] = captured.get("warmup_updates", 0) + 1
            return original_step(optimizer, *args, **kwargs)
        trainer = captured["trainer"]
        teacher = trainer.source_teacher
        if teacher is not None:
            current = {n: p.detach().clone() for n, p in teacher.actor.named_parameters()}
            assert all(not p.requires_grad and p.grad is None for p in teacher.actor.parameters())
            if "teacher" in captured:
                for name, p in current.items():
                    torch.testing.assert_close(p, captured["teacher"][name], rtol=0, atol=0)
            captured["teacher"] = current
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(LengthTrainer, "__init__", capture)
    monkeypatch.setattr(GoldActor, "forward", check_lengths)
    monkeypatch.setattr(torch.optim.AdamW, "step", check_teacher)
    result = runtime.train(settings, "cpu")
    trainer = captured["trainer"]
    assert isinstance(trainer, GRPOTrainer)
    assert result["steps"] == 2
    assert result["ce_warmup_steps"] == captured["warmup_updates"] == 1
    assert (tmp_path / "out/length_ce_warmup.pt").is_file()
    assert captured["word_requests"] == {2, 12}
    assert (tmp_path / "out/validation_length_ce_baseline.jsonl").is_file()
    sft = torch.load(settings["sft_checkpoint"], map_location="cpu", weights_only=False)["model"]
    adapted = torch.load(tmp_path / "out/length_ce_warmup.pt", map_location="cpu", weights_only=False)["model"]
    assert any(not torch.equal(sft[name], value) for name, value in adapted.items())
    for name, value in captured["actor"].items():
        torch.testing.assert_close(value, adapted[name.removeprefix("policy.seam.")], rtol=0, atol=0)
    assert [row["policy_version"] for row in result["rollouts"]] == [0, 1]
    assert trainer._likelihood_source is None
    assert (trainer.source_teacher is not None) == bool(source_weight)
    if trainer.source_teacher is not None:
        for name, value in trainer.source_teacher.actor.named_parameters():
            torch.testing.assert_close(value, sft[name.removeprefix("policy.seam.")], rtol=0, atol=0)
    model = trainer.accelerator.unwrap_model(trainer.model)
    assert any(
        not torch.equal(p, captured["actor"][n])
        for n, p in model.named_parameters()
        if n.startswith("policy.seam.encoder.")
    )
    metrics = [json.loads(row) for row in (tmp_path / "out/training_metrics.jsonl").read_text().splitlines()]
    assert any(row.get("behavior_logp_mae", 1) < 1e-4 for row in metrics)
    validation = [json.loads(row) for row in (tmp_path / "out/validation_epoch_001.jsonl").read_text().splitlines()]
    assert [row["target_words"] for row in validation] == [2, 12]
    assert (tmp_path / "out/last.pt").is_file()
    assert (tmp_path / "out/length_grpo_config.yaml").is_file()
    assert (tmp_path / "out/best_selection.json").is_file()
    from seam_length_grpo.config import load
    from seam_length_grpo.data import LengthCollator, prepare_eval

    exported = load(tmp_path / "out/length_grpo_config.yaml")
    inference = prepare_eval(exported)
    reloaded_collator = LengthCollator((tokenizer, tokenizer), inference["data"], exported["length_control"])
    assert reloaded_collator.source_limit == trainer.collator.source_limit
    assert reloaded_collator.encoder_limit == trainer.collator.encoder_limit
    assert exported["base_config"] == str(tmp_path / "out/resolved_config.yaml")
