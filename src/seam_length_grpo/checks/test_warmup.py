import copy
from types import SimpleNamespace

import pytest
import torch
from eviseq_afmr.modeling.model import EviSeqAFMR
from fixtures import fixture, settings_file
from seam_grpo.actor import SEAMActor
from seam_length_grpo.config import WARMUP_DEFAULTS, load
from seam_length_grpo.policy import LengthPolicy
from seam_length_grpo.warmup import GoldActor, GoldTrainer, warmup_steps

from seam_length_grpo import train as runtime


def test_warmup_global_token_normalization_matches_full_batch():
    cfg, tok, length, rows, collator = fixture()
    actor = SEAMActor(EviSeqAFMR(cfg))
    actor.policy = LengthPolicy(actor.policy.seam)
    reference = copy.deepcopy(actor)
    full = collator(rows)
    denominator = full["labels"][:, 1:].ne(-100).sum()
    # The same implementation is used inside one DDP forward per microbatch.
    trainer = SimpleNamespace(accelerator=SimpleNamespace(num_processes=1))
    wrapped = GoldActor(actor)
    for row in rows:
        batch = collator([row])
        assert batch["target_words"] == [row["target_words"]]
        assert str(row["target_words"]) in batch["decoder_instructions"][0]
        loss = GoldTrainer.compute_loss(trainer, wrapped, {"batch": batch}, num_items_in_batch=denominator)
        loss.backward()
    reference.policy(full, temperature=1.0, ce=True).backward()
    for (name, p), (_, q) in zip(actor.named_parameters(), reference.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=3e-5, msg=name)


def test_warmup_schedule_and_zero_skip():
    settings = {
        "ce_warmup": dict(WARMUP_DEFAULTS),
        "training": {"prompt_batch_size": 2, "gradient_accumulation_steps": 4},
    }
    assert warmup_steps(16000, settings, 2) == 50
    settings["ce_warmup"].update(fraction=0)
    assert warmup_steps(16000, settings, 2) == 0
    settings["ce_warmup"].update(max_steps=3)
    assert warmup_steps(16000, settings, 2) == 3


def test_warmup_can_be_disabled_without_any_gold_stage(tmp_path, monkeypatch):
    settings, tokenizer, _ = settings_file(tmp_path, source_weight=0, max_steps=1)
    settings["ce_warmup"] = {"fraction": 0, "max_steps": 0}
    monkeypatch.setattr(runtime, "_tokenizers", lambda config: (tokenizer, tokenizer))

    def forbidden(*args, **kwargs):
        raise AssertionError("A disabled warm-up executed gold adaptation")

    monkeypatch.setattr(GoldTrainer, "train", forbidden)
    result = runtime.train(settings, "cpu")
    assert result["ce_warmup_steps"] == 0
    assert not (tmp_path / "out/length_ce_warmup.pt").exists()
    assert (tmp_path / "out/validation_sft_baseline.jsonl").is_file()


@pytest.mark.parametrize("value", [{"fraction": -1}, {"fraction": True}, {"fraction": float("nan")}, {"max_steps": -1}])
def test_invalid_warmup_config_is_rejected(tmp_path, value):
    import yaml

    settings, tokenizer, path = settings_file(tmp_path, source_weight=0)
    raw = yaml.safe_load(path.read_text())
    raw["ce_warmup"] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="ce_warmup"):
        load(path)
