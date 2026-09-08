"""Gold-only training: one forward, parameter updates and removed-recipe guards."""

import pytest
import torch
from eviseq_update.config import validate_config
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.runtime import build_loaders
from eviseq_update.training.checkpoint import save_checkpoint
from eviseq_update.training.engine import AFMRTrainer
from eviseq_update.training.optimizer import build_optimizer, set_stage_trainability
from test_semantic_read import model_config


def config():
    cfg = model_config("independent_bounded")
    cfg["training"].update(lr_scheduler="cosine", lr_warmup_ratio=0.05)
    return cfg


def tensors(cfg):
    batch = next(iter(build_loaders(cfg, max_train_examples=2)["train"]))
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}


@pytest.mark.parametrize("autocast", [False, True])
def test_single_pass_gold_training_updates_all_components(autocast, monkeypatch):
    import eviseq_update.evaluation.generate as generation

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not generate text")

    monkeypatch.setattr(generation, "generate_greedy", forbidden)
    monkeypatch.setattr(generation, "generate_sampled", forbidden)
    cfg = config()
    validate_config(cfg)
    model = EviSeqAFMR(cfg).train()
    batch = tensors(cfg)
    calls, encoder_calls = [], []
    hook = model.decoder.backbone.register_forward_hook(lambda *_: calls.append(1))
    encoder_hook = model.encoder.register_forward_hook(lambda *_: encoder_calls.append(1))
    set_stage_trainability(model, "full_finetune")
    optimizer = build_optimizer(model, cfg, "full_finetune")
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            output = model(**batch, return_logits=False)
        assert len(calls) == len(encoder_calls) == step + 1
        assert output.loss is output.loss_ce
        output.loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.dtype == torch.float32, name
        optimizer.step()
        for component in ("encoder", "decoder", "bridge"):
            assert any(
                name.startswith(component) and not torch.equal(before[name], p) for name, p in model.named_parameters()
            )
        if step:
            for name, p in model.decoder.grounded_copy.named_parameters():
                assert p.grad.abs().any(), name
    hook.remove()
    encoder_hook.remove()


def test_removed_noise_config_rejected():
    cfg = config()
    cfg["training"]["neftune_noise_alpha"] = 1.0
    with pytest.raises(ValueError, match="neftune_noise_alpha"):
        validate_config(cfg)


def test_removed_noise_checkpoint_cannot_silently_resume_as_ce(tmp_path):
    cfg = config()
    cfg["experiment"]["output_dir"] = str(tmp_path / "fit")
    model = EviSeqAFMR(cfg)
    cfg["training"]["neftune_noise_alpha"] = 1.0
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, None, cfg, epoch=1, step=1)
    del cfg["training"]["neftune_noise_alpha"]
    with pytest.raises(ValueError, match="removed NEFTune"):
        AFMRTrainer(model, cfg, "cpu").fit([], resume_checkpoint=str(path))
