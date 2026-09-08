import pytest
from eviseq_update.config import validate_config
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.training.checkpoint import save_checkpoint
from eviseq_update.training.engine import AFMRTrainer
from eviseq_update.training.schedule import lr_multiplier
from test_supervised_training import config


def test_lr_warmup_cosine_and_legacy_linear():
    assert [lr_multiplier(i, 10) for i in (0, 5, 10)] == [1, 0.5, 0]
    assert [lr_multiplier(i, 10, "cosine", 0.2) for i in (0, 1, 2, 6, 10)] == [0, 0.5, 1, 0.5, 0]
    assert lr_multiplier(0, 1, "cosine", 0.1) == 1
    assert lr_multiplier(1, 1, "cosine", 0.1) == 0


@pytest.mark.parametrize("key,value", [("lr_scheduler", "unknown"), ("lr_warmup_ratio", -1), ("lr_warmup_ratio", 1)])
def test_invalid_schedule_rejected(key, value):
    cfg = config()
    cfg["training"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


@pytest.mark.parametrize("key,value", [("lr_scheduler", "linear"), ("lr_warmup_ratio", 0.1)])
def test_resume_cannot_silently_change_recipe(tmp_path, key, value):
    cfg = config()
    cfg["experiment"]["output_dir"] = str(tmp_path / "fit")
    model = EviSeqAFMR(cfg)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, None, cfg, epoch=1, step=1)
    cfg["training"][key] = value
    with pytest.raises(ValueError, match=key):
        AFMRTrainer(model, cfg, "cpu").fit([], resume_checkpoint=str(path))
