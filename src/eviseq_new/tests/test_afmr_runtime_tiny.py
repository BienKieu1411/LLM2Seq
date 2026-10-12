from pathlib import Path

import pytest
import torch
from eviseq_afmr.config import load_config
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _write_resolved_config, build_loaders, evaluate
from eviseq_afmr.training.checkpoint import load_checkpoint, save_checkpoint
from eviseq_afmr.training.engine import AFMRTrainer, _lr_multiplier


def _config(tmp_path: Path) -> dict:
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["generation"]["max_new_tokens"] = 4
    config["training"]["save_best"] = True
    return config


@pytest.mark.parametrize("grounded_copy", [False, True])
def test_tiny_runtime_trains_saves_best_and_streams_eval(tmp_path: Path, grounded_copy):
    config = _config(tmp_path)
    config["decoder"]["grounded_copy"]["enabled"] = grounded_copy
    _write_resolved_config(config, tmp_path)
    config_path = tmp_path / "resolved_config.yaml"
    loaders = build_loaders(config, max_train_examples=4, max_validation_examples=2)
    test_batch = next(iter(build_loaders(config, split="test")["test"]))
    assert "allocation_target" not in test_batch

    model = EviSeqAFMR(config)
    trainer = AFMRTrainer(model, config, torch.device("cpu"))
    trainer.fit(loaders["train"], loaders["validation"])
    output_dir = Path(config["experiment"]["output_dir"])
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "best.pt").is_file()

    config["training"]["full_finetune_epochs"] = 2
    resumed_model = EviSeqAFMR(config)
    resumed_trainer = AFMRTrainer(resumed_model, config, torch.device("cpu"))
    resumed_trainer.fit(loaders["train"], loaders["validation"], resume_checkpoint=str(output_dir / "last.pt"))
    metadata = load_checkpoint(output_dir / "last.pt", resumed_model, config=config)
    assert metadata["stage"] == "full_finetune"
    assert metadata["stage_epoch"] == 2

    predictions = output_dir / "test.jsonl"
    result = evaluate(config_path, output_dir / "last.pt", predictions, split="test", batch_size=2, device="cpu")
    assert result["empty_prediction_rate"] >= 0.0
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
    resumed = evaluate(config_path, output_dir / "last.pt", predictions, split="test", batch_size=2, device="cpu")
    assert resumed == result


def test_init_checkpoint_starts_fresh_epoch_and_validation_selection(tmp_path: Path):
    config = _config(tmp_path)
    config["experiment"]["output_dir"] = str(tmp_path / "finetune")
    config["training"]["interface_warmup_epochs"] = 0
    config["training"]["full_finetune_epochs"] = 1
    assert config["training"]["lr_scheduler"] == "cosine"

    source_model = EviSeqAFMR(config)
    source_checkpoint = tmp_path / "source_best.pt"
    save_checkpoint(
        source_checkpoint,
        source_model,
        None,
        config,
        epoch=5,
        step=123,
        best_metric=-1e9,
        stage="full_finetune",
        stage_epoch=5,
    )
    loaders = build_loaders(config, max_train_examples=4, max_validation_examples=2)
    trainer = AFMRTrainer(EviSeqAFMR(config), config, torch.device("cpu"))
    trainer.fit(loaders["train"], loaders["validation"], init_checkpoint=str(source_checkpoint))

    output_dir = Path(config["experiment"]["output_dir"])
    state = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 1
    assert state["stage_epoch"] == 1
    assert state["step"] > 0
    assert state["best_metric"] > -1e9
    assert _lr_multiplier(0, 10, "cosine") == 1.0
    assert _lr_multiplier(5, 10, "cosine") == pytest.approx(0.5)
    assert _lr_multiplier(10, 10, "cosine") == pytest.approx(0.0)
