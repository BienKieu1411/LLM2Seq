from pathlib import Path

import torch
from eviseq_afmr.config import load_config
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import build_loaders, evaluate
from eviseq_afmr.training.checkpoint import load_checkpoint
from eviseq_afmr.training.engine import AFMRTrainer


def _config(tmp_path: Path) -> dict:
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["generation"]["max_new_tokens"] = 4
    config["training"]["save_best"] = True
    return config


def test_tiny_runtime_trains_saves_best_and_streams_eval(tmp_path: Path):
    config = _config(tmp_path)
    config_path = Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml"
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
