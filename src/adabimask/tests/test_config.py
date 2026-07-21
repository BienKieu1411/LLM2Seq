from pathlib import Path

from adabimask.config import apply_model_size, load_config


def test_ablation_config_inherits_base():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "ablations" / "middle_k8.yaml")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3.5-0.8B"
    assert config["model"]["train_base"] is True
    assert config["training"]["epochs"] == 10
    assert config["training"]["interface_warmup_epochs"] == 2
    assert config["training"]["batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["data"]["train_file"] == "data/processed/train.jsonl"
    assert config["mask"]["mode"] == "fixed"
    assert config["mask"]["fixed_strategy"] == "middle"
    assert config["mask"]["budget_groups"] == 2


def test_model_size_override_updates_both_halves_and_separates_outputs():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "base.yaml")
    apply_model_size(config, "2B")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3.5-2B"
    assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3.5-2B"
    assert config["experiment"]["output_dir"] == "runs/adabimask/base_2b"


def test_a100_smoke_reaches_both_training_stages_on_a_small_subset():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "a100_smoke.yaml")
    assert config["training"]["epochs"] == 2
    assert config["training"]["interface_warmup_epochs"] == 1
    assert config["data"]["max_train_samples"] == 256
    assert config["training"]["batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 8
