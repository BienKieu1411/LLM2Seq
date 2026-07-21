import tempfile
from pathlib import Path

from genbridge.config import apply_model_size, load_config


def test_config_inheritance_and_model_scale_override():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "base.yaml").write_text("model:\n  encoder_name: old\nexperiment:\n  output_dir: runs/base\n", encoding="utf-8")
        (root / "child.yaml").write_text("_base_: base.yaml\ntraining:\n  epochs: 10\n", encoding="utf-8")
        config = load_config(root / "child.yaml")
        apply_model_size(config, "2B")
        assert config["model"]["encoder_name"] == "Qwen/Qwen3.5-2B"
        assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3.5-2B"
        assert config["decoder"]["num_layers"] == 24
        assert config["training"]["batch_size"] == 8
        assert config["training"]["gradient_accumulation_steps"] == 4
        assert config["training"]["full_lr"] == 8.0e-6
        assert config["experiment"]["output_dir"] == "runs/base_2b"


def test_qwen3_06b_profile_uses_its_native_28_layers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "base.yaml"
        path.write_text("model:\n  encoder_name: old\n", encoding="utf-8")
        config = load_config(path)
        apply_model_size(config, "0.6B")
        assert config["model"]["encoder_name"] == "Qwen/Qwen3-0.6B"
        assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3-0.6B"
        assert config["decoder"]["num_layers"] == 28
        assert config["training"]["batch_size"] == 32
        assert config["training"]["gradient_accumulation_steps"] == 1
