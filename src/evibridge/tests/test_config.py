import tempfile
from pathlib import Path

from evibridge.config import apply_model_size, load_config


def test_config_inheritance_and_model_scale_override():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "base.yaml").write_text("model:\n  encoder_name: old\nexperiment:\n  output_dir: runs/base\n", encoding="utf-8")
        (root / "child.yaml").write_text("_base_: base.yaml\ntraining:\n  epochs: 10\n", encoding="utf-8")
        config = load_config(root / "child.yaml")
        apply_model_size(config, "2B")
        assert config["model"]["encoder_name"] == "Qwen/Qwen3.5-2B"
        assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3.5-2B"
        assert config["experiment"]["output_dir"] == "runs/base_2b"
