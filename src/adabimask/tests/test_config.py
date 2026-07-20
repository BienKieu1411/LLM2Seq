from pathlib import Path

from adabimask.config import load_config


def test_ablation_config_inherits_base():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "ablations" / "middle_k8.yaml")
    assert config["model"]["encoder_name"] == "Qwen/Qwen3-0.6B-Base"
    assert config["mask"]["mode"] == "fixed"
    assert config["mask"]["fixed_strategy"] == "middle"
    assert config["mask"]["budget_groups"] == 2
