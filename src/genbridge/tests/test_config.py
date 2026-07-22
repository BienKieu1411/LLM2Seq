import tempfile
from pathlib import Path

from genbridge.config import MODEL_PROFILES, apply_model_size, load_config


def test_only_qwen3_dense_profiles_are_exposed():
    assert set(MODEL_PROFILES) == {"0.6B", "1.7B", "4B"}
    assert all("Qwen3-" in str(profile["name"]) for profile in MODEL_PROFILES.values())


def test_main_config_uses_four_adapter_layers_and_fp32_master_weights():
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(path)
    assert config["bridge"]["token_num_layers"] == 4
    assert config["bridge"]["use_adapter_rope"] is True
    assert config["bridge"]["rope_theta"] == 1_000_000.0
    assert config["bridge"]["balance_salience_loss"] is True
    assert config["bridge"]["plan_adapter_gate_init"] == 0.1
    assert config["model"]["dtype"] == "float32"
    assert config["training"]["bf16"] is True
    assert config["training"]["require_fp32_master_weights"] is True
    assert config["evaluation"]["model_dtype"] == "bfloat16"


def test_config_inheritance_and_model_scale_override():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "base.yaml").write_text(
            "model:\n  encoder_name: old\nexperiment:\n  output_dir: runs/base\n", encoding="utf-8"
        )
        (root / "child.yaml").write_text("_base_: base.yaml\ntraining:\n  epochs: 10\n", encoding="utf-8")
        config = load_config(root / "child.yaml")
        apply_model_size(config, "1.7B")
        assert config["model"]["encoder_name"] == "Qwen/Qwen3-1.7B"
        assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3-1.7B"
        assert config["decoder"]["num_layers"] == 28
        assert config["training"]["batch_size"] == 8
        assert config["training"]["gradient_accumulation_steps"] == 4
        assert config["training"]["full_lr"] == 8.0e-6
        assert config["experiment"]["output_dir"] == "runs/base_1_7b"


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


def test_embedding_encoder_ablation_keeps_regular_generative_decoder():
    path = Path(__file__).resolve().parents[1] / "configs" / "qwen3_embedding_enc0_6b_dec0_6b.yaml"
    config = load_config(path)
    assert config["model"]["encoder_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["model"]["allow_mixed_checkpoints"] is True
    assert config["decoder"]["pretrained_name"] == "Qwen/Qwen3-0.6B"
