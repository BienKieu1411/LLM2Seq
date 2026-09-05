import random

import numpy as np
import pytest
import torch
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint


def _config():
    return {
        "model": {"encoder_name": "tiny", "decoder_name": "tiny"},
        "architecture": {"name": "afmr_v1", "depth_taps": 0, "depth_rank": 4, "feature_rank": 4, "focus_windows": [4]},
        "decoder": {"cross_attention_every": 1},
    }


def test_checkpoint_roundtrip_and_structural_guard(tmp_path):
    model = AdaptiveFullMemoryResidualBridge(
        8,
        8,
        {
            "controller_dim": 4,
            "depth_taps": 0,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 4,
            "focus_windows": [4],
            "focus_strength_max": 1.0,
            "focus_strength_init": 0.1,
            "depth_gate_max": 0.15,
            "depth_gate_init": 0.02,
            "feature_gate_max": 0.2,
            "feature_gate_init": 0.02,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, _config(), epoch=3, step=12)
    restored = AdaptiveFullMemoryResidualBridge(
        8,
        8,
        {
            "controller_dim": 4,
            "depth_taps": 0,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 4,
            "focus_windows": [4],
            "focus_strength_max": 1.0,
            "focus_strength_init": 0.1,
            "depth_gate_max": 0.15,
            "depth_gate_init": 0.02,
            "feature_gate_max": 0.2,
            "feature_gate_init": 0.02,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )
    metadata = load_checkpoint(path, restored, config=_config())
    assert metadata["epoch"] == 3 and metadata["step"] == 12
    legacy = torch.load(path, weights_only=False)
    legacy["architecture_spec"]["graph_version"] = "afmr_content_windows_lowrank_v2"
    torch.save(legacy, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(tmp_path / "legacy.pt", restored, config=_config())
    bad = _config()
    bad["architecture"]["focus_windows"] = [8]
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, restored, config=bad)


def test_architecture_spec_excludes_runtime_paths():
    config = _config()
    config["training"] = {"batch_size": 1}
    config["data"] = {"train_file": "a"}
    spec = architecture_spec(config)
    assert "batch_size" not in spec and "train_file" not in spec


def test_checkpoint_roundtrip_restores_rng_state(tmp_path):
    model = AdaptiveFullMemoryResidualBridge(
        8,
        8,
        {
            "controller_dim": 4,
            "depth_taps": 0,
            "depth_rank": 4,
            "feature_rank": 4,
            "focus_hidden": 4,
            "focus_windows": [4],
            "focus_strength_max": 1.0,
            "focus_strength_init": 0.1,
            "depth_gate_max": 0.15,
            "depth_gate_init": 0.02,
            "feature_gate_max": 0.2,
            "feature_gate_init": 0.02,
            "temperature_init": 1.0,
            "temperature_min": 0.5,
            "temperature_max": 2.0,
        },
    )
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    save_checkpoint(tmp_path / "rng.pt", model, None, _config(), epoch=1, step=1)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    load_checkpoint(tmp_path / "rng.pt", model, config=_config())
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
