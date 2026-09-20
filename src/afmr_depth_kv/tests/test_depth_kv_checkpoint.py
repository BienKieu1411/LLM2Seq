import random

import numpy as np
import pytest
import torch

from eviseq_afmr.modeling.afmr import LayerwiseCoupledDepthBridge
from eviseq_afmr.training.checkpoint import architecture_spec, load_checkpoint, save_checkpoint


def _architecture() -> dict:
    return {
        "name": "layerwise_coupled_depth_kv",
        "bridge_mode": "depth_kv",
        "controller_dim": 4,
        "depth_taps": 3,
        "depth_rank": 4,
        "depth_gate_init": 0.1,
        "depth_gate_max": 1.0,
        "residual_max_relative_rms": 0.25,
        "final_tap_bias": 1.5,
    }


def _config() -> dict:
    return {
        "model": {"encoder_name": "tiny", "decoder_name": "tiny"},
        "architecture": _architecture(),
        "decoder": {"cross_attention_every": 1, "cross_gate_max": 1.0},
    }


def _model() -> LayerwiseCoupledDepthBridge:
    return LayerwiseCoupledDepthBridge(8, 8, 2, _architecture())


def test_checkpoint_roundtrip_and_structural_guard(tmp_path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, _config(), epoch=3, step=12)
    restored = _model()
    metadata = load_checkpoint(path, restored, config=_config())
    assert metadata["epoch"] == 3 and metadata["step"] == 12

    incompatible = _config()
    incompatible["architecture"]["residual_max_relative_rms"] = 0.5
    with pytest.raises(ValueError, match="architecture_spec"):
        load_checkpoint(path, restored, config=incompatible)


def test_architecture_spec_excludes_runtime_paths() -> None:
    config = _config()
    config["training"] = {"batch_size": 1}
    config["data"] = {"train_file": "a"}
    spec = architecture_spec(config)
    assert "batch_size" not in spec and "train_file" not in spec


def test_checkpoint_roundtrip_restores_rng_state(tmp_path) -> None:
    model = _model()
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
