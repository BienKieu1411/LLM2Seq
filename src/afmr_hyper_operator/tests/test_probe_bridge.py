import copy
from pathlib import Path

import torch

from eviseq_afmr.config import load_config
from eviseq_afmr.runtime import build_loaders
from eviseq_afmr.training.checkpoint import save_checkpoint
from scripts.probe_bridge import _counterfactual, run_probe
from eviseq_afmr.modeling.model import EviSeqAFMR


def _tiny_checkpoint(tmp_path: Path) -> tuple[dict, Path]:
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["training"]["validation_batch_size"] = 2
    model = EviSeqAFMR(config).eval()
    with torch.no_grad():
        model.bridge.focus_output.weight.fill_(0.1)
        model.bridge.feature_up.weight.normal_(std=0.02)
        model.bridge.depth_out.weight.normal_(std=0.02)
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, model, None, config, epoch=1, step=2)
    return config, checkpoint


def test_probe_reports_counterfactual_scope_and_zeroes_prior_routes(tmp_path: Path):
    config, checkpoint = _tiny_checkpoint(tmp_path)
    report = run_probe(
        Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml",
        checkpoint,
        batch_size=2,
        max_examples=2,
        device="cpu",
    )
    assert report["scope"] == "within_checkpoint_counterfactual"
    assert "not retrained" in report["note"]
    assert report["contextual_value_still_active"] is False
    assert set(report["results"]) == {
        "full",
        "zero_source_bias",
        "zero_depth_feature_residuals",
        "zero_source_bias_and_residuals",
    }
    assert report["results"]["full"]["prior"]["mean_abs"] > 0
    assert report["results"]["zero_source_bias"]["prior"]["mean_abs"] == 0
    assert report["results"]["zero_source_bias_and_residuals"]["prior"]["max_abs"] == 0
    assert report["results"]["zero_depth_feature_residuals"]["hooked_modules"] == ["depth_out", "feature_up"]
    assert all(torch.isfinite(torch.tensor(result["ce"])) for result in report["results"].values())
    assert config["model"]["encoder_name"] == "__tiny__"


def test_counterfactual_restores_model_hooks_and_copy_bias():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["decoder"]["grounded_copy"]["enabled"] = True
    model = EviSeqAFMR(config).eval()
    loader = build_loaders(config, split="validation", max_validation_examples=1)["validation"]
    batch = next(iter(loader))
    inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        baseline = model(**inputs, return_logits=False)
        with _counterfactual(model, "zero_source_bias"):
            counterfactual = model(**inputs, return_logits=False)
        restored = model(**inputs, return_logits=False)
    assert baseline.bridge.copy_state is not None
    assert counterfactual.bridge.copy_state is not None
    assert counterfactual.bridge.source_bias.eq(0).all()
    assert counterfactual.bridge.copy_state.bias.eq(0).all()
    assert baseline.bridge.source_bias.abs().sum() == 0
    torch.testing.assert_close(restored.bridge.source_bias, baseline.bridge.source_bias)


def test_zero_both_routes_recovers_matched_direct_projection_logits():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    direct_config = copy.deepcopy(config)
    direct_config["architecture"]["bridge_mode"] = "direct_projection"
    torch.manual_seed(61)
    full = EviSeqAFMR(config).eval()
    torch.manual_seed(61)
    direct = EviSeqAFMR(direct_config).eval()
    with torch.no_grad():
        full.bridge.focus_output.weight.normal_(std=0.2)
        full.bridge.feature_up.weight.normal_(std=0.1)
        full.bridge.depth_out.weight.normal_(std=0.1)
    batch = next(iter(build_loaders(config, split="validation", max_validation_examples=1)["validation"]))
    inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        with _counterfactual(full, "zero_source_bias_and_residuals"):
            ablated = full(**inputs)
        control = direct(**inputs)
    torch.testing.assert_close(ablated.logits, control.logits, rtol=0, atol=0)
