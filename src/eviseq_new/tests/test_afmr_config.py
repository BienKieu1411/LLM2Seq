from pathlib import Path

import pytest
from eviseq_afmr.config import load_config, resolve_path


def test_smoke_config_is_valid():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    assert config["architecture"]["name"] == "afmr_v1"
    assert config["generation"]["num_beams"] == 1


def test_packaged_paths_do_not_depend_on_cwd_or_existing_files(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_pubmed.yaml")
    expected = Path(__file__).parents[1] / "datasets/pubmed/train.jsonl"
    monkeypatch.chdir(tmp_path)
    assert resolve_path(config["data"]["train_file"], config) == expected.resolve()


def test_legacy_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("removed_section: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown AFMR top-level"):
        load_config(path)


def test_removed_objective_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("objective: {allocation_weight: 0.1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown AFMR top-level"):
        load_config(path)


def test_focus_windows_must_be_strictly_increasing(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {encoder_name: e, decoder_name: d}
encoder: {backend: pretrained_native}
architecture: {name: afmr_v1, focus_windows: [128, 32], depth_gate_init: 0.02, depth_gate_max: 0.15, feature_gate_init: 0.02, feature_gate_max: 0.2, focus_strength_init: 0.1, focus_strength_max: 1.0, temperature_init: 1.0, temperature_min: 0.5, temperature_max: 2.0}
decoder: {cross_attention_every: 1, initialize_cross_from_self: true}
training: {batch_size: 1, gradient_accumulation_steps: 1}
data: {train_file: a, validation_file: b, test_file: c, source_field: text, target_field: summary}
generation: {num_beams: 1, do_sample: false}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_config(path)
