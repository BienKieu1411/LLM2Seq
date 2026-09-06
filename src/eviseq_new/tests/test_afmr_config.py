from pathlib import Path

import pytest
from eviseq_afmr.config import load_config, resolve_path


def test_smoke_config_is_valid():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    assert config["architecture"]["name"] == "afmr_value_anchor"
    assert config["generation"]["num_beams"] == 1


def test_pubmed_recipe_matches_t5gemma_prompt_and_decode_contract():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_pubmed.yaml")
    assert config["data"]["encoder_prefix"] == (
        "Summarize the following biomedical research article into a concise, factual abstract. "
        "Preserve the key objective, methods, results, and conclusion; do not add information.\nArticle:\n"
    )
    assert config["data"]["decoder_prompt"]
    assert config["data"]["decoder_chat_template"] is True
    assert config["data"]["detokenize"] is True
    assert config["model"]["dtype"] == "float32"
    assert config["model"]["compute_dtype"] == "bfloat16"
    assert config["generation"]["min_new_tokens"] == 32
    assert config["generation"]["repetition_penalty"] == 1.05
    assert config["generation"]["no_repeat_ngram_size"] == 3


@pytest.mark.parametrize(
    ("afmr_name", "t5_name", "total_epochs"),
    (
        ("afmr_pubmed.yaml", "pubmed_full_1b_1b_4096.yaml", 4),
        ("afmr_cnndm.yaml", "cnndm_full_1b_1b_4096.yaml", 6),
        ("afmr_wikilingua.yaml", "wikilingua_full_3072.yaml", 6),
    ),
)
def test_task_recipes_match_t5gemma_protocol(afmr_name, t5_name, total_epochs):
    import yaml

    root = Path(__file__).parents[3]
    afmr = load_config(Path(__file__).parents[1] / "configs" / afmr_name)
    t5 = yaml.safe_load((root / "src" / "T5Gemma" / "configs" / t5_name).read_text(encoding="utf-8"))
    assert afmr["data"]["encoder_prefix"] == t5["data"]["source_prefix"]
    assert afmr["data"]["decoder_prompt"]
    for key in ("max_new_tokens", "min_new_tokens", "repetition_penalty", "no_repeat_ngram_size"):
        assert afmr["generation"][key] == t5["generation"][key]
    assert afmr["training"]["interface_warmup_epochs"] + afmr["training"]["full_finetune_epochs"] == total_epochs


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
