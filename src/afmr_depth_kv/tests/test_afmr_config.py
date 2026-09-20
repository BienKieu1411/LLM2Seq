from pathlib import Path

import pytest
from eviseq_afmr.config import load_config, resolve_path, validate_config


def test_smoke_config_is_valid():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    assert config["architecture"]["name"] == "layerwise_coupled_depth_kv"
    assert config["generation"]["num_beams"] == 1


def test_8192_avg_config_exposes_system_prompt_and_context_budget():
    config = load_config(Path(__file__).parents[1] / "configs" / "8192_avg.yaml")
    assert config["data"]["system_prompt_field"] == "system_prompt"
    assert "Bạn là trợ lý chuyên tóm tắt" in config["data"]["system_prompt"]
    assert config["data"]["max_source_length"] == 8192
    assert config["data"]["decoder_chat_template"] is True


def test_pubmed_recipe_matches_t5gemma_prompt_and_decode_contract():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_pubmed.yaml")
    assert config["training"]["salience_loss_weight"] == 0.0
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
    assert config["generation"]["do_sample"] is False
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_k"] == 0
    assert config["generation"]["top_p"] == 1.0


def test_arxiv_recipe_uses_long_context_and_greedy_decode_contract():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_arxiv.yaml")
    assert config["data"]["encoder_prefix"] == (
        "Summarize the following scientific research article into a concise, factual abstract. "
        "Preserve the key objective, methods, results, and conclusion; do not add information.\nArticle:\n"
    )
    assert config["data"]["max_source_length"] == 8192
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["detokenize"] is True
    assert config["training"]["interface_warmup_epochs"] + config["training"]["full_finetune_epochs"] == 5
    assert config["generation"]["do_sample"] is False
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_k"] == 0
    assert config["generation"]["top_p"] == 1.0


def test_sampling_config_requires_positive_temperature():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["generation"].update(do_sample=True, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        validate_config(config)


def test_direct_projection_bridge_ablation_is_valid_and_checkpoint_distinguishable():
    from eviseq_afmr.training.checkpoint import architecture_spec

    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["architecture"]["bridge_mode"] = "direct_projection"
    validate_config(config)
    assert config["architecture"]["bridge_mode"] == "direct_projection"
    assert architecture_spec(config)["bridge_mode"] == "direct_projection"


def test_unknown_bridge_mode_is_rejected():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["architecture"]["bridge_mode"] = "unknown"
    with pytest.raises(ValueError, match="bridge_mode"):
        validate_config(config)


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


def test_ce_only_contract_rejects_salience_loss():
    config = load_config(Path(__file__).parents[1] / "configs" / "afmr_smoke.yaml")
    config["training"]["salience_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="CE only"):
        validate_config(config)
