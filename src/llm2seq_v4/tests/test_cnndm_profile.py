"""Offline contract tests for the V4 CNN/DailyMail profile."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from llm2seq_v4.config import load_config, validate_config


def test_cnndm_profile_matches_t5gemma_without_wikilingua_leakage() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/cnndm_qwen3_embedding_0_6b_psb_4096.yaml")
    t5_config = yaml.safe_load((root.parent / "T5Gemma/configs/cnndm_full_1b_1b_4096.yaml").read_text(encoding="utf-8"))

    assert config["adapter"]["num_bidirectional_layers"] == 6
    assert config["training"]["interface_warmup_epochs"] == 1
    assert config["training"]["full_finetune_epochs"] == 5
    assert config["training"]["interface_warmup_epochs"] + config["training"]["full_finetune_epochs"] == 6
    assert config["training"]["batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["validation_batch_size"] == 16
    assert config["generation"]["batch_size"] == 16
    assert config["data"]["max_source_length"] == 4096
    assert config["data"]["max_target_length"] == 256
    assert config["data"]["clean_wikihow_metadata"] is False
    assert config["data"]["source_prefix"] == t5_config["data"]["source_prefix"]
    for field in ("min_new_tokens", "max_new_tokens", "repetition_penalty", "no_repeat_ngram_size"):
        assert config["generation"][field] == t5_config["generation"][field]

    assert set(config["benchmark"]) == {"parameters", "reference_only"}
    assert config["benchmark"]["reference_only"]["rouge2"] == 19.396
    assert config["benchmark"]["reference_only"]["comparability"] == ("pending_exact_test_count_and_fingerprint")


def test_reference_only_benchmark_is_validated_but_does_not_require_a_guessed_count() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/cnndm_qwen3_embedding_0_6b_psb_4096.yaml")
    assert "num_examples" not in config["benchmark"]["reference_only"]

    invalid = copy.deepcopy(config)
    invalid["benchmark"]["reference_only"]["rouge2"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["benchmark"]["reference_only"]["backend"] = "rouge-score"
    with pytest.raises(ValueError, match="Perl ROUGE-1.5.5"):
        validate_config(invalid)


def test_cnndm_modes_are_exposed_without_hub_uploads_or_false_paper_comparison() -> None:
    root = Path(__file__).parents[1]
    v4_script = (root / "run.sh").read_text(encoding="utf-8")
    project_script = (root.parent / "run.sh").read_text(encoding="utf-8")
    for mode in ("cnndm-prepare", "cnndm-smoke", "cnndm|cnndm-full", "cnndm-train", "cnndm-eval"):
        assert mode in v4_script
    for mode in ("llm2seq-v4-cnndm-prepare", "llm2seq-v4-cnndm-smoke", "llm2seq-v4-cnndm"):
        assert mode in project_script
    assert "push_to_hub" not in v4_script
    assert 'target = load_config(sys.argv[1]).get("benchmark", {}).get("paper", {})' in v4_script
