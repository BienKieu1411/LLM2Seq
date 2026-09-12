from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from decoder_baselines.config import validate_config
from decoder_baselines.data import CausalCollator, CausalSummarizationDataset, encode_prompt
from decoder_baselines.evaluate import _filter_logits
from decoder_baselines.suite import build_run_config


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[int]]:
        values = [3 + (ord(char) % 47) for char in str(text)] or [3]
        if add_special_tokens:
            values = [1, *values]
        if truncation and max_length is not None:
            values = values[: int(max_length)]
        return {"input_ids": values}


def _config_data() -> dict[str, Any]:
    return {
        "source_field": "text",
        "target_field": "summary",
        "id_field": "id",
        "source_prefix": "Summarize:\nArticle:\n",
        "prompt_suffix": "",
        "max_source_length": 128,
        "max_target_length": 32,
        "max_sequence_length": 192,
        "clean_text": True,
    }


def test_prompt_masking_preserves_t5gemma_source_instruction(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "source", "summary": "target"}) + "\n", encoding="utf-8")
    tokenizer = FakeTokenizer()
    data = _config_data()
    dataset = CausalSummarizationDataset(path, tokenizer, data)
    item = dataset[0]
    prompt_length = len(encode_prompt(tokenizer, "source", data))
    assert torch.all(item["labels"][:prompt_length] == -100)
    assert torch.equal(item["labels"][prompt_length:], item["input_ids"][prompt_length:])
    collated = CausalCollator(tokenizer.pad_token_id)([item, item])
    assert collated["input_ids"].shape[0] == 2
    assert collated["labels"].shape == collated["input_ids"].shape


def test_top_k_and_top_p_filtering_keeps_valid_logits() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    filtered = _filter_logits(logits, top_k=2, top_p=1.0)
    assert torch.isfinite(filtered[0, 2:]).all()
    assert torch.isneginf(filtered[0, :2]).all()
    nucleus = _filter_logits(logits, top_k=0, top_p=0.7)
    assert torch.isfinite(nucleus).any()


def test_suite_materializes_nemotron_ar_recipe() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "nemotron_diffusion_3b", "pubmed")
    validate_config(config)
    assert config["model"]["family"] == "nemotron_diffusion"
    assert config["model"]["diffusion_paradigm"] == "autoregressive"
    assert config["data"]["source_prefix"].startswith("Summarize the following biomedical")
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_k"] == 0
    assert config["generation"]["top_p"] == 1.0


def test_suite_materializes_qwen3_06b_recipe() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "qwen3_0_6b", "pubmed")
    validate_config(config)
    assert config["model"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert config["model"]["family"] == "causal_lm"
    assert config["training"]["per_device_train_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["generation"]["batch_size"] == 8


def test_training_and_eval_batch_sizes_are_independent() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    expected = {
        "qwen3_0_6b": (4, 8),
        "qwen3_4b": (2, 4),
        "qwen3_8b": (1, 2),
        "llama3_3b": (4, 4),
        "llama3_8b": (1, 2),
        "nemotron_diffusion_3b": (2, 1),
        "nemotron_diffusion_8b": (1, 1),
    }
    for model_name, (train_batch, eval_batch) in expected.items():
        config, _ = build_run_config(suite, suite_path, model_name, "pubmed")
        validate_config(config)
        assert config["training"]["per_device_train_batch_size"] == train_batch
        assert config["generation"]["batch_size"] == eval_batch


def test_all_models_share_t5gemma_decode_controls() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    for model_name in suite["model_order"]:
        config, _ = build_run_config(suite, suite_path, model_name, "pubmed")
        validate_config(config)
        generation = config["generation"]
        assert generation["do_sample"] is False
        assert generation["temperature"] == 0.0
        assert generation["top_k"] == 0
        assert generation["top_p"] == 1.0


def test_arxiv_context_budget_covers_source_and_target() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "qwen3_0_6b", "arxiv")
    validate_config(config)
    assert config["data"]["max_source_length"] == 8096
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["max_sequence_length"] == 9216
