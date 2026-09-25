from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from decoder_baselines.config import validate_config
from decoder_baselines.data import CausalCollator, CausalSummarizationDataset, encode_prompt, read_jsonl, record_texts
from decoder_baselines.evaluate import (
    _config_from_suite,
    _filter_logits,
    _no_repeat_ngram_mask,
)
from decoder_baselines.train import _read_distributed_context
from decoder_baselines.suite import _distributed_train_command, _parse_gpu_ids, build_run_config


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


def test_jsonl_without_id_gets_stable_row_identifier(tmp_path: Path) -> None:
    path = tmp_path / "no_id.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"input": "source one", "content": "target one"}),
                json.dumps({"input": "source two", "content": "target two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_jsonl(path)
    data = {"source_field": "input", "target_field": "content", "id_field": "id", "clean_text": True}
    assert record_texts(rows[0], data)[0] == "row-00000001"
    assert record_texts(rows[1], data)[0] == "row-00000002"


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
    assert config["model"]["model_id"] == "nvidia/Nemotron-Labs-Diffusion-3B"
    assert config["model"]["name_or_path"].endswith("Nemotron-Labs-Diffusion-3B")
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
        "qwen3_1_7b": (2, 4),
        "qwen3_4b": (2, 4),
        "qwen3_8b": (1, 2),
        "llama3_3b": (4, 4),
        "llama3_8b": (1, 2),
        "nemotron_diffusion_3b": (2, 4),
        "nemotron_diffusion_8b": (1, 2),
    }
    for model_name, (train_batch, eval_batch) in expected.items():
        config, _ = build_run_config(suite, suite_path, model_name, "pubmed")
        validate_config(config)
        assert config["training"]["per_device_train_batch_size"] == train_batch
        assert config["generation"]["batch_size"] == eval_batch


def test_generation_batch_matrix_tracks_dataset_length() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    expected = {
        "qwen3_0_6b": {"pubmed": 8, "arxiv": 4, "booksum": 2, "govreport": 2},
        "qwen3_1_7b": {"pubmed": 4, "arxiv": 2, "booksum": 1, "govreport": 1},
        "qwen3_4b": {"pubmed": 4, "arxiv": 2, "booksum": 1, "govreport": 1},
        "nemotron_diffusion_3b": {"pubmed": 4, "arxiv": 2, "booksum": 1, "govreport": 1},
    }
    for model_name, datasets in expected.items():
        for dataset_name, batch_size in datasets.items():
            config, _ = build_run_config(suite, suite_path, model_name, dataset_name)
            validate_config(config)
            assert config["generation"]["batch_size"] == batch_size
            assert not isinstance(config["generation"]["batch_size"], dict)


def test_llama3_3b_uses_instruct_checkpoint() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "llama3_3b", "pubmed")
    validate_config(config)
    assert config["model"]["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert config["model"]["name_or_path"].endswith("Llama-3.2-3B-Instruct")


def test_dataset_exposes_length_estimates_for_padding_bucketing(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "short", "text": "a", "summary": "b"}),
                json.dumps({"id": "long", "text": "a" * 30, "summary": "b" * 10}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = CausalSummarizationDataset(path, FakeTokenizer(), _config_data())
    assert dataset.length_estimates == [2, 40]


def test_model_resolution_never_falls_back_to_hub_id(monkeypatch: Any) -> None:
    monkeypatch.delenv("QWEN3_0_6B_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "qwen3_0_6b", "pubmed")
    assert config["model"]["name_or_path"] != config["model"]["model_id"]
    assert Path(config["model"]["name_or_path"]).is_absolute()


def test_qwen3_17b_uses_local_model_and_suite_eval_config(monkeypatch: Any) -> None:
    monkeypatch.delenv("QWEN3_1_7B_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    config = _config_from_suite(suite_path, "qwen3_1_7b", "pubmed")
    validate_config(config)
    assert config["model"]["model_id"] == "Qwen/Qwen3-1.7B"
    assert Path(config["model"]["name_or_path"]).name == "Qwen3-1.7B"
    assert config["model"]["name_or_path"] != config["model"]["model_id"]


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


def test_evaluate_can_materialize_config_directly_from_suite() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    config = _config_from_suite(suite_path, "qwen3_4b", "pubmed")
    assert config["model"]["model_id"] == "Qwen/Qwen3-4B"
    assert config["data"]["source_prefix"].startswith("Summarize the following biomedical")
    assert config["generation"]["batch_size"] == 4
    assert config["generation"]["temperature"] == 0.0
    assert config["generation"]["top_k"] == 0
    assert config["generation"]["top_p"] == 1.0


def test_no_repeat_ngram_mask_matches_rowwise_semantics() -> None:
    generated = torch.tensor([[1, 2, 1, 2, 1], [3, 4, 5, 3, 4]])
    logits = torch.zeros(2, 8)
    masked = _no_repeat_ngram_mask(logits, generated, ngram_size=3)
    assert torch.isneginf(masked[0, 2])
    assert torch.isneginf(masked[1, 5])
    assert torch.isfinite(masked[0, 3])


def test_arxiv_context_budget_covers_source_and_target() -> None:
    suite_path = Path(__file__).parents[1] / "configs" / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    config, _ = build_run_config(suite, suite_path, "qwen3_0_6b", "arxiv")
    validate_config(config)
    assert config["data"]["max_source_length"] == 8096
    assert config["data"]["max_target_length"] == 512
    assert config["data"]["max_sequence_length"] == 9216


def test_gpu_spec_supports_two_distinct_devices() -> None:
    assert _parse_gpu_ids("0,1") == ["0", "1"]
    assert _parse_gpu_ids("3") == ["3"]


def test_ddp_launcher_wraps_only_training_command() -> None:
    base = ["python", "-m", "decoder_baselines.train", "--config", "run.yaml"]
    assert _distributed_train_command(base, world_size=1, python_executable="python") == base
    assert _distributed_train_command(
        base,
        world_size=2,
        python_executable="python",
        torchrun_path="/usr/bin/torchrun",
    ) == [
        "/usr/bin/torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m",
        "decoder_baselines.train",
        "--config",
        "run.yaml",
    ]


def test_distributed_context_reads_torchrun_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    context = _read_distributed_context()
    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert context.enabled is True
    assert context.is_main is False
