from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from direct_qwen.config import load_config
from direct_qwen.data import (
    CausalCollator,
    DirectCausalDataset,
    build_prompt_ids,
    left_pad_prompts,
)
from direct_qwen.evaluate import _checkpoint_contract, generate_prompt_batch, generation_contract
from direct_qwen.provenance import data_manifest
from direct_qwen.training import load_tokenizer_and_model, prepare_full_finetune
from llm2seq_v2.checkpoint import save_last_checkpoint

from llm2seq_v2.data import decoder_seed_ids, encode_source

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/wikilingua.yaml"


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    chat_template = "fake-chat"
    padding_side = "right"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self._pad_token = "<pad>"

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value
        self.pad_token_id = self.eos_token_id

    @property
    def eos_token(self) -> str:
        return "<eos>"

    def _ids(self, text: Any) -> list[int]:
        return [3 + (ord(character) % 43) for character in str(text)] or [3]

    def __call__(
        self,
        text: Any,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        return [47, *self._ids(messages[0]["content"]), 48]

    def batch_decode(self, values: torch.Tensor, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        return [" ".join(str(int(item)) for item in row if int(item) not in {0, 1, 2}) for row in values]


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.lm_head = torch.nn.Linear(8, 64, bias=False)
        self.config = SimpleNamespace(max_position_embeddings=4096, use_cache=False)
        self.gradient_checkpointing_called = False

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.gradient_checkpointing_called = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask
        assert use_cache is False
        logits = self.lm_head(self.embedding(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


def _small_data() -> dict[str, Any]:
    config = load_config(CONFIG)
    data = dict(config["data"])
    data.update(
        {
            "source_prefix": "SRC:",
            "decoder_instruction": "SUM",
            "decoder_prefix": "OUT:",
            "max_source_length": 40,
            "max_target_length": 12,
        }
    )
    return data


def test_config_is_exact_strong_full_finetune_control() -> None:
    config = load_config(CONFIG)
    assert config["model"]["base_model_id"] == "Qwen/Qwen3-0.6B"
    assert config["model"]["local_files_only"] is True
    assert config["training"]["mode"] == "full_finetune"
    assert config["training"]["num_train_epochs"] == 14
    assert config["training"]["batch_size"] == 32
    assert config["training"]["gradient_accumulation_steps"] == 4
    assert config["training"]["learning_rate"] == 1e-5
    assert config["checkpoint"] == {"save_best": False, "save_each_epoch": False, "save_last": True}
    assert config["data"]["max_source_length"] == 3072
    assert config["data"]["max_target_length"] == 384
    assert generation_contract(config) == {
        "max_new_tokens": 256,
        "min_new_tokens": 16,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
    }


def test_config_requires_an_existing_reference_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["contract"]["reference_config"] = "missing/config.yaml"
    path = tmp_path / "missing_reference.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Frozen EviSeq config not found"):
        load_config(path)


def test_canonical_split_sizes_are_recorded() -> None:
    manifest = data_manifest(load_config(CONFIG))
    assert manifest["train"]["num_examples"] == 13999
    assert manifest["validation"]["num_examples"] == 1680
    assert manifest["test"]["num_examples"] == 3901


def test_prompt_is_exact_source_segment_then_decoder_seed() -> None:
    tokenizer = FakeTokenizer()
    data = _small_data()
    source = "Một. Hai."
    source_ids, _, _ = encode_source(tokenizer, source, data)
    seed = decoder_seed_ids(tokenizer, data)
    prompt = build_prompt_ids(tokenizer, source, data)
    assert prompt == source_ids + seed
    assert source_ids[-1] == tokenizer.eos_token_id
    assert prompt[len(source_ids) :] == seed


def test_dataset_masks_every_prompt_token_and_supervises_target(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "x", "source": "Nguồn.", "target": "Đích."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tokenizer = FakeTokenizer()
    data = _small_data()
    dataset = DirectCausalDataset(path, tokenizer, data, max_sequence_length=128)
    item = dataset[0]
    prompt_length = len(build_prompt_ids(tokenizer, "Nguồn.", data))
    assert torch.all(item["labels"][:prompt_length] == -100)
    assert torch.equal(item["labels"][prompt_length:], item["input_ids"][prompt_length:])
    assert int(item["labels"][-1]) == tokenizer.eos_token_id


def test_dataset_refuses_silent_combined_context_truncation(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"source": "abc", "target": "xyz"}) + "\n", encoding="utf-8")
    dataset = DirectCausalDataset(path, FakeTokenizer(), _small_data(), max_sequence_length=3)
    with pytest.raises(ValueError, match="refusing to silently truncate"):
        dataset[0]


def test_training_collator_right_pads_and_keeps_ignore_mask() -> None:
    collator = CausalCollator(0, pad_to_multiple_of=4)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "labels": torch.tensor([-100, 5, 6]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.tensor([-100, 8]),
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[4, 5, 6, 0], [7, 8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 5, 6, -100], [-100, 8, -100, -100]]


def test_ce_backward_reaches_embedding_and_lm_head() -> None:
    model = TinyCausalLM()
    batch = {
        "input_ids": torch.tensor([[4, 5, 6, 7]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 6, 7]]),
    }
    (model(**batch, use_cache=False).loss / 2).backward()
    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert float(model.embedding.weight.grad.abs().sum()) > 0
    assert float(model.lm_head.weight.grad.abs().sum()) > 0


def test_left_padding_and_generation_strip_the_entire_padded_prompt() -> None:
    class Generator(torch.nn.Module):
        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            assert kwargs["do_sample"] is False
            assert kwargs["num_beams"] == 1
            suffix = torch.tensor([[51, 52], [53, 54]], device=input_ids.device)
            return torch.cat([input_ids, suffix], dim=1)

    tokenizer = FakeTokenizer()
    prompts = [[9, 10], [11, 12, 13]]
    padded, mask = left_pad_prompts(prompts, tokenizer.pad_token_id)
    assert padded.tolist() == [[0, 9, 10], [11, 12, 13]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    generated = generate_prompt_batch(
        Generator(), tokenizer, prompts, generation_contract(load_config(CONFIG)), torch.device("cpu")
    )
    assert generated.tolist() == [[51, 52], [53, 54]]


def test_loader_passes_local_only_and_never_a_hub_token() -> None:
    calls: dict[str, Any] = {}

    class TokenizerFactory:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> FakeTokenizer:
            calls["tokenizer"] = (name, kwargs)
            return FakeTokenizer()

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> TinyCausalLM:
            calls["model"] = (name, kwargs)
            return TinyCausalLM()

    config = load_config(CONFIG)
    tokenizer, model = load_tokenizer_and_model(
        config,
        tokenizer_factory=TokenizerFactory,
        model_factory=ModelFactory,
    )
    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(model, TinyCausalLM)
    for _, kwargs in calls.values():
        assert kwargs["local_files_only"] is True
        assert "token" not in kwargs
    assert calls["model"][1]["dtype"] == torch.float32


def test_full_finetune_manifest_and_last_checkpoint_are_complete(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    model = TinyCausalLM()
    model.embedding.weight.requires_grad_(False)
    manifest = prepare_full_finetune(model, config)
    assert model.gradient_checkpointing_called is True
    assert manifest["full_finetune"] is True
    assert manifest["trainable_parameter_elements"] == manifest["unique_parameter_elements"]

    checkpoint = tmp_path / "last.pt"
    data = {name: {"num_examples": 1} for name in ("train", "validation", "test")}
    save_last_checkpoint(model, checkpoint, config, 14, 7, data)
    (tmp_path / "parameter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    from direct_qwen.provenance import tokenizer_manifest

    tokenizer = FakeTokenizer()
    (tmp_path / "tokenizer_manifest.json").write_text(
        json.dumps(tokenizer_manifest(tokenizer, config)), encoding="utf-8"
    )
    payload, saved = _checkpoint_contract(TinyCausalLM(), tokenizer, checkpoint, config)
    assert payload["checkpoint_role"] == "last"
    assert payload["epoch"] == 14
    assert saved["full_finetune"] is True


def test_runner_is_offline_and_contains_no_upload_operation() -> None:
    script = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "push_to_hub" not in script
    assert "huggingface-cli" not in script
    assert "curl " not in script
    assert "wget " not in script
