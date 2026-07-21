"""Dataset utilities for encoder-decoder and direct-causal experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


_WIKIHOW_IMAGE_OBJECT = re.compile(r'\{\s*"\s*smallUrl\s*"[^{}]*\}', flags=re.IGNORECASE)


def clean_wikihow_metadata(text: str) -> str:
    """Remove serialized wikiHow image objects without deleting surrounding text."""

    cleaned = _WIKIHOW_IMAGE_OBJECT.sub(" ", str(text))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _configured_text(text: Any, data_config: Dict[str, Any]) -> str:
    value = str(text)
    if bool(data_config.get("clean_wikihow_metadata", False)):
        return clean_wikihow_metadata(value)
    return value


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def prompt_token_ids(tokenizer: Any, source: str, data_config: Dict[str, Any]) -> List[int]:
    """Build one concise dataset-specific prompt and preserve its output label."""

    source_prefix = str(data_config.get("source_prefix", "Summarize concisely.\nDocument:\n"))
    target_prefix = str(data_config.get("target_prefix", "\nSummary:\n"))
    max_length = int(data_config.get("max_source_length", 3072))
    if bool(data_config.get("use_chat_template", True)) and getattr(tokenizer, "chat_template", None):
        sentinel = "<ADABIMASK_SOURCE_7F3A>"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{source_prefix}{sentinel}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if sentinel not in rendered:
            raise RuntimeError("Tokenizer chat template changed the source sentinel")
        prompt_start, prompt_end = rendered.split(sentinel, maxsplit=1)
        prefix_ids = tokenizer(prompt_start, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(f"{prompt_end}{target_prefix}", add_special_tokens=False)["input_ids"]
    else:
        prefix_ids = tokenizer(source_prefix, add_special_tokens=True)["input_ids"]
        suffix_ids = tokenizer(target_prefix, add_special_tokens=False)["input_ids"]
    source_budget = max(1, max_length - len(prefix_ids) - len(suffix_ids))
    source_ids = tokenizer(
        source,
        add_special_tokens=False,
        truncation=True,
        max_length=source_budget,
    )["input_ids"]
    return list(prefix_ids) + list(source_ids) + list(suffix_ids)


class PromptedSeq2SeqDataset(Dataset):
    """Encoder-decoder supervision with the exact same prompt at train/test."""

    def __init__(self, path: str | Path, tokenizer: Any, data_config: Dict[str, Any]):
        self.examples = read_jsonl(path)
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_target_length = int(data_config.get("max_target_length", 384))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        source = _configured_text(example["source"], self.data_config)
        target = _configured_text(example["target"], self.data_config)
        source_ids = prompt_token_ids(self.tokenizer, source, self.data_config)
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_target_length - 1),
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids = list(target_ids) + [self.tokenizer.eos_token_id]
        start_token_id = self.tokenizer.bos_token_id
        if start_token_id is None:
            start_token_id = self.tokenizer.pad_token_id
        if start_token_id is None:
            start_token_id = self.tokenizer.eos_token_id
        if start_token_id is None:
            raise ValueError("Tokenizer has no BOS, PAD, or EOS token for decoder start")

        input_ids = torch.tensor(source_ids, dtype=torch.long)
        labels = torch.tensor(target_ids, dtype=torch.long)
        decoder_input_ids = torch.cat(
            [torch.tensor([start_token_id], dtype=torch.long), labels[:-1]], dim=0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


class Seq2SeqCollator:
    """Dynamic source/target padding without depending on the old llm2seq package."""

    def __init__(
        self,
        pad_token_id: int,
        max_source_length: int = 3072,
        max_target_length: int = 384,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = int(pad_token_id)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        self.label_pad_token_id = int(label_pad_token_id)

    @staticmethod
    def _pad(
        tensors: List[torch.Tensor],
        max_length: int,
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_length = min(max(tensor.numel() for tensor in tensors), max_length)
        padded = []
        masks = []
        for tensor in tensors:
            tensor = tensor[:target_length]
            valid_length = tensor.numel()
            padding = torch.full(
                (target_length - valid_length,),
                pad_value,
                dtype=tensor.dtype,
            )
            padded.append(torch.cat([tensor, padding]))
            masks.append(
                torch.cat(
                    [
                        torch.ones(valid_length, dtype=torch.long),
                        torch.zeros(target_length - valid_length, dtype=torch.long),
                    ]
                )
            )
        return torch.stack(padded), torch.stack(masks)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = self._pad(
            [feature["input_ids"] for feature in features],
            self.max_source_length,
            self.pad_token_id,
        )
        decoder_input_ids, decoder_attention_mask = self._pad(
            [feature["decoder_input_ids"] for feature in features],
            self.max_target_length,
            self.pad_token_id,
        )
        labels, _ = self._pad(
            [feature["labels"] for feature in features],
            self.max_target_length,
            self.label_pad_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


class DirectSummarizationDataset(Dataset):
    """Mask prompt tokens and supervise only direct-Qwen summary tokens."""

    def __init__(self, path: str | Path, tokenizer: Any, data_config: Dict[str, Any]):
        self.examples = read_jsonl(path)
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_source_length = int(data_config.get("max_source_length", 3072))
        self.max_target_length = int(data_config.get("max_target_length", 384))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        source = _configured_text(example["source"], self.data_config)
        target = _configured_text(example["target"], self.data_config)
        prompt_ids = prompt_token_ids(self.tokenizer, source, self.data_config)
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_target_length - 1),
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


class DirectCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].numel() for feature in features)
        batch_ids, batch_masks, batch_labels = [], [], []
        for feature in features:
            length = feature["input_ids"].numel()
            pad_length = max_length - length
            batch_ids.append(
                torch.cat([feature["input_ids"], torch.full((pad_length,), self.pad_token_id, dtype=torch.long)])
            )
            batch_masks.append(torch.cat([feature["attention_mask"], torch.zeros(pad_length, dtype=torch.long)]))
            batch_labels.append(torch.cat([feature["labels"], torch.full((pad_length,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_masks),
            "labels": torch.stack(batch_labels),
        }
