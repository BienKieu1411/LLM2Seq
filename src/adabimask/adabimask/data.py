"""Dataset utilities for encoder-decoder and direct-causal experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


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


class DirectSummarizationDataset(Dataset):
    """Mask prompt tokens and supervise only direct-Qwen summary tokens."""

    def __init__(self, path: str | Path, tokenizer: Any, data_config: Dict[str, Any]):
        self.examples = read_jsonl(path)
        self.tokenizer = tokenizer
        self.source_prefix = str(data_config.get("source_prefix", "Summarize the following document:\n"))
        self.target_prefix = str(data_config.get("target_prefix", "\nSummary:\n"))
        self.max_source_length = int(data_config.get("max_source_length", 3072))
        self.max_target_length = int(data_config.get("max_target_length", 384))

    def __len__(self) -> int:
        return len(self.examples)

    def _prompt(self, source: str) -> str:
        return f"{self.source_prefix}{source}{self.target_prefix}"

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt_ids = self.tokenizer(
            self._prompt(str(example["source"])),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_length,
        )["input_ids"]
        target_ids = self.tokenizer(
            str(example["target"]),
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
