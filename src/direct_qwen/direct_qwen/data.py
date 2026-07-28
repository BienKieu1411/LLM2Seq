"""Causal examples built from the two exact EviSeq prompt segments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from llm2seq_v2.data import clean_text, decoder_seed_ids, encode_source, read_jsonl


def build_prompt_ids(tokenizer: Any, source: str, data_config: Dict[str, Any]) -> List[int]:
    """Concatenate the exact EviSeq source input and exact EviSeq decoder seed.

    The terminal source EOS is deliberately retained. Removing it or wrapping
    the document in a new chat message would create a different prompt control.
    """

    source_ids, _, _ = encode_source(tokenizer, source, data_config)
    seed = decoder_seed_ids(tokenizer, data_config)
    prompt = [int(value) for value in (*source_ids, *seed)]
    if not prompt:
        raise ValueError("Direct Qwen prompt cannot be empty")
    return prompt


def encode_target(tokenizer: Any, target: str, max_target_length: int) -> List[int]:
    eos_id = tokenizer.eos_token_id
    budget = int(max_target_length) - int(eos_id is not None)
    ids = list(
        tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, budget),
        )["input_ids"]
    )
    if eos_id is not None:
        ids.append(int(eos_id))
    if not ids:
        raise ValueError("Direct Qwen target cannot be empty")
    return [int(value) for value in ids]


class DirectCausalDataset(Dataset):
    """Prompt-masked causal language-model examples."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        data_config: Dict[str, Any],
        *,
        max_sequence_length: int,
        max_examples: int = 0,
    ) -> None:
        self.examples = read_jsonl(path, max_examples=max_examples)
        self.tokenizer = tokenizer
        self.config = data_config
        self.max_sequence_length = int(max_sequence_length)
        self.clean_metadata = bool(data_config.get("clean_wikihow_metadata", True))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.examples[index]
        source = clean_text(row["source"], self.clean_metadata)
        target = clean_text(row["target"], self.clean_metadata)
        prompt = build_prompt_ids(self.tokenizer, source, self.config)
        target_ids = encode_target(
            self.tokenizer,
            target,
            int(self.config["max_target_length"]),
        )
        ids = prompt + target_ids
        if len(ids) > self.max_sequence_length:
            raise ValueError(
                f"Combined EviSeq prompt/target length {len(ids)} exceeds the direct model context "
                f"{self.max_sequence_length}; refusing to silently truncate a controlled baseline"
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor([-100] * len(prompt) + target_ids, dtype=torch.long),
        }


class CausalCollator:
    """Right-pad causal training examples; labels use the standard -100 mask."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))

    def __call__(self, examples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        longest = max(int(item["input_ids"].numel()) for item in examples)
        length = int(math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        input_ids = torch.full((len(examples), length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
        labels = torch.full((len(examples), length), -100, dtype=torch.long)
        for row, item in enumerate(examples):
            size = int(item["input_ids"].numel())
            input_ids[row, :size] = item["input_ids"]
            attention_mask[row, :size] = item["attention_mask"]
            labels[row, :size] = item["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def left_pad_prompts(
    prompts: Sequence[Sequence[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left padding is required for batched decoder-only generation."""

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Generation prompts must be non-empty")
    length = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), length), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        values = torch.tensor(list(prompt), dtype=torch.long)
        input_ids[row, length - len(prompt) :] = values
        attention_mask[row, length - len(prompt) :] = 1
    return input_ids, attention_mask
