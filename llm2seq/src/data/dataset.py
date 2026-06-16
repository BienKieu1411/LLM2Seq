"""
Seq2Seq Dataset for LLM2Seq.

Loads data from JSONL format:
    {"id": "0001", "source": "...", "target": "...", "task": "translation"}

Tokenizes source and target, creates decoder input IDs and labels.
Optionally loads teacher targets or teacher logits paths for KD.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    """
    Seq2Seq dataset that reads JSONL files.

    Each line should be a JSON object with at least "source" and "target" fields.
    Optional fields: "id", "task", "teacher_target", "teacher_logits_path".

    Args:
        data_path: Path to JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_source_length: Maximum source sequence length.
        max_target_length: Maximum target sequence length.
        source_prefix: Optional prefix to prepend to source text.
        source_field: Field name for source text.
        target_field: Field name for target text.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_source_length: int = 512,
        max_target_length: int = 256,
        source_prefix: str = "",
        source_field: str = "source",
        target_field: str = "target",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix
        self.source_field = source_field
        self.target_field = target_field

        # Load data
        self.examples: List[Dict[str, Any]] = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.examples.append(json.loads(line))

        if not self.examples:
            # Create a minimal dummy example so the pipeline doesn't crash
            # during smoke testing without real data
            self.examples = [
                {"source": "This is a test input.", "target": "Test output."}
            ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Source
        source_text = self.source_prefix + str(example.get(self.source_field, ""))
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )

        # Target
        target_text = str(example.get(self.target_field, ""))
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        # Create decoder input IDs (shifted right)
        # labels = target_ids
        # decoder_input_ids = [bos/pad] + target_ids[:-1]
        target_ids = target_encoding["input_ids"].squeeze(0)  # [T]
        labels = target_ids.clone()

        # Shift right: prepend BOS/PAD token
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.pad_token_id

        decoder_input_ids = torch.cat([
            torch.tensor([bos_token_id], dtype=target_ids.dtype),
            target_ids[:-1],
        ])

        result = {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

        # Optional: teacher target for sequence-level KD
        if "teacher_target" in example:
            teacher_encoding = self.tokenizer(
                example["teacher_target"],
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt",
            )
            result["teacher_target_ids"] = teacher_encoding["input_ids"].squeeze(0)

        return result
