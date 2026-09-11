from __future__ import annotations

import math
import unicodedata
from typing import Any

import torch

COPY_INPUT_KEYS = (
    "copy_token_ids",
    "copy_token_mask",
    "copy_encoder_indices",
    "copy_token_indices",
    "copy_alignment_weights",
)


def _word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return character == "_" or character.isalnum() or category.startswith("M")


def _truncation_splits_token(source: str, boundary: int) -> bool:
    """Return whether a character truncation boundary falls inside a word token."""
    if not 0 < boundary < len(source):
        return False
    left, right = source[boundary - 1], source[boundary]
    if _word_character(left) and _word_character(right):
        return True
    connectors = {"'", "’", "-", "‐", "‑"}
    if left in connectors and boundary >= 2:
        return _word_character(source[boundary - 2]) and _word_character(right)
    if right in connectors and boundary + 1 < len(source):
        return _word_character(left) and _word_character(source[boundary + 1])
    return False


def align_copy_tokens(source: str, prefix_length: int, encoder_offsets, tokenizer: Any) -> dict[str, list]:
    source = str(source)
    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    spans = []
    for index, pair in enumerate(encoder_offsets):
        if len(pair) != 2:
            raise ValueError("encoder offsets must contain (start, end) pairs")
        raw_start, raw_end = (int(value) for value in pair)
        if raw_end <= raw_start or raw_end <= prefix_length:
            continue
        start = max(0, raw_start - prefix_length)
        end = min(len(source), raw_end - prefix_length)
        if end > start:
            spans.append((index, start, end))
    spans.sort(key=lambda item: (item[1], item[2], item[0]))
    visible_end = max((end for _, _, end in spans), default=0)
    if not spans:
        return dict(copy_token_ids=[], copy_encoder_indices=[], copy_token_indices=[], copy_alignment_weights=[])
    encoded = tokenizer(source[:visible_end], add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in encoded:
        raise ValueError("Grounded copy requires a fast decoder tokenizer with offset mapping")
    special = set(getattr(tokenizer, "all_special_ids", ()))
    special.update(
        value
        for value in (
            getattr(tokenizer, name, None) for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")
        )
        if value is not None
    )
    ids, enc_indices, token_indices, weights = [], [], [], []
    cursor = 0
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        token = int(token)
        start, end = int(start), int(end)
        if token in special or end <= start:
            continue
        if _truncation_splits_token(source, visible_end) and end >= visible_end:
            continue
        start, end = max(0, start), min(visible_end, end)
        if end <= start:
            continue
        while cursor < len(spans) and spans[cursor][2] <= start:
            cursor += 1
        overlap = []
        covered = set()
        j = cursor
        while j < len(spans) and spans[j][1] < end:
            index, left, right = spans[j]
            left, right = max(left, start), min(right, end)
            if right > left:
                overlap.append((index, right - left))
                covered.update(range(left, right))
            j += 1
        if not overlap or any(not source[k].isspace() and k not in covered for k in range(start, end)):
            continue
        destination = len(ids)
        ids.append(int(token))
        total = sum(size for _, size in overlap)
        for index, size in overlap:
            enc_indices.append(index)
            token_indices.append(destination)
            weights.append(size / total)
    return dict(
        copy_token_ids=ids,
        copy_encoder_indices=enc_indices,
        copy_token_indices=token_indices,
        copy_alignment_weights=weights,
    )


def pad_copy_alignments(rows: list[dict[str, list]]) -> dict[str, torch.Tensor]:
    required = ("copy_token_ids", "copy_encoder_indices", "copy_token_indices", "copy_alignment_weights")
    normalized_rows = []
    for row_index, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"copy alignment row {row_index} is missing {missing}")
        token_ids = [int(value) for value in row["copy_token_ids"]]
        encoder_indices = [int(value) for value in row["copy_encoder_indices"]]
        token_indices = [int(value) for value in row["copy_token_indices"]]
        alignment_weights = [float(value) for value in row["copy_alignment_weights"]]
        edge_lengths = {len(encoder_indices), len(token_indices), len(alignment_weights)}
        if len(edge_lengths) != 1:
            raise ValueError(f"copy alignment row {row_index} edge arrays must have equal lengths")
        if any(value < 0 for value in token_ids + encoder_indices + token_indices):
            raise ValueError(f"copy alignment row {row_index} contains a negative index or token ID")
        if token_indices and (not token_ids or max(token_indices) >= len(token_ids)):
            raise ValueError(f"copy alignment row {row_index} points outside candidate tokens")
        if any(not math.isfinite(value) or value < 0.0 for value in alignment_weights):
            raise ValueError(f"copy alignment row {row_index} weights must be finite and non-negative")
        normalized_rows.append(
            {
                "copy_token_ids": token_ids,
                "copy_encoder_indices": encoder_indices,
                "copy_token_indices": token_indices,
                "copy_alignment_weights": alignment_weights,
            }
        )
    rows = normalized_rows
    result = {}
    for key in ("copy_token_ids", "copy_encoder_indices", "copy_token_indices", "copy_alignment_weights"):
        width = max(1, max((len(row[key]) for row in rows), default=0))
        dtype = torch.float32 if key == "copy_alignment_weights" else torch.long
        tensor = torch.zeros(len(rows), width, dtype=dtype)
        for i, row in enumerate(rows):
            tensor[i, : len(row[key])] = torch.tensor(row[key], dtype=dtype)
        result[key] = tensor
    result["copy_token_mask"] = (
        torch.arange(result["copy_token_ids"].shape[1])[None, :]
        < torch.tensor([len(row["copy_token_ids"]) for row in rows])[:, None]
    )
    return result
