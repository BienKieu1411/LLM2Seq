"""Two-tokenizer, sentence-aligned summarization data for LLM2Seq-v2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WIKIHOW_IMAGE = re.compile(r'\{\s*"\s*smallUrl\s*"[^{}]*\}', flags=re.IGNORECASE)


def read_jsonl(path: str | Path, max_examples: int = 0) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if "source" not in row or "target" not in row:
                raise ValueError(f"Missing source/target at {path}:{line_number}")
            rows.append(row)
            if max_examples > 0 and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def dataset_fingerprint(path: str | Path, max_examples: int = 0) -> Dict[str, Any]:
    rows = read_jsonl(path, max_examples=max_examples)
    digest = hashlib.sha256()
    for row in rows:
        canonical = {key: row.get(key) for key in ("id", "source", "target")}
        digest.update(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "path": str(Path(path).resolve()),
        "num_examples": len(rows),
        "sha256": digest.hexdigest(),
    }


def clean_text(text: Any, enabled: bool) -> str:
    value = str(text)
    if enabled:
        value = _WIKIHOW_IMAGE.sub(" ", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\s+([.,;:!?])", r"\1", value)
    return value.strip()


def split_units(text: str) -> List[str]:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in _SENTENCE.split(str(text).strip()) if part.strip()]


def _tokens(text: str) -> List[str]:
    return [value.lower() for value in _WORD.findall(text)]


def _ngrams(tokens: Sequence[str], order: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _f1(candidate: Counter, reference: Counter) -> float:
    overlap = sum((candidate & reference).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, sum(candidate.values()))
    recall = overlap / max(1, sum(reference.values()))
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def greedy_evidence_labels(
    units: Sequence[str],
    target: str,
    max_units: int = 12,
    rouge1_weight: float = 0.5,
    rouge2_weight: float = 0.5,
) -> List[float]:
    """Greedy sentence coverage labels; reference is used during training only."""

    if not units:
        return []
    target_tokens = _tokens(target)
    reference = (_ngrams(target_tokens, 1), _ngrams(target_tokens, 2))
    counters = [(_ngrams(_tokens(unit), 1), _ngrams(_tokens(unit), 2)) for unit in units]
    budget = min(len(units), max(1, len(split_units(target))), max(1, int(max_units)))
    selected: List[int] = []
    remaining = set(range(len(units)))
    current = (Counter(), Counter())

    def score(pair: Tuple[Counter, Counter]) -> float:
        return rouge1_weight * _f1(pair[0], reference[0]) + rouge2_weight * _f1(pair[1], reference[1])

    current_score = 0.0
    for _ in range(budget):
        winner = None
        winner_score = current_score
        for index in sorted(remaining):
            candidate = (current[0] + counters[index][0], current[1] + counters[index][1])
            value = score(candidate)
            if value > winner_score + 1e-12:
                winner, winner_score = index, value
        if winner is None:
            break
        selected.append(winner)
        remaining.remove(winner)
        current = (current[0] + counters[winner][0], current[1] + counters[winner][1])
        current_score = winner_score
    if not selected:
        individual = [score(pair) for pair in counters]
        if max(individual, default=0.0) <= 0.0:
            return [-1.0] * len(units)
        selected = [max(range(len(individual)), key=individual.__getitem__)]
    chosen = set(selected)
    return [1.0 if index in chosen else 0.0 for index in range(len(units))]


def encode_source(
    tokenizer: Any,
    source: str,
    config: Dict[str, Any],
) -> Tuple[List[int], List[int], List[str]]:
    prefix = str(config.get("source_prefix", ""))
    separator = str(config.get("sentence_separator", "\n"))
    max_length = int(config.get("max_source_length", 3072))
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    special_prefix_ids: List[int] = []
    special_suffix_ids: List[int] = []
    if bool(config.get("source_add_special_tokens", False)):
        cache_name = "_llm2seq_v3_default_special_wrapper"
        cached = getattr(tokenizer, cache_name, None)
        if cached is None:
            probe = "LLM2Seq"
            plain_probe = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
            wrapped_probe = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
            start = next(
                (
                    index
                    for index in range(len(wrapped_probe) - len(plain_probe) + 1)
                    if wrapped_probe[index : index + len(plain_probe)] == plain_probe
                ),
                None,
            )
            if start is None:
                raise RuntimeError("Tokenizer default special-token wrapper is not prefix/suffix aligned")
            cached = (wrapped_probe[:start], wrapped_probe[start + len(plain_probe) :])
            setattr(tokenizer, cache_name, cached)
        special_prefix_ids = list(cached[0])
        special_suffix_ids = list(cached[1])
    eos_id = tokenizer.eos_token_id if bool(config.get("append_source_eos", True)) else None
    eos_budget = int(eos_id is not None)
    special_budget = len(special_prefix_ids) + len(special_suffix_ids)
    budget = max(1, max_length - len(prefix_ids) - eos_budget - special_budget)
    source_ids: List[int] = []
    unit_ids: List[int] = []
    visible_units: List[str] = []
    for unit in split_units(source):
        complete = list(tokenizer(unit + separator, add_special_tokens=False)["input_ids"])
        remaining = budget - len(source_ids)
        if remaining <= 0:
            break
        encoded = complete[:remaining]
        if not encoded:
            break
        visible_units.append(
            tokenizer.decode(encoded, skip_special_tokens=True).strip() if len(encoded) < len(complete) else unit
        )
        unit_index = len(visible_units)
        source_ids.extend(encoded)
        unit_ids.extend([unit_index] * len(encoded))
        if len(encoded) < len(complete):
            break
    ids = special_prefix_ids + prefix_ids + source_ids + special_suffix_ids
    aligned = [0] * (len(special_prefix_ids) + len(prefix_ids)) + unit_ids + [0] * len(special_suffix_ids)
    if eos_id is not None and len(ids) < max_length:
        ids.append(int(eos_id))
        aligned.append(0)
    return ids[:max_length], aligned[:max_length], visible_units


def decoder_seed_ids(tokenizer: Any, config: Dict[str, Any]) -> List[int]:
    instruction = str(config.get("decoder_instruction", "")).strip()
    prefix = str(config.get("decoder_prefix", ""))
    use_chat = bool(config.get("use_decoder_chat_template", True))
    if use_chat and instruction and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(config.get("enable_thinking", False)),
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        seed = [int(value) for value in ids]
    else:
        start = tokenizer.bos_token_id
        if start is None:
            start = tokenizer.pad_token_id
        if start is None:
            start = tokenizer.eos_token_id
        if start is None:
            raise ValueError("Decoder tokenizer has no BOS/PAD/EOS token")
        seed = [int(start)]
    seed.extend(int(value) for value in tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if not seed:
        raise ValueError("Decoder seed cannot be empty")
    return seed


class SummarizationDataset(Dataset):
    """Token-level encoder memory plus decoder-native target tokenization."""

    def __init__(
        self,
        path: str | Path,
        encoder_tokenizer: Any,
        decoder_tokenizer: Any,
        data_config: Dict[str, Any],
        max_examples: int = 0,
        precompute_evidence: bool = True,
    ):
        self.examples = read_jsonl(path, max_examples=max_examples)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = data_config
        self.max_target_length = int(data_config.get("max_target_length", 384))
        self.clean_metadata = bool(data_config.get("clean_wikihow_metadata", True))
        self.seed = decoder_seed_ids(decoder_tokenizer, data_config)
        self.evidence_cache: List[List[float]] | None = None
        if precompute_evidence:
            self.evidence_cache = []
            for row in self.examples:
                source = clean_text(row["source"], self.clean_metadata)
                target = clean_text(row["target"], self.clean_metadata)
                self.evidence_cache.append(
                    greedy_evidence_labels(
                        split_units(source),
                        target,
                        max_units=int(data_config.get("oracle_max_units", 12)),
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.examples[index]
        source = clean_text(row["source"], self.clean_metadata)
        target = clean_text(row["target"], self.clean_metadata)
        source_ids, unit_ids, visible_units = encode_source(self.encoder_tokenizer, source, self.config)
        full_units = split_units(source)
        cached = self.evidence_cache[index] if self.evidence_cache is not None else None
        if cached is not None and visible_units == full_units:
            evidence = cached
        else:
            evidence = greedy_evidence_labels(
                visible_units,
                target,
                max_units=int(self.config.get("oracle_max_units", 12)),
            )

        eos_id = self.decoder_tokenizer.eos_token_id
        target_budget = self.max_target_length - int(eos_id is not None)
        target_ids = list(
            self.decoder_tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, target_budget),
            )["input_ids"]
        )
        if eos_id is not None:
            target_ids.append(int(eos_id))
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        decoder_input = torch.cat([torch.tensor(self.seed, dtype=torch.long), target_tensor[:-1]])
        labels = torch.cat(
            [
                torch.full((len(self.seed) - 1,), -100, dtype=torch.long),
                target_tensor,
            ]
        )
        if decoder_input.numel() != labels.numel():
            raise RuntimeError("Shifted decoder inputs and labels must have equal length")
        return {
            "input_ids": torch.tensor(source_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(source_ids), dtype=torch.long),
            "unit_ids": torch.tensor(unit_ids, dtype=torch.long),
            "evidence_labels": torch.tensor(evidence, dtype=torch.float32),
            "decoder_input_ids": decoder_input,
            "decoder_attention_mask": torch.ones_like(decoder_input),
            "labels": labels,
        }


class Seq2SeqCollator:
    def __init__(self, encoder_pad_id: int, decoder_pad_id: int, max_source_length: int, max_decoder_length: int):
        self.encoder_pad_id = int(encoder_pad_id)
        self.decoder_pad_id = int(decoder_pad_id)
        self.max_source_length = int(max_source_length)
        self.max_decoder_length = int(max_decoder_length)

    @staticmethod
    def _pad(values: Iterable[torch.Tensor], length: int, fill: float) -> torch.Tensor:
        rows = []
        for value in values:
            value = value[:length]
            padding = torch.full((length - value.numel(),), fill, dtype=value.dtype)
            rows.append(torch.cat([value, padding]))
        return torch.stack(rows)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        source_length = min(self.max_source_length, max(item["input_ids"].numel() for item in features))
        decoder_length = min(
            self.max_decoder_length,
            max(item["decoder_input_ids"].numel() for item in features),
        )
        unit_count = max(1, max(item["evidence_labels"].numel() for item in features))
        return {
            "input_ids": self._pad((item["input_ids"] for item in features), source_length, self.encoder_pad_id),
            "attention_mask": self._pad((item["attention_mask"] for item in features), source_length, 0),
            "unit_ids": self._pad((item["unit_ids"] for item in features), source_length, 0),
            "evidence_labels": self._pad(
                (item["evidence_labels"] for item in features),
                unit_count,
                -1.0,
            ),
            "decoder_input_ids": self._pad(
                (item["decoder_input_ids"] for item in features),
                decoder_length,
                self.decoder_pad_id,
            ),
            "decoder_attention_mask": self._pad(
                (item["decoder_attention_mask"] for item in features),
                decoder_length,
                0,
            ),
            "labels": self._pad((item["labels"] for item in features), decoder_length, -100),
        }
