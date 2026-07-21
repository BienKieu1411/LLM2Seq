"""Evidence-unit-aligned data and automatic supervision for summarization."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


_WIKIHOW_IMAGE_OBJECT = re.compile(r'\{\s*"\s*smallUrl\s*"[^{}]*\}', flags=re.IGNORECASE)
_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def clean_wikihow_metadata(text: str) -> str:
    cleaned = _WIKIHOW_IMAGE_OBJECT.sub(" ", str(text))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _configured_text(text: Any, data_config: Dict[str, Any]) -> str:
    value = str(text)
    return clean_wikihow_metadata(value) if bool(data_config.get("clean_wikihow_metadata", False)) else value


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def split_sentences(text: str) -> List[str]:
    """Prefer dataset line boundaries, with punctuation fallback for generic corpora."""

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [piece.strip() for piece in _SENTENCE_BOUNDARY.split(str(text).strip()) if piece.strip()]


def split_evidence_units(text: str, data_config: Dict[str, Any]) -> List[str]:
    """Create dataset-configurable evidence units without changing the model.

    Sentence units suit news and procedural summaries. Grouped sentences are
    useful for long scientific/legal documents, while paragraph units preserve
    discourse boundaries when the source contains explicit blank lines.
    """

    mode = str(data_config.get("evidence_unit", "sentence"))
    if mode == "sentence":
        return split_sentences(text)
    if mode == "sentence_group":
        sentences = split_sentences(text)
        group_size = max(1, int(data_config.get("sentences_per_unit", 3)))
        return [" ".join(sentences[start : start + group_size]) for start in range(0, len(sentences), group_size)]
    if mode == "paragraph":
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", str(text)) if part.strip()]
        return paragraphs if len(paragraphs) > 1 else split_sentences(text)
    raise ValueError(f"Unknown data.evidence_unit: {mode}")


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in _WORD.findall(text)]


def _ngrams(tokens: Sequence[str], order: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _counter_f1(candidate: Counter, reference: Counter) -> float:
    overlap = sum((candidate & reference).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(candidate.values()))
    recall = overlap / max(1, sum(reference.values()))
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def _counter_overlap(
    candidate_unigrams: Counter,
    candidate_bigrams: Counter,
    reference_unigrams: Counter,
    reference_bigrams: Counter,
    rouge1_weight: float = 0.5,
    rouge2_weight: float = 0.5,
) -> float:
    return rouge1_weight * _counter_f1(
        candidate_unigrams, reference_unigrams
    ) + rouge2_weight * _counter_f1(candidate_bigrams, reference_bigrams)


def greedy_evidence_labels(
    source_units: Sequence[str],
    target: str,
    max_evidence_units: int = 12,
    budget_mode: str = "target_units",
    fixed_budget: int = 3,
    rouge1_weight: float = 0.5,
    rouge2_weight: float = 0.5,
) -> List[float]:
    """Create extractive-oracle labels using greedy ROUGE-1/2 coverage.

    The labels are derived only for training/evaluation diagnostics. At test
    time the learned salience head predicts evidence without a reference.
    """

    if not source_units:
        return []
    if rouge1_weight < 0 or rouge2_weight < 0 or rouge1_weight + rouge2_weight <= 0:
        raise ValueError("Oracle ROUGE weights must be non-negative with a positive sum")
    weight_sum = rouge1_weight + rouge2_weight
    rouge1_weight /= weight_sum
    rouge2_weight /= weight_sum
    target_tokens = _tokens(target)
    reference_unigrams = _ngrams(target_tokens, 1)
    reference_bigrams = _ngrams(target_tokens, 2)
    candidate_counters = []
    for unit in source_units:
        sentence_tokens = _tokens(unit)
        candidate_counters.append((_ngrams(sentence_tokens, 1), _ngrams(sentence_tokens, 2)))
    if budget_mode == "target_units":
        requested_budget = max(1, len(split_sentences(target)))
    elif budget_mode == "fixed":
        requested_budget = max(1, int(fixed_budget))
    elif budget_mode == "all":
        requested_budget = len(source_units)
    else:
        raise ValueError(f"Unknown oracle budget mode: {budget_mode}")
    budget = min(len(source_units), int(max_evidence_units), requested_budget)
    selected: List[int] = []
    remaining = set(range(len(source_units)))
    current_score = 0.0
    selected_unigrams: Counter = Counter()
    selected_bigrams: Counter = Counter()
    for _ in range(budget):
        best_index = None
        best_score = current_score
        for index in sorted(remaining):
            candidate_unigrams, candidate_bigrams = candidate_counters[index]
            score = _counter_overlap(
                selected_unigrams + candidate_unigrams,
                selected_bigrams + candidate_bigrams,
                reference_unigrams,
                reference_bigrams,
                rouge1_weight,
                rouge2_weight,
            )
            if score > best_score + 1e-12:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        selected_unigrams += candidate_counters[best_index][0]
        selected_bigrams += candidate_counters[best_index][1]
        current_score = best_score
    if not selected:
        # Always supply a positive example when lexical overlap exists only at
        # character/tokenization edge cases.
        scores = [
            _counter_overlap(
                unigrams,
                bigrams,
                reference_unigrams,
                reference_bigrams,
                rouge1_weight,
                rouge2_weight,
            )
            for unigrams, bigrams in candidate_counters
        ]
        selected = [max(range(len(scores)), key=scores.__getitem__)]
    selected_set = set(selected)
    return [1.0 if index in selected_set else 0.0 for index in range(len(source_units))]


def prompted_source_features(
    tokenizer: Any,
    source: str,
    data_config: Dict[str, Any],
) -> tuple[List[int], List[int], List[str]]:
    """Build the prompt while retaining token-to-evidence-unit alignment."""

    source_prefix = str(data_config.get("source_prefix", "Summarize concisely.\nDocument:\n"))
    target_prefix = str(data_config.get("target_prefix", "\nSummary:\n"))
    max_length = int(data_config.get("max_source_length", 3072))
    if bool(data_config.get("use_chat_template", True)) and getattr(tokenizer, "chat_template", None):
        sentinel = "<EVIBRIDGE_SOURCE_7F3A>"
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

    budget = max(1, max_length - len(prefix_ids) - len(suffix_ids))
    separator = str(data_config.get("sentence_separator", "\n"))
    units = split_evidence_units(source, data_config)
    source_ids: List[int] = []
    unit_ids: List[int] = []
    used_units: List[str] = []
    for unit in units:
        complete = list(tokenizer(f"{unit}{separator}", add_special_tokens=False)["input_ids"])
        remaining = budget - len(source_ids)
        if remaining <= 0:
            break
        encoded = complete[:remaining]
        if not encoded:
            break
        used_units.append(unit)
        current_id = len(used_units)
        source_ids.extend(encoded)
        unit_ids.extend([current_id] * len(encoded))
        if len(encoded) < len(complete):
            break
    ids = list(prefix_ids) + source_ids + list(suffix_ids)
    aligned = [0] * len(prefix_ids) + unit_ids + [0] * len(suffix_ids)
    return ids, aligned, used_units


def prompt_token_ids(tokenizer: Any, source: str, data_config: Dict[str, Any]) -> List[int]:
    return prompted_source_features(tokenizer, source, data_config)[0]


class EvidenceSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        data_config: Dict[str, Any],
        precompute_evidence: bool = False,
        max_examples: int = 0,
    ):
        self.examples = read_jsonl(path)
        if max_examples > 0:
            self.examples = self.examples[: int(max_examples)]
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_target_length = int(data_config.get("max_target_length", 384))
        self.max_evidence_units = int(data_config.get("oracle_max_units", 12))
        self.oracle_budget_mode = str(data_config.get("oracle_budget_mode", "target_units"))
        self.oracle_fixed_units = int(data_config.get("oracle_fixed_units", 3))
        self.rouge1_weight = float(data_config.get("oracle_rouge1_weight", 0.5))
        self.rouge2_weight = float(data_config.get("oracle_rouge2_weight", 0.5))
        self.evidence_cache: List[List[float]] | None = None
        self.truncated_evidence_cache: Dict[int, List[float]] = {}
        if precompute_evidence:
            self.evidence_cache = []
            for example in self.examples:
                source = _configured_text(example["source"], self.data_config)
                target = _configured_text(example["target"], self.data_config)
                self.evidence_cache.append(
                    greedy_evidence_labels(
                        split_evidence_units(source, self.data_config),
                        target,
                        self.max_evidence_units,
                        self.oracle_budget_mode,
                        self.oracle_fixed_units,
                        self.rouge1_weight,
                        self.rouge2_weight,
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        source = _configured_text(example["source"], self.data_config)
        target = _configured_text(example["target"], self.data_config)
        source_ids, unit_ids, units = prompted_source_features(
            self.tokenizer, source, self.data_config
        )
        cached = self.evidence_cache[index] if self.evidence_cache is not None else None
        if cached is not None and len(cached) == len(units):
            evidence = cached
        elif index in self.truncated_evidence_cache:
            evidence = self.truncated_evidence_cache[index]
        else:
            evidence = greedy_evidence_labels(
                units,
                target,
                self.max_evidence_units,
                self.oracle_budget_mode,
                self.oracle_fixed_units,
                self.rouge1_weight,
                self.rouge2_weight,
            )
            self.truncated_evidence_cache[index] = evidence
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_target_length - 1),
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids = list(target_ids) + [self.tokenizer.eos_token_id]
        start_id = self.tokenizer.bos_token_id
        if start_id is None:
            start_id = self.tokenizer.pad_token_id
        if start_id is None:
            start_id = self.tokenizer.eos_token_id
        if start_id is None:
            raise ValueError("Tokenizer has no BOS/PAD/EOS decoder start token")
        labels = torch.tensor(target_ids, dtype=torch.long)
        return {
            "input_ids": torch.tensor(source_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(source_ids), dtype=torch.long),
            "unit_ids": torch.tensor(unit_ids, dtype=torch.long),
            "evidence_labels": torch.tensor(evidence, dtype=torch.float32),
            "decoder_input_ids": torch.cat([torch.tensor([start_id]), labels[:-1]]),
            "labels": labels,
        }


class EvidenceSeq2SeqCollator:
    def __init__(self, pad_token_id: int, max_source_length: int, max_target_length: int):
        self.pad_token_id = int(pad_token_id)
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)

    @staticmethod
    def _pad_1d(tensors: Iterable[torch.Tensor], length: int, value: float) -> torch.Tensor:
        result = []
        for tensor in tensors:
            tensor = tensor[:length]
            padding = torch.full((length - tensor.numel(),), value, dtype=tensor.dtype)
            result.append(torch.cat([tensor, padding]))
        return torch.stack(result)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        source_length = min(max(item["input_ids"].numel() for item in features), self.max_source_length)
        target_length = min(max(item["labels"].numel() for item in features), self.max_target_length)
        evidence_length = max(1, max(item["evidence_labels"].numel() for item in features))
        return {
            "input_ids": self._pad_1d((item["input_ids"] for item in features), source_length, self.pad_token_id),
            "attention_mask": self._pad_1d((item["attention_mask"] for item in features), source_length, 0),
            "unit_ids": self._pad_1d((item["unit_ids"] for item in features), source_length, 0),
            "evidence_labels": self._pad_1d((item["evidence_labels"] for item in features), evidence_length, -1.0),
            "decoder_input_ids": self._pad_1d((item["decoder_input_ids"] for item in features), target_length, self.pad_token_id),
            "decoder_attention_mask": self._pad_1d((torch.ones_like(item["decoder_input_ids"]) for item in features), target_length, 0),
            "labels": self._pad_1d((item["labels"] for item in features), target_length, -100),
        }


class DirectSummarizationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        data_config: Dict[str, Any],
        max_examples: int = 0,
    ):
        self.examples = read_jsonl(path)
        if max_examples > 0:
            self.examples = self.examples[: int(max_examples)]
        self.tokenizer = tokenizer
        self.data_config = data_config
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
            target_ids = list(target_ids) + [self.tokenizer.eos_token_id]
        return {
            "input_ids": torch.tensor(prompt_ids + target_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids) + len(target_ids), dtype=torch.long),
            "labels": torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long),
        }


class DirectCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        length = max(item["input_ids"].numel() for item in features)
        pad = EvidenceSeq2SeqCollator._pad_1d
        return {
            "input_ids": pad((item["input_ids"] for item in features), length, self.pad_token_id),
            "attention_mask": pad((item["attention_mask"] for item in features), length, 0),
            "labels": pad((item["labels"] for item in features), length, -100),
        }
