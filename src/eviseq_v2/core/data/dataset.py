"""Configurable JSONL text-to-text data for EviSeq."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WIKIHOW_IMAGE = re.compile(r'\{\s*"\s*smallUrl\s*"[^{}]*\}', flags=re.IGNORECASE)
LOGGER = logging.getLogger("eviseq.data.dataset")


class _TemplateValues(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"Template field {key!r} is missing from the JSON record")


def _field_value(row: Dict[str, Any], field: str, separator: str) -> str:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Field {field!r} is missing from the JSON record")
        value = value[part]
    if isinstance(value, list):
        return separator.join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_record(row: Dict[str, Any], data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Map an arbitrary JSON object to EviSeq's source/target/id interface."""

    mapper_path = str(data_config.get("record_mapper", "")).strip()
    if mapper_path:
        module_name, function_name = mapper_path.split(":", 1)
        mapped = getattr(import_module(module_name), function_name)(row, data_config)
        if not isinstance(mapped, Mapping) or "source" not in mapped or "target" not in mapped:
            raise TypeError("Custom record mapper must return a mapping with source and target")
        return {
            "id": mapped.get("id"),
            "source": str(mapped["source"]),
            "target": str(mapped["target"]),
        }

    list_separator = str(data_config.get("list_separator", "\n"))
    template_values = _TemplateValues(
        {
            key: list_separator.join(str(item) for item in value) if isinstance(value, list) else value
            for key, value in row.items()
        }
    )
    source_template = str(data_config.get("source_template", "")).strip()
    target_template = str(data_config.get("target_template", "")).strip()
    if source_template:
        source = source_template.format_map(template_values)
    else:
        source = _field_value(row, str(data_config.get("source_field", "source")), list_separator)
    if target_template:
        target = target_template.format_map(template_values)
    else:
        target = _field_value(row, str(data_config.get("target_field", "target")), list_separator)
    id_field = str(data_config.get("id_field", "id"))
    try:
        example_id: Any = _field_value(row, id_field, list_separator)
    except KeyError:
        example_id = None
    return {"id": example_id, "source": source, "target": target}


def read_jsonl(
    path: str | Path,
    max_examples: int = 0,
    data_config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
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
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            try:
                rows.append(normalize_record(row, data_config or {}))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Cannot map text-to-text fields at {path}:{line_number}: {exc}") from exc
            if max_examples > 0 and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def clean_text(text: Any, enabled: bool) -> str:
    value = str(text)
    if enabled:
        value = _WIKIHOW_IMAGE.sub(" ", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\s+([.,;:!?])", r"\1", value)
    return value.strip()


def split_units(text: str) -> List[str]:
    return [unit for unit, _, _ in split_units_with_spans(text)]


def split_units_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Split text exactly like ``split_units`` while retaining character spans."""

    value = str(text)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) > 1:
        units: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"[^\n]+", value):
            unit = match.group(0).strip()
            if not unit:
                continue
            start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            units.append((unit, start, start + len(unit)))
        return units

    units = []
    start = 0
    for separator in _SENTENCE.finditer(value):
        end = separator.start()
        segment = value[start:end]
        unit = segment.strip()
        if unit:
            unit_start = start + len(segment) - len(segment.lstrip())
            units.append((unit, unit_start, unit_start + len(unit)))
        start = separator.end()
    segment = value[start:]
    unit = segment.strip()
    if unit:
        unit_start = start + len(segment) - len(segment.lstrip())
        units.append((unit, unit_start, unit_start + len(unit)))
    return units


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
    budget: int | None = None,
) -> List[float]:
    """Greedy sentence coverage labels; reference is used during training only."""

    if not units:
        return []
    target_tokens = _tokens(target)
    reference = (_ngrams(target_tokens, 1), _ngrams(target_tokens, 2))
    counters = [(_ngrams(_tokens(unit), 1), _ngrams(_tokens(unit), 2)) for unit in units]
    if budget is None:
        budget = max(1, len(split_units(target)))
    budget = min(len(units), max(1, int(budget)), max(1, int(max_units)))
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
    eos_id = tokenizer.eos_token_id
    eos_budget = int(eos_id is not None)
    if len(prefix_ids) + eos_budget >= max_length:
        raise ValueError(
            "source_prefix leaves no room for article tokens: "
            f"prefix={len(prefix_ids)} eos={eos_budget} max_source_length={max_length}"
        )
    budget = max(1, max_length - len(prefix_ids) - eos_budget)
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
    ids = prefix_ids + source_ids
    aligned = [0] * len(prefix_ids) + unit_ids
    if eos_id is not None and len(ids) < max_length:
        ids.append(int(eos_id))
        aligned.append(0)
    return ids[:max_length], aligned[:max_length], visible_units


def visible_target_sentences(
    tokenizer: Any,
    content_target_ids: Sequence[int],
    sentence_ids: Sequence[int],
) -> List[str]:
    """Decode exactly the target content supervised after truncation.

    Sentence evidence labels must not use words that were cut off by
    ``max_target_length``.  Grouping the *actual* retained token ids by their
    offset-derived sentence ids keeps the evidence target and decoder loss
    aligned even when the final reference sentence is partial.
    """

    if len(content_target_ids) != len(sentence_ids):
        raise ValueError("content_target_ids and sentence_ids must be aligned")
    grouped: Dict[int, List[int]] = {}
    for token_id, sentence_id in zip(content_target_ids, sentence_ids):
        if int(sentence_id) > 0:
            grouped.setdefault(int(sentence_id), []).append(int(token_id))
    # Keep row ``r`` tied to sentence id ``r + 1`` even if decoding a span
    # yields only special/empty tokens.  Dropping an empty group would shift
    # every later evidence-label row and train sentence r against sentence
    # r+1's oracle evidence.
    sentences = [""] * max(grouped, default=0)
    for sentence_id, token_ids in grouped.items():
        sentences[sentence_id - 1] = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return sentences


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


def target_sentence_ids(
    tokenizer: Any,
    target: str,
    target_ids: Sequence[int],
    *,
    require_offsets: bool = False,
) -> List[int]:
    """Map every target token to its reference sentence, using fast offsets.

    A conservative all-one fallback is appropriate for generic text-to-text
    tasks.  Sentence-aligned evidence supervision instead requests strict
    offsets: silently treating a multi-sentence abstract as one sentence
    would turn the intended objective back into global InfoNCE.
    """

    spans = split_units_with_spans(target)
    fallback = [1] * len(target_ids)
    if not spans or not target_ids:
        return fallback
    try:
        encoded = tokenizer(
            target,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=len(target_ids),
        )
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError) as error:
        if require_offsets:
            raise ValueError(
                "Sentence-aligned evidence supervision requires a fast decoder tokenizer with offsets"
            ) from error
        return fallback
    if encoded_ids != [int(value) for value in target_ids] or len(offsets) != len(target_ids):
        if require_offsets:
            raise ValueError(
                "Sentence-aligned target offsets do not match the truncated decoder target ids; "
                "refusing to collapse the objective to one summary sentence"
            )
        return fallback

    sentence_ids: List[int] = []
    sentence_index = 0
    for start, end in offsets:
        if end <= start:
            sentence_ids.append(max(1, sentence_index + 1))
            continue
        midpoint = start + (end - start - 1) // 2
        while sentence_index + 1 < len(spans) and midpoint >= spans[sentence_index][2]:
            sentence_index += 1
        sentence_ids.append(sentence_index + 1)
    return sentence_ids


class Text2TextDataset(Dataset):
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
        self.examples = read_jsonl(path, max_examples=max_examples, data_config=data_config)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = data_config
        self.max_target_length = int(data_config.get("max_target_length", 384))
        self.clean_metadata = bool(data_config.get("clean_wikihow_metadata", False))
        self.supervise_evidence = bool(data_config.get("supervise_evidence", True))
        self.sentence_evidence_supervision = bool(data_config.get("sentence_evidence_supervision", False))
        self.sentence_evidence_max_units = int(data_config.get("sentence_evidence_max_units", 1))
        self.sentence_evidence_use_union_as_salience = bool(
            data_config.get("sentence_evidence_use_union_as_salience", False)
        )
        self.seed = decoder_seed_ids(decoder_tokenizer, data_config)
        self.evidence_cache: List[List[float]] | None = None
        if (
            precompute_evidence
            and self.supervise_evidence
            and not (self.sentence_evidence_supervision and self.sentence_evidence_use_union_as_salience)
        ):
            self.evidence_cache = []
            total_examples = len(self.examples)
            progress_interval = max(1, total_examples // 20)
            LOGGER.info("precomputing evidence labels: 0/%d", total_examples)
            for index, row in enumerate(self.examples, start=1):
                source = clean_text(row["source"], self.clean_metadata)
                target = clean_text(row["target"], self.clean_metadata)
                self.evidence_cache.append(
                    greedy_evidence_labels(
                        split_units(source),
                        target,
                        max_units=int(data_config.get("oracle_max_units", 12)),
                    )
                )
                if index % progress_interval == 0 or index == total_examples:
                    LOGGER.info("precomputing evidence labels: %d/%d", index, total_examples)

    def __len__(self) -> int:
        return len(self.examples)

    def source_length_estimates(self) -> List[int]:
        """Cheap source-length proxy for dynamic-padding buckets."""

        cap = max(1, int(self.config.get("max_source_length", 3072)) * 4)
        return [min(cap, len(clean_text(row["source"], self.clean_metadata))) for row in self.examples]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.examples[index]
        source = clean_text(row["source"], self.clean_metadata)
        target = clean_text(row["target"], self.clean_metadata)
        source_ids, unit_ids, visible_units = encode_source(self.encoder_tokenizer, source, self.config)
        full_units = split_units(source)
        cached = self.evidence_cache[index] if self.evidence_cache is not None else None
        use_sentence_union = (
            self.sentence_evidence_supervision
            and self.sentence_evidence_use_union_as_salience
            and self.supervise_evidence
        )
        if not self.supervise_evidence:
            evidence = [-1.0] * len(visible_units)
        elif use_sentence_union:
            # Filled after each visible target sentence receives its own
            # oracle labels.  Avoid computing an unused global greedy oracle.
            evidence = []
        elif cached is not None and visible_units == full_units:
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
        content_target_ids = target_ids[:-1] if eos_id is not None else target_ids
        sentence_ids = target_sentence_ids(
            self.decoder_tokenizer,
            target,
            content_target_ids,
            require_offsets=self.sentence_evidence_supervision and self.supervise_evidence,
        )
        if eos_id is not None:
            # EOS helps CE finish the sequence but carries no semantic
            # evidence.  Leaving it out of sentence pooling avoids a short
            # final sentence being dominated by its EOS prediction state.
            sentence_ids.append(0)
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
        decoder_sentence_ids = torch.cat(
            [
                torch.zeros((len(self.seed) - 1,), dtype=torch.long),
                torch.tensor(sentence_ids, dtype=torch.long),
            ]
        )
        if decoder_sentence_ids.numel() != labels.numel():
            raise RuntimeError("Decoder sentence ids must align with shifted decoder labels")
        if self.sentence_evidence_supervision and self.supervise_evidence:
            target_units = visible_target_sentences(
                self.decoder_tokenizer,
                content_target_ids,
                sentence_ids[: len(content_target_ids)],
            )
            sentence_evidence = [
                greedy_evidence_labels(
                    visible_units,
                    sentence,
                    max_units=self.sentence_evidence_max_units,
                    budget=self.sentence_evidence_max_units,
                )
                for sentence in target_units
            ]
            if not sentence_evidence:
                sentence_evidence = [[-1.0] * len(visible_units)]
            if use_sentence_union:
                evidence = []
                for unit_index in range(len(visible_units)):
                    labels_for_unit = [row[unit_index] for row in sentence_evidence if unit_index < len(row)]
                    if any(value > 0.5 for value in labels_for_unit):
                        evidence.append(1.0)
                    elif any(value >= 0.0 for value in labels_for_unit):
                        evidence.append(0.0)
                    else:
                        evidence.append(-1.0)
        else:
            sentence_evidence = [[-1.0] * len(visible_units)]
        return {
            "input_ids": torch.tensor(source_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(source_ids), dtype=torch.long),
            "unit_ids": torch.tensor(unit_ids, dtype=torch.long),
            "evidence_labels": torch.tensor(evidence, dtype=torch.float32),
            "decoder_input_ids": decoder_input,
            "decoder_attention_mask": torch.ones_like(decoder_input),
            "labels": labels,
            "target_sentence_ids": decoder_sentence_ids,
            "sentence_evidence_labels": torch.tensor(sentence_evidence, dtype=torch.float32),
        }


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Shuffle random pools and batch examples with similar source lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int = 42,
        bucket_size_multiplier: int = 50,
        drop_last: bool = False,
    ):
        if not lengths:
            raise ValueError("Length bucketing requires a non-empty dataset")
        if batch_size <= 0 or bucket_size_multiplier <= 0:
            raise ValueError("Length-bucket batch and multiplier must be positive")
        self.lengths = [int(value) for value in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size_multiplier = int(bucket_size_multiplier)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shuffled = torch.randperm(len(self.lengths), generator=generator).tolist()
        pool_size = self.batch_size * self.bucket_size_multiplier
        batches: List[List[int]] = []
        for start in range(0, len(shuffled), pool_size):
            pool = shuffled[start : start + pool_size]
            pool.sort(key=self.lengths.__getitem__, reverse=True)
            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        for index in torch.randperm(len(batches), generator=generator).tolist():
            yield batches[index]


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
        sentence_count = max(1, max(item["sentence_evidence_labels"].shape[0] for item in features))
        sentence_evidence = torch.full((len(features), sentence_count, unit_count), -1.0, dtype=torch.float32)
        for index, item in enumerate(features):
            values = item["sentence_evidence_labels"]
            sentence_evidence[index, : values.shape[0], : values.shape[1]] = values
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
            "target_sentence_ids": self._pad((item["target_sentence_ids"] for item in features), decoder_length, 0),
            "sentence_evidence_labels": sentence_evidence,
        }
