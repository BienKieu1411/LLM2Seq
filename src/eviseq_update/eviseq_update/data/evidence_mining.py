"""Deterministic lexical ambiguity mining for evidence contrastive training.

This module intentionally uses only normalized source/reference text and the
two tokenizers.  It never loads model weights or generates a candidate summary.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .copy_alignment import align_copy_tokens_with_offsets
from .evidence import EvidenceAnnotation, EvidenceUnit, ids_sha256, text_sha256
from .schema import CanonicalRecord

DEFAULT_MINING_CONFIG = {
    "min_positive_score": 0.30,
    "max_negative_score": 0.15,
    "positive_band": 0.05,
    "min_context_gap": 0.20,
    "min_context_matches": 2,
    "max_units_per_example": 32,
    "max_positive_sentences": 4,
    "max_negative_sentences": 4,
}

_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_NUMBER_UNIT = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?)(?:\s*(?:%|[A-Za-zµμ][A-Za-z0-9µμ/^.-]*))?", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "within",
        "without",
        "we",
        "they",
        "their",
        "our",
        "these",
        "those",
        "than",
        "then",
        "also",
        "using",
        "use",
    }
)
_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "et al.", "fig.", "dr.", "mr.", "mrs.", "ms.", "vs.", "no."})


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str
    normalized: str
    is_number: bool = False


def _normalise_anchor(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _sentence_spans(text: str) -> tuple[TextSpan, ...]:
    """Split while preserving offsets and avoiding decimal/known abbreviation cuts."""

    result: list[TextSpan] = []
    start = 0
    for position, character in enumerate(text):
        if character not in ".!?":
            continue
        if (
            character == "."
            and position > 0
            and position + 1 < len(text)
            and text[position - 1].isdigit()
            and text[position + 1].isdigit()
        ):
            continue
        prefix = text[max(start, position - 12) : position + 1].casefold()
        if character == "." and any(prefix.endswith(item) for item in _ABBREVIATIONS):
            continue
        if position + 1 < len(text) and not text[position + 1].isspace():
            continue
        end = position + 1
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            value = text[start:end]
            result.append(TextSpan(start, end, value, value.casefold()))
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        value = text[start:]
        result.append(TextSpan(start, len(text), value, value.casefold()))
    return tuple(result)


def _candidate_spans(text: str) -> tuple[TextSpan, ...]:
    """Content words plus numbers/units, with number spans taking precedence."""

    result: list[TextSpan] = []
    occupied: list[tuple[int, int]] = []
    for match in _NUMBER_UNIT.finditer(text):
        value = match.group()
        result.append(TextSpan(match.start(), match.end(), value, _normalise_anchor(value), True))
        occupied.append((match.start(), match.end()))
    for match in _WORD.finditer(text):
        if any(left < match.end() and match.start() < right for left, right in occupied):
            continue
        value = match.group()
        normalized = value.casefold()
        if len(normalized) < 3 or normalized in _STOPWORDS:
            continue
        result.append(TextSpan(match.start(), match.end(), value, normalized))
    return tuple(sorted(result, key=lambda item: (item.start, item.end)))


def _content_terms(text: str, excluded: TextSpan | None = None) -> list[str]:
    output = []
    for term in _candidate_spans(text):
        if excluded is not None and term.start < excluded.end and excluded.start < term.end:
            continue
        output.append(term.normalized)
    return output


def _f1(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = Counter(left), Counter(right)
    matches = sum((a & b).values())
    if not matches:
        return 0.0
    precision, recall = matches / sum(a.values()), matches / sum(b.values())
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _context_score(reference: list[str], source: list[str]) -> float:
    if not reference or not source:
        return 0.0
    lcs = _lcs_length(reference, source)
    lcs_f1 = (
        0.0
        if not lcs
        else 2.0 * (lcs / len(reference)) * (lcs / len(source)) / (lcs / len(reference) + lcs / len(source))
    )
    return 0.5 * _f1(reference, source) + 0.5 * lcs_f1


def _inside(sentence: TextSpan, term: TextSpan) -> bool:
    return sentence.start <= term.start and term.end <= sentence.end


def _spans_from_positions(positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    values = sorted(set(int(value) for value in positions))
    if not values:
        return ()
    spans: list[tuple[int, int]] = []
    begin = previous = values[0]
    for position in values[1:]:
        if position == previous + 1:
            previous = position
            continue
        spans.append((begin, previous + 1))
        begin = previous = position
    spans.append((begin, previous + 1))
    return tuple(spans)


def _token_offsets(tokenizer: Any, text: str, maximum: int) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=maximum
    )
    if "offset_mapping" not in encoded:
        raise ValueError("Evidence mining requires fast tokenizers with offset mapping")
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return ids, offsets


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _encoder_positions(encoder_offsets, prefix_length: int, sentence: TextSpan) -> tuple[int, ...]:
    values = []
    for index, (start, end) in enumerate(encoder_offsets):
        relative = (int(start) - prefix_length, int(end) - prefix_length)
        if relative[1] > relative[0] and _overlaps(relative, (sentence.start, sentence.end)):
            values.append(index)
    return tuple(values)


def build_annotation(
    row_index: int,
    record: CanonicalRecord,
    collator: Any,
    mining_config: dict[str, Any] | None = None,
) -> tuple[EvidenceAnnotation, dict[str, int]]:
    """Create one annotation and auditable skip counters for a normalized record."""

    settings = dict(DEFAULT_MINING_CONFIG)
    if mining_config:
        settings.update(mining_config)
    source_ids, _content, encoder_offsets = collator._encode_source(record, return_offsets=True)
    target_ids, target_offsets = _token_offsets(
        collator.decoder_tokenizer, record.target, max(1, collator.max_target_length - 1)
    )
    copy = align_copy_tokens_with_offsets(
        record.source, len(collator.encoder_prefix), encoder_offsets, collator.decoder_tokenizer
    )
    source_sentences = _sentence_spans(record.source)
    target_sentences = _sentence_spans(record.target)
    source_terms = _candidate_spans(record.source)
    skips: Counter[str] = Counter()
    units: list[EvidenceUnit] = []
    for unit in _candidate_spans(record.target):
        if len(units) >= int(settings["max_units_per_example"]):
            skips["unit_cap"] += 1
            break
        target_sentence = next((sentence for sentence in target_sentences if _inside(sentence, unit)), None)
        if target_sentence is None:
            skips["missing_target_sentence"] += 1
            continue
        source_occurrences = [term for term in source_terms if term.normalized == unit.normalized]
        matching_sentences = [
            sentence for sentence in source_sentences if any(_inside(sentence, term) for term in source_occurrences)
        ]
        if len(matching_sentences) < 2:
            skips["fewer_than_two_source_contexts"] += 1
            continue
        target_local = TextSpan(
            unit.start - target_sentence.start,
            unit.end - target_sentence.start,
            unit.text,
            unit.normalized,
            unit.is_number,
        )
        reference_terms = _content_terms(target_sentence.text, target_local)
        candidates: list[tuple[TextSpan, float]] = []
        for sentence in matching_sentences:
            local_occurrence = next(term for term in source_occurrences if _inside(sentence, term))
            local = TextSpan(
                local_occurrence.start - sentence.start,
                local_occurrence.end - sentence.start,
                local_occurrence.text,
                local_occurrence.normalized,
                local_occurrence.is_number,
            )
            candidates.append((sentence, _context_score(reference_terms, _content_terms(sentence.text, local))))
        best = max(score for _, score in candidates)
        positives = [
            sentence
            for sentence, score in candidates
            if score >= max(float(settings["min_positive_score"]), best - float(settings["positive_band"]))
            and len(set(reference_terms) & set(_content_terms(sentence.text))) >= int(settings["min_context_matches"])
        ][: int(settings["max_positive_sentences"])]
        negatives = [
            sentence
            for sentence, score in candidates
            if score <= float(settings["max_negative_score"]) and score <= best - float(settings["min_context_gap"])
        ][: int(settings["max_negative_sentences"])]
        if not positives or not negatives:
            skips["insufficient_positive_or_negative"] += 1
            continue
        positive_occurrences = [
            term for term in source_occurrences if any(_inside(sentence, term) for sentence in positives)
        ]
        negative_occurrences = [
            term for term in source_occurrences if any(_inside(sentence, term) for sentence in negatives)
        ]
        positive_ranges = [(item.start, item.end) for item in positive_occurrences]
        negative_ranges = [(item.start, item.end) for item in negative_occurrences]
        if set(positive_ranges) & set(negative_ranges):
            skips["ambiguous_source_sentence"] += 1
            continue
        target_positions = [
            index
            for index, offset in enumerate(target_offsets)
            if _overlaps(offset, (unit.start, unit.end)) and offset[1] > offset[0]
        ]
        copy_positive: dict[int, tuple[int, ...]] = {}
        copy_negative: dict[int, tuple[int, ...]] = {}
        for target_position in target_positions:
            token_id = target_ids[target_position]
            positive_positions = tuple(
                index
                for index, (candidate_id, offset) in enumerate(zip(copy["copy_token_ids"], copy["copy_offsets"]))
                if candidate_id == token_id and any(_overlaps(offset, source_range) for source_range in positive_ranges)
            )
            negative_positions = tuple(
                index
                for index, (candidate_id, offset) in enumerate(zip(copy["copy_token_ids"], copy["copy_offsets"]))
                if candidate_id == token_id and any(_overlaps(offset, source_range) for source_range in negative_ranges)
            )
            if positive_positions and negative_positions and not set(positive_positions) & set(negative_positions):
                copy_positive[target_position] = positive_positions
                copy_negative[target_position] = negative_positions
        semantic_positive = _spans_from_positions(
            position
            for sentence in positives
            for position in _encoder_positions(encoder_offsets, len(collator.encoder_prefix), sentence)
        )
        semantic_negative = _spans_from_positions(
            position
            for sentence in negatives
            for position in _encoder_positions(encoder_offsets, len(collator.encoder_prefix), sentence)
        )
        if not target_positions or not copy_positive or not semantic_positive or not semantic_negative:
            skips["tokenization_or_truncation"] += 1
            continue
        min_positive = min(score for sentence, score in candidates if sentence in positives)
        max_negative = max(score for sentence, score in candidates if sentence in negatives)
        confidence = min(1.0, max(0.0, min_positive * (min_positive - max_negative)))
        units.append(
            EvidenceUnit(
                confidence,
                tuple(target_positions),
                tuple(sorted(copy_positive.items())),
                tuple(sorted(copy_negative.items())),
                semantic_positive,
                semantic_negative,
            )
        )
    annotation = EvidenceAnnotation(
        row_index,
        record.example_id,
        text_sha256(record.source),
        text_sha256(record.target),
        ids_sha256(source_ids),
        ids_sha256(target_ids),
        tuple(units),
    )
    return annotation, dict(skips)
