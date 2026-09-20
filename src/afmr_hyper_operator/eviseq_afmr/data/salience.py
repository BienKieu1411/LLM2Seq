"""Training-only lexical evidence labels for the visible encoder source."""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_SENTENCE_END = frozenset(".!?。！？…")


def _is_sentence_boundary(text: str, left: int, right: int) -> bool:
    """Return whether the gap between two words starts a new sentence.

    Commas and other intra-sentence punctuation remain transparent, matching
    the historical lexical-label behavior.  A decimal point is also kept
    transparent so values such as ``1.2 mg`` are not split into sentences.
    """

    for index in range(left, right):
        character = text[index]
        if character in "\r\n":
            return True
        if character not in _SENTENCE_END:
            continue
        if character == ".":
            previous = text[index - 1] if index > 0 else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if previous.isdigit() and following.isdigit():
                continue
            if previous.isalpha() and following.isalpha():
                continue
        return True
    return False


def _word_groups(words: Sequence[re.Match[str]], text: str) -> list[list[int]]:
    """Group word-match indices without crossing sentence boundaries."""

    groups: list[list[int]] = []
    for index, word in enumerate(words):
        if not groups or _is_sentence_boundary(text, words[index - 1].end(), word.start()):
            groups.append([])
        groups[-1].append(index)
    return groups


def lexical_source_salience(
    source: str,
    target: str,
    encoder_offsets: Sequence[tuple[int, int]],
    content_mask: Sequence[bool],
    *,
    prefix_length: int,
    target_visible_end: int | None = None,
) -> tuple[list[float], bool]:
    """Weight visible source tokens by matched gold bigrams and trigrams.

    The label is a weak, train-only proxy for evidence, not a factuality label.
    Matching is case-insensitive, confined to source text visible after encoder
    truncation, and does not form n-grams across sentence-ending punctuation or
    newlines. Rows with no positive or no negative content token are excluded
    from the auxiliary loss.
    """

    labels = [0.0] * len(encoder_offsets)
    visible_end = max(
        (end - prefix_length for (_, end), content in zip(encoder_offsets, content_mask) if content),
        default=0,
    )
    if visible_end <= 0:
        return labels, False

    # Include one extra character so a word cut in the middle by truncation
    # cannot be mistaken for a complete word.
    visible_text = source[: visible_end + 1]
    visible_words = [match for match in _WORD.finditer(visible_text) if match.end() <= visible_end]
    target_end = len(target) if target_visible_end is None else min(len(target), target_visible_end)
    target_text = target[: target_end + 1]
    target_matches = [match for match in _WORD.finditer(target_text) if match.end() <= target_end]
    if len(visible_words) < 2 or len(target_matches) < 2:
        return labels, False

    source_groups = _word_groups(visible_words, visible_text)
    target_groups = _word_groups(target_matches, target_text)
    reference_ngrams = {
        tuple(target_matches[index].group().casefold() for index in group[start : start + n])
        for group in target_groups
        for n in (2, 3)
        for start in range(len(group) - n + 1)
    }
    source_words = [match.group().casefold() for match in visible_words]
    positive_words = [0.0] * len(source_words)
    for group in source_groups:
        for n in (2, 3):
            for start in range(len(group) - n + 1):
                positions = group[start : start + n]
                if tuple(source_words[position] for position in positions) in reference_ngrams:
                    for position in positions:
                        positive_words[position] = max(positive_words[position], float(n - 1))

    positive_spans = [
        (match.start(), match.end(), weight) for match, weight in zip(visible_words, positive_words) if weight > 0
    ]
    span_index = 0
    for token_index, ((start, end), content) in enumerate(zip(encoder_offsets, content_mask)):
        if not content:
            continue
        start, end = max(0, start - prefix_length), end - prefix_length
        while span_index < len(positive_spans) and positive_spans[span_index][1] <= start:
            span_index += 1
        index = span_index
        while index < len(positive_spans):
            left, right, weight = positive_spans[index]
            if left >= end:
                break
            if left < end and start < right:
                labels[token_index] = max(labels[token_index], weight)
            index += 1

    positive_count = sum(value > 0 for value in labels)
    content_count = sum(bool(value) for value in content_mask)
    return labels, 0 < positive_count < content_count
