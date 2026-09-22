"""Small, dependency-light helpers for inspecting summarization JSONL files."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ALIASES = ("source", "input", "document", "article", "text")
DEFAULT_TARGET_ALIASES = ("target", "output", "summary", "abstract", "content")


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Yield ``(line_number, raw_line, object)`` for non-empty JSONL lines."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is JSON {type(value).__name__}, expected an object")
            yield line_number, raw_line, value


def get_field(row: Mapping[str, Any], field: str, aliases: tuple[str, ...] = ()) -> Any:
    """Read a dotted field, then fall back to common aliases when it is absent."""

    def nested(name: str) -> Any:
        value: Any = row
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return value

    value = nested(field)
    if value is not None:
        return value
    for alias in aliases:
        value = nested(alias)
        if value is not None:
            return value
    return None


def as_text(value: Any, *, separator: str = "\n") -> str:
    """Convert the string/list forms used by the project to plain text."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return separator.join(as_text(item, separator=separator) for item in value if item not in (None, "")).strip()
    if value is None:
        return ""
    return str(value).strip()


def record_texts(
    row: Mapping[str, Any],
    *,
    source_field: str,
    target_field: str,
    separator: str = "\n",
) -> tuple[str, str]:
    source = as_text(get_field(row, source_field, DEFAULT_SOURCE_ALIASES), separator=separator)
    target = as_text(get_field(row, target_field, DEFAULT_TARGET_ALIASES), separator=separator)
    return source, target


def _detokenize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = unicodedata.normalize("NFKC", line).replace("``", '"').replace("''", '"')
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?%])", r"\1", line)
        line = re.sub(r"([\(\[\{])\s+", r"\1", line)
        line = re.sub(r"\s+([\)\]\}])", r"\1", line)
        line = re.sub(r"\s+n['’]t\b", "n't", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m))\b", r"\1", line, flags=re.IGNORECASE)
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def _detokenize_value(value: Any, *, separator: str) -> Any:
    if isinstance(value, str):
        return _detokenize_text(value)
    if isinstance(value, list):
        return [_detokenize_value(item, separator=separator) for item in value]
    if isinstance(value, tuple):
        return tuple(_detokenize_value(item, separator=separator) for item in value)
    return value


def _find_field_path(row: Mapping[str, Any], field: str, aliases: tuple[str, ...]) -> tuple[str, ...] | None:
    def exists(path: str) -> bool:
        value: Any = row
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return False
            value = value[part]
        return True

    for candidate in (field, *aliases):
        if exists(candidate):
            return tuple(candidate.split("."))
    return None


def _set_nested(row: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = row
    for part in path[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            return
        target = child
    target[path[-1]] = value


def detokenize_record_fields(
    row: dict[str, Any],
    *,
    source_field: str,
    target_field: str,
    separator: str = "\n",
) -> dict[str, Any]:
    """Detokenize only source/target fields while preserving every other field."""

    for field, aliases in (
        (source_field, DEFAULT_SOURCE_ALIASES),
        (target_field, DEFAULT_TARGET_ALIASES),
    ):
        path = _find_field_path(row, field, aliases)
        if path is None:
            continue
        current: Any = row
        for part in path:
            current = current[part]
        _set_nested(row, path, _detokenize_value(current, separator=separator))
    return row


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def flatten_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Return leaf values using dotted paths, while retaining empty objects/lists."""

    if isinstance(value, dict):
        if not value:
            return {prefix or "<root>": value}
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_fields(child, path))
        return flattened
    return {prefix or "<root>": value}


def compact_value(value: Any, max_chars: int) -> Any:
    """Keep schema/sample reports readable without dropping field names."""

    if max_chars <= 0:
        return value
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    if len(rendered) <= max_chars:
        return value
    return rendered[: max(0, max_chars - 3)] + "..."


class TokenCounter:
    """Count model tokens, or fall back to non-whitespace token-like units."""

    def __init__(self, tokenizer: Any | None = None):
        self.tokenizer = tokenizer
        self.description = "whitespace" if tokenizer is None else tokenizer.name_or_path

    def __call__(self, text: str) -> int:
        if self.tokenizer is None:
            return len(re.findall(r"\S+", text))
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                raise ValueError("tokenizer returned a batch for one text")
            ids = ids[0]
        return len(ids)


def load_tokenizer(path: str | None, *, allow_download: bool = False, trust_remote_code: bool = False) -> Any | None:
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - only reached in minimal environments
        raise RuntimeError("--tokenizer requires transformers; install it in the training environment") from exc
    return AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=not allow_download,
        trust_remote_code=trust_remote_code,
    )


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "mean": sum(values) / len(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_counters(
    *,
    tokenizer: str | None,
    source_tokenizer: str | None,
    target_tokenizer: str | None,
    allow_download: bool,
    trust_remote_code: bool,
) -> tuple[TokenCounter, TokenCounter]:
    shared = load_tokenizer(tokenizer, allow_download=allow_download, trust_remote_code=trust_remote_code)
    # Reuse the shared instance when possible. This avoids loading a large tokenizer twice.
    source = (
        load_tokenizer(source_tokenizer, allow_download=allow_download, trust_remote_code=trust_remote_code)
        if source_tokenizer
        else shared
    )
    target = (
        load_tokenizer(target_tokenizer, allow_download=allow_download, trust_remote_code=trust_remote_code)
        if target_tokenizer
        else shared
    )
    return TokenCounter(source), TokenCounter(target)
