"""Versioned JSONL storage for offline teacher-generated text and logits.

The cache deliberately has no dependency on ``torch`` or ``transformers``.
That keeps loading a finished cache an offline operation and, importantly,
does not instantiate the student model (or any model) during cache reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

CACHE_KIND = "eviseq_kd_teacher_cache"
CACHE_VERSION = 2
LEGACY_CACHE_VERSIONS = frozenset({1, CACHE_VERSION})
TOPK_ALIGNMENT = {
    "tokenization": "teacher",
    "rows": "teacher_topk_ids[i] and teacher_topk_logits[i] correspond to pseudo_token_ids[i]",
    "logits": "raw pre-temperature teacher logits; rows are selected from a full forward on prompt plus generated sequence",
    "position": "for row i, forward logits at prompt_sequence_width - 1 + teacher_topk_positions[i] predict the generated token",
    "special_tokens": "EOS is included exactly once; PAD and other teacher special tokens are omitted from pseudo_token_ids and top-k rows",
}


def source_hash(text: str) -> str:
    """Return the canonical digest used to bind a teacher row to its source."""

    return sha256(str(text).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert tokenizer metadata to deterministic JSON-safe primitives."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return the effective tokenizer vocabulary size, including added tokens."""

    try:
        size = int(len(tokenizer))
        if size > 0:
            return size
    except (TypeError, ValueError):
        pass
    try:
        vocabulary = tokenizer.get_vocab()
        size = int(len(vocabulary))
        if size > 0:
            return size
    except (AttributeError, TypeError, ValueError):
        pass
    value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        raise ValueError("Teacher tokenizer does not expose a vocabulary size")
    size = int(value)
    if size <= 0:
        raise ValueError(f"Teacher tokenizer vocabulary size must be positive, got {size}")
    return size


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the tokenizer state that determines token IDs and prompt formatting.

    ``name_or_path`` is intentionally not part of the digest: two local copies
    of the same tokenizer should be compatible.  The vocabulary, special-token
    mapping, added vocabulary, chat template, and fast-tokenizer JSON are.
    """

    vocabulary: Any = None
    try:
        vocabulary = tokenizer.get_vocab()
    except AttributeError:
        pass
    backend_json: Any = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        try:
            backend_json = to_str()
        except Exception:  # pragma: no cover - third-party tokenizer fallback
            backend_json = None
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": tokenizer_vocab_size(tokenizer),
        "vocab": vocabulary,
        "added_vocab": getattr(tokenizer, "get_added_vocab", lambda: None)(),
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", []) or []),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "backend_tokenizer": backend_json,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _int_list(value: Any, field_name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [int(item) for item in value]


def _matrix(value: Any, field_name: str, cast: Any) -> list[list[Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of rows")
    rows: list[list[Any]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"{field_name} must contain only list rows")
        rows.append([cast(item) for item in row])
    return rows


@dataclass(frozen=True)
class TeacherRecord:
    example_id: Optional[str]
    source_hash: str
    pseudo_target: str
    pseudo_token_ids: list[int] = field(default_factory=list)
    teacher_topk_ids: list[list[int]] = field(default_factory=list)
    teacher_topk_logits: list[list[float]] = field(default_factory=list)
    # These fields are appended after the original six fields so old callers
    # that construct TeacherRecord positionally remain source-compatible.
    teacher_generated_token_ids: list[int] = field(default_factory=list)
    teacher_topk_positions: list[int] = field(default_factory=list)
    prompt_token_count: int = 0
    prompt_sequence_width: int = 0
    generated_token_count: int = 0
    gold_token_ids: list[int] = field(default_factory=list)
    gold_topk_ids: list[list[int]] = field(default_factory=list)
    gold_topk_logits: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherRecord":
        raw_id = value.get("example_id")
        example_id = None if raw_id is None or not str(raw_id).strip() else str(raw_id).strip()
        pseudo_token_ids = _int_list(value.get("pseudo_token_ids", []), "pseudo_token_ids")
        topk_ids = _matrix(value.get("teacher_topk_ids", []), "teacher_topk_ids", int)
        topk_logits = _matrix(value.get("teacher_topk_logits", []), "teacher_topk_logits", float)
        generated_ids = _int_list(value.get("teacher_generated_token_ids", []), "teacher_generated_token_ids")
        positions = _int_list(value.get("teacher_topk_positions", []), "teacher_topk_positions")
        gold_token_ids = _int_list(value.get("gold_token_ids", []), "gold_token_ids")
        gold_topk_ids = _matrix(value.get("gold_topk_ids", []), "gold_topk_ids", int)
        gold_topk_logits = _matrix(value.get("gold_topk_logits", []), "gold_topk_logits", float)
        generated_count = int(value.get("generated_token_count", len(pseudo_token_ids)))
        return cls(
            example_id=example_id,
            source_hash=str(value.get("source_hash", "")),
            pseudo_target=str(value.get("pseudo_target", "")),
            pseudo_token_ids=pseudo_token_ids,
            teacher_topk_ids=topk_ids,
            teacher_topk_logits=topk_logits,
            teacher_generated_token_ids=generated_ids,
            teacher_topk_positions=positions,
            prompt_token_count=int(value.get("prompt_token_count", 0)),
            prompt_sequence_width=int(value.get("prompt_sequence_width", 0)),
            generated_token_count=generated_count,
            gold_token_ids=gold_token_ids,
            gold_topk_ids=gold_topk_ids,
            gold_topk_logits=gold_topk_logits,
        )

    @property
    def topk_width(self) -> int:
        """Return the cached top-k width, or zero for a text-only row."""

        if not self.teacher_topk_ids:
            return 0
        return len(self.teacher_topk_ids[0])

    def validate(
        self,
        *,
        expected_top_k: int | None = None,
        vocab_size: int | None = None,
        require_alignment: bool = False,
    ) -> None:
        """Validate row shape and the token-position alignment contract."""

        if self.prompt_token_count < 0 or self.prompt_sequence_width < 0 or self.generated_token_count < 0:
            raise ValueError("Teacher cache prompt/generated lengths must be non-negative")
        if self.prompt_sequence_width and self.prompt_token_count > self.prompt_sequence_width:
            raise ValueError("Teacher cache prompt_token_count exceeds prompt_sequence_width")
        if self.generated_token_count and self.generated_token_count != len(self.pseudo_token_ids):
            raise ValueError("Teacher cache generated_token_count does not match pseudo_token_ids")
        if any(token_id < 0 for token_id in self.pseudo_token_ids):
            raise ValueError("Teacher cache pseudo_token_ids must be non-negative")

        has_ids = bool(self.teacher_topk_ids)
        has_logits = bool(self.teacher_topk_logits)
        if has_ids != has_logits:
            raise ValueError("Teacher cache top-k IDs and logits must be supplied together")
        if has_ids:
            if len(self.teacher_topk_ids) != len(self.teacher_topk_logits):
                raise ValueError("Teacher cache top-k IDs/logits row counts must match")
            width = len(self.teacher_topk_ids[0])
            if width <= 0:
                raise ValueError("Teacher cache top-k width must be positive")
            if expected_top_k is not None and expected_top_k > 0 and width != expected_top_k:
                raise ValueError(f"Teacher cache top-k width mismatch: expected {expected_top_k}, found {width}")
            for ids, logits in zip(self.teacher_topk_ids, self.teacher_topk_logits):
                if len(ids) != width or len(logits) != width:
                    raise ValueError("Teacher cache top-k rows must have a constant width")
                if any(token_id < 0 for token_id in ids):
                    raise ValueError("Teacher cache top-k IDs must be non-negative")
                if len(set(ids)) != len(ids):
                    raise ValueError("Teacher cache top-k IDs must not contain duplicates")
                if any(not math.isfinite(value) for value in logits):
                    raise ValueError("Teacher cache top-k logits must be finite")
                if vocab_size is not None and any(token_id >= vocab_size for token_id in ids):
                    raise ValueError("Teacher cache top-k ID exceeds the teacher vocabulary size")
            if require_alignment and len(self.teacher_topk_ids) != len(self.pseudo_token_ids):
                raise ValueError("Teacher top-k rows must align one-to-one with pseudo_token_ids")

        if self.teacher_generated_token_ids or self.teacher_topk_positions:
            if len(self.teacher_topk_positions) != len(self.teacher_topk_ids):
                raise ValueError("teacher_topk_positions must align with teacher top-k rows")
            if any(
                position < 0 or position >= len(self.teacher_generated_token_ids)
                for position in self.teacher_topk_positions
            ):
                raise ValueError("teacher_topk_positions contains an invalid generated-sequence offset")
            if require_alignment and len(self.teacher_topk_positions) == len(self.pseudo_token_ids):
                for pseudo_id, position in zip(self.pseudo_token_ids, self.teacher_topk_positions):
                    if self.teacher_generated_token_ids[position] != pseudo_id:
                        raise ValueError("Teacher top-k position does not point to the corresponding pseudo token")

        gold_has_ids = bool(self.gold_topk_ids)
        gold_has_logits = bool(self.gold_topk_logits)
        if gold_has_ids != gold_has_logits:
            raise ValueError("Gold teacher top-k IDs and logits must be supplied together")
        if gold_has_ids:
            if len(self.gold_topk_ids) != len(self.gold_topk_logits):
                raise ValueError("Gold teacher top-k IDs/logits row counts must match")
            width = len(self.gold_topk_ids[0])
            if width <= 0:
                raise ValueError("Gold teacher top-k width must be positive")
            if expected_top_k is not None and expected_top_k > 0 and width != expected_top_k:
                raise ValueError(f"Gold teacher top-k width mismatch: expected {expected_top_k}, found {width}")
            if require_alignment and len(self.gold_topk_ids) != len(self.gold_token_ids):
                raise ValueError("Gold teacher top-k rows must align one-to-one with gold_token_ids")
            for ids, logits in zip(self.gold_topk_ids, self.gold_topk_logits):
                if len(ids) != width or len(logits) != width:
                    raise ValueError("Gold teacher top-k rows must have a constant width")
                if any(token_id < 0 for token_id in ids):
                    raise ValueError("Gold teacher top-k IDs must be non-negative")
                if len(set(ids)) != len(ids):
                    raise ValueError("Gold teacher top-k IDs must not contain duplicates")
                if any(not math.isfinite(value) for value in logits):
                    raise ValueError("Gold teacher top-k logits must be finite")
                if vocab_size is not None and any(token_id >= vocab_size for token_id in ids):
                    raise ValueError("Gold teacher top-k ID exceeds the teacher vocabulary size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_hash": self.source_hash,
            "pseudo_target": self.pseudo_target,
            "pseudo_token_ids": list(self.pseudo_token_ids),
            "teacher_topk_ids": [list(row) for row in self.teacher_topk_ids],
            "teacher_topk_logits": [list(row) for row in self.teacher_topk_logits],
            "teacher_generated_token_ids": list(self.teacher_generated_token_ids),
            "teacher_topk_positions": list(self.teacher_topk_positions),
            "prompt_token_count": self.prompt_token_count,
            "prompt_sequence_width": self.prompt_sequence_width,
            "generated_token_count": self.generated_token_count,
            "gold_token_ids": list(self.gold_token_ids),
            "gold_topk_ids": [list(row) for row in self.gold_topk_ids],
            "gold_topk_logits": [list(row) for row in self.gold_topk_logits],
        }


class TeacherCache:
    """In-memory indexed view of a cache with metadata validation helpers."""

    def __init__(self, metadata: dict[str, Any], records: list[TeacherRecord]):
        self.metadata = dict(metadata)
        self.records = list(records)
        expected_top_k = int(self.metadata.get("top_k", 0) or 0)
        vocab_size_value = self.metadata.get("teacher_model_vocab_size", self.metadata.get("tokenizer_vocab_size"))
        vocab_size = int(vocab_size_value) if vocab_size_value is not None else None
        has_topk = bool(self.metadata.get("has_topk", False)) or any(
            record.teacher_topk_ids or record.gold_topk_ids for record in records
        )
        has_gold_topk = bool(self.metadata.get("has_gold_topk", False)) or any(
            record.gold_topk_ids for record in records
        )
        require_alignment = bool(self.metadata.get("topk_alignment"))
        widths: set[int] = set()
        for record in records:
            record.validate(
                expected_top_k=expected_top_k or None,
                vocab_size=vocab_size,
                require_alignment=require_alignment,
            )
            if record.topk_width:
                widths.add(record.topk_width)
            elif has_topk:
                raise ValueError("Teacher cache has top-k rows for only some records")
            if has_gold_topk and not record.gold_topk_ids:
                raise ValueError("Teacher cache has gold top-k rows for only some records")
        if len(widths) > 1:
            raise ValueError("Teacher cache top-k width must be constant across records")
        if expected_top_k > 0 and widths and widths != {expected_top_k}:
            raise ValueError(f"Teacher cache top-k width mismatch: metadata={expected_top_k}, rows={sorted(widths)}")

        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for record in records:
            if record.example_id is None:
                continue
            if record.example_id in seen_ids:
                duplicate_ids.add(record.example_id)
            seen_ids.add(record.example_id)
        duplicates = sorted(duplicate_ids)
        if duplicates:
            preview = ", ".join(repr(value) for value in duplicates[:10])
            suffix = "..." if len(duplicates) > 10 else ""
            raise ValueError(f"Teacher cache contains duplicate example IDs: {preview}{suffix}")
        self._by_id = {record.example_id: record for record in records if record.example_id is not None}

    def __len__(self) -> int:
        return len(self.records)

    def get(
        self,
        example_id: Any,
        index: int,
        *,
        source_hash: str = "",
        require_source_match: bool = True,
        allow_index_fallback: bool = False,
    ) -> TeacherRecord:
        key = None if example_id is None else str(example_id).strip() or None
        if key is not None:
            record = self._by_id.get(key)
            if record is None:
                raise KeyError(f"Teacher cache has no record for example ID {key!r}")
        elif allow_index_fallback:
            if index < 0 or index >= len(self.records):
                raise IndexError(f"Teacher cache has no record for dataset index {index}")
            record = self.records[index]
        else:
            raise KeyError(
                f"Teacher cache lookup requires a matching example ID; no ID was supplied for dataset index {index}"
            )
        if require_source_match:
            if not source_hash:
                raise ValueError(f"Source hash is required for teacher cache example {key or index}")
            if not record.source_hash:
                raise ValueError(f"Teacher cache record {key or index} has no source hash; rebuild the cache")
            if source_hash != record.source_hash:
                raise ValueError(
                    f"Teacher cache source hash mismatch for example {key or index}; "
                    "the cache and dataset are not the same snapshot"
                )
        return record


def write_cache(path: str | Path, metadata: dict[str, Any], records: Iterable[TeacherRecord]) -> None:
    """Atomically write a version-2 cache while retaining text-only compatibility."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    normalized_metadata = dict(metadata)
    normalized_metadata["record_count"] = len(materialized)
    has_topk = any(record.teacher_topk_ids or record.gold_topk_ids for record in materialized)
    has_gold_topk = any(record.gold_topk_ids for record in materialized)
    if has_topk and not all(record.teacher_topk_ids and record.teacher_topk_logits for record in materialized):
        raise ValueError("A teacher cache cannot mix top-k and text-only records")
    if has_gold_topk and not all(record.gold_topk_ids and record.gold_topk_logits for record in materialized):
        raise ValueError("A teacher cache cannot mix gold top-k and text-only records")
    normalized_metadata.setdefault("has_topk", has_topk)
    normalized_metadata.setdefault("has_gold_topk", has_gold_topk)
    if "top_k" not in normalized_metadata:
        widths = {record.topk_width for record in materialized if record.topk_width}
        if len(widths) > 1:
            raise ValueError("Teacher cache top-k width must be constant across records")
        normalized_metadata["top_k"] = next(iter(widths), 0)
    if has_topk:
        # A manually-created legacy row may predate the explicit alignment
        # contract.  Builder-produced caches always provide this metadata.
        normalized_metadata.setdefault("topk_alignment", metadata.get("topk_alignment"))
        if normalized_metadata["topk_alignment"] is None:
            normalized_metadata.pop("topk_alignment")
    require_alignment = bool(normalized_metadata.get("topk_alignment"))
    for record in materialized:
        record.validate(
            expected_top_k=int(normalized_metadata.get("top_k", 0) or 0) or None,
            vocab_size=(
                int(normalized_metadata["teacher_model_vocab_size"])
                if normalized_metadata.get("teacher_model_vocab_size") is not None
                else None
            ),
            require_alignment=require_alignment,
        )

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": CACHE_KIND, "version": CACHE_VERSION, "metadata": normalized_metadata}) + "\n")
        for record in materialized:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_cache(path: str | Path) -> TeacherCache:
    """Load a cache without importing a tokenizer, student, or teacher model."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher cache not found: {path}")
    metadata: dict[str, Any] | None = None
    records: list[TeacherRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if metadata is None:
                if not isinstance(value, dict) or value.get("kind") != CACHE_KIND:
                    raise ValueError(f"Unsupported teacher cache header at {path}:{line_number}")
                version = int(value.get("version", 0))
                if version not in LEGACY_CACHE_VERSIONS:
                    raise ValueError(f"Unsupported teacher cache header at {path}:{line_number}")
                metadata = dict(value.get("metadata", {}))
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Teacher cache record must be an object at {path}:{line_number}")
                try:
                    records.append(TeacherRecord.from_dict(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid teacher cache record at {path}:{line_number}: {exc}") from exc
    if metadata is None:
        raise ValueError(f"Teacher cache is empty: {path}")
    if not records:
        raise ValueError(f"Teacher cache contains no records: {path}")
    expected_records = int(metadata.get("record_count", 0))
    if expected_records > 0 and expected_records != len(records):
        raise ValueError(
            f"Teacher cache record count mismatch at {path}: metadata={expected_records}, actual={len(records)}"
        )
    return TeacherCache(metadata, records)
