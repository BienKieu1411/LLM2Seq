"""Versioned JSONL storage for offline teacher-generated text."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Optional


def source_hash(text: str) -> str:
    """Return the canonical digest used to bind a teacher row to its source."""

    return sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TeacherRecord:
    example_id: Optional[str]
    source_hash: str
    pseudo_target: str
    pseudo_token_ids: list[int] = field(default_factory=list)
    teacher_topk_ids: list[list[int]] = field(default_factory=list)
    teacher_topk_logits: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherRecord":
        raw_id = value.get("example_id")
        example_id = None if raw_id is None or not str(raw_id).strip() else str(raw_id).strip()
        return cls(
            example_id=example_id,
            source_hash=str(value.get("source_hash", "")),
            pseudo_target=str(value.get("pseudo_target", "")),
            pseudo_token_ids=[int(item) for item in value.get("pseudo_token_ids", [])],
            teacher_topk_ids=[[int(item) for item in row] for row in value.get("teacher_topk_ids", [])],
            teacher_topk_logits=[[float(item) for item in row] for row in value.get("teacher_topk_logits", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_hash": self.source_hash,
            "pseudo_target": self.pseudo_target,
            "pseudo_token_ids": self.pseudo_token_ids,
            "teacher_topk_ids": self.teacher_topk_ids,
            "teacher_topk_logits": self.teacher_topk_logits,
        }


class TeacherCache:
    """In-memory indexed view of a cache with metadata validation helpers."""

    def __init__(self, metadata: dict[str, Any], records: list[TeacherRecord]):
        self.metadata = dict(metadata)
        self.records = list(records)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "eviseq_kd_teacher_cache", "version": 1, "metadata": metadata}) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_cache(path: str | Path) -> TeacherCache:
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
                if value.get("kind") != "eviseq_kd_teacher_cache" or int(value.get("version", 0)) != 1:
                    raise ValueError(f"Unsupported teacher cache header at {path}:{line_number}")
                metadata = dict(value.get("metadata", {}))
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Teacher cache record must be an object at {path}:{line_number}")
                records.append(TeacherRecord.from_dict(value))
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
