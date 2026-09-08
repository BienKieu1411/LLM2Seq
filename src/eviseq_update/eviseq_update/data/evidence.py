"""Typed, sparse contracts for evidence-contrastive supervision.

The cache stores only deterministic source/reference alignments.  It contains
no model scores, generated text, or trainable state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_TENSOR_KEYS = (
    "evidence_unit_batch_index",
    "evidence_unit_confidence",
    "evidence_unit_valid",
    "evidence_target_unit",
    "evidence_target_hidden_pos",
    "evidence_target_weight",
    "evidence_copy_owner",
    "evidence_copy_source_position",
    "evidence_copy_positive",
    "evidence_semantic_owner",
    "evidence_semantic_source_position",
    "evidence_semantic_positive",
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ids_sha256(values: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _positions(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"Evidence {field} must be a list of non-negative integer positions")
    positions = tuple(int(item) for item in value)
    if len(set(positions)) != len(positions):
        raise ValueError(f"Evidence {field} contains duplicate positions")
    return positions


def _spans(value: Any, field: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Evidence {field} must be a list of [start, end] spans")
    spans: list[tuple[int, int]] = []
    for span in value:
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not isinstance(span[0], int)
            or not isinstance(span[1], int)
            or span[0] < 0
            or span[1] <= span[0]
        ):
            raise ValueError(f"Evidence {field} contains an invalid span")
        spans.append((int(span[0]), int(span[1])))
    return tuple(spans)


def _copy_map(value: Any, field: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evidence {field} must map target positions to source positions")
    result: list[tuple[int, tuple[int, ...]]] = []
    for target, positions in value.items():
        try:
            target_position = int(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Evidence {field} has a non-integer target position") from exc
        if target_position < 0:
            raise ValueError(f"Evidence {field} has a negative target position")
        result.append((target_position, _positions(positions, field)))
    result.sort(key=lambda item: item[0])
    if len({target for target, _ in result}) != len(result):
        raise ValueError(f"Evidence {field} has duplicate target positions")
    return tuple(result)


@dataclass(frozen=True)
class EvidenceUnit:
    confidence: float
    target_positions: tuple[int, ...]
    copy_positive_by_target: tuple[tuple[int, tuple[int, ...]], ...]
    copy_negative_by_target: tuple[tuple[int, tuple[int, ...]], ...]
    semantic_positive_spans: tuple[tuple[int, int], ...]
    semantic_negative_spans: tuple[tuple[int, int], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceUnit":
        try:
            confidence = float(value["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Evidence unit requires finite confidence") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Evidence confidence must lie in [0, 1]")
        target_positions = _positions(value.get("target_positions", []), "target_positions")
        positive = _copy_map(value.get("copy_positive_by_target", {}), "copy_positive_by_target")
        negative = _copy_map(value.get("copy_negative_by_target", {}), "copy_negative_by_target")
        if not set(target for target, _ in positive).issubset(target_positions):
            raise ValueError("Copy-positive target position is absent from the evidence unit")
        if not set(target for target, _ in negative).issubset(target_positions):
            raise ValueError("Copy-negative target position is absent from the evidence unit")
        return cls(
            confidence,
            target_positions,
            positive,
            negative,
            _spans(value.get("semantic_positive_spans", []), "semantic_positive_spans"),
            _spans(value.get("semantic_negative_spans", []), "semantic_negative_spans"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "target_positions": list(self.target_positions),
            "copy_positive_by_target": {str(key): list(value) for key, value in self.copy_positive_by_target},
            "copy_negative_by_target": {str(key): list(value) for key, value in self.copy_negative_by_target},
            "semantic_positive_spans": [list(span) for span in self.semantic_positive_spans],
            "semantic_negative_spans": [list(span) for span in self.semantic_negative_spans],
        }

    @property
    def copy_positive(self) -> dict[int, tuple[int, ...]]:
        return dict(self.copy_positive_by_target)

    @property
    def copy_negative(self) -> dict[int, tuple[int, ...]]:
        return dict(self.copy_negative_by_target)


@dataclass(frozen=True)
class EvidenceAnnotation:
    row_index: int
    record_id: str
    normalized_source_sha256: str
    normalized_target_sha256: str
    source_input_ids_sha256: str
    target_input_ids_sha256: str
    units: tuple[EvidenceUnit, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceAnnotation":
        try:
            row_index = int(value["row_index"])
            record_id = str(value.get("record_id", ""))
            hashes = tuple(
                str(value[key])
                for key in (
                    "normalized_source_sha256",
                    "normalized_target_sha256",
                    "source_input_ids_sha256",
                    "target_input_ids_sha256",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Evidence annotation is missing its cache identity") from exc
        if row_index < 0 or any(len(item) != 64 for item in hashes):
            raise ValueError("Evidence annotation has an invalid cache identity")
        units_raw = value.get("units", [])
        if not isinstance(units_raw, list):
            raise ValueError("Evidence annotation units must be a list")
        return cls(row_index, record_id, *hashes, tuple(EvidenceUnit.from_mapping(unit) for unit in units_raw))

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "record_id": self.record_id,
            "normalized_source_sha256": self.normalized_source_sha256,
            "normalized_target_sha256": self.normalized_target_sha256,
            "source_input_ids_sha256": self.source_input_ids_sha256,
            "target_input_ids_sha256": self.target_input_ids_sha256,
            "units": [unit.as_dict() for unit in self.units],
        }

    def validate_record(self, row_index: int, record_id: str, source: str, target: str) -> None:
        if self.row_index != row_index:
            raise ValueError(f"Evidence cache row mismatch: expected {row_index}, found {self.row_index}")
        if self.record_id != record_id:
            raise ValueError(f"Evidence cache record ID mismatch at row {row_index}")
        if self.normalized_source_sha256 != text_sha256(source) or self.normalized_target_sha256 != text_sha256(target):
            raise ValueError(f"Evidence cache content hash mismatch at row {row_index}; rebuild the cache")


def empty_evidence_tensors() -> dict[str, torch.Tensor]:
    return {
        "evidence_unit_batch_index": torch.empty(0, dtype=torch.long),
        "evidence_unit_confidence": torch.empty(0, dtype=torch.float32),
        "evidence_unit_valid": torch.empty(0, dtype=torch.bool),
        "evidence_target_unit": torch.empty(0, dtype=torch.long),
        "evidence_target_hidden_pos": torch.empty(0, dtype=torch.long),
        "evidence_target_weight": torch.empty(0, dtype=torch.float32),
        "evidence_copy_owner": torch.empty(0, dtype=torch.long),
        "evidence_copy_source_position": torch.empty(0, dtype=torch.long),
        "evidence_copy_positive": torch.empty(0, dtype=torch.bool),
        "evidence_semantic_owner": torch.empty(0, dtype=torch.long),
        "evidence_semantic_source_position": torch.empty(0, dtype=torch.long),
        "evidence_semantic_positive": torch.empty(0, dtype=torch.bool),
    }
