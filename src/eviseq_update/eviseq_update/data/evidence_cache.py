"""Atomic JSONL sidecar cache for deterministic evidence annotations."""

from __future__ import annotations

import hashlib
import json
import os
from array import array
from pathlib import Path
from typing import Any, Iterable

from .evidence import EVIDENCE_SCHEMA_VERSION, EvidenceAnnotation


def cache_config_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint every tokenizer/position choice that can invalidate a sidecar."""

    evidence = config.get("training", {}).get("evidence_contrastive", {})
    payload = {
        "data": config.get("data", {}),
        "model": config.get("model", {}),
        "mining": evidence.get("mining", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_cache_manifest(path: str | Path, config: dict[str, Any]) -> None:
    cache = Path(path)
    manifest_path = cache.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evidence manifest not found beside {cache}; rebuild the cache")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid evidence manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Evidence manifest schema does not match this training code; rebuild the cache")
    if manifest.get("cache_file") != cache.name:
        raise ValueError("Evidence manifest points to a different cache file")
    expected = cache_config_fingerprint(config)
    if manifest.get("config_fingerprint") != expected:
        raise ValueError("Evidence cache configuration/tokenizer fingerprint mismatch; rebuild the cache")
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if manifest.get("cache_sha256") != digest:
        raise ValueError("Evidence cache file changed after preparation; rebuild the cache")


class EvidenceCacheReader:
    def __init__(self, path: str | Path, expected_rows: int | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Evidence cache not found: {self.path}; run scripts/prepare_evidence.py first")
        self.offsets = array("Q")
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if expected_rows is not None and len(self.offsets) < expected_rows:
            raise ValueError(
                f"Evidence cache has {len(self.offsets)} rows but dataset needs {expected_rows}; rebuild the cache"
            )
        self._handle = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, row_index: int) -> EvidenceAnnotation:
        if not 0 <= row_index < len(self.offsets):
            raise IndexError(row_index)
        if self._pid != os.getpid() or self._handle is None:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._pid = os.getpid()
        self._handle.seek(self.offsets[row_index])
        try:
            row = json.loads(self._handle.readline())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evidence cache row {row_index} in {self.path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Evidence cache row {row_index} is not an object")
        return EvidenceAnnotation.from_mapping(row)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_cache(output_dir: str | Path, rows: Iterable[EvidenceAnnotation], manifest: dict[str, Any]) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "evidence.jsonl"
    temporary = destination / "evidence.jsonl.tmp"
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for count, annotation in enumerate(rows, start=1):
            handle.write(json.dumps(annotation.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, cache)
    result = dict(manifest)
    result.update(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        record_count=count,
        cache_file=cache.name,
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
    )
    temporary_manifest = destination / "manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_manifest, destination / "manifest.json")
    return cache
