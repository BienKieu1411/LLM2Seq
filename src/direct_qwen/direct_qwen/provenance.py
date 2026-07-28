"""Data, parameter and artifact provenance shared by train/evaluation."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn

from llm2seq_v2.data import dataset_fingerprint, read_jsonl

from .config import SRC_ROOT


def resolve_from_src(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SRC_ROOT / path
    return path.resolve()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    data = config["data"]
    limits = config.get("limits", {})
    return {
        split: dataset_fingerprint(
            resolve_from_src(data[f"{split}_file"]),
            int(limits.get(f"max_{split}_examples", 0)),
        )
        for split in ("train", "validation", "test")
    }


def _normalise(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).casefold()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_splits(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fail on exact source/pair leakage without importing the paper package."""

    data = config["data"]
    limits = config.get("limits", {})
    rows = {
        split: read_jsonl(
            resolve_from_src(data[f"{split}_file"]),
            max_examples=int(limits.get(f"max_{split}_examples", 0)),
        )
        for split in ("train", "validation", "test")
    }
    signatures: Dict[str, Dict[str, set[str]]] = {}
    report: Dict[str, Any] = {"splits": {}, "cross_split": {}}
    for split, values in rows.items():
        ids = [_digest(_normalise(row.get("id", ""))) for row in values]
        sources = [_digest(_normalise(row["source"])) for row in values]
        pairs = [_digest(_normalise(row["source"]) + "\0" + _normalise(row["target"])) for row in values]
        signatures[split] = {"id": set(ids), "source": set(sources), "pair": set(pairs)}
        report["splits"][split] = {
            "num_examples": len(values),
            "duplicate_ids": sum(count - 1 for count in Counter(ids).values() if count > 1),
            "duplicate_sources": sum(count - 1 for count in Counter(sources).values() if count > 1),
            "duplicate_pairs": sum(count - 1 for count in Counter(pairs).values() if count > 1),
        }
    violations = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps = {kind: len(signatures[left][kind] & signatures[right][kind]) for kind in ("id", "source", "pair")}
        report["cross_split"][f"{left}__{right}"] = overlaps
        if overlaps["source"] or overlaps["pair"]:
            violations.append((left, right, overlaps))
    report["passed"] = not violations
    if violations:
        raise ValueError(f"Exact cross-split leakage: {violations}")
    return report


def parameter_manifest(model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    parameters = list(model.parameters())
    total = int(sum(parameter.numel() for parameter in parameters))
    trainable = int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))
    return {
        "base_model_id": config["model"]["base_model_id"],
        "direct_contract_sha256": config["_meta"]["direct_contract_sha256"],
        "unique_parameter_elements": total,
        "trainable_parameter_elements": trainable,
        "trainable_ratio_percent": 100.0 * trainable / max(1, total),
        "full_finetune": trainable == total,
    }


def tokenizer_manifest(tokenizer: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "base_model_id": config["model"]["base_model_id"],
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", config["model"]["name_or_path"])),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1) or -1),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def fingerprint_identity(value: Dict[str, Any]) -> tuple[int, str]:
    return int(value.get("num_examples", -1)), str(value.get("sha256", ""))
