"""Exact split-leakage and canonical-data audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from .data import read_jsonl


def _normalise(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).casefold()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit(config: Dict[str, Any], fail_on_cross_split: bool = True) -> Dict[str, Any]:
    data = config["data"]
    limits = config.get("limits", {})
    rows: Dict[str, list[Dict[str, Any]]] = {
        split: read_jsonl(
            data[f"{split}_file"],
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
        key = f"{left}__{right}"
        overlaps = {kind: len(signatures[left][kind] & signatures[right][kind]) for kind in ("id", "source", "pair")}
        report["cross_split"][key] = overlaps
        if overlaps["source"] or overlaps["pair"]:
            violations.append((key, overlaps))
    report["passed"] = not violations
    if violations and fail_on_cross_split:
        raise ValueError(f"Exact cross-split content leakage: {violations}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(load_config(args.config))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
