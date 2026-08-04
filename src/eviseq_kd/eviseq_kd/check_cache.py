"""Inspect a teacher cache without requiring a tokenizer or package install."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import load_cache


def check_cache(cache_path: str | Path) -> dict[str, object]:
    cache = load_cache(cache_path)
    return {
        "cache": str(Path(cache_path).resolve()),
        "teacher_model": cache.metadata.get("teacher_model", ""),
        "records": len(cache),
        "split": cache.metadata.get("split", ""),
        "max_new_tokens": cache.metadata.get("max_new_tokens", 0),
        "num_beams": cache.metadata.get("num_beams", 0),
        "has_topk": bool(cache.metadata.get("has_topk", False)),
        "has_gold_topk": bool(cache.metadata.get("has_gold_topk", False)),
        "top_k": int(cache.metadata.get("top_k", 0) or 0),
        "tokenizer_fingerprint": cache.metadata.get(
            "teacher_tokenizer_fingerprint", cache.metadata.get("tokenizer_fingerprint", "")
        ),
        "tokenizer_vocab_size": cache.metadata.get("teacher_tokenizer_vocab_size", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an EviSeq-KD teacher cache")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    print(json.dumps(check_cache(args.cache), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
