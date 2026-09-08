#!/usr/bin/env python3
"""Prepare a deterministic evidence sidecar without loading model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eviseq_update.config import load_config, resolve_path
from eviseq_update.data.collate import SummarizationCollator
from eviseq_update.data.dataset import JsonlSummarizationDataset
from eviseq_update.data.evidence_cache import cache_config_fingerprint, write_cache
from eviseq_update.data.evidence_mining import DEFAULT_MINING_CONFIG, build_annotation
from eviseq_update.runtime import _tokenizers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-size", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = dict(config["data"])
    path = resolve_path(data[f"{args.split}_file"], config)
    dataset = JsonlSummarizationDataset(path, data)
    encoder, decoder = _tokenizers(config)
    collator = SummarizationCollator(
        encoder, decoder, data, grounded_copy=bool(config["decoder"].get("grounded_copy", {}).get("enabled", False))
    )
    if not collator.grounded_copy:
        raise ValueError("Evidence contrastive preparation requires decoder.grounded_copy.enabled=true")
    evidence_config = config["training"].get("evidence_contrastive", {})
    mining = dict(DEFAULT_MINING_CONFIG)
    mining.update(evidence_config.get("mining", {}))
    rows = []
    skips: Counter[str] = Counter()
    units = 0
    records_with_units = 0
    audit_rows = []
    for row_index in range(len(dataset)):
        annotation, row_skips = build_annotation(row_index, dataset[row_index], collator, mining)
        rows.append(annotation)
        skips.update(row_skips)
        units += len(annotation.units)
        records_with_units += bool(annotation.units)
        if args.audit_size and len(audit_rows) < args.audit_size:
            audit_rows.append(
                {
                    "row_index": row_index,
                    "record_id": annotation.record_id,
                    "source": dataset[row_index].source,
                    "target": dataset[row_index].target,
                    "annotation": annotation.as_dict(),
                    "skip_reason_counts": row_skips,
                }
            )
    manifest = {
        "split": args.split,
        "dataset_path": str(path),
        "config_fingerprint": cache_config_fingerprint(config),
        "mining": mining,
        "units": units,
        "skip_reason_counts": dict(sorted(skips.items())),
        "audit_size_requested": args.audit_size,
        "tokenizer": {"encoder": str(config["model"]["encoder_name"]), "decoder": str(config["model"]["decoder_name"])},
    }
    cache = write_cache(args.output_dir, rows, manifest)
    output_dir = Path(args.output_dir)
    summary = {
        "cache": str(cache),
        "records": len(rows),
        "records_with_units": records_with_units,
        "coverage": records_with_units / max(1, len(rows)),
        "units": units,
        "skip_reason_counts": dict(sorted(skips.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit_rows:
        with (output_dir / "audit_examples.jsonl").open("w", encoding="utf-8") as handle:
            for row in audit_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
