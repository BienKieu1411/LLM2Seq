"""Offline candidate generation for Phase 3 BRIO-like ranking.

Usage:
    python -m eviseq_v2.generate_candidates \
        --config configs/pubmed.yaml \
        --checkpoint runs/eviseq_v2/pubmed/last.pt \
        --output candidates/pubmed_candidates.jsonl \
        --max-examples 20000 \
        --seed 42

Generates 1 greedy + 3 sampled candidates per document.
Computes weighted ROUGE quality scores.
Deduplicates candidates per document.
Skips documents with < 2 unique candidates.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .checkpoint import load_last_checkpoint
from .config import load_config
from .data import clean_text, decoder_seed_ids, encode_source, read_jsonl
from .generation import generate, generate_sampled
from .metrics import rouge_per_example
from .model import EviSeq
from .runtime import _device, _tokenizers

LOGGER = logging.getLogger("eviseq_v2.generate_candidates")


def _compute_quality(
    candidate: str,
    reference: str,
    weights: Dict[str, float],
) -> float:
    """Compute weighted ROUGE quality score for a candidate."""
    scores = rouge_per_example([candidate], [reference])
    if not scores:
        return 0.0
    s = scores[0]
    return (
        weights.get("r1", 0.2) * s["rouge1"]
        + weights.get("r2", 0.6) * s["rouge2"]
        + weights.get("rL", 0.2) * s["rougeL"]
    )


def _pad_source_batch(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    data: Dict[str, Any],
    device: torch.device,
) -> tuple:
    """Prepare a padded batch of source tokens."""
    encoded_list = []
    units_list = []
    for row in rows:
        source = clean_text(row["source"], bool(data.get("clean_wikihow_metadata", True)))
        ids, unit_ids, _ = encode_source(tokenizer, source, data)
        encoded_list.append(torch.tensor(ids, dtype=torch.long))
        units_list.append(torch.tensor(unit_ids, dtype=torch.long))

    max_len = max(v.numel() for v in encoded_list)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    unit_ids_tensor = torch.zeros((len(rows), max_len), dtype=torch.long)

    for i, (ids, uids) in enumerate(zip(encoded_list, units_list)):
        input_ids[i, : ids.numel()] = ids
        attention_mask[i, : ids.numel()] = 1
        unit_ids_tensor[i, : uids.numel()] = uids

    return (
        input_ids.to(device),
        attention_mask.to(device),
        unit_ids_tensor.to(device),
    )


def generate_candidates(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_examples: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate candidate summaries for ranking training.

    For each document, generates:
    - 1 greedy candidate (do_sample=False)
    - 3 sampled candidates (do_sample=True, top_p=0.90, temperature=[0.7, 0.9, 1.1])

    Deduplicates within each document.
    Computes quality scores via weighted ROUGE.
    Saves as JSONL with: source, target, candidates[{text, quality}]
    """
    random.seed(seed)
    torch.manual_seed(seed)

    config = load_config(config_path)
    device = _device()
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)

    model = EviSeq(config)
    load_last_checkpoint(model, checkpoint_path)
    model.to(device).eval()

    data = config["data"]
    ranking = config.get("ranking", {})
    quality_weights = ranking.get("quality_weights", {"r1": 0.2, "r2": 0.6, "rL": 0.2})
    generation = config.get("generation", {})
    batch_size = max(1, int(generation.get("batch_size", 2)))

    clean_metadata = bool(data.get("clean_wikihow_metadata", True))
    rows = read_jsonl(data["train_file"], max_examples=max_examples)
    decoder_seed = decoder_seed_ids(decoder_tokenizer, data)

    # Sampling configurations
    sample_configs = [
        {"do_sample": False, "temperature": 1.0, "top_p": 1.0},
        {"do_sample": True, "temperature": 0.7, "top_p": 0.90},
        {"do_sample": True, "temperature": 0.9, "top_p": 0.90},
        {"do_sample": True, "temperature": 1.1, "top_p": 0.90},
    ]

    max_new_tokens = int(generation.get("max_new_tokens", 256))
    min_new_tokens = int(generation.get("min_new_tokens", 16))

    bf16 = device.type == "cuda" and bool(config.get("training", {}).get("bf16", True))

    def autocast():
        if bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_skipped = 0
    started = time.perf_counter()

    LOGGER.info("Generating candidates for %d documents", len(rows))

    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            input_ids, attention_mask, unit_ids = _pad_source_batch(
                batch_rows,
                encoder_tokenizer,
                data,
                device,
            )

            # Generate for each sampling config
            all_candidates: List[List[str]] = [[] for _ in batch_rows]

            for scfg in sample_configs:
                with autocast():
                    if scfg["do_sample"]:
                        generated = generate_sampled(
                            model,
                            input_ids,
                            attention_mask,
                            decoder_seed,
                            unit_ids=unit_ids,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            eos_token_id=decoder_tokenizer.eos_token_id,
                            pad_token_id=decoder_tokenizer.pad_token_id,
                            temperature=scfg["temperature"],
                            top_p=scfg["top_p"],
                        )
                    else:
                        generated = generate(
                            model,
                            input_ids,
                            attention_mask,
                            decoder_seed,
                            unit_ids=unit_ids,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            eos_token_id=decoder_tokenizer.eos_token_id,
                            pad_token_id=decoder_tokenizer.pad_token_id,
                        )

                decoded = decoder_tokenizer.batch_decode(generated, skip_special_tokens=True)
                for i, text in enumerate(decoded):
                    all_candidates[i].append(text.strip())

            # Process each document
            for i, row in enumerate(batch_rows):
                reference = clean_text(row["target"], clean_metadata)
                seen_texts = set()
                unique_candidates = []

                for cand_text in all_candidates[i]:
                    normalized = " ".join(cand_text.lower().split())
                    if normalized in seen_texts or not cand_text.strip():
                        continue
                    seen_texts.add(normalized)

                    quality = _compute_quality(cand_text, reference, quality_weights)
                    unique_candidates.append(
                        {
                            "text": cand_text,
                            "quality": round(quality, 4),
                        }
                    )

                if len(unique_candidates) < 2:
                    total_skipped += 1
                    continue

                record = {
                    "source": row["source"],
                    "target": row["target"],
                    "candidates": unique_candidates,
                }
                if "id" in row:
                    record["id"] = row["id"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_generated += 1

            if (start // batch_size + 1) % 10 == 0:
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "Progress: %d/%d documents, %d generated, %d skipped, %.1fs elapsed",
                    start + len(batch_rows),
                    len(rows),
                    total_generated,
                    total_skipped,
                    elapsed,
                )

    elapsed = time.perf_counter() - started
    stats = {
        "total_documents": len(rows),
        "generated": total_generated,
        "skipped": total_skipped,
        "elapsed_seconds": round(elapsed, 2),
        "output_file": str(output),
    }
    LOGGER.info("Candidate generation complete: %s", json.dumps(stats))

    # Write stats sidecar
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate summaries for ranking training")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to last.pt checkpoint")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-examples", type=int, default=0, help="Max training documents (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    generate_candidates(
        args.config,
        args.checkpoint,
        args.output,
        max_examples=args.max_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
