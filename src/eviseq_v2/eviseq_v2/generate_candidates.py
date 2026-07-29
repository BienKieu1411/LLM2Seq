"""Offline candidate generation for Phase 3 BRIO-like ranking.

Usage:
    python -m eviseq_v2.generate_candidates \
        --config configs/pubmed.yaml \
        --checkpoint runs/eviseq_v2/pubmed/last.pt \
        --output candidates/pubmed_candidates.jsonl \
        --max-examples 40000 \
        --seed 42

Generates 1 greedy + configurable sampled candidates per document.
Computes weighted ROUGE quality scores.
Deduplicates candidates per document.
Skips documents with < 2 unique candidates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from .data import clean_text, decoder_seed_ids, encode_source
from .generation import generate_from_memory, generate_sampled_from_memory
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


def _sample_rows(path: str, maximum: int, seed: int) -> List[Dict[str, Any]]:
    """Deterministically reservoir-sample rows without retaining full PubMed."""

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if "source" not in row or "target" not in row:
                raise ValueError(f"Missing source/target at {path}:{line_number}")
            seen += 1
            if maximum <= 0:
                rows.append(row)
            elif len(rows) < maximum:
                rows.append(row)
            else:
                replacement = rng.randrange(seen)
                if replacement < maximum:
                    rows[replacement] = row
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    rng.shuffle(rows)
    return rows


def _canonical_candidate_text(text: str, tokenizer: Any, max_target_length: int) -> str:
    """Match the exact content truncation later used by ranking teacher forcing."""

    content_budget = max(1, int(max_target_length) - int(tokenizer.eos_token_id is not None))
    ids = tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=True,
        max_length=content_budget,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


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


@torch.inference_mode()
def generate_candidates(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_examples: Optional[int] = None,
    seed: int = 42,
    num_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate candidate summaries for ranking training.

    For each document, generates:
    - 1 greedy candidate (do_sample=False)
    - N-1 sampled candidates in one decoder-parallel call

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
    checkpoint_payload = load_last_checkpoint(model, checkpoint_path)
    checkpoint_architecture = checkpoint_payload.get("config", {}).get("_meta", {}).get("architecture_sha256")
    if checkpoint_architecture != config.get("_meta", {}).get("architecture_sha256"):
        raise RuntimeError("Candidate-generation config does not match the Phase-2 checkpoint architecture")
    model.to(device).eval()

    data = config["data"]
    ranking = config.get("ranking", {})
    quality_weights = ranking.get("quality_weights", {"r1": 0.2, "r2": 0.6, "rL": 0.2})
    generation = config.get("generation", {})
    batch_size = max(1, int(ranking.get("generation_batch_size", 2)))
    requested_candidates = int(num_candidates if num_candidates is not None else ranking.get("num_candidates", 10))
    if requested_candidates < 2:
        raise ValueError("ranking.num_candidates must be at least 2")

    clean_metadata = bool(data.get("clean_wikihow_metadata", True))
    configured_maximum = int(ranking.get("max_examples", 0))
    maximum = configured_maximum if max_examples is None else int(max_examples)
    rows = _sample_rows(data["train_file"], maximum, seed)
    decoder_seed = decoder_seed_ids(decoder_tokenizer, data)

    max_new_tokens = int(ranking.get("candidate_max_new_tokens", generation.get("max_new_tokens", 256)))
    min_new_tokens = int(ranking.get("candidate_min_new_tokens", generation.get("min_new_tokens", 16)))
    repetition_penalty = float(generation.get("repetition_penalty", 1.05))
    no_repeat_ngram_size = int(generation.get("no_repeat_ngram_size", 3))
    sampling_temperature = float(ranking.get("sampling_temperature", 0.9))
    sampling_top_k = int(ranking.get("sampling_top_k", 0))
    sampling_top_p = float(ranking.get("sampling_top_p", 0.95))
    sampling_repetition_penalty = float(ranking.get("sampling_repetition_penalty", 1.0))
    sampling_no_repeat_ngram_size = int(ranking.get("sampling_no_repeat_ngram_size", 0))

    bf16 = device.type == "cuda" and bool(config.get("training", {}).get("bf16", True))

    def autocast():
        if bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_skipped = 0
    candidate_counts: List[int] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    LOGGER.info("Generating candidates for %d documents", len(rows))

    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        cursor = 0
        completed_batches = 0
        active_batch_size = batch_size
        while cursor < len(rows):
            batch_rows = rows[cursor : cursor + active_batch_size]
            input_ids = attention_mask = unit_ids = None
            adapter_output = greedy = sampled = None
            try:
                input_ids, attention_mask, unit_ids = _pad_source_batch(
                    batch_rows,
                    encoder_tokenizer,
                    data,
                    device,
                )

                # Encode each long article once. Sampled candidates are decoded
                # in parallel over an expanded decoder batch. The whole function
                # runs under inference_mode, including this encoder call.
                with autocast():
                    adapter_output = model.encode(input_ids, attention_mask, unit_ids=unit_ids)
                    greedy = generate_from_memory(
                        model,
                        adapter_output,
                        decoder_seed,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        eos_token_id=decoder_tokenizer.eos_token_id,
                        pad_token_id=decoder_tokenizer.pad_token_id,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    sampled = generate_sampled_from_memory(
                        model,
                        adapter_output,
                        decoder_seed,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        eos_token_id=decoder_tokenizer.eos_token_id,
                        pad_token_id=decoder_tokenizer.pad_token_id,
                        temperature=sampling_temperature,
                        top_k=sampling_top_k,
                        top_p=sampling_top_p,
                        repetition_penalty=sampling_repetition_penalty,
                        no_repeat_ngram_size=sampling_no_repeat_ngram_size,
                        num_return_sequences=requested_candidates - 1,
                    )
            except torch.cuda.OutOfMemoryError:
                del input_ids, attention_mask, unit_ids, adapter_output, greedy, sampled
                gc.collect()
                torch.cuda.empty_cache()
                if active_batch_size <= 1:
                    raise
                previous_batch_size = active_batch_size
                active_batch_size = max(1, active_batch_size // 2)
                LOGGER.warning(
                    "CUDA OOM at document %d; retrying the same documents with batch=%d (was %d)",
                    cursor,
                    active_batch_size,
                    previous_batch_size,
                )
                continue

            if greedy is None or sampled is None:
                raise RuntimeError("Candidate generation returned no token sequences")

            greedy_text = decoder_tokenizer.batch_decode(greedy, skip_special_tokens=True)
            flat_sampled = sampled.reshape(-1, sampled.shape[-1])
            sampled_text = decoder_tokenizer.batch_decode(flat_sampled, skip_special_tokens=True)
            sampled_text = [
                sampled_text[index * (requested_candidates - 1) : (index + 1) * (requested_candidates - 1)]
                for index in range(len(batch_rows))
            ]
            all_candidates = [[greedy_text[index], *sampled_text[index]] for index in range(len(batch_rows))]

            # Process each document
            for i, row in enumerate(batch_rows):
                reference = clean_text(row["target"], clean_metadata)
                seen_texts = set()
                unique_texts: List[str] = []

                for raw_candidate in all_candidates[i]:
                    cand_text = _canonical_candidate_text(
                        raw_candidate,
                        decoder_tokenizer,
                        int(data.get("max_target_length", 384)),
                    )
                    normalized = " ".join(cand_text.lower().split())
                    if normalized in seen_texts or not cand_text.strip():
                        continue
                    seen_texts.add(normalized)

                    unique_texts.append(cand_text)

                if len(unique_texts) < 2:
                    total_skipped += 1
                    continue

                component_scores = rouge_per_example(unique_texts, [reference] * len(unique_texts))
                unique_candidates = []
                for cand_text, components in zip(unique_texts, component_scores):
                    quality = sum(
                        float(quality_weights.get(name, 0.0)) * components[key]
                        for name, key in (
                            ("r1", "rouge1"),
                            ("r2", "rouge2"),
                            ("rL", "rougeL"),
                        )
                    )
                    unique_candidates.append(
                        {
                            "text": cand_text,
                            "quality": round(quality, 4),
                            "rouge2": round(float(components["rouge2"]), 4),
                            "rougeL": round(float(components["rougeL"]), 4),
                        }
                    )

                record = {
                    "source": row["source"],
                    "target": row["target"],
                    "candidates": unique_candidates,
                }
                if "id" in row:
                    record["id"] = row["id"]
                serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                handle.write(serialized)
                digest.update(serialized.encode("utf-8"))
                total_generated += 1
                candidate_counts.append(len(unique_candidates))

            cursor += len(batch_rows)
            completed_batches += 1
            if completed_batches % 10 == 0 or cursor == len(rows):
                elapsed = time.perf_counter() - started
                documents_per_second = cursor / max(elapsed, 1.0e-6)
                eta_seconds = (len(rows) - cursor) / max(documents_per_second, 1.0e-6)
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                LOGGER.info(
                    "Progress: %d/%d documents, %d generated, %d skipped, batch=%d, "
                    "%.3f docs/s, peak_vram=%.2f GiB, ETA %.1f min",
                    cursor,
                    len(rows),
                    total_generated,
                    total_skipped,
                    active_batch_size,
                    documents_per_second,
                    peak_vram_gb,
                    eta_seconds / 60.0,
                )

    temporary.replace(output)
    elapsed = time.perf_counter() - started
    stats = {
        "schema_version": 2,
        "total_documents": len(rows),
        "generated": total_generated,
        "skipped": total_skipped,
        "requested_candidates_per_document": requested_candidates,
        "unique_candidates_mean": round(sum(candidate_counts) / max(1, len(candidate_counts)), 4),
        "unique_candidates_min": min(candidate_counts, default=0),
        "unique_candidates_max": max(candidate_counts, default=0),
        "candidate_file_sha256": digest.hexdigest(),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint_payload.get("global_step", -1)),
        "architecture_sha256": str(config.get("_meta", {}).get("architecture_sha256", "")),
        "quality_backend": "rouge==1.0.0",
        "quality_weights": quality_weights,
        "sampling": {
            "temperature": sampling_temperature,
            "top_k": sampling_top_k,
            "top_p": sampling_top_p,
            "repetition_penalty": sampling_repetition_penalty,
            "no_repeat_ngram_size": sampling_no_repeat_ngram_size,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        },
        "elapsed_seconds": round(elapsed, 2),
        "peak_vram_gib": round(
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
            3,
        ),
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
    parser.add_argument("--max-examples", type=int, default=None, help="Max random training documents (0=all)")
    parser.add_argument("--num-candidates", type=int, default=None, help="Total candidates per document")
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
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
