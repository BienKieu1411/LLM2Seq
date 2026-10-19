#!/usr/bin/env python3
"""Compare nested groups of 1/4/8 sampled SEAM candidates on 100 train rows.

Gold-based selections are oracle diagnostics, not deployable rerankers.
The first 1/4/8 outputs of the SAME n=8 request form each comparison group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eviseq_new"))

import evaluate_rouge_vietnamese as legacy

METRICS = ("rouge1", "rouge2", "rougeL")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_train(path, data_config, count, seed):
    """Seeded reservoir sample; keep physical line numbers for traceability."""
    from eviseq_afmr.data.normalization import detokenize
    from eviseq_afmr.data.schema import CanonicalRecord

    rng, selected, total = random.Random(seed), [], 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total += 1
            slot = total - 1 if total <= count else rng.randrange(total)
            if slot >= count:
                continue
            record = CanonicalRecord.from_mapping(
                json.loads(line),
                source_field=data_config.get("source_field", "text"),
                target_field=data_config.get("target_field", "summary"),
                id_field=data_config.get("id_field", "id"),
                separator=data_config.get("list_separator", "\n"),
            )
            row = {
                "line_number": line_number,
                "id": record.example_id or str(line_number),
                # Serving applies the resolved config's source detokenization.
                "source": record.source,
                "reference": detokenize(record.target) if data_config.get("detokenize") else record.target,
            }
            if total <= count:
                selected.append(row)
            else:
                selected[slot] = row
    if total < count:
        raise ValueError(f"Requested {count} train rows, but only {total} are present")
    return sorted(selected, key=lambda row: row["line_number"])


def generate(args, config, samples, budget):
    destination = args.output_dir / "candidates.jsonl"
    saved = []
    if destination.exists():
        with destination.open(encoding="utf-8") as stream:
            saved = [json.loads(line) for line in stream if line.strip()]
    if len(saved) > len(samples):
        raise ValueError("Saved candidate count exceeds selected sample count")
    for index, row in enumerate(saved):
        if any(row.get(key) != samples[index][key] for key in samples[index]) or len(row["candidates"]) != 8:
            raise ValueError(f"Saved candidates disagree with selected sample at row {index}")
    if len(saved) == len(samples):
        print("[generate] All candidates already saved; skipping model loading", flush=True)
        return saved

    from eviseq_afmr.config import resolve_path
    from seam_vllm import SEAMLLM
    from seam_vllm._prompt import configured_decoder_prompt_ids
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_path(config["model"]["decoder_name"], config)),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        local_files_only=True,
    )
    decoder_limit = len(configured_decoder_prompt_ids(tokenizer, config["data"])) + budget
    # vLLM reserves against the configured MM maximum, even for lower request caps.
    source_limit = int(config["data"]["max_source_length"])
    model = SEAMLLM(
        args.config,
        args.checkpoint,
        seed=args.seed,
        max_model_len=decoder_limit,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(args.max_num_batched_tokens, source_limit, decoder_limit, args.max_num_seqs),
        gpu_memory_utilization=args.gpu_memory_utilization,
        encoder_max_batch_tokens=args.encoder_max_batch_tokens,
        mm_processor_cache_gb=0,
    )
    started = time.monotonic()
    with destination.open("a", encoding="utf-8") as stream:
        for start in range(len(saved), len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            # Per-source seeds keep unfinished requests reproducible on resume.
            params = [
                SamplingParams(
                    n=8,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1 if args.top_k == 0 else args.top_k,
                    max_tokens=budget,
                    min_tokens=0,
                    repetition_penalty=1.0,
                    seed=args.seed + start + offset,
                )
                for offset in range(len(batch))
            ]
            outputs = model.generate([row["source"] for row in batch], params, max_input_tokens=args.max_input_tokens)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned an incorrect source count")
            for row, output in zip(batch, outputs, strict=True):
                completions = sorted(output.outputs, key=lambda completion: completion.index)
                if [completion.index for completion in completions] != list(range(8)):
                    raise RuntimeError("vLLM did not return eight indexed candidates")
                result = {
                    **row,
                    "candidates": [
                        {
                            "index": completion.index,
                            "prediction": completion.text,
                            "tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                        }
                        for completion in completions
                    ],
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                saved.append(result)
            print(
                f"[generate] {len(saved)}/{len(samples)} sources saved | elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return saved


def score_candidates(rows, rouge_path, output_dir):
    """One Perl call for all 800 candidates, with per-candidate F1 scores."""
    from evaluate_rouge import _parse_per_example
    from pyrouge import Rouge155

    legacy.validate_rouge_install(rouge_path)
    with tempfile.TemporaryDirectory(prefix="seam-candidates-rouge-") as temporary:
        root = Path(temporary)
        system, reference = root / "system", root / "reference"
        system.mkdir()
        reference.mkdir()
        for row_index, row in enumerate(rows):
            for candidate_index, candidate in enumerate(row["candidates"]):
                key = f"{row_index * 8 + candidate_index:06d}"
                raw = root / "prediction.txt"
                raw.write_text(candidate["prediction"], encoding="utf-8")
                (system / f"Model.{key}.txt").write_text(legacy.clean(legacy.read_prediction(raw)), encoding="utf-8")
                # Each JSONL target is ONE reference, including multiline targets.
                gold = " ".join(line.strip() for line in row["reference"].splitlines() if line.strip())
                (reference / f"Reference.A.{key}.txt").write_text(legacy.clean(gold), encoding="utf-8")
        rouge = Rouge155(str(rouge_path))
        rouge.system_dir, rouge.model_dir = str(system), str(reference)
        rouge.system_filename_pattern = r"Model\.(\d+)\.txt"
        rouge.model_filename_pattern = "Reference.A.#ID#.txt"
        raw_output = rouge.convert_and_evaluate(rouge_args=f"-e {rouge_path / 'data'} -a -m -n 2 -d")
        (output_dir / "rouge_raw.txt").write_text(raw_output, encoding="utf-8")
        details = _parse_per_example(raw_output, len(rows) * 8)
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row["candidates"]):
            detail = details[row_index * 8 + candidate_index]
            candidate["rouge"] = {metric: detail[metric]["f1"] for metric in METRICS}
            candidate["weighted_rouge"] = (
                candidate["rouge"]["rouge1"] + 2 * candidate["rouge"]["rouge2"] + candidate["rouge"]["rougeL"]
            ) / 4
    with (output_dir / "scored_candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(rows):
    def average(candidates):
        return {
            metric: sum(candidate["rouge"][metric] for candidate in candidates) / len(candidates) for metric in METRICS
        }

    groups = {}
    for count in (1, 4, 8):
        pools = [row["candidates"][:count] for row in rows]
        selected = [max(pool, key=lambda item: item["weighted_rouge"]) for pool in pools]
        best_r2 = [max(pool, key=lambda item: item["rouge"]["rouge2"]) for pool in pools]
        groups[str(count)] = {
            "random_candidate_expectation": average([candidate for pool in pools for candidate in pool]),
            "oracle_weighted_rouge": average(selected),
            "oracle_rouge2": average(best_r2),
            "oracle_selected_indices": [candidate["index"] for candidate in selected],
            "mean_unique_candidates": sum(len({item["prediction"] for item in pool}) for pool in pools) / len(pools),
            "length_capped_fraction": sum(item["finish_reason"] == "length" for pool in pools for item in pool)
            / (len(pools) * count),
        }
    return groups


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, help="Default: data.train_file from resolved config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="Default: generation budget from resolved config")
    parser.add_argument("--batch-size", type=int, default=8, help="Sources per batch; each produces 8 candidates")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--encoder-max-batch-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--rouge-path", type=Path, default=legacy.ROUGE_PATH)
    parser.add_argument("--score-only", action="store_true", help="Score saved candidates without loading any model")
    return parser.parse_args()


def main():
    from eviseq_afmr.config import load_config, resolve_path

    args = parse_args()
    if (
        min(
            args.num_samples,
            args.batch_size,
            args.max_num_seqs,
            args.encoder_max_batch_tokens,
            args.max_num_batched_tokens,
        )
        < 1
    ):
        raise ValueError("Sample/batch/token counts must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Require temperature > 0, 0 < top-p <= 1 and top-k >= 0")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("Require 0 < gpu-memory-utilization < 1")
    config = load_config(args.config, train_only=True)
    train_file = args.train_file or resolve_path(config["data"]["train_file"], config)
    samples = sample_train(train_file, config["data"], args.num_samples, args.seed)
    budget = args.max_new_tokens if args.max_new_tokens is not None else int(config["generation"]["max_new_tokens"])
    if (
        budget < 1
        or args.max_input_tokens is not None
        and not 0 < args.max_input_tokens <= config["data"]["max_source_length"]
    ):
        raise ValueError("Invalid output budget or input limit beyond resolved config")
    signature = {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "sample_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "train_file": str(train_file.expanduser().resolve()),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": budget,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("Output directory belongs to a different experiment; choose a new --output-dir")
    elif (args.output_dir / "candidates.jsonl").exists():
        raise ValueError("Candidate file has no manifest; choose a new --output-dir")
    else:
        write_json(manifest, signature)
    write_json(args.output_dir / "selected_train_samples.json", samples)
    if args.score_only:
        with (args.output_dir / "candidates.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != len(samples) or any(
            any(row.get(key) != sample[key] for key in sample) or len(row["candidates"]) != 8
            for row, sample in zip(rows, samples, strict=True)
        ):
            raise ValueError("Saved candidates are incomplete or differ from selected samples")
    else:
        rows = generate(args, config, samples, budget)
    groups = summarize(score_candidates(rows, args.rouge_path.expanduser().resolve(), args.output_dir))
    report = {
        "split": "train",
        "samples": len(rows),
        "scale": "0_to_1",
        "protocol": "Perl ROUGE-1.5.5 -a -m -n 2 -d; macro mean of per-example F1",
        "selection": "Nested first 1/4/8 of the same n=8 request; gold-based oracle is diagnostic only",
        "oracle_weights": {"rouge1": 1, "rouge2": 2, "rougeL": 1},
        "generation": signature,
        "groups": groups,
    }
    write_json(args.output_dir / "comparison.json", report)
    for count, group in groups.items():
        for mode in ("random_candidate_expectation", "oracle_weighted_rouge", "oracle_rouge2"):
            values = " / ".join(f"{100 * group[mode][metric]:.2f}" for metric in METRICS)
            print(f"N={count} | {mode} | R1/R2/RL={values}", flush=True)
    print(f"Report: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
