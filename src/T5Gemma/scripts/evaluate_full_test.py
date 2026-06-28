#!/usr/bin/env python3
"""Generate full-test predictions and metrics for the T5Gemma LoRA baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

T5GEMMA_ROOT = Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    env_file = Path(os.environ.get("ENV_FILE", T5GEMMA_ROOT / "env.txt"))
    if not env_file.is_absolute():
        env_file = T5GEMMA_ROOT.parent / env_file
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path, limit: int = -1) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and len(examples) >= limit:
                break
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def torch_dtype_from_config(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if lowered in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def word_count(text: str) -> int:
    return len(text.split())


def repeated_ngram_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    compute_bertscore: bool = False,
    bertscore_model_type: str = "xlm-roberta-large",
) -> Dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=False,
    )
    rouge_totals = {name: 0.0 for name in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for name in rouge_totals:
            rouge_totals[name] += scores[name].fmeasure

    n = max(1, len(predictions))
    metrics: Dict[str, Any] = {name: round((value / n) * 100.0, 4) for name, value in rouge_totals.items()}
    bleu = corpus_bleu(predictions, [references])
    chrf = corpus_chrf(predictions, [references])
    metrics.update(
        {
            "bleu": round(bleu.score, 4),
            "bleu_bp": round(bleu.bp, 6),
            "chrf": round(chrf.score, 4),
            "num_examples": len(predictions),
        }
    )

    pred_words = [word_count(pred) for pred in predictions]
    ref_words = [word_count(ref) for ref in references]
    length_ratios = [pred_len / max(1, ref_len) for pred_len, ref_len in zip(pred_words, ref_words)]
    repeat_rates = [repeated_ngram_rate(pred, n=3) for pred in predictions]
    metrics.update(
        {
            "prediction_words_mean": round(safe_mean(pred_words), 4),
            "reference_words_mean": round(safe_mean(ref_words), 4),
            "length_ratio_mean": round(safe_mean(length_ratios), 6),
            "empty_prediction_rate": round(
                100.0 * safe_mean([1.0 if not pred.strip() else 0.0 for pred in predictions]), 4
            ),
            "too_short_rate": round(100.0 * safe_mean([1.0 if ratio < 0.5 else 0.0 for ratio in length_ratios]), 4),
            "too_long_rate": round(100.0 * safe_mean([1.0 if ratio > 1.5 else 0.0 for ratio in length_ratios]), 4),
            "repeated_trigram_rate_mean": round(100.0 * safe_mean(repeat_rates), 4),
        }
    )

    if sources is not None:
        source_words = [word_count(src) for src in sources]
        compression_ratios = [pred_len / max(1, src_len) for pred_len, src_len in zip(pred_words, source_words)]
        metrics.update(
            {
                "source_words_mean": round(safe_mean(source_words), 4),
                "compression_ratio_mean": round(safe_mean(compression_ratios), 6),
            }
        )

    if compute_bertscore:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            predictions,
            references,
            model_type=bertscore_model_type,
            verbose=True,
        )
        metrics.update(
            {
                "bertscore_model_type": bertscore_model_type,
                "bertscore_precision": round(float(precision.mean().item()) * 100.0, 4),
                "bertscore_recall": round(float(recall.mean().item()) * 100.0, 4),
                "bertscore_f1": round(float(f1.mean().item()) * 100.0, 4),
            }
        )
    return metrics


def generation_value(args: argparse.Namespace, raw_cfg: Dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return raw_cfg.get("generation", {}).get(name, default)


def resolve_adapter_source(raw_cfg: Dict[str, Any], adapter: Optional[str]) -> Tuple[str, Optional[str], str]:
    local_candidates: List[Path] = []
    if adapter:
        local_candidates.append(Path(adapter))
    output_dir = Path(raw_cfg["project"]["output_dir"])
    local_candidates.extend([output_dir / "best_adapter", output_dir / "final_adapter"])
    for path in local_candidates:
        if path.exists():
            return str(path), None, str(path)

    hf_cfg = raw_cfg.get("huggingface", {})
    repo_id = os.environ.get("HF_REPO_ID") or hf_cfg.get("repo_id")
    if not repo_id:
        raise FileNotFoundError(
            f"No local adapter found in {[str(p) for p in local_candidates]} and HF_REPO_ID is not set."
        )
    path_in_repo = str(hf_cfg.get("path_in_repo", "checkpoints/t5gemma2_1b_1b_lora_wikilingua")).strip("/")
    subfolder = f"{path_in_repo}/best_adapter"
    return repo_id, subfolder, f"{repo_id}/{subfolder}"


def maybe_upload_eval_outputs(raw_cfg: Dict[str, Any], output_dir: Path) -> None:
    hf_cfg = raw_cfg.get("huggingface", {})
    if not bool(hf_cfg.get("enabled", False)) or not bool(hf_cfg.get("push_eval_outputs", True)):
        return
    repo_id = os.environ.get("HF_REPO_ID") or hf_cfg.get("repo_id")
    token = os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        print("Skipping eval output upload because HF_REPO_ID or HF_TOKEN is not set.", file=sys.stderr)
        return
    from huggingface_hub import HfApi

    repo_type = hf_cfg.get("repo_type", "model")
    base_path = str(hf_cfg.get("path_in_repo", "checkpoints/t5gemma2_1b_1b_lora_wikilingua")).strip("/")
    path_in_repo = f"{base_path}/eval_outputs/full_test"
    try:
        api = HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=bool(hf_cfg.get("private", False)),
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(output_dir),
            path_in_repo=path_in_repo,
            commit_message="Upload T5Gemma full-test evaluation outputs",
        )
        print(f"Uploaded eval outputs -> {repo_id}/{path_in_repo}")
    except Exception as exc:
        if bool(hf_cfg.get("fail_on_error", True)):
            raise
        print(f"HF eval upload failed: {exc}", file=sys.stderr)


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--output_dir", default="T5Gemma/eval_outputs/full_test")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--compute_bertscore", action="store_true")
    parser.add_argument("--bertscore_model_type", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    test_file = Path(args.test_file or raw_cfg["data"].get("test_file", "T5Gemma/data/processed/test.jsonl"))
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_info_path = output_dir / "eval_run_info.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = os.environ.get("HF_TOKEN")
    model_name = raw_cfg["model"]["model_name_or_path"]
    trust_remote_code = bool(raw_cfg["model"].get("trust_remote_code", True))
    dtype = torch_dtype_from_config(raw_cfg["model"].get("torch_dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    base_model.config.use_cache = bool(raw_cfg["model"].get("use_cache_for_eval", True))

    adapter_source, adapter_subfolder, adapter_label = resolve_adapter_source(raw_cfg, args.adapter)
    print(f"Loading LoRA adapter: {adapter_label}")
    if adapter_subfolder:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_source,
            subfolder=adapter_subfolder,
            token=token,
        )
    else:
        model = PeftModel.from_pretrained(base_model, adapter_source, token=token)
    model.to(device)
    model.eval()

    examples = load_jsonl(test_file, limit=args.limit)
    batch_size = int(args.batch_size or raw_cfg.get("generation", {}).get("eval_batch_size", 1))
    generation_settings = {
        "max_new_tokens": int(generation_value(args, raw_cfg, "max_new_tokens", 256)),
        "min_new_tokens": int(generation_value(args, raw_cfg, "min_new_tokens", 32)),
        "num_beams": int(generation_value(args, raw_cfg, "num_beams", 1)),
        "do_sample": bool(generation_value(args, raw_cfg, "do_sample", False)),
        "temperature": float(generation_value(args, raw_cfg, "temperature", 0.0)),
        "top_k": int(generation_value(args, raw_cfg, "top_k", 0)),
        "top_p": float(generation_value(args, raw_cfg, "top_p", 1.0)),
        "repetition_penalty": float(generation_value(args, raw_cfg, "repetition_penalty", 1.15)),
        "no_repeat_ngram_size": int(generation_value(args, raw_cfg, "no_repeat_ngram_size", 3)),
    }
    generate_kwargs = {
        "max_new_tokens": generation_settings["max_new_tokens"],
        "min_new_tokens": generation_settings["min_new_tokens"],
        "num_beams": generation_settings["num_beams"],
        "do_sample": generation_settings["do_sample"],
        "repetition_penalty": generation_settings["repetition_penalty"],
        "no_repeat_ngram_size": generation_settings["no_repeat_ngram_size"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_settings["do_sample"]:
        generate_kwargs.update(
            {
                "temperature": max(1e-5, generation_settings["temperature"]),
                "top_k": generation_settings["top_k"],
                "top_p": generation_settings["top_p"],
            }
        )

    source_prefix = raw_cfg.get("data", {}).get("source_prefix", "")
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    latencies: List[float] = []
    new_token_counts: List[float] = []
    total_new_tokens = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with predictions_path.open("w", encoding="utf-8") as out_f:
        for offset in tqdm(range(0, len(examples), batch_size), desc="Generating"):
            batch = examples[offset : offset + batch_size]
            batch_sources = [row["source"] for row in batch]
            batch_refs = [row["target"] for row in batch]
            enc = tokenizer(
                [source_prefix + source for source in batch_sources],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=int(raw_cfg["data"]["max_source_length"]),
            ).to(device)

            sync_device(device)
            generation_start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**enc, **generate_kwargs)
            sync_device(device)
            batch_elapsed = time.perf_counter() - generation_start

            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for row, source, reference, prediction, ids in zip(batch, batch_sources, batch_refs, decoded, output_ids):
                if tokenizer.pad_token_id is None:
                    new_tokens = int(ids.numel())
                else:
                    new_tokens = int((ids != tokenizer.pad_token_id).sum().item())
                total_new_tokens += new_tokens
                predictions.append(prediction)
                references.append(reference)
                sources.append(source)
                latencies.append(batch_elapsed / max(1, len(batch)))
                new_token_counts.append(float(new_tokens))

                out_f.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "source": source,
                            "reference": reference,
                            "prediction": prediction,
                            "generated_tokens": new_tokens,
                            "latency_seconds": batch_elapsed / max(1, len(batch)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_f.flush()

    elapsed = time.perf_counter() - start
    metrics = compute_metrics(
        predictions,
        references,
        sources=sources,
        compute_bertscore=args.compute_bertscore or bool(raw_cfg.get("evaluation", {}).get("compute_bertscore", False)),
        bertscore_model_type=args.bertscore_model_type
        or raw_cfg.get("evaluation", {}).get("bertscore_model_type", "xlm-roberta-large"),
    )
    decode_elapsed = max(1e-9, sum(latencies))
    metrics.update(
        {
            "elapsed_seconds": round(elapsed, 3),
            "examples_per_second": round(len(predictions) / max(1e-9, elapsed), 6),
            "generated_tokens_per_second": round(total_new_tokens / max(1e-9, elapsed), 6),
            "decode_mode": "autoregressive_generate",
            "decode_elapsed_seconds": round(decode_elapsed, 3),
            "decode_examples_per_second": round(len(predictions) / max(1e-9, decode_elapsed), 6),
            "decode_generated_tokens_per_second": round(total_new_tokens / max(1e-9, decode_elapsed), 6),
            "seconds_per_generated_token": round(decode_elapsed / max(1, total_new_tokens), 8),
            "latency_seconds_mean": round(safe_mean(latencies), 6),
            "latency_seconds_median": round(safe_median(latencies), 6),
            "latency_seconds_p95": round(percentile(latencies, 95), 6),
            "latency_seconds_min": round(min(latencies) if latencies else 0.0, 6),
            "latency_seconds_max": round(max(latencies) if latencies else 0.0, 6),
            "generated_tokens_total": int(total_new_tokens),
            "generated_tokens_mean": round(safe_mean(new_token_counts), 4),
            "decode_steps_total": float(total_new_tokens),
            "decode_steps_mean": round(safe_mean(new_token_counts), 6),
            "tokens_per_decode_step": 1.0,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0,
            "adapter": adapter_label,
            "base_model": model_name,
            "config": str(config_path),
            "test_file": str(test_file),
            "predictions_file": str(predictions_path),
            "generation": generation_settings,
            "source_prefix": source_prefix,
            "eval_batch_size": batch_size,
        }
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with run_info_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter": adapter_label,
                "base_model": model_name,
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "generation": generation_settings,
                "metrics_file": str(metrics_path),
                "predictions_file": str(predictions_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    maybe_upload_eval_outputs(raw_cfg, output_dir)


if __name__ == "__main__":
    main()
