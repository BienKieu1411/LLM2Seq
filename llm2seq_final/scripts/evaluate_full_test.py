#!/usr/bin/env python3
"""Generate predictions on the full test set and compute summarization metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

H200_ROOT = Path(__file__).resolve().parents[1]
if str(H200_ROOT) not in sys.path:
    sys.path.insert(0, str(H200_ROOT))

import torch
import yaml
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from llm2seq.src.inference.generate import autoregressive_generate
from llm2seq.src.inference.generate_mtp import mtp_generate
from llm2seq.src.models.llm2seq_model import LLM2Seq, LLM2SeqConfig


PHASE_REMOTE_DIRS = {
    "h200_phase1_warmup": "checkpoints/h200_phase1_warmup",
    "h200_phase2_lora_encoder": "checkpoints/h200_phase2_lora_encoder",
    "h200_phase3_mtp_self_distill": "checkpoints/h200_phase3_mtp_self_distill",
}


def load_env_file() -> None:
    env_file = Path(os.environ.get("ENV_FILE", H200_ROOT / "env.txt"))
    if not env_file.is_absolute():
        env_file = H200_ROOT.parent / env_file
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


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


def infer_stage_from_path(path: str) -> Optional[str]:
    lowered = path.lower()
    if "phase1" in lowered or "warmup" in lowered:
        return "h200_phase1_warmup"
    if "phase2" in lowered or "lora_encoder" in lowered:
        return "h200_phase2_lora_encoder"
    if "phase3" in lowered or "mtp_self_distill" in lowered:
        return "h200_phase3_mtp_self_distill"
    return None


def hf_download_checkpoint(
    raw_cfg: Dict[str, Any],
    local_hint: str,
    stage_override: Optional[str] = None,
) -> Path:
    if os.environ.get("HF_AUTO_DOWNLOAD_CHECKPOINTS", "true").lower() not in {"1", "true", "yes", "on"}:
        raise FileNotFoundError(local_hint)

    hf_cfg = raw_cfg.get("huggingface", {})
    repo_id = hf_cfg.get("repo_id") or os.environ.get("HF_REPO_ID")
    repo_type = hf_cfg.get("repo_type", "model")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise FileNotFoundError(f"{local_hint}; HF_REPO_ID is not set for fallback download.")

    stage = stage_override or str(raw_cfg.get("training", {}).get("stage", "")) or infer_stage_from_path(local_hint)
    remote_dir = PHASE_REMOTE_DIRS.get(stage or "")
    if stage_override is None:
        remote_dir = str(hf_cfg.get("path_in_repo") or remote_dir or "").strip("/")
    if not remote_dir:
        raise FileNotFoundError(f"{local_hint}; cannot infer HF checkpoint path.")

    from huggingface_hub import hf_hub_download

    cache_dir = Path(os.environ.get("HF_CHECKPOINT_CACHE", "runs/hf_checkpoints"))
    if not cache_dir.is_absolute():
        cache_dir = H200_ROOT.parent / cache_dir
    names = [Path(local_hint).name]
    for fallback_name in ("best.pt", "final.pt"):
        if fallback_name not in names:
            names.append(fallback_name)

    last_error: Optional[BaseException] = None
    for name in names:
        remote_file = f"{remote_dir}/{name}"
        try:
            print(f"Downloading HF checkpoint: {repo_id}/{remote_file}", file=sys.stderr)
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    filename=remote_file,
                    local_dir=str(cache_dir),
                )
            )
        except Exception as exc:
            last_error = exc
            print(f"HF checkpoint not available: {remote_file} ({exc})", file=sys.stderr)
    raise FileNotFoundError(f"{local_hint}; could not download HF checkpoint.") from last_error


def find_checkpoint(output_dir: Path, checkpoint: str | None, raw_cfg: Dict[str, Any]) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            return hf_download_checkpoint(raw_cfg, checkpoint)
        return path
    for name in ("best.pt", "final.pt"):
        path = output_dir / name
        if path.exists():
            return path
    return hf_download_checkpoint(raw_cfg, str(output_dir / "best.pt"))


def is_allowed_missing_key(key: str, allowed_prefixes: tuple[str, ...], stage: str, context: str) -> bool:
    if not any(key.startswith(prefix) for prefix in allowed_prefixes):
        return False
    if context == "delta" and stage == "h200_phase3_mtp_self_distill" and key.startswith("encoder.") and "lora_" in key:
        pass
    elif stage != "h200_phase1_warmup" and key.startswith("encoder.") and "lora_" in key:
        return False
    if context == "delta" and key.startswith("mtp_module.blocks."):
        return False
    if context == "delta" and key.startswith("mtp_module.heads."):
        return False
    return True


def load_model_state_checked(
    model: LLM2Seq,
    state_dict: Dict[str, torch.Tensor],
    stage: str,
    context: str = "checkpoint",
) -> None:
    """Fail fast if an evaluation checkpoint is missing required weights."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    allowed_prefixes: tuple[str, ...] = ()
    allowed_exact: set[str] = set()
    if stage in {"h200_phase1_warmup", "h200_phase2_lora_encoder"}:
        # Phase 1 checkpoints omit the frozen base encoder by design.
        allowed_prefixes = ("encoder.",)
    if stage == "h200_phase3_mtp_self_distill":
        # These are aliases of decoder.embed_tokens/lm_head when sharing weights.
        allowed_exact = {"mtp_module.embed_tokens.weight", "mtp_module.lm_head.weight"}
        if context == "base":
            allowed_prefixes = ("encoder.", "mtp_module.")
        elif context == "delta":
            allowed_prefixes = ("encoder.", "adaptor.", "decoder.", "lm_head.", "mtp_module.embed_tokens.", "mtp_module.lm_head.")

    bad_missing = [
        key for key in missing
        if key not in allowed_exact
        and not is_allowed_missing_key(key, allowed_prefixes, stage, context)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match config stage={stage}. "
            f"Bad missing keys ({len(bad_missing)}): {', '.join(bad_missing[:20])}. "
            f"Unexpected keys ({len(unexpected)}): {', '.join(unexpected[:20])}."
        )


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


def aggregate_mtp_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {}

    numeric_keys = [
        "acceptance_rate",
        "average_accepted_length",
        "average_emitted_length",
        "emitted_tokens",
        "speedup_vs_autoregressive",
        "num_steps",
    ]
    result: Dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(metrics[key]) for metrics in metrics_list if key in metrics]
        if values:
            result[f"mtp_{key}_mean"] = round(safe_mean(values), 6)
            result[f"mtp_{key}_median"] = round(safe_median(values), 6)

    cumulative = [
        metrics.get("cumulative_acceptance_rates", [])
        for metrics in metrics_list
        if metrics.get("cumulative_acceptance_rates")
    ]
    if cumulative:
        max_len = max(len(values) for values in cumulative)
        result["mtp_cumulative_acceptance_rates_mean"] = [
            round(
                safe_mean([float(values[i]) for values in cumulative if i < len(values)]),
                6,
            )
            for i in range(max_len)
        ]

    result["mtp_verified_with_main"] = all(bool(metrics.get("verified_with_main", False)) for metrics in metrics_list)
    return result


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
    metrics: Dict[str, Any] = {
        name: round((value / n) * 100.0, 4)
        for name, value in rouge_totals.items()
    }
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
    length_ratios = [
        pred_len / max(1, ref_len)
        for pred_len, ref_len in zip(pred_words, ref_words)
    ]
    repeat_rates = [repeated_ngram_rate(pred, n=3) for pred in predictions]
    metrics.update(
        {
            "prediction_words_mean": round(safe_mean(pred_words), 4),
            "reference_words_mean": round(safe_mean(ref_words), 4),
            "length_ratio_mean": round(safe_mean(length_ratios), 6),
            "empty_prediction_rate": round(100.0 * safe_mean([1.0 if not pred.strip() else 0.0 for pred in predictions]), 4),
            "too_short_rate": round(100.0 * safe_mean([1.0 if ratio < 0.5 else 0.0 for ratio in length_ratios]), 4),
            "too_long_rate": round(100.0 * safe_mean([1.0 if ratio > 1.5 else 0.0 for ratio in length_ratios]), 4),
            "repeated_trigram_rate_mean": round(100.0 * safe_mean(repeat_rates), 4),
        }
    )

    if sources is not None:
        source_words = [word_count(src) for src in sources]
        compression_ratios = [
            pred_len / max(1, src_len)
            for pred_len, src_len in zip(pred_words, source_words)
        ]
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


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        help="Optional earlier-phase checkpoint to load before the current compact delta checkpoint.",
    )
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--output_dir", default="llm2seq_h200/eval_outputs")
    parser.add_argument(
        "--decode_mode",
        choices=["autoregressive", "mtp_verified"],
        default="autoregressive",
        help="autoregressive uses the main head; mtp_verified uses main-head-constrained MTP drafts.",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--compute_bertscore", action="store_true")
    parser.add_argument("--bertscore_model_type", default="xlm-roberta-large")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = LLM2SeqConfig(raw_cfg)

    project_output_dir = Path(raw_cfg["project"]["output_dir"])
    checkpoint_path = find_checkpoint(project_output_dir, args.checkpoint, raw_cfg)
    test_file = Path(args.test_file or raw_cfg["data"].get("test_file", "llm2seq_h200/data/processed/test.jsonl"))
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_info_path = output_dir / "eval_run_info.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = LLM2Seq(cfg, vocab_size=len(tokenizer))
    stage = str(raw_cfg.get("training", {}).get("stage", ""))
    base_checkpoint_path: Optional[Path] = None
    if args.base_checkpoint:
        base_checkpoint_path = Path(args.base_checkpoint)
        if not base_checkpoint_path.exists():
            base_stage = infer_stage_from_path(str(base_checkpoint_path)) or "h200_phase2_lora_encoder"
            base_checkpoint_path = hf_download_checkpoint(raw_cfg, str(base_checkpoint_path), stage_override=base_stage)
        base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
        load_model_state_checked(
            model,
            base_checkpoint["model_state_dict"],
            stage=stage,
            context="base",
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if (
        stage == "h200_phase3_mtp_self_distill"
        and base_checkpoint_path is None
        and not any(key.startswith("adaptor.") or key.startswith("decoder.") for key in state_dict)
    ):
        raise ValueError(
            "Phase 3 checkpoint is trainable-only and needs --base_checkpoint "
            "pointing to the Phase 2 LoRA best.pt."
        )
    load_model_state_checked(
        model,
        state_dict,
        stage=stage,
        context="delta" if base_checkpoint_path is not None else "checkpoint",
    )
    model.to(device)
    model.eval()

    examples = load_jsonl(test_file, limit=args.limit)
    generation_settings = {
        "max_new_tokens": int(generation_value(args, raw_cfg, "max_new_tokens", 256)),
        "min_new_tokens": int(generation_value(args, raw_cfg, "min_new_tokens", 32)),
        "do_sample": bool(generation_value(args, raw_cfg, "do_sample", False)),
        "temperature": float(generation_value(args, raw_cfg, "temperature", 0.0)),
        "top_k": int(generation_value(args, raw_cfg, "top_k", 0)),
        "top_p": float(generation_value(args, raw_cfg, "top_p", 1.0)),
        "repetition_penalty": float(generation_value(args, raw_cfg, "repetition_penalty", 1.15)),
        "no_repeat_ngram_size": int(generation_value(args, raw_cfg, "no_repeat_ngram_size", 3)),
    }
    if args.decode_mode == "mtp_verified":
        if model.mtp_module is None:
            raise ValueError("decode_mode=mtp_verified requires a checkpoint/config with model.use_mtp=true.")
        if generation_settings["do_sample"]:
            raise ValueError("decode_mode=mtp_verified supports deterministic greedy generation only; set do_sample=false.")

    source_prefix = raw_cfg.get("data", {}).get("source_prefix", "")

    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    latencies: List[float] = []
    new_token_counts: List[float] = []
    decode_steps: List[float] = []
    mtp_metrics_list: List[Dict[str, Any]] = []
    total_new_tokens = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with predictions_path.open("w", encoding="utf-8") as out_f:
        for example in tqdm(examples, desc="Generating"):
            source = example["source"]
            reference = example["target"]
            enc = tokenizer(
                source_prefix + source,
                return_tensors="pt",
                truncation=True,
                max_length=raw_cfg["data"]["max_source_length"],
            ).to(device)

            sync_device(device)
            generation_start = time.perf_counter()
            sample_mtp_metrics: Dict[str, Any] = {}
            if args.decode_mode == "autoregressive":
                out_ids = autoregressive_generate(
                    model,
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=generation_settings["max_new_tokens"],
                    min_new_tokens=generation_settings["min_new_tokens"],
                    do_sample=generation_settings["do_sample"],
                    temperature=generation_settings["temperature"],
                    top_k=generation_settings["top_k"],
                    top_p=generation_settings["top_p"],
                    repetition_penalty=generation_settings["repetition_penalty"],
                    no_repeat_ngram_size=generation_settings["no_repeat_ngram_size"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
            else:
                mtp_result = mtp_generate(
                    model,
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=generation_settings["max_new_tokens"],
                    min_new_tokens=generation_settings["min_new_tokens"],
                    verify_with_main=True,
                    repetition_penalty=generation_settings["repetition_penalty"],
                    no_repeat_ngram_size=generation_settings["no_repeat_ngram_size"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
                out_ids = mtp_result["generated_ids"]
                sample_mtp_metrics = mtp_result.get("metrics", {})
                mtp_metrics_list.append(sample_mtp_metrics)
            sync_device(device)
            latency = time.perf_counter() - generation_start

            prediction = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            new_tokens = int(out_ids.ne(tokenizer.pad_token_id).sum().item())
            total_new_tokens += new_tokens
            latencies.append(latency)
            new_token_counts.append(float(new_tokens))
            if sample_mtp_metrics:
                decode_steps.append(float(sample_mtp_metrics.get("num_steps", 0.0)))
            else:
                decode_steps.append(float(new_tokens))
            predictions.append(prediction)
            references.append(reference)
            sources.append(source)
            source_words = word_count(source)
            reference_words = word_count(reference)
            prediction_words = word_count(prediction)
            row = {
                "id": example.get("id"),
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "source_chars": len(source),
                "reference_chars": len(reference),
                "prediction_chars": len(prediction),
                "source_words": source_words,
                "reference_words": reference_words,
                "prediction_words": prediction_words,
                "length_ratio": round(prediction_words / max(1, reference_words), 6),
                "compression_ratio": round(prediction_words / max(1, source_words), 6),
                "repeated_trigram_rate": round(repeated_ngram_rate(prediction, n=3), 6),
                "decode_mode": args.decode_mode,
                "latency_seconds": round(latency, 6),
                "generated_tokens": new_tokens,
                "decode_steps": round(decode_steps[-1], 6),
            }
            if sample_mtp_metrics:
                row["mtp_metrics"] = sample_mtp_metrics
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    sync_device(device)
    elapsed = time.perf_counter() - start
    peak_gpu_memory_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3)
        if device.type == "cuda"
        else 0.0
    )
    metrics = compute_metrics(
        predictions,
        references,
        sources=sources,
        compute_bertscore=args.compute_bertscore,
        bertscore_model_type=args.bertscore_model_type,
    )
    metrics.update(
        {
            "elapsed_seconds": round(elapsed, 3),
            "examples_per_second": round(len(examples) / max(elapsed, 1e-9), 6),
            "generated_tokens_per_second": round(total_new_tokens / max(elapsed, 1e-9), 3),
            "decode_mode": args.decode_mode,
            "decode_elapsed_seconds": round(sum(latencies), 3),
            "decode_examples_per_second": round(len(examples) / max(sum(latencies), 1e-9), 6),
            "decode_generated_tokens_per_second": round(total_new_tokens / max(sum(latencies), 1e-9), 3),
            "seconds_per_generated_token": round(sum(latencies) / max(total_new_tokens, 1), 8),
            "latency_seconds_mean": round(safe_mean(latencies), 6),
            "latency_seconds_median": round(safe_median(latencies), 6),
            "latency_seconds_p95": round(percentile(latencies, 95), 6),
            "latency_seconds_min": round(min(latencies), 6) if latencies else 0.0,
            "latency_seconds_max": round(max(latencies), 6) if latencies else 0.0,
            "generated_tokens_total": total_new_tokens,
            "generated_tokens_mean": round(safe_mean(new_token_counts), 4),
            "decode_steps_total": round(sum(decode_steps), 6),
            "decode_steps_mean": round(safe_mean(decode_steps), 6),
            "tokens_per_decode_step": round(total_new_tokens / max(sum(decode_steps), 1e-9), 6),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "test_file": str(test_file),
            "predictions_file": str(predictions_path),
            "generation": generation_settings,
            "source_prefix": source_prefix,
        }
    )
    metrics.update(aggregate_mtp_metrics(mtp_metrics_list))

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with run_info_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "base_checkpoint": str(base_checkpoint_path) if base_checkpoint_path else None,
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "test_file": str(test_file),
                "output_dir": str(output_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
