"""Last-checkpoint evaluation for the controlled direct-Qwen baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from llm2seq_v2.checkpoint import load_last_checkpoint
from llm2seq_v2.metrics import rouge_scores

from llm2seq_v2.data import clean_text, read_jsonl

from .config import load_config
from .data import build_prompt_ids, left_pad_prompts
from .provenance import (
    parameter_manifest,
    resolve_from_src,
    tokenizer_manifest,
)
from .training import device, load_tokenizer_and_model, model_context_length


def generation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    generation = config["generation"]
    return {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "min_new_tokens": int(generation["min_new_tokens"]),
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": float(generation["repetition_penalty"]),
        "no_repeat_ngram_size": int(generation["no_repeat_ngram_size"]),
    }


def generate_prompt_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    generation: Dict[str, Any],
    target: torch.device,
) -> torch.Tensor:
    """Generate greedily and return only tokens produced after the padded prompt."""

    input_ids, attention_mask = left_pad_prompts(prompts, tokenizer.pad_token_id)
    input_ids = input_ids.to(target)
    attention_mask = attention_mask.to(target)
    prompt_width = int(input_ids.shape[1])
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation["max_new_tokens"]),
        min_new_tokens=int(generation["min_new_tokens"]),
        num_beams=1,
        do_sample=False,
        repetition_penalty=float(generation["repetition_penalty"]),
        no_repeat_ngram_size=int(generation["no_repeat_ngram_size"]),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if outputs.ndim != 2 or outputs.shape[0] != len(prompts) or outputs.shape[1] < prompt_width:
        raise RuntimeError("Causal generation returned an invalid token tensor")
    return outputs[:, prompt_width:]


def _diagnostics(predictions: List[str], references: List[str], sources: List[str]) -> Dict[str, float]:
    prediction_lengths = [len(value.split()) for value in predictions]
    reference_lengths = [len(value.split()) for value in references]
    source_lengths = [len(value.split()) for value in sources]
    ratios = [prediction / max(1, reference) for prediction, reference in zip(prediction_lengths, reference_lengths)]
    normalized = [" ".join(value.lower().split()) for value in predictions if value.strip()]
    prefixes = Counter(" ".join(value.split()[:5]) for value in normalized)

    def repeated(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 3:
            return 0.0
        ngrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
        return 1.0 - len(set(ngrams)) / len(ngrams)

    count = max(1, len(predictions))
    return {
        "num_examples": len(predictions),
        "prediction_words_mean": round(sum(prediction_lengths) / count, 4),
        "reference_words_mean": round(sum(reference_lengths) / count, 4),
        "length_ratio_mean": round(sum(ratios) / count, 6),
        "empty_prediction_rate": round(100.0 * sum(not value.strip() for value in predictions) / count, 4),
        "too_short_rate": round(100.0 * sum(value < 0.5 for value in ratios) / count, 4),
        "too_long_rate": round(100.0 * sum(value > 1.5 for value in ratios) / count, 4),
        "repeated_trigram_rate_mean": round(100.0 * sum(repeated(value) for value in predictions) / count, 4),
        "source_words_mean": round(sum(source_lengths) / count, 4),
        "unique_prediction_rate": round(100.0 * len(set(normalized)) / max(1, len(normalized)), 4),
        "dominant_prefix_5gram_rate": round(100.0 * max(prefixes.values(), default=0) / max(1, len(normalized)), 4),
    }


def _autocast(target: torch.device, config: Dict[str, Any]):
    enabled = target.type == "cuda" and str(config["model"].get("eval_torch_dtype", "bfloat16")) in {
        "bfloat16",
        "bf16",
    }
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _checkpoint_contract(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint: Path,
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if checkpoint.name != "last.pt":
        raise ValueError("Direct Qwen evaluates the canonical last.pt only")
    if (checkpoint.parent / "RUNNING").exists():
        raise RuntimeError("Refusing to evaluate an incomplete direct-Qwen run")
    payload = load_last_checkpoint(model, checkpoint)
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("Checkpoint has no resolved config")
    if int(payload.get("epoch", -1)) != int(config["training"]["num_train_epochs"]):
        raise RuntimeError("Checkpoint is not the completed fourteen-epoch control")
    runtime_parameters = parameter_manifest(model, config)
    parameter_path = checkpoint.parent / "parameter_manifest.json"
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Missing parameter provenance: {parameter_path}")
    saved_parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    if runtime_parameters["unique_parameter_elements"] != saved_parameters.get("unique_parameter_elements"):
        raise RuntimeError("Instantiated direct model does not match the training parameter count")
    if saved_parameters.get("full_finetune") is not True:
        raise RuntimeError("Checkpoint manifest does not prove full fine-tuning")
    tokenizer_path = checkpoint.parent / "tokenizer_manifest.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer provenance: {tokenizer_path}")
    saved_tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if tokenizer_manifest(tokenizer, config) != saved_tokenizer:
        raise RuntimeError("Evaluation tokenizer does not match the training tokenizer")
    return payload, saved_parameters


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    *,
    split: str = "validation",
    max_samples: int = 0,
    paper_test: bool = False,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split == "test" and not paper_test:
        raise ValueError("Full test is locked; pass --paper-test after freezing the direct baseline")
    if paper_test and (split != "test" or int(max_samples) != 0):
        raise ValueError("Paper test must evaluate the complete test split")

    config = load_config(config_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    tokenizer, model = load_tokenizer_and_model(config, evaluation=True)
    if model_context_length(model) < int(config["model"]["minimum_context_length"]):
        raise RuntimeError("Evaluation model context is shorter than the frozen prompt contract")
    payload, saved_parameters = _checkpoint_contract(model, tokenizer, checkpoint, config)
    target = device()
    model.to(target).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data = config["data"]
    configured_limit = int(config.get("limits", {}).get(f"max_{split}_examples", 0))
    effective_limit = int(max_samples) if int(max_samples) > 0 else configured_limit
    split_path = resolve_from_src(data[f"{split}_file"])
    rows = read_jsonl(split_path, max_examples=effective_limit)
    generation = generation_contract(config)
    batch_size = int(config["generation"]["batch_size"])
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    clean_metadata = bool(data.get("clean_wikihow_metadata", True))
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_sources = [clean_text(row["source"], clean_metadata) for row in batch_rows]
            batch_references = [clean_text(row["target"], clean_metadata) for row in batch_rows]
            prompts = [build_prompt_ids(tokenizer, source, data) for source in batch_sources]
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            began = time.perf_counter()
            with _autocast(target, config):
                generated = generate_prompt_batch(model, tokenizer, prompts, generation, target)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            elapsed = time.perf_counter() - began
            decoded = [value.strip() for value in tokenizer.batch_decode(generated, skip_special_tokens=True)]
            predictions.extend(decoded)
            references.extend(batch_references)
            sources.extend(batch_sources)
            latencies.extend([elapsed / max(1, len(batch_rows))] * len(batch_rows))
            for row, source, reference, prediction in zip(batch_rows, batch_sources, batch_references, decoded):
                records.append(
                    {
                        "id": row.get("id"),
                        "source": source,
                        "reference": reference,
                        "prediction": prediction,
                    }
                )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics: Dict[str, Any] = {
        **rouge_scores(predictions, references),
        **_diagnostics(predictions, references, sources),
        "rouge_backend": "rouge==1.0.0 (HeterSumGraph diagnostic)",
        "rouge_preprocessing": "NFC + lowercase + stored whitespace tokenization",
        "evaluation_split": split,
        "paper_test": bool(paper_test),
        "checkpoint": str(checkpoint),
        "checkpoint_role": payload.get("checkpoint_role"),
        "checkpoint_parameters_match_model": True,
        "base_model": config["model"]["base_model_id"],
        "checkpoint_base_model": payload.get("config", {}).get("model", {}).get("base_model_id"),
        "total_parameters": int(saved_parameters["unique_parameter_elements"]),
        "training_parameters": int(saved_parameters["trainable_parameter_elements"]),
        "full_finetune": saved_parameters["full_finetune"],
        "predictions_file": str(output),
        "generation": generation,
        "source_prefix": str(data["source_prefix"]),
        "decoder_instruction": str(data["decoder_instruction"]),
        "decoder_prefix": str(data["decoder_prefix"]),
        "prompt_layout": config["contract"]["prompt_layout"],
        "max_source_length": int(data["max_source_length"]),
        "max_target_length": int(data["max_target_length"]),
        "latency_seconds_mean": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "decode_elapsed_seconds": round(sum(latencies), 3),
        "decode_examples_per_second": round(len(rows) / max(1e-9, sum(latencies)), 6),
    }
    diagnostic = config.get("benchmark", {}).get("diagnostic", {})
    if all(name in diagnostic for name in ("rouge1", "rouge2", "rougeL")):
        metrics["diagnostic_gap_to_t5gemma"] = {
            name: round(float(metrics[name]) - float(diagnostic[name]), 4) for name in ("rouge1", "rouge2", "rougeL")
        }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--paper-test", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
        paper_test=args.paper_test,
    )


if __name__ == "__main__":
    main()
