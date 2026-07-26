"""Generate last.pt predictions and fast diagnostic ROUGE scores."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch

from .checkpoint import load_last_checkpoint
from .config import load_config
from .data import clean_text, dataset_fingerprint, decoder_seed_ids, encode_source, read_jsonl
from .generation import generate
from .metrics import rouge_scores
from .model import LLM2SeqV3
from .training import _device, _tokenizers


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
        "empty_prediction_rate": round(100 * sum(not value.strip() for value in predictions) / count, 4),
        "too_short_rate": round(100 * sum(value < 0.5 for value in ratios) / count, 4),
        "too_long_rate": round(100 * sum(value > 1.5 for value in ratios) / count, 4),
        "repeated_trigram_rate_mean": round(100 * sum(repeated(value) for value in predictions) / count, 4),
        "source_words_mean": round(sum(source_lengths) / count, 4),
        "unique_prediction_rate": round(100 * len(set(normalized)) / max(1, len(normalized)), 4),
        "dominant_prefix_5gram_rate": round(100 * max(prefixes.values(), default=0) / max(1, len(normalized)), 4),
    }


def _pad_source(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    data: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    encoded = []
    units = []
    sources = []
    references = []
    clean_metadata = bool(data.get("clean_wikihow_metadata", True))
    for row in rows:
        source = clean_text(row["source"], clean_metadata)
        reference = clean_text(row["target"], clean_metadata)
        ids, unit_ids, _ = encode_source(tokenizer, source, data)
        encoded.append(torch.tensor(ids, dtype=torch.long))
        units.append(torch.tensor(unit_ids, dtype=torch.long))
        sources.append(source)
        references.append(reference)
    length = max(value.numel() for value in encoded)
    input_ids = torch.full((len(rows), length), int(tokenizer.pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
    unit_tensor = torch.zeros((len(rows), length), dtype=torch.long)
    for index, (ids, unit_ids) in enumerate(zip(encoded, units)):
        input_ids[index, : ids.numel()] = ids
        attention_mask[index, : ids.numel()] = 1
        unit_tensor[index, : unit_ids.numel()] = unit_ids
    return (
        input_ids.to(device),
        attention_mask.to(device),
        unit_tensor.to(device),
        sources,
        references,
    )


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_samples: int = 0,
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.name != "last.pt":
        raise ValueError("LLM2Seq-v3 evaluates last.pt only")
    config = load_config(config_path)
    device = _device()
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = LLM2SeqV3(config)
    payload = load_last_checkpoint(model, checkpoint)
    model.to(device).eval()
    data = config["data"]
    configured_limit = int(config.get("limits", {}).get("max_test_examples", 0))
    effective_limit = int(max_samples) if int(max_samples) > 0 else configured_limit
    rows = read_jsonl(data["test_file"], max_examples=effective_limit)
    generation = config.get("generation", {})
    batch_size = int(generation.get("batch_size", 64))
    decoder_seed = decoder_seed_ids(decoder_tokenizer, data)
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    routing_sum = None
    routing_observations = 0
    bf16 = device.type == "cuda" and bool(config.get("training", {}).get("bf16", True))

    parameter_summary = model.parameter_summary()
    deployable_parameters = int(parameter_summary["total"] - parameter_summary["alignment_head"])
    training_parameters = int(parameter_summary["total"])
    checkpoint_parameter_count = int(payload.get("parameter_element_count", -1))
    checkpoint_parameters_match_model = checkpoint_parameter_count == training_parameters
    parameter_target = config.get("benchmark", {}).get("parameters", {})
    target_declared_parameters = int(parameter_target.get("target_declared_parameters", 0))
    current_test_fingerprint = dataset_fingerprint(data["test_file"], effective_limit)
    checkpoint_test_fingerprint = payload.get("data_manifest", {}).get("test", {})
    checkpoint_test_matches_current = int(checkpoint_test_fingerprint.get("num_examples", -1)) == int(
        current_test_fingerprint["num_examples"]
    ) and str(checkpoint_test_fingerprint.get("sha256", "")) == str(current_test_fingerprint["sha256"])

    def autocast():
        if bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        input_ids, attention_mask, unit_ids, batch_sources, batch_references = _pad_source(
            batch_rows,
            encoder_tokenizer,
            data,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        with autocast():
            generated = generate(
                model,
                input_ids,
                attention_mask,
                decoder_seed,
                unit_ids=unit_ids,
                max_new_tokens=int(generation.get("max_new_tokens", 256)),
                min_new_tokens=int(generation.get("min_new_tokens", 16)),
                eos_token_id=decoder_tokenizer.eos_token_id,
                pad_token_id=decoder_tokenizer.pad_token_id,
                repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
                no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
            )
        batch_routing, batch_observations = model.decoder.generation_routing_statistics()
        if batch_observations > 0:
            weighted_routing = batch_routing.detach().float().cpu() * batch_observations
            routing_sum = weighted_routing if routing_sum is None else routing_sum + weighted_routing
            routing_observations += batch_observations
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        decoded = decoder_tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(value.strip() for value in decoded)
        references.extend(batch_references)
        sources.extend(batch_sources)
        latencies.extend([elapsed / len(batch_rows)] * len(batch_rows))
        for row, prediction, reference in zip(batch_rows, decoded, batch_references):
            records.append(
                {
                    "id": row.get("id"),
                    "prediction": prediction.strip(),
                    "reference": reference,
                }
            )
    metrics: Dict[str, Any] = {
        **rouge_scores(predictions, references),
        **_diagnostics(predictions, references, sources),
        "checkpoint": str(checkpoint),
        "checkpoint_role": payload.get("checkpoint_role"),
        "rouge_backend": "rouge==1.0.0 (diagnostic; paper score is Perl ROUGE-1.5.5)",
        "rouge_preprocessing": "NFC + lowercase + stored whitespace tokenization",
        "latency_seconds_mean": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "decode_elapsed_seconds": round(sum(latencies), 3),
        "decode_examples_per_second": round(len(rows) / max(1e-9, sum(latencies)), 6),
        "encoder_name": config["model"]["encoder_name"],
        "encoder_hidden_size": int(model.encoder.hidden_size),
        "encoder_hidden_layers": int(model.encoder.num_hidden_layers),
        "encoder_bidirectional": bool(model.encoder.is_bidirectional),
        "encoder_attention_mode_declared": str(config["model"].get("encoder_attention_mode", "auto")),
        "encoder_revision": config["model"].get("encoder_revision"),
        "encoder_trust_remote_code": bool(config["model"].get("encoder_trust_remote_code", True)),
        "decoder_name": config["model"]["decoder_name"],
        "cross_attention_every": int(config.get("decoder", {}).get("cross_attention_every", 1)),
        "memory_bank_count": int(config.get("decoder", {}).get("memory_bank_count", 1)),
        "adapter_bidirectional_layers": int(config.get("adapter", {}).get("num_bidirectional_layers", 0)),
        "final_graph": "one_encoder_one_adapter_one_decoder",
        "version": "llm2seq_v3",
        "contrastive_learning": bool(config.get("objectives", {}).get("use_contrastive", True)),
        "prompt_alignment": bool(config.get("objectives", {}).get("use_prompt_alignment", True)),
        "source_swap": bool(config.get("objectives", {}).get("use_source_swap", True)),
        "label_smoothing": float(config.get("objectives", {}).get("label_smoothing", 0.1)),
        "memory_routing_mode": str(config.get("decoder", {}).get("memory_routing_mode", "attention_output")),
        "query_adaptive_routing": bool(config.get("decoder", {}).get("query_adaptive_routing", False)),
        "parameter_summary": parameter_summary,
        "deployable_parameters": deployable_parameters,
        "training_parameters": training_parameters,
        "checkpoint_parameter_count": checkpoint_parameter_count,
        "checkpoint_parameters_match_model": checkpoint_parameters_match_model,
        "parameter_target_name": str(parameter_target.get("target_name", "")),
        "parameter_target_declared": target_declared_parameters,
        "parameter_target_is_rounded": bool(parameter_target.get("target_is_rounded", False)),
        "parameter_budget_reached": (
            deployable_parameters < target_declared_parameters if target_declared_parameters > 0 else None
        ),
        "test_data_fingerprint": current_test_fingerprint,
        "checkpoint_test_data_fingerprint": checkpoint_test_fingerprint,
        "checkpoint_test_matches_current": checkpoint_test_matches_current,
    }
    if routing_sum is not None and routing_observations > 0:
        per_layer_routing = routing_sum / routing_observations
    else:
        per_layer_routing = model.decoder.memory_routing_per_layer().detach().float().cpu()
    routing = per_layer_routing.mean(dim=0)
    if routing.numel() == 1:
        metrics["memory_route_mean"] = {"summary": round(float(routing[0]), 6)}
    else:
        bank_names = ("lexical", "semantic", "summary")
        metrics["memory_route_mean"] = {name: round(float(routing[index]), 6) for index, name in enumerate(bank_names)}
        entropy = -(routing * routing.clamp_min(1e-8).log()).sum() / math.log(routing.numel())
        metrics["memory_routing_entropy"] = round(float(entropy), 6)
        metrics["memory_routing_observations"] = routing_observations
        metrics["memory_route_per_cross_layer"] = [
            {name: round(float(row[index]), 6) for index, name in enumerate(bank_names)} for row in per_layer_routing
        ]
    benchmark = config.get("benchmark", {})
    diagnostic_benchmark = benchmark.get("diagnostic", {})
    paper_benchmark = benchmark.get("paper", {})
    locked_test = benchmark.get("data", {}).get("test", {})
    locked_test_matches = int(locked_test.get("num_examples", -1)) == int(
        current_test_fingerprint["num_examples"]
    ) and str(locked_test.get("sha256", "")) == str(current_test_fingerprint["sha256"])
    metrics["locked_test_data_fingerprint"] = locked_test
    metrics["locked_test_matches"] = locked_test_matches
    if all(name in diagnostic_benchmark for name in ("rouge1", "rouge2", "rougeL")):
        expected_examples = int(diagnostic_benchmark.get("num_examples", len(rows)))
        comparable = len(rows) == expected_examples
        metrics["diagnostic_benchmark_name"] = str(diagnostic_benchmark.get("name", "T5Gemma"))
        metrics["diagnostic_benchmark_backend"] = str(diagnostic_benchmark.get("backend", "rouge==1.0.0"))
        metrics["diagnostic_benchmark_comparable"] = comparable
        if comparable:
            metrics["gap_to_t5gemma_diagnostic"] = {
                name: round(float(metrics[name]) - float(diagnostic_benchmark[name]), 4)
                for name in ("rouge1", "rouge2", "rougeL")
            }
            metrics["rouge2_diagnostic_target_reached"] = float(metrics["rouge2"]) >= float(
                diagnostic_benchmark["rouge2"]
            )
    if all(name in paper_benchmark for name in ("rouge1", "rouge2", "rougeL")):
        metrics["paper_benchmark"] = {
            "name": str(paper_benchmark.get("name", "T5Gemma")),
            "backend": str(paper_benchmark.get("backend", "Perl ROUGE-1.5.5")),
            "rouge1": float(paper_benchmark["rouge1"]),
            "rouge2": float(paper_benchmark["rouge2"]),
            "rougeL": float(paper_benchmark["rougeL"]),
            "num_examples": int(paper_benchmark.get("num_examples", len(rows))),
        }
        metrics["paper_comparison_status"] = "pending_perl_rouge155"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if "gap_to_t5gemma_diagnostic" in metrics:
        gap_path = output.parent / "t5gemma_diagnostic_gap_report.json"
        gap_path.write_text(
            json.dumps(
                {
                    "candidate_checkpoint": str(checkpoint),
                    "candidate": {name: metrics[name] for name in ("rouge1", "rouge2", "rougeL")},
                    "candidate_backend": metrics["rouge_backend"],
                    "target_name": metrics["diagnostic_benchmark_name"],
                    "target_backend": metrics["diagnostic_benchmark_backend"],
                    "target": {name: float(diagnostic_benchmark[name]) for name in ("rouge1", "rouge2", "rougeL")},
                    "candidate_minus_target": metrics["gap_to_t5gemma_diagnostic"],
                    "rouge2_target_reached": metrics["rouge2_diagnostic_target_reached"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.max_samples)


if __name__ == "__main__":
    main()
