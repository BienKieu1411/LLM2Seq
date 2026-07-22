"""Generate summaries once after training and compute standard ROUGE."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from .backbone import torch_dtype
from .checkpoint import load_checkpoint
from .config import MODEL_PROFILES, apply_model_size, load_config
from .data import (
    clean_wikihow_metadata,
    decoder_prompt_ids,
    greedy_evidence_labels,
    prompt_token_ids,
    prompted_source_features,
    read_jsonl,
)
from .metrics import heter_sum_graph_rouge
from .training import build_experiment


def _rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    return heter_sum_graph_rouge(
        predictions,
        references,
        include_rouge_lsum=True,
        digits=4,
    )


def _generation_diagnostics(
    predictions: List[str],
    references: List[str],
    sources: List[str],
) -> Dict[str, float]:
    def word_count(text: str) -> int:
        return len(str(text).split())

    def repeated_ngram_rate(text: str, order: int = 3) -> float:
        tokens = str(text).split()
        if len(tokens) < order:
            return 0.0
        ngrams = [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]
        return 1.0 - len(set(ngrams)) / len(ngrams)

    count = max(1, len(predictions))
    prediction_lengths = [word_count(text) for text in predictions]
    reference_lengths = [word_count(text) for text in references]
    source_lengths = [word_count(text) for text in sources]
    length_ratios = [
        prediction / max(1, reference) for prediction, reference in zip(prediction_lengths, reference_lengths)
    ]
    compression_ratios = [prediction / max(1, source) for prediction, source in zip(prediction_lengths, source_lengths)]
    normalized_predictions = [" ".join(text.lower().split()) for text in predictions]
    nonempty = [text for text in normalized_predictions if text]
    prefixes = Counter(" ".join(text.split()[:5]) for text in nonempty)
    dominant_prefix_rate = max(prefixes.values(), default=0) / max(1, len(nonempty))
    return {
        "num_examples": len(predictions),
        "prediction_words_mean": round(sum(prediction_lengths) / count, 4),
        "reference_words_mean": round(sum(reference_lengths) / count, 4),
        "length_ratio_mean": round(sum(length_ratios) / count, 6),
        "empty_prediction_rate": round(
            100.0 * sum(not text.strip() for text in predictions) / count,
            4,
        ),
        "too_short_rate": round(
            100.0 * sum(ratio < 0.5 for ratio in length_ratios) / count,
            4,
        ),
        "too_long_rate": round(
            100.0 * sum(ratio > 1.5 for ratio in length_ratios) / count,
            4,
        ),
        "repeated_trigram_rate_mean": round(
            100.0 * sum(repeated_ngram_rate(text) for text in predictions) / count,
            4,
        ),
        "source_words_mean": round(sum(source_lengths) / count, 4),
        "compression_ratio_mean": round(sum(compression_ratios) / count, 6),
        "unique_prediction_rate": round(
            100.0 * len(set(nonempty)) / max(1, len(nonempty)),
            4,
        ),
        "dominant_prefix_5gram_rate": round(100.0 * dominant_prefix_rate, 4),
    }


def _salience_diagnostics(
    probabilities: List[float],
    labels: List[int],
) -> Dict[str, float]:
    """Thresholded and ranking diagnostics for held-out oracle evidence.

    References create labels only after generation and are never passed to the
    model. Average precision is included because class-balanced BCE is not
    probability-calibrated and a fixed 0.5 threshold can be misleading.
    """

    if not probabilities or len(probabilities) != len(labels):
        return {}
    predicted = [probability >= 0.5 for probability in probabilities]
    positive = [bool(label) for label in labels]
    true_positive = sum(prediction and gold for prediction, gold in zip(predicted, positive))
    false_positive = sum(prediction and not gold for prediction, gold in zip(predicted, positive))
    false_negative = sum(not prediction and gold for prediction, gold in zip(predicted, positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    ranked = sorted(zip(probabilities, positive), key=lambda pair: pair[0], reverse=True)
    positives_seen = 0
    precision_sum = 0.0
    for rank, (_, gold) in enumerate(ranked, start=1):
        if gold:
            positives_seen += 1
            precision_sum += positives_seen / rank
    average_precision = precision_sum / max(1, sum(positive))
    return {
        "salience_precision": round(100.0 * precision, 4),
        "salience_recall": round(100.0 * recall, 4),
        "salience_f1": round(100.0 * f1, 4),
        "salience_average_precision": round(100.0 * average_precision, 4),
        "salience_predicted_positive_rate": round(100.0 * sum(predicted) / len(predicted), 4),
        "salience_oracle_positive_rate": round(100.0 * sum(positive) / len(positive), 4),
        "salience_scored_units": len(labels),
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generated_lengths(
    generated: torch.Tensor,
    eos_token_id: int | None,
) -> List[int]:
    lengths: List[int] = []
    for row in generated:
        if eos_token_id is None:
            lengths.append(int(row.numel()))
            continue
        endings = row.eq(eos_token_id).nonzero(as_tuple=False)
        lengths.append(int(endings[0].item()) + 1 if endings.numel() else int(row.numel()))
    return lengths


def _decode_performance(
    latencies: List[float],
    generated_lengths: List[int],
) -> Dict[str, float]:
    elapsed = max(1e-9, sum(latencies))
    example_count = len(generated_lengths)
    total_tokens = sum(generated_lengths)
    per_example_latency = list(latencies)
    ordered_latency = sorted(per_example_latency)
    p95_index = max(
        0,
        min(len(ordered_latency) - 1, math.ceil(0.95 * len(ordered_latency)) - 1),
    )
    return {
        "decode_elapsed_seconds": round(elapsed, 3),
        "decode_examples_per_second": round(example_count / elapsed, 6),
        "decode_generated_tokens_per_second": round(total_tokens / elapsed, 6),
        "seconds_per_generated_token": round(elapsed / max(1, total_tokens), 8),
        "latency_seconds_mean": round(
            float(statistics.mean(per_example_latency)) if per_example_latency else 0.0,
            6,
        ),
        "latency_seconds_median": round(
            float(statistics.median(per_example_latency)) if per_example_latency else 0.0,
            6,
        ),
        "latency_seconds_p95": round(
            ordered_latency[p95_index] if ordered_latency else 0.0,
            6,
        ),
        "latency_seconds_min": round(min(per_example_latency, default=0.0), 6),
        "latency_seconds_max": round(max(per_example_latency, default=0.0), 6),
        "generated_tokens_total": int(total_tokens),
        "generated_tokens_mean": round(total_tokens / max(1, example_count), 4),
        "decode_steps_total": int(total_tokens),
        "decode_steps_mean": round(total_tokens / max(1, example_count), 6),
        "tokens_per_decode_step": 1.0,
    }


def _plan_gate_diagnostics(
    layer_sums: List[float],
    layer_counts: List[float],
    step_sums: List[float],
    step_counts: List[float],
) -> Dict[str, Any]:
    if not layer_sums or not step_sums:
        return {}
    per_layer = [total / max(1.0, count) for total, count in zip(layer_sums, layer_counts)]
    per_step = [total / max(1.0, count) for total, count in zip(step_sums, step_counts)]
    early_boundary = min(16, len(step_sums))
    early_sum = sum(step_sums[:early_boundary])
    early_count = sum(step_counts[:early_boundary])
    late_sum = sum(step_sums[early_boundary:])
    late_count = sum(step_counts[early_boundary:])
    result: Dict[str, Any] = {
        "plan_gate_generation_mean": round(
            sum(step_sums) / max(1.0, sum(step_counts)),
            6,
        ),
        "plan_gate_early_16_mean": round(early_sum / max(1.0, early_count), 6),
        "plan_gate_per_cross_layer": [round(value, 6) for value in per_layer],
        "plan_gate_by_generation_step": [round(value, 6) for value in per_step],
        "plan_gate_scored_layer_queries": int(sum(step_counts)),
    }
    if late_count > 0:
        result["plan_gate_late_after_16_mean"] = round(late_sum / late_count, 6)
    return result


@torch.inference_mode()
def _generate_encoder_decoder(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> tuple[List[str], Dict[str, float], Dict[str, float]]:
    from .generation import autoregressive_generate

    data = config.get("data", {}) or {}
    generation = config.get("generation", {}) or {}
    batch_size = int(generation.get("batch_size", 4))
    tokenizer.padding_side = "left"
    predictions: List[str] = []
    salience_probabilities: List[float] = []
    salience_labels: List[int] = []
    per_example_latencies: List[float] = []
    generated_lengths: List[int] = []
    plan_layer_sums: List[float] = []
    plan_layer_counts: List[float] = []
    plan_step_sums: List[float] = []
    plan_step_counts: List[float] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        token_features = []
        aligned_ids = []
        visible_units = []
        for example in batch:
            ids, unit_ids, units = prompted_source_features(tokenizer, str(example["source"]), data)
            token_features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            aligned_ids.append(unit_ids)
            visible_units.append(units)
        encoded = tokenizer.pad(token_features, return_tensors="pt", padding=True)
        unit_tensor = torch.zeros_like(encoded["input_ids"])
        for row, ids in enumerate(aligned_ids):
            unit_tensor[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
        encoded = encoded.to(device)
        unit_tensor = unit_tensor.to(device)
        _sync_device(device)
        batch_start = time.perf_counter()
        bridge_output = model.encode(
            encoded["input_ids"],
            encoded["attention_mask"],
            unit_ids=unit_tensor,
            return_bridge_output=True,
        )
        generated, generation_diagnostics = autoregressive_generate(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            unit_ids=unit_tensor,
            bridge_output=bridge_output,
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            do_sample=False,
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id,
            decoder_prefix_ids=decoder_prompt_ids(tokenizer, data),
            use_cache=True,
            return_diagnostics=True,
        )
        _sync_device(device)
        batch_elapsed = time.perf_counter() - batch_start
        batch_lengths = _generated_lengths(generated, tokenizer.eos_token_id)
        generated_lengths.extend(batch_lengths)
        per_example_latencies.extend([batch_elapsed / max(1, len(batch))] * len(batch))
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        batch_layer_sums = generation_diagnostics["plan_gate_layer_sums"]
        batch_layer_counts = generation_diagnostics["plan_gate_layer_counts"]
        if batch_layer_sums:
            if not plan_layer_sums:
                plan_layer_sums = [0.0] * len(batch_layer_sums)
                plan_layer_counts = [0.0] * len(batch_layer_counts)
            if len(batch_layer_sums) != len(plan_layer_sums):
                raise RuntimeError("Cross-attention layer count changed between batches")
            plan_layer_sums = [total + float(value) for total, value in zip(plan_layer_sums, batch_layer_sums)]
            plan_layer_counts = [total + float(value) for total, value in zip(plan_layer_counts, batch_layer_counts)]
        batch_step_sums = generation_diagnostics["plan_gate_step_sums"]
        batch_step_counts = generation_diagnostics["plan_gate_step_counts"]
        if len(plan_step_sums) < len(batch_step_sums):
            extension = len(batch_step_sums) - len(plan_step_sums)
            plan_step_sums.extend([0.0] * extension)
            plan_step_counts.extend([0.0] * extension)
        for index, (total, count) in enumerate(zip(batch_step_sums, batch_step_counts)):
            plan_step_sums[index] += float(total)
            plan_step_counts[index] += float(count)
        # References are consulted only after the complete model generation
        # and timing path. They score salience; they never affect memory or
        # autoregressive token selection.
        if bridge_output.salience_logits is not None:
            for row, (example, units) in enumerate(zip(batch, visible_units)):
                oracle = greedy_evidence_labels(
                    units,
                    str(example["target"]),
                    int(data.get("oracle_max_units", 12)),
                    str(data.get("oracle_budget_mode", "target_units")),
                    int(data.get("oracle_fixed_units", 3)),
                    float(data.get("oracle_rouge1_weight", 0.5)),
                    float(data.get("oracle_rouge2_weight", 0.5)),
                )
                width = min(len(oracle), bridge_output.salience_logits.shape[1])
                for probability, label in zip(
                    torch.sigmoid(bridge_output.salience_logits[row, :width].float()).tolist(),
                    oracle[:width],
                ):
                    if label >= 0:
                        salience_probabilities.append(float(probability))
                        salience_labels.append(int(label > 0.5))
    return (
        [prediction.strip() for prediction in predictions],
        {
            **_salience_diagnostics(salience_probabilities, salience_labels),
            **_plan_gate_diagnostics(
                plan_layer_sums,
                plan_layer_counts,
                plan_step_sums,
                plan_step_counts,
            ),
        },
        _decode_performance(per_example_latencies, generated_lengths),
    )


@torch.inference_mode()
def _generate_direct(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> tuple[List[str], Dict[str, float]]:
    from transformers import LogitsProcessorList

    from .generation import (
        OutputOnlyNoRepeatNGramLogitsProcessor,
        OutputOnlyRepetitionPenaltyLogitsProcessor,
    )

    data = config.get("data", {}) or {}
    generation = config.get("generation", {}) or {}
    batch_size = int(generation.get("batch_size", 4))
    tokenizer.padding_side = "left"
    predictions = []
    per_example_latencies: List[float] = []
    generated_lengths: List[int] = []
    for start in range(0, len(examples), batch_size):
        features = []
        for example in examples[start : start + batch_size]:
            ids = prompt_token_ids(tokenizer, str(example["source"]), data)
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        encoded = tokenizer.pad(features, return_tensors="pt", padding=True).to(device)
        prompt_length = encoded["input_ids"].shape[1]
        processors = LogitsProcessorList()
        repetition_penalty = float(generation.get("repetition_penalty", 1.05))
        no_repeat_ngram_size = int(generation.get("no_repeat_ngram_size", 3))
        if repetition_penalty != 1.0:
            processors.append(
                OutputOnlyRepetitionPenaltyLogitsProcessor(
                    repetition_penalty,
                    prompt_length,
                )
            )
        if no_repeat_ngram_size > 0:
            processors.append(
                OutputOnlyNoRepeatNGramLogitsProcessor(
                    no_repeat_ngram_size,
                    prompt_length,
                )
            )
        _sync_device(device)
        batch_start = time.perf_counter()
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            logits_processor=processors,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        _sync_device(device)
        batch_elapsed = time.perf_counter() - batch_start
        continuation = generated[:, prompt_length:]
        batch_lengths = _generated_lengths(continuation, tokenizer.eos_token_id)
        generated_lengths.extend(batch_lengths)
        per_example_latencies.extend([batch_elapsed / max(1, len(features))] * len(features))
        predictions.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return (
        [prediction.strip() for prediction in predictions],
        _decode_performance(per_example_latencies, generated_lengths),
    )


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    kind: str,
) -> tuple[Dict[str, float], List[str], List[str]]:
    data = config.get("data", {}) or {}
    if bool(data.get("clean_wikihow_metadata", False)):
        examples = [
            {
                **example,
                "source": clean_wikihow_metadata(str(example["source"])),
                "target": clean_wikihow_metadata(str(example["target"])),
            }
            for example in examples
        ]
    architecture_metrics: Dict[str, float] = {}
    performance_metrics: Dict[str, float] = {}
    if kind == "encoder_decoder":
        predictions, architecture_metrics, performance_metrics = _generate_encoder_decoder(
            model, tokenizer, examples, config, device
        )
    else:
        predictions, performance_metrics = _generate_direct(model, tokenizer, examples, config, device)
    references = [str(example["target"]) for example in examples]
    sources = [str(example["source"]) for example in examples]
    metrics = _rouge(predictions, references)
    metrics.update(_generation_diagnostics(predictions, references, sources))
    metrics.update(architecture_metrics)
    metrics.update(performance_metrics)
    return metrics, predictions, references


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_samples: int | None,
    model_size: str | None = None,
) -> None:
    checkpoint = Path(checkpoint_path)
    running_marker = checkpoint.parent / "RUNNING"
    if running_marker.exists():
        raise RuntimeError(f"Refusing to evaluate {checkpoint}: {running_marker} indicates an incomplete run")
    config = load_config(config_path)
    apply_model_size(config, model_size)
    evaluation_config = config.get("evaluation", {}) or {}
    evaluation_dtype_name = str(
        evaluation_config.get(
            "model_dtype",
            (config.get("model", {}) or {}).get("dtype", "bfloat16"),
        )
    )
    config.setdefault("model", {})["dtype"] = evaluation_dtype_name
    # Evaluation needs neither the train dataset nor its costly extractive
    # oracle precomputation.
    model, tokenizer, _, _, _ = build_experiment(config, load_datasets=False)
    checkpoint_payload = load_checkpoint(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    examples = read_jsonl((config.get("data", {}) or {})["test_file"])
    if max_samples is not None:
        examples = examples[:max_samples]
    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    dtype = torch_dtype(evaluation_dtype_name)
    use_autocast = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
        metrics, predictions, references = evaluate_examples(model, tokenizer, examples, config, device, kind)
    if kind == "encoder_decoder":
        metrics["cross_gate_mean"] = round(float(model.decoder.cross_gate_mean().item()), 6)
        metrics["cross_residual_ratio"] = round(float(model.decoder.cross_residual_ratio_mean().item()), 6)
        metrics["plan_gate_last_step_mean"] = round(float(model.decoder.plan_gate_mean().item()), 6)
        metrics["plan_gate_mean"] = metrics.get(
            "plan_gate_generation_mean",
            metrics["plan_gate_last_step_mean"],
        )
        metrics["plan_gate_cross_layer_indices"] = list(model.decoder.cross_attention_indices)
        metrics["token_adapter_gate"] = round(float(torch.tanh(model.bridge.token_adapter_gate.float()).item()), 6)
        metrics["plan_adapter_gate"] = round(float(torch.tanh(model.bridge.plan_adapter_gate.float()).item()), 6)
        if hasattr(model.bridge, "unit_broadcast_gate"):
            metrics["unit_broadcast_gate"] = round(
                float(torch.tanh(model.bridge.unit_broadcast_gate.float()).item()), 6
            )
    metrics["eval_batch_size"] = int((config.get("generation", {}) or {}).get("batch_size", 4))
    metrics["peak_gpu_memory_mb"] = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else 0.0
    )
    metrics["rouge_backend"] = "rouge==1.0.0 (HeterSumGraph)"
    metrics["rouge_preprocessing"] = "NFC + lowercase + stored whitespace tokenization"
    training_manifest = (
        checkpoint_payload.get("config", {}).get("_data_manifest") if isinstance(checkpoint_payload, dict) else None
    )
    if training_manifest:
        metrics["training_data_manifest"] = training_manifest

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example, prediction, reference in zip(examples, predictions, references):
            handle.write(
                json.dumps(
                    {"id": example.get("id"), "prediction": prediction, "reference": reference}, ensure_ascii=False
                )
                + "\n"
            )
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-size", choices=sorted(MODEL_PROFILES), default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.max_samples, args.model_size)


if __name__ == "__main__":
    main()
