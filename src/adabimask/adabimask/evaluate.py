"""Generate summaries and compute ROUGE for either controlled model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .backbone import torch_dtype
from .checkpoint import load_checkpoint
from .config import QWEN35_MODEL_SIZES, apply_model_size, load_config
from .data import clean_wikihow_metadata, prompt_token_ids, read_jsonl
from .training import build_experiment


def _rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for name in totals:
            totals[name] += scores[name].fmeasure
    denominator = max(1, len(predictions))
    return {name: round(100.0 * value / denominator, 4) for name, value in totals.items()}


@torch.inference_mode()
def _generate_encoder_decoder(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> List[str]:
    from .generation import autoregressive_generate

    data = config.get("data", {}) or {}
    generation = config.get("generation", {}) or {}
    batch_size = int(generation.get("batch_size", 8))
    predictions: List[str] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        features = []
        for example in batch:
            ids = prompt_token_ids(tokenizer, str(example["source"]), data)
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        encoded = tokenizer.pad(
            features,
            return_tensors="pt",
            padding=True,
        ).to(device)
        generated = autoregressive_generate(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            do_sample=False,
            repetition_penalty=float(generation.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [prediction.strip() for prediction in predictions]


@torch.inference_mode()
def _generate_direct(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> List[str]:
    data = config.get("data", {}) or {}
    generation = config.get("generation", {}) or {}
    batch_size = int(generation.get("batch_size", 8))
    tokenizer.padding_side = "left"
    predictions: List[str] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        features = []
        for example in batch:
            ids = prompt_token_ids(tokenizer, str(example["source"]), data)
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        encoded = tokenizer.pad(
            features,
            return_tensors="pt",
            padding=True,
        ).to(device)
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            do_sample=False,
            repetition_penalty=float(generation.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        predictions.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return [prediction.strip() for prediction in predictions]


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
    if kind == "encoder_decoder":
        predictions = _generate_encoder_decoder(model, tokenizer, examples, config, device)
    else:
        predictions = _generate_direct(model, tokenizer, examples, config, device)
    references = [str(example["target"]) for example in examples]
    return _rouge(predictions, references), predictions, references


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    max_samples: int | None,
    model_size: str | None = None,
) -> None:
    config = load_config(config_path)
    apply_model_size(config, model_size)
    model, tokenizer, _, _, _ = build_experiment(config)
    load_checkpoint(model, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data = config.get("data", {}) or {}
    examples = read_jsonl(data["test_file"])
    if max_samples is not None:
        examples = examples[:max_samples]
    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    dtype_name = str((config.get("model", {}) or {}).get("dtype", "bfloat16"))
    inference_dtype = torch_dtype(dtype_name)
    use_autocast = device.type == "cuda" and inference_dtype in {torch.bfloat16, torch.float16}
    with torch.autocast(
        device_type=device.type,
        dtype=inference_dtype,
        enabled=use_autocast,
    ):
        metrics, predictions, references = evaluate_examples(
            model,
            tokenizer,
            examples,
            config,
            device,
            kind,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example, prediction, reference in zip(examples, predictions, references):
            handle.write(
                json.dumps(
                    {"id": example.get("id"), "prediction": prediction, "reference": reference},
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output, args.max_samples, args.model_size)


if __name__ == "__main__":
    main()
