"""Generate summaries once after training and compute standard ROUGE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .backbone import torch_dtype
from .checkpoint import load_checkpoint
from .config import MODEL_PROFILES, apply_model_size, load_config
from .data import (
    clean_wikihow_metadata,
    decoder_prompt_ids,
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
    batch_size = int(generation.get("batch_size", 4))
    tokenizer.padding_side = "left"
    predictions: List[str] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        token_features = []
        aligned_ids = []
        for example in batch:
            ids, unit_ids, _ = prompted_source_features(tokenizer, str(example["source"]), data)
            token_features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            aligned_ids.append(unit_ids)
        encoded = tokenizer.pad(token_features, return_tensors="pt", padding=True)
        unit_tensor = torch.zeros_like(encoded["input_ids"])
        for row, ids in enumerate(aligned_ids):
            unit_tensor[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long)
        encoded = encoded.to(device)
        unit_tensor = unit_tensor.to(device)
        generated = autoregressive_generate(
            model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            unit_ids=unit_tensor,
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
    batch_size = int(generation.get("batch_size", 4))
    tokenizer.padding_side = "left"
    predictions = []
    for start in range(0, len(examples), batch_size):
        features = []
        for example in examples[start : start + batch_size]:
            ids = prompt_token_ids(tokenizer, str(example["source"]), data)
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        encoded = tokenizer.pad(features, return_tensors="pt", padding=True).to(device)
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            do_sample=False,
            repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
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
    predictions = (
        _generate_encoder_decoder(model, tokenizer, examples, config, device)
        if kind == "encoder_decoder"
        else _generate_direct(model, tokenizer, examples, config, device)
    )
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
    examples = read_jsonl((config.get("data", {}) or {})["test_file"])
    if max_samples is not None:
        examples = examples[:max_samples]
    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    dtype = torch_dtype(str((config.get("model", {}) or {}).get("dtype", "bfloat16")))
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        metrics, predictions, references = evaluate_examples(
            model, tokenizer, examples, config, device, kind
        )
    metrics["rouge_backend"] = "rouge==1.0.0 (HeterSumGraph)"
    metrics["rouge_preprocessing"] = "NFC + lowercase + stored whitespace tokenization"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example, prediction, reference in zip(examples, predictions, references):
            handle.write(json.dumps({"id": example.get("id"), "prediction": prediction, "reference": reference}, ensure_ascii=False) + "\n")
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
