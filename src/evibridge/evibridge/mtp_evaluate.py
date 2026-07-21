"""Evaluate verified Phase-3 decoding and benchmark it against greedy AR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .checkpoint import load_checkpoint
from .config import QWEN35_MODEL_SIZES, apply_model_size, load_config
from .data import clean_wikihow_metadata, prompted_source_features, read_jsonl
from .evaluate import _rouge
from .generation import autoregressive_generate
from .mtp import load_mtp_checkpoint
from .mtp_generation import MTPGenerationOutput, verified_mtp_generate
from .training import build_experiment


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _features(
    tokenizer: Any,
    example: Dict[str, Any],
    data_config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids, unit_ids, _ = prompted_source_features(
        tokenizer,
        str(example["source"]),
        data_config,
    )
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    units = torch.tensor([unit_ids], dtype=torch.long, device=device)
    return input_ids, attention_mask, units


def _generation_kwargs(tokenizer: Any, generation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_new_tokens": int(generation.get("max_new_tokens", 256)),
        "min_new_tokens": int(generation.get("min_new_tokens", 0)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.05)),
        "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 3)),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id,
    }


@torch.inference_mode()
def _run_ar(
    model: Any,
    features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    kwargs: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    _sync(device)
    started = time.perf_counter()
    generated = autoregressive_generate(
        model,
        input_ids=features[0],
        attention_mask=features[1],
        unit_ids=features[2],
        do_sample=False,
        use_cache=True,
        **kwargs,
    )
    _sync(device)
    return generated, time.perf_counter() - started


@torch.inference_mode()
def _run_mtp(
    model: Any,
    features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    kwargs: Dict[str, Any],
    generation: Dict[str, Any],
) -> MTPGenerationOutput:
    return verified_mtp_generate(
        model,
        input_ids=features[0],
        attention_mask=features[1],
        unit_ids=features[2],
        fallback_probe_steps=int(generation.get("mtp_fallback_probe_steps", 8)),
        fallback_min_accepted_drafts=float(
            generation.get("mtp_fallback_min_accepted_drafts", 1.0)
        ),
        **kwargs,
    )


def evaluate_mtp(
    config_path: str,
    checkpoint_path: str,
    mtp_checkpoint_path: str,
    output_path: str,
    max_samples: int | None,
    model_size: str | None = None,
    compare_ar: bool = True,
    warmup_samples: int = 1,
) -> None:
    config = load_config(config_path)
    apply_model_size(config, model_size)
    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    if kind != "encoder_decoder":
        raise ValueError("Verified MTP is only defined for EviBridge encoder-decoder checkpoints")

    model, tokenizer, _, _, _ = build_experiment(config)
    load_checkpoint(model, checkpoint_path)
    predictor = model.enable_mtp()
    mtp_payload = load_mtp_checkpoint(predictor, mtp_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictor.eval()
    predictor.draft_head.prepare(model.lm_head)

    data_config = config.get("data", {}) or {}
    generation = config.get("generation", {}) or {}
    examples = read_jsonl(data_config["test_file"])
    if max_samples is not None:
        examples = examples[:max_samples]
    if bool(data_config.get("clean_wikihow_metadata", False)):
        examples = [
            {
                **example,
                "source": clean_wikihow_metadata(str(example["source"])),
                "target": clean_wikihow_metadata(str(example["target"])),
            }
            for example in examples
        ]
    if not examples:
        raise ValueError("No examples to evaluate")

    kwargs = _generation_kwargs(tokenizer, generation)
    amp_dtype = torch.bfloat16 if str((config.get("model", {}) or {}).get("dtype", "bfloat16")) == "bfloat16" else torch.float16
    amp_enabled = device.type == "cuda"

    # Exclude one-time CUDA/kernel initialization from the measured examples.
    for example in examples[: max(0, min(warmup_samples, len(examples)))]:
        features = _features(tokenizer, example, data_config, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _run_mtp(model, features, kwargs, generation)
            if compare_ar:
                _run_ar(model, features, kwargs, device)

    predictions: List[str] = []
    references: List[str] = []
    rows: List[Dict[str, Any]] = []
    mtp_seconds = 0.0
    ar_seconds = 0.0
    emitted_tokens = 0.0
    decoder_calls = 0.0
    drafted_tokens = 0.0
    accepted_drafts = 0.0
    fallback_count = 0
    exact_matches = 0

    for example in examples:
        features = _features(tokenizer, example, data_config, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            mtp_output = _run_mtp(model, features, kwargs, generation)
            ar_ids = None
            elapsed_ar = 0.0
            if compare_ar:
                ar_ids, elapsed_ar = _run_ar(model, features, kwargs, device)

        elapsed_mtp = float(mtp_output.metrics["encode_seconds"]) + float(
            mtp_output.metrics["decode_seconds"]
        )
        mtp_seconds += elapsed_mtp
        ar_seconds += elapsed_ar
        emitted_tokens += float(mtp_output.metrics["emitted_tokens"])
        decoder_calls += float(mtp_output.metrics["decoder_calls"])
        accepted_drafts += float(mtp_output.metrics["accepted_drafts"])
        drafted_tokens += float(mtp_output.metrics["drafted_tokens"])
        fallback_count += int(bool(mtp_output.metrics["fallback_triggered"]))

        exact = True
        if ar_ids is not None:
            exact = torch.equal(mtp_output.generated_ids, ar_ids)
            exact_matches += int(exact)
            if not exact:
                raise RuntimeError(
                    f"Verified MTP diverged from greedy AR on example {example.get('id')}; "
                    "this is a cache/constraint bug, not an acceptable quality trade-off"
                )
        prediction = tokenizer.decode(mtp_output.generated_ids[0], skip_special_tokens=True).strip()
        reference = str(example["target"])
        predictions.append(prediction)
        references.append(reference)
        rows.append(
            {
                "id": example.get("id"),
                "prediction": prediction,
                "reference": reference,
                "exact_ar_match": exact if compare_ar else None,
                "mtp": mtp_output.metrics,
                "ar_seconds": elapsed_ar if compare_ar else None,
            }
        )

    metrics: Dict[str, Any] = _rouge(predictions, references)
    metrics.update(
        {
            "examples": len(examples),
            "mtp_seconds": round(mtp_seconds, 6),
            "mtp_tokens_per_second": round(emitted_tokens / max(mtp_seconds, 1e-12), 4),
            "mtp_tokens_per_decoder_call": round(emitted_tokens / max(decoder_calls, 1.0), 4),
            "mtp_draft_acceptance_rate": round(accepted_drafts / max(drafted_tokens, 1.0), 6),
            "mtp_fallback_rate": round(fallback_count / len(examples), 6),
            "verified_with_main": True,
            "base_checkpoint": str(checkpoint_path),
            "mtp_checkpoint": str(mtp_checkpoint_path),
            "phase3_epoch": mtp_payload.get("epoch"),
        }
    )
    if compare_ar:
        metrics.update(
            {
                "ar_seconds": round(ar_seconds, 6),
                "wall_clock_speedup": round(ar_seconds / max(mtp_seconds, 1e-12), 4),
                "exact_ar_match_rate": round(exact_matches / len(examples), 6),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mtp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--no-compare-ar", action="store_true")
    args = parser.parse_args()
    evaluate_mtp(
        args.config,
        args.checkpoint,
        args.mtp_checkpoint,
        args.output,
        args.max_samples,
        args.model_size,
        not args.no_compare_ar,
        args.warmup_samples,
    )


if __name__ == "__main__":
    main()
