"""Greedy or candidate generation for one completed decoder baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .data import encode_prompt, left_pad_prompts, read_jsonl, record_texts
from .metrics import rouge_scores
from .train import _context_length, _dtype, _load_tokenizer_and_model


def _filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    """Apply top-k then nucleus filtering to a batch of next-token logits."""

    filtered = logits.clone()
    if top_k > 0:
        k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered.scatter_(
            -1,
            sorted_indices,
            sorted_logits.masked_fill(remove, -torch.inf),
        )
    return filtered


def _no_repeat_ngram_mask(logits: torch.Tensor, generated: torch.Tensor, ngram_size: int) -> torch.Tensor:
    if ngram_size <= 0 or generated.shape[1] < ngram_size - 1:
        return logits
    result = logits.clone()
    for row in range(generated.shape[0]):
        tokens = generated[row].tolist()
        if len(tokens) < ngram_size - 1:
            continue
        prefix = tuple(tokens[-(ngram_size - 1) :])
        banned = {
            tokens[index + ngram_size - 1]
            for index in range(len(tokens) - ngram_size + 1)
            if tuple(tokens[index : index + ngram_size - 1]) == prefix
        }
        if banned:
            result[row, list(banned)] = -torch.inf
    return result


def _next_token(
    logits: torch.Tensor,
    *,
    generation: dict[str, Any],
    generated: torch.Tensor,
    eos_token_id: int | None,
    step: int,
) -> torch.Tensor:
    scores = logits.float()
    penalty = float(generation.get("repetition_penalty", 1.0))
    if penalty != 1.0 and generated.numel():
        for row in range(scores.shape[0]):
            seen = torch.unique(generated[row])
            values = scores[row, seen]
            scores[row, seen] = torch.where(values < 0, values * penalty, values / penalty)
    scores = _no_repeat_ngram_mask(scores, generated, int(generation.get("no_repeat_ngram_size", 0)))
    if eos_token_id is not None and step < int(generation.get("min_new_tokens", 0)):
        scores[:, int(eos_token_id)] = -torch.inf
    do_sample = bool(generation.get("do_sample", False))
    if not do_sample:
        return scores.argmax(dim=-1, keepdim=True)
    temperature = float(generation.get("temperature", 0.0))
    if temperature <= 0:
        raise ValueError("Sampling requires generation.temperature > 0")
    scores = _filter_logits(
        scores / temperature, top_k=int(generation.get("top_k", 0)), top_p=float(generation.get("top_p", 1.0))
    )
    invalid = ~torch.isfinite(scores).any(dim=-1)
    if invalid.any():
        scores[invalid] = logits[invalid]
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_nemotron_ar(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tokenizer: Any,
    generation: dict[str, Any],
) -> torch.Tensor:
    """Generate with Nemotron-Labs-Diffusion's explicit AR cache API.

    Nemotron exposes ``AutoModel.ar_generate`` rather than the standard
    ``AutoModelForCausalLM.generate``.  Reproducing its small cache loop here
    lets the baseline honor the same temperature/top-k/top-p controls while
    keeping the model in autoregressive mode (no diffusion training objective).
    """

    if prompt_ids.shape[0] != 1:
        raise ValueError("Nemotron AR generation uses batch_size=1 to avoid padding ambiguity")
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - depends on Transformers version
        raise RuntimeError("Nemotron AR generation requires transformers.cache_utils.DynamicCache") from exc
    for layer in getattr(model.encoder, "layers", []):
        attention = getattr(layer, "self_attn", None)
        if hasattr(attention, "diffusion_lm"):
            attention.diffusion_lm = False
    device = prompt_ids.device
    prompt_length = int(prompt_ids.shape[1])
    cache = DynamicCache()
    positions = torch.arange(prompt_length, device=device)
    encoded = model.encoder(
        input_ids=prompt_ids,
        position_ids=positions.unsqueeze(0),
        past_key_values=cache,
        use_cache=True,
        cache_position=positions,
    )
    cache = encoded.past_key_values
    logits = model.diffusion_head(encoded.last_hidden_state[:, -1, :])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[torch.Tensor] = []
    for step in range(int(generation["max_new_tokens"])):
        generated_tensor = (
            torch.cat(generated, dim=1) if generated else torch.empty((1, 0), dtype=torch.long, device=device)
        )
        token = _next_token(
            logits,
            generation=generation,
            generated=generated_tensor,
            eos_token_id=eos_token_id,
            step=step,
        )
        generated.append(token)
        if (
            eos_token_id is not None
            and step + 1 >= int(generation.get("min_new_tokens", 0))
            and bool((token == eos_token_id).all())
        ):
            break
        if step + 1 >= int(generation["max_new_tokens"]):
            break
        cache_position = torch.tensor([prompt_length + step], device=device)
        encoded = model.encoder(
            input_ids=token,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        cache = encoded.past_key_values
        logits = model.diffusion_head(encoded.last_hidden_state[:, -1, :])
    return torch.cat([prompt_ids, *generated], dim=1)


def _generation_kwargs(generation: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    values = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "min_new_tokens": int(generation.get("min_new_tokens", 0)),
        "num_beams": 1,
        "do_sample": bool(generation.get("do_sample", False)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 0)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if values["do_sample"]:
        values.update(
            {
                "temperature": float(generation["temperature"]),
                "top_k": int(generation.get("top_k", 0)),
                "top_p": float(generation.get("top_p", 1.0)),
            }
        )
    return values


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    max_examples: int = 0,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    config = load_config(config_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_path}")
    tokenizer, model = _load_tokenizer_and_model(
        {**config, "model": {**config["model"], "name_or_path": str(checkpoint_path)}}, evaluation=True
    )
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    context_length = _context_length(model)
    data = config["data"]
    configured_limit = int(config.get("limits", {}).get(f"max_{split}_examples", 0))
    limit = int(max_examples) if int(max_examples) > 0 else configured_limit
    rows = read_jsonl(data[f"{split}_file"], max_examples=limit)
    generation = dict(config["generation"])
    prompts: list[list[int]] = []
    sources: list[str] = []
    references: list[str] = []
    identifiers: list[str] = []
    for row in rows:
        identifier, source, reference = record_texts(row, data)
        prompt = encode_prompt(
            tokenizer,
            source,
            data,
            max_total_length=max(1, context_length - int(generation["max_new_tokens"])),
        )
        prompts.append(prompt)
        identifiers.append(identifier)
        sources.append(source)
        references.append(reference)

    predictions: list[str] = []
    batch_size = int(generation["batch_size"])
    if str(config["model"].get("family", "causal_lm")) == "nemotron_diffusion":
        batch_size = 1
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        input_ids, attention = left_pad_prompts(batch_prompts, tokenizer.pad_token_id)
        input_ids = input_ids.to(target)
        attention = attention.to(target)
        if str(config["model"].get("family", "causal_lm")) == "nemotron_diffusion":
            outputs = generate_nemotron_ar(model, input_ids, tokenizer, generation)
        else:
            kwargs = _generation_kwargs(generation, tokenizer)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention, **kwargs)
        width = int(input_ids.shape[1])
        decoded = tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
        predictions.extend(value.strip() for value in decoded)

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for identifier, source, reference, prediction in zip(identifiers, sources, references, predictions):
            handle.write(
                json.dumps(
                    {"id": identifier, "source": source, "reference": reference, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = {
        **rouge_scores(predictions, references),
        "model_id": config["model"].get("model_id", config["model"].get("name_or_path")),
        "model_family": config["model"].get("family", "causal_lm"),
        "split": split,
        "num_examples": len(predictions),
        "predictions_file": str(output_path),
        "generation": generation,
        "prompt_protocol": "t5gemma_source_prefix_plus_causal_target_masking",
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one decoder-only baseline checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.config, args.checkpoint, args.output, split=args.split, max_examples=args.max_examples),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
