"""Greedy generation helper with append-only JSONL resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch


@torch.inference_mode()
def generate_greedy(
    model: Any, batch: dict[str, Any], tokenizer: Any, max_new_tokens: int, min_new_tokens: int = 0
) -> tuple[list[str], torch.Tensor]:
    model.eval()
    bridge = model.encode_source(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        torch.full(
            (batch["input_ids"].shape[0],), max_new_tokens, device=batch["input_ids"].device, dtype=torch.float32
        ),
    )
    prompt_mask = batch["decoder_prompt_mask"].bool()
    if prompt_mask.shape[1] == 0 or not bool(prompt_mask.any(-1).all()):
        raise ValueError("Greedy generation requires a non-empty decoder prompt per sample")
    width = prompt_mask.shape[1]
    positions = torch.arange(width, device=prompt_mask.device)[None, :]
    order = torch.argsort(torch.where(prompt_mask, positions + width, positions), dim=-1)
    token_ids = batch["decoder_prompt_ids"].gather(1, order)
    decode_mask = prompt_mask.gather(1, order)
    past = None
    finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
    model.decoder.eval()
    model.decoder.prepare_cross_cache(bridge.memory)
    try:
        for step in range(max_new_tokens):
            current = token_ids if past is None else token_ids[:, -1:]
            logits, past, _ = model.decoder(
                current,
                bridge.memory,
                bridge.memory_mask,
                bridge.source_bias,
                decode_mask,
                past_key_values=past,
                use_cache=True,
            )
            scores = logits[:, -1].float()
            eos = getattr(tokenizer, "eos_token_id", None)
            if step < min_new_tokens:
                if eos is not None:
                    scores[:, int(eos)] = -float("inf")
            next_token = scores.argmax(dim=-1)
            next_token = torch.where(finished, int(getattr(tokenizer, "pad_token_id", 0) or 0), next_token)
            decode_mask = torch.cat((decode_mask, (~finished)[:, None]), dim=1)
            if eos is not None:
                finished = finished | next_token.eq(int(eos))
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
            if eos is not None and step >= min_new_tokens and bool(finished.all()):
                break
    finally:
        model.decoder.clear_cross_cache()
    texts = tokenizer.batch_decode(token_ids[:, batch["decoder_prompt_ids"].shape[1] :], skip_special_tokens=True)
    return list(texts), token_ids


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def existing_ids(path: str | Path) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()
    found: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                found.add(str(row.get("id", "")))
    return found
