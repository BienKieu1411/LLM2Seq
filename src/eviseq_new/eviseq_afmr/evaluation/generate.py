"""Greedy generation helper with append-only JSONL resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch


def _apply_repetition_penalty(scores: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> None:
    """Match Transformers' RepetitionPenaltyLogitsProcessor for greedy decode."""
    if penalty == 1.0:
        return
    values = scores.gather(1, token_ids)
    scores.scatter_(1, token_ids, torch.where(values < 0, values * penalty, values / penalty))


def _apply_no_repeat_ngram(scores: torch.Tensor, token_ids: torch.Tensor, ngram_size: int) -> None:
    if ngram_size <= 0 or token_ids.shape[1] < ngram_size:
        return
    ngrams = token_ids.unfold(1, ngram_size, 1)
    if ngram_size == 1:
        matches = torch.ones_like(ngrams[..., -1], dtype=torch.bool)
    else:
        matches = (ngrams[..., :-1] == token_ids[:, None, -(ngram_size - 1) :]).all(-1)
    penalties = torch.zeros_like(matches, dtype=scores.dtype).masked_fill(matches, -float("inf"))
    scores.scatter_add_(1, ngrams[..., -1], penalties)


def _no_repeat_ngram_tokens(token_ids: torch.Tensor, ngram_size: int) -> list[list[int]]:
    """Return tokens banned by Transformers' NoRepeatNGramLogitsProcessor."""
    if ngram_size <= 0:
        return [[] for _ in range(token_ids.shape[0])]
    banned: list[list[int]] = []
    for row in token_ids.tolist():
        if len(row) + 1 < ngram_size:
            banned.append([])
            continue
        ngrams: dict[tuple[int, ...], set[int]] = {}
        for start in range(len(row) - ngram_size + 1):
            ngram = tuple(row[start : start + ngram_size])
            ngrams.setdefault(ngram[:-1], set()).add(ngram[-1])
        prefix = tuple(row[-(ngram_size - 1) :]) if ngram_size > 1 else ()
        banned.append(sorted(ngrams.get(prefix, set())))
    return banned


@torch.inference_mode()
def generate_greedy(
    model: Any,
    batch: dict[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    compact_finished: bool = True,
) -> tuple[list[str], torch.Tensor]:
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")
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
    active_rows = torch.arange(token_ids.shape[0], device=token_ids.device)
    memory, memory_mask, source_bias = bridge.memory, bridge.memory_mask, bridge.source_bias
    try:
        model.decoder.prepare_cross_cache(memory)
        for step in range(max_new_tokens):
            history = token_ids.index_select(0, active_rows)
            current = history if past is None else history[:, -1:]
            logits, past, _ = model.decoder(
                current,
                memory,
                memory_mask,
                source_bias,
                decode_mask.index_select(0, active_rows),
                past_key_values=past,
                use_cache=True,
            )
            scores = logits[:, -1].float()
            _apply_repetition_penalty(scores, history, float(repetition_penalty))
            _apply_no_repeat_ngram(scores, history, int(no_repeat_ngram_size))
            eos = getattr(tokenizer, "eos_token_id", None)
            if step < min_new_tokens:
                if eos is not None:
                    scores[:, int(eos)] = -float("inf")
            next_token = torch.full(
                (token_ids.shape[0],),
                int(getattr(tokenizer, "pad_token_id", 0) or 0),
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
            next_token[active_rows] = scores.argmax(dim=-1)
            next_token = torch.where(finished, int(getattr(tokenizer, "pad_token_id", 0) or 0), next_token)
            decode_mask = torch.cat((decode_mask, (~finished)[:, None]), dim=1)
            if eos is not None:
                finished = finished | next_token.eq(int(eos))
            token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
            if eos is not None and step >= min_new_tokens and bool(finished.all()):
                break
            if (
                compact_finished
                and eos is not None
                and hasattr(past, "batch_select_indices")
                and hasattr(model.decoder, "select_cross_cache")
            ):
                surviving = (~finished.index_select(0, active_rows)).nonzero(as_tuple=True)[0]
                if surviving.numel() != active_rows.numel():
                    past.batch_select_indices(surviving)
                    model.decoder.select_cross_cache(surviving)
                    active_rows = active_rows.index_select(0, surviving)
                    memory = memory.index_select(0, surviving)
                    memory_mask = memory_mask.index_select(0, surviving)
                    source_bias = source_bias.index_select(0, surviving)
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
