"""Greedy generation over one reusable encoded source memory."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def _blocked_tokens(tokens: Sequence[int], order: int) -> list[int]:
    if order <= 0 or len(tokens) < order:
        return []
    prefix_length = order - 1
    prefix = tuple(tokens[-prefix_length:]) if prefix_length else ()
    blocked = []
    for index in range(len(tokens) - order + 1):
        ngram = tuple(tokens[index : index + order])
        if ngram[:-1] == prefix:
            blocked.append(ngram[-1])
    return blocked


def _apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float) -> None:
    """Apply the usual greedy repetition penalty without host synchronisation.

    ``tokens.tolist()`` inside the decoding loop makes one GPU-to-CPU
    synchronisation for every generated token and every row.  PubMed decoding
    has long outputs and batches, so that Python path dominated wall-clock
    time.  Gather/scatter preserves the scalar rule exactly while staying on
    the device.  Repeated token ids are harmless because each duplicate reads
    the same unmodified logit and writes the same penalised value.
    """

    if penalty == 1.0 or tokens.numel() == 0:
        return
    scores = logits.gather(1, tokens)
    penalised = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits.scatter_(1, tokens, penalised)


def _apply_no_repeat_ngram(logits: torch.Tensor, tokens: torch.Tensor, order: int) -> None:
    """Ban repeated n-grams for a complete greedy batch entirely on GPU."""

    if order <= 0 or tokens.shape[1] < order:
        return
    # Every existing n-gram whose prefix matches the current suffix contributes
    # its final token to the forbidden set.  This is the same condition as
    # ``_blocked_tokens`` but avoids a ``.tolist()`` synchronisation per row.
    windows = tokens.unfold(1, order, 1)
    prefix_width = order - 1
    if prefix_width:
        current_prefix = tokens[:, -prefix_width:].unsqueeze(1)
        matched = windows[:, :, :-1].eq(current_prefix).all(dim=-1)
    else:
        matched = torch.ones(windows.shape[:2], dtype=torch.bool, device=tokens.device)
    blocked = windows[:, :, -1]
    rows = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1).expand_as(blocked)
    logits[rows[matched], blocked[matched]] = float("-inf")


@torch.inference_mode()
def _generate_from_memory(
    model,
    adapter_output: Any,
    decoder_seed: Sequence[int],
    max_new_tokens: int = 256,
    min_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    if not decoder_seed:
        raise ValueError("decoder_seed cannot be empty")
    model.decoder.prepare_cross_attention_cache(adapter_output.memory)
    batch = adapter_output.memory.shape[0]
    seed = torch.tensor(list(decoder_seed), device=adapter_output.memory.device, dtype=torch.long)
    prompt_length = seed.numel()
    # ``torch.cat`` on the complete history once per decoded token repeatedly
    # reallocates and copies a [batch, prompt + step] tensor.  Keep one fixed
    # greedy-history buffer instead; ``past_key_values`` still ensures the
    # decoder itself sees only the new token after its first forward pass.
    capacity = prompt_length + int(max_new_tokens)
    fill_value = int(pad_token_id if pad_token_id is not None else (eos_token_id if eos_token_id is not None else 0))
    generated = torch.full(
        (batch, capacity),
        fill_value,
        device=adapter_output.memory.device,
        dtype=torch.long,
    )
    generated[:, :prompt_length] = seed.unsqueeze(0)
    current_length = prompt_length
    finished = torch.zeros(batch, device=adapter_output.memory.device, dtype=torch.bool)
    past = None
    try:
        for step in range(int(max_new_tokens)):
            decoder_input = (
                generated[:, current_length - 1 : current_length] if past is not None else generated[:, :current_length]
            )
            states, past = model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=adapter_output.memory,
                encoder_attention_mask=adapter_output.memory_mask,
                encoder_attention_bias=adapter_output.attention_bias,
                past_key_values=past,
                use_cache=True,
            )
            logits = model.lm_head(states[:, -1, :]).float()
            if eos_token_id is not None and step < int(min_new_tokens):
                logits[:, int(eos_token_id)] = float("-inf")
            output_tokens = generated[:, prompt_length:current_length]
            _apply_repetition_penalty(logits, output_tokens, float(repetition_penalty))
            _apply_no_repeat_ngram(logits, output_tokens, int(no_repeat_ngram_size))
            next_token = logits.argmax(dim=-1, keepdim=True)
            if pad_token_id is not None:
                next_token = next_token.masked_fill(finished.unsqueeze(1), int(pad_token_id))
            generated[:, current_length] = next_token.squeeze(1)
            current_length += 1
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(int(eos_token_id))
                should_check_finished = (step + 1) % 8 == 0 or step + 1 == int(max_new_tokens)
                if should_check_finished and bool(finished.all()):
                    break
    finally:
        model.decoder.clear_cross_attention_cache()
    return generated[:, prompt_length:current_length]


@torch.inference_mode()
def generate_from_memory(
    model,
    adapter_output: Any,
    decoder_seed: Sequence[int],
    max_new_tokens: int = 256,
    min_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    """Greedy-decode from an already computed encoder/bridge output."""

    return _generate_from_memory(
        model,
        adapter_output,
        decoder_seed,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


@torch.inference_mode()
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_seed: Sequence[int],
    unit_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    adapter_output = model.encode(input_ids, attention_mask, unit_ids=unit_ids)
    return generate_from_memory(model, adapter_output, decoder_seed, **kwargs)
