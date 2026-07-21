"""Standalone autoregressive generation for EviBridge models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


@torch.inference_mode()
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    unit_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.eval()
    batch_size = input_ids.size(0)
    memory, memory_mask = model.encode(
        input_ids,
        attention_mask,
        unit_ids=unit_ids,
        return_attention_mask=True,
    )
    if bos_token_id is None:
        bos_token_id = eos_token_id or 0
    generated = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    past_key_values = None

    for step in range(max_new_tokens):
        decoder_input = generated[:, -1:] if use_cache and past_key_values is not None else generated
        decoder_states, next_cache = model.decoder(
            input_ids=decoder_input,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
        )
        if use_cache:
            past_key_values = next_cache
        logits = model.lm_head(decoder_states[:, -1, :])

        if eos_token_id is not None and step + 1 < min_new_tokens:
            logits[:, eos_token_id] = float("-inf")
        if repetition_penalty != 1.0:
            for batch_index in range(batch_size):
                previous = torch.unique(generated[batch_index])
                scores = logits[batch_index, previous]
                logits[batch_index, previous] = torch.where(
                    scores < 0,
                    scores * repetition_penalty,
                    scores / repetition_penalty,
                )
        if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
            for batch_index in range(batch_size):
                tokens = generated[batch_index].tolist()
                prefix_size = no_repeat_ngram_size - 1
                current_prefix = tuple(tokens[-prefix_size:]) if prefix_size else ()
                blocked = []
                for index in range(len(tokens) - no_repeat_ngram_size + 1):
                    ngram = tuple(tokens[index : index + no_repeat_ngram_size])
                    if ngram[:-1] == current_prefix:
                        blocked.append(ngram[-1])
                if blocked:
                    logits[batch_index, blocked] = float("-inf")

        if temperature <= 0:
            do_sample = False
        elif temperature != 1.0:
            logits = logits / temperature
        if do_sample and top_k > 0:
            threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if do_sample and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            remove = remove.scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(remove, float("-inf"))

        if do_sample:
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        if pad_token_id is not None:
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token_id)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None:
            finished |= next_token.squeeze(1).eq(eos_token_id)
            if finished.all():
                break

    return generated[:, 1:]
