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
    generated = torch.tensor(list(decoder_seed), device=adapter_output.memory.device, dtype=torch.long)
    generated = generated.unsqueeze(0).expand(batch, -1).clone()
    prompt_length = generated.shape[1]
    finished = torch.zeros(batch, device=adapter_output.memory.device, dtype=torch.bool)
    past = None
    try:
        for step in range(int(max_new_tokens)):
            decoder_input = generated[:, -1:] if past is not None else generated
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
            if repetition_penalty != 1.0 or no_repeat_ngram_size > 0:
                for batch_index in range(batch):
                    output_tokens = generated[batch_index, prompt_length:]
                    if repetition_penalty != 1.0 and output_tokens.numel():
                        previous = torch.unique(output_tokens)
                        scores = logits[batch_index, previous]
                        logits[batch_index, previous] = torch.where(
                            scores < 0,
                            scores * repetition_penalty,
                            scores / repetition_penalty,
                        )
                    blocked = _blocked_tokens(output_tokens.tolist(), int(no_repeat_ngram_size))
                    if blocked:
                        logits[batch_index, blocked] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            if pad_token_id is not None:
                next_token = next_token.masked_fill(finished.unsqueeze(1), int(pad_token_id))
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(int(eos_token_id))
                should_check_finished = (step + 1) % 8 == 0 or step + 1 == int(max_new_tokens)
                if should_check_finished and bool(finished.all()):
                    break
    finally:
        model.decoder.clear_cross_attention_cache()
    return generated[:, prompt_length:]


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
