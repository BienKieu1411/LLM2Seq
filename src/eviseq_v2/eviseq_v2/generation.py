"""Greedy and sampled generation over one reusable encoded source memory."""

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


def _repeat_batch(value: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.repeat_interleave(int(repeats), dim=0)


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
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if not decoder_seed:
        raise ValueError("decoder_seed cannot be empty")
    if do_sample and temperature <= 0.0:
        raise ValueError("temperature must be positive for sampling")
    if do_sample and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if do_sample and int(top_k) < 0:
        raise ValueError("top_k must be non-negative")
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
            if do_sample:
                logits = logits / float(temperature)
                if top_k > 0:
                    retained = min(int(top_k), logits.shape[-1])
                    candidate_logits, candidate_indices = torch.topk(logits, retained, dim=-1)
                    if top_p < 1.0:
                        cumulative_probs = torch.cumsum(torch.softmax(candidate_logits, dim=-1), dim=-1)
                        remove = cumulative_probs > float(top_p)
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        candidate_logits = candidate_logits.masked_fill(remove, float("-inf"))
                    sampled_index = torch.multinomial(torch.softmax(candidate_logits, dim=-1), num_samples=1)
                    next_token = candidate_indices.gather(1, sampled_index)
                elif top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_sorted = cumulative_probs > float(top_p)
                    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                    remove_sorted[..., 0] = False
                    remove = torch.zeros_like(remove_sorted)
                    remove.scatter_(1, sorted_indices, remove_sorted)
                    logits = logits.masked_fill(remove, float("-inf"))
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                else:
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
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
def generate_sampled_from_memory(
    model,
    adapter_output: Any,
    decoder_seed: Sequence[int],
    max_new_tokens: int = 256,
    min_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 0.90,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
    num_return_sequences: int = 1,
) -> torch.Tensor:
    """Nucleus-sample one or more candidates from one encoded source batch.

    The source is encoded exactly once.  ``num_return_sequences`` expands only
    the decoder-side memory, which is substantially cheaper than rerunning the
    long-document encoder for every candidate.
    """
    repeats = int(num_return_sequences)
    if repeats <= 0:
        raise ValueError("num_return_sequences must be positive")
    batch = adapter_output.memory.shape[0]
    expanded = type(adapter_output)(
        memory=_repeat_batch(adapter_output.memory, repeats),
        memory_mask=_repeat_batch(adapter_output.memory_mask, repeats),
        attention_bias=_repeat_batch(adapter_output.attention_bias, repeats),
        salience_logits=None,
        loss_salience=adapter_output.loss_salience,
    )
    generated = _generate_from_memory(
        model,
        expanded,
        decoder_seed,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return generated.reshape(batch, repeats, generated.shape[-1])


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


@torch.inference_mode()
def generate_sampled(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_seed: Sequence[int],
    unit_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    adapter_output = model.encode(input_ids, attention_mask, unit_ids=unit_ids)
    generated = generate_sampled_from_memory(
        model,
        adapter_output,
        decoder_seed,
        num_return_sequences=1,
        **kwargs,
    )
    return generated[:, 0]
