"""Greedy generation that conditions every decoder layer on adapter memory."""

from __future__ import annotations

from typing import Optional, Sequence

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
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_seed: Sequence[int],
    unit_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    min_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    if not decoder_seed:
        raise ValueError("decoder_seed cannot be empty")
    adapter_output = model.encode(input_ids, attention_mask, unit_ids=unit_ids)
    phrase_pointer = getattr(model, "phrase_pointer", None)
    if phrase_pointer is not None and unit_ids is None:
        raise ValueError("unit_ids are required when the phrase pointer is enabled")
    source_copy_mask = attention_mask.bool().clone()
    if unit_ids is not None:
        source_copy_mask &= unit_ids.gt(0)
    try:
        if phrase_pointer is not None:
            phrase_pointer.prepare_source_cache(adapter_output.memory)
        model.decoder.prepare_cross_attention_cache(adapter_output.memory)
    except Exception:
        # A partially prepared cache must never leak into the next batch.
        if phrase_pointer is not None:
            phrase_pointer.clear_source_cache()
        model.decoder.clear_cross_attention_cache()
        raise
    batch = input_ids.shape[0]
    generated = torch.tensor(list(decoder_seed), device=input_ids.device, dtype=torch.long)
    generated = generated.unsqueeze(0).expand(batch, -1).clone()
    prompt_length = generated.shape[1]
    summary_prefix = adapter_output.summary_prefix
    summary_prefix_mask = adapter_output.summary_prefix_mask
    finished = torch.zeros(batch, device=input_ids.device, dtype=torch.bool)
    past = None
    routing_sum = None
    routing_observations = 0
    pointer_responsibility = None
    phrase_mode_sum = None
    phrase_mode_observations = 0
    try:
        for step in range(int(max_new_tokens)):
            decoder_input = generated[:, -1:] if past is not None else generated
            # On the first call the decoder prepends the latent mask itself.
            # Cached calls need the complete mask for prefix + cached tokens +
            # the current token even though the prefix embeddings are not
            # inserted a second time.
            decoder_attention_mask = torch.ones_like(generated)
            if past is not None and summary_prefix is not None:
                prefix_mask = summary_prefix_mask
                if prefix_mask is None:
                    prefix_mask = torch.ones(
                        summary_prefix.shape[:2],
                        dtype=decoder_attention_mask.dtype,
                        device=decoder_attention_mask.device,
                    )
                else:
                    prefix_mask = prefix_mask.to(
                        device=decoder_attention_mask.device,
                        dtype=decoder_attention_mask.dtype,
                    )
                decoder_attention_mask = torch.cat(
                    [prefix_mask, decoder_attention_mask],
                    dim=1,
                )
            states, past = model.decoder(
                input_ids=decoder_input,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=adapter_output.memory,
                encoder_attention_mask=adapter_output.memory_mask,
                encoder_attention_bias=adapter_output.attention_bias,
                summary_prefix=summary_prefix,
                summary_prefix_mask=summary_prefix_mask,
                past_key_values=past,
                use_cache=True,
            )
            per_layer_routing = model.decoder.memory_routing_per_layer().detach().float()
            observations = int(states.shape[0] * states.shape[1])
            weighted_routing = per_layer_routing * observations
            routing_sum = weighted_routing if routing_sum is None else routing_sum + weighted_routing
            routing_observations += observations
            logits = model.lm_head(states[:, -1, :]).float()
            if phrase_pointer is not None:
                pointer_step = phrase_pointer.generation_step(
                    decoder_state=states[:, -1, :],
                    lm_logits=logits,
                    source_memory=adapter_output.memory,
                    source_token_ids=input_ids,
                    source_unit_ids=unit_ids,
                    source_copy_mask=source_copy_mask,
                    previous_responsibility=pointer_responsibility,
                    attention_bias=adapter_output.attention_bias,
                )
                logits = pointer_step.log_probabilities
                observed_modes = pointer_step.mode_probabilities.detach().float().sum(dim=0)
                phrase_mode_sum = observed_modes if phrase_mode_sum is None else phrase_mode_sum + observed_modes
                phrase_mode_observations += batch
            if eos_token_id is not None and step < int(min_new_tokens):
                logits[:, int(eos_token_id)] = float("-inf")
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
            if phrase_pointer is not None:
                pointer_responsibility = phrase_pointer.posterior_source_responsibility(
                    pointer_step,
                    next_token.squeeze(1),
                    input_ids,
                )
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(int(eos_token_id))
                if bool(finished.all()):
                    break
    finally:
        if routing_sum is not None and routing_observations > 0:
            model.decoder.set_generation_routing_statistics(
                routing_sum / routing_observations,
                routing_observations,
            )
        if phrase_pointer is not None and phrase_mode_sum is not None and phrase_mode_observations > 0:
            phrase_pointer.set_generation_statistics(
                phrase_mode_sum / phrase_mode_observations,
                phrase_mode_observations,
            )
        if phrase_pointer is not None:
            phrase_pointer.clear_source_cache()
        model.decoder.clear_cross_attention_cache()
    return generated[:, prompt_length:]
