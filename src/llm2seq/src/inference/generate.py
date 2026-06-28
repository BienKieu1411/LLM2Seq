"""
Autoregressive generation for LLM2Seq (without MTP).

Supports:
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- KV-cache for efficient generation
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
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
    """
    Standard autoregressive generation without MTP.

    Args:
        model: LLM2Seq model.
        input_ids: [B, S_src] — source token IDs.
        attention_mask: [B, S_src] — source attention mask.
        max_new_tokens: Maximum number of tokens to generate.
        min_new_tokens: Minimum number of tokens before EOS is allowed.
        do_sample: If True, sample from the filtered distribution. If False, use greedy decoding.
        temperature: Sampling temperature (1.0 = no scaling).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p nucleus filtering (1.0 = disabled).
        repetition_penalty: Penalty for tokens already generated (1.0 = disabled).
        no_repeat_ngram_size: Block repeated n-grams of this size (0 = disabled).
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.
        bos_token_id: Beginning of sequence token ID.
        use_cache: Whether to use KV-cache.

    Returns:
        generated_ids: [B, T_generated] — generated token IDs.
    """
    model.eval()
    device = input_ids.device
    bsz = input_ids.size(0)

    # Encode source once
    h_mem, memory_attention_mask = model.encode(input_ids, attention_mask, return_attention_mask=True)

    # Initialize decoder input with BOS token
    if bos_token_id is None:
        bos_token_id = eos_token_id or 0
    generated = torch.full((bsz, 1), bos_token_id, dtype=torch.long, device=device)

    # Track which sequences have finished
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    past_key_values = None

    for step in range(max_new_tokens):
        # Determine input for this step
        if use_cache and past_key_values is not None:
            decoder_input = generated[:, -1:]  # Only last token
        else:
            decoder_input = generated

        # Forward through decoder
        decoder_states, past_key_values_out = model.decoder(
            input_ids=decoder_input,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
        )

        if use_cache:
            past_key_values = past_key_values_out

        # Get logits for last position
        logits = model.lm_head(decoder_states[:, -1, :])  # [B, V]

        if eos_token_id is not None and step + 1 < min_new_tokens:
            logits[:, eos_token_id] = float("-inf")

        # Penalize repeated tokens.
        if repetition_penalty and repetition_penalty != 1.0:
            for batch_idx in range(bsz):
                previous_tokens = set(generated[batch_idx].tolist())
                for token_id in previous_tokens:
                    if logits[batch_idx, token_id] < 0:
                        logits[batch_idx, token_id] *= repetition_penalty
                    else:
                        logits[batch_idx, token_id] /= repetition_penalty

        # Block tokens that would create a repeated n-gram.
        if no_repeat_ngram_size and generated.size(1) >= no_repeat_ngram_size:
            for batch_idx in range(bsz):
                prefix = generated[batch_idx].tolist()
                ngram_prefix = tuple(prefix[-(no_repeat_ngram_size - 1) :])
                blocked_tokens = []
                for i in range(len(prefix) - no_repeat_ngram_size + 1):
                    ngram = tuple(prefix[i : i + no_repeat_ngram_size])
                    if ngram[:-1] == ngram_prefix:
                        blocked_tokens.append(ngram[-1])
                if blocked_tokens:
                    logits[batch_idx, blocked_tokens] = float("-inf")

        if temperature is None or temperature <= 0:
            do_sample = False
        elif temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if do_sample and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if do_sample and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample or greedy
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        # Mask finished sequences with pad token
        if pad_token_id is not None:
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token_id)

        generated = torch.cat([generated, next_token], dim=1)

        # Check for EOS
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

    # Remove the initial BOS token
    return generated[:, 1:]
