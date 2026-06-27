"""
MTP-based generation for LLM2Seq.

Uses Multi-Token Prediction heads to propose draft tokens, then verifies the
draft with the main head. The default path is main-head-constrained greedy
decoding, so accepted output is identical to greedy autoregressive decoding.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .confidence_adaptive import confidence_adaptive_accept, compute_acceptance_metrics


@torch.no_grad()
def mtp_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    confidence_threshold: float = 0.9,
    verify_with_main: bool = True,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    fallback_to_autoregressive: bool = False,
    fallback_after_steps: int = 1,
    fallback_min_emitted_length: float = 3.8,
) -> dict:
    """
    MTP-accelerated generation.

    Default verified mode:
    1. Main head predicts y_t.
    2. MTP heads draft y_{t+1}, ..., y_{t+K}.
    3. Main head verifies the candidate block in one decoder pass.
    4. Accept the longest draft prefix matching the main head; at mismatch,
       append the main-head token for that position and stop the block.

    Args:
        model: LLM2Seq model with MTP module.
        input_ids: [B, S_src] — source token IDs.
        attention_mask: [B, S_src] — source attention mask.
        max_new_tokens: Maximum tokens to generate.
        min_new_tokens: Minimum tokens before EOS is allowed.
        confidence_threshold: Threshold for accepting MTP drafts.
        verify_with_main: If True, use main-head verification. If False, use
            the older confidence-only heuristic.
        repetition_penalty: Same greedy decoding penalty as autoregressive_generate.
        no_repeat_ngram_size: Same repeated n-gram block as autoregressive_generate.
        eos_token_id: EOS token ID.
        pad_token_id: PAD token ID.
        bos_token_id: BOS token ID.

    Returns:
        Dict with:
            - "generated_ids": [B, T] — generated token IDs.
            - "num_steps": Number of decoding steps taken.
            - "metrics": Acceptance rate metrics.
    """
    model.eval()

    if model.mtp_module is None:
        raise ValueError("Model does not have an MTP module. Use autoregressive_generate instead.")

    if verify_with_main:
        return _mtp_generate_verified(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            fallback_to_autoregressive=fallback_to_autoregressive,
            fallback_after_steps=fallback_after_steps,
            fallback_min_emitted_length=fallback_min_emitted_length,
        )

    return _mtp_generate_confidence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        confidence_threshold=confidence_threshold,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
    )


@torch.no_grad()
def _mtp_generate_confidence(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    confidence_threshold: float,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    bos_token_id: Optional[int],
    fallback_to_autoregressive: bool = False,
    fallback_after_steps: int = 1,
    fallback_min_emitted_length: float = 3.8,
) -> dict:
    """Legacy confidence-threshold MTP decoding without main-head verification."""
    device = input_ids.device
    bsz = input_ids.size(0)

    # Encode source once
    h_mem, memory_attention_mask = model.encode(
        input_ids, attention_mask, return_attention_mask=True
    )

    # Initialize
    if bos_token_id is None:
        bos_token_id = eos_token_id or 0
    generated = torch.full((bsz, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    num_steps = 0
    accepted_lengths: List[int] = []
    num_mtp_heads = model.cfg.mtp_num_heads

    total_generated = 0

    while total_generated < max_new_tokens:
        # Forward decoder (without cache for simplicity with multi-token append)
        decoder_states, _ = model.decoder(
            input_ids=generated,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            use_cache=False,
        )

        # Get hidden state at last position
        last_hidden = decoder_states[:, -1:, :]  # [B, 1, D]

        # Main head prediction
        main_logits = model.lm_head(last_hidden)  # [B, 1, V]
        main_probs = torch.softmax(main_logits, dim=-1)
        main_confidence, main_token = main_probs.max(dim=-1)  # [B, 1]

        # MTP draft predictions
        from ..models.mtp_heads import ParallelMTPHeads
        from ..models.mtp_cascaded import CascadedMTP

        if isinstance(model.mtp_module, ParallelMTPHeads):
            draft_results = model.mtp_module.get_draft_tokens_with_confidence(last_hidden)
        elif isinstance(model.mtp_module, CascadedMTP):
            draft_results = model.mtp_module.generate_draft(last_hidden, main_token)
        else:
            draft_results = []

        # Confidence-adaptive accept
        accepted_tokens, num_accepted = confidence_adaptive_accept(
            main_token=main_token,
            main_confidence=main_confidence,
            draft_results=draft_results,
            confidence_threshold=confidence_threshold,
        )

        accepted_lengths.append(num_accepted)

        # Append accepted tokens
        for token in accepted_tokens:
            if pad_token_id is not None:
                token = token.masked_fill(finished.unsqueeze(1), pad_token_id)
            generated = torch.cat([generated, token], dim=1)
            total_generated += 1

            # Check EOS
            if eos_token_id is not None:
                finished = finished | (token.squeeze(1) == eos_token_id)

            if total_generated >= max_new_tokens:
                break

        num_steps += 1

        if finished.all():
            break

    # Compute metrics
    metrics = compute_acceptance_metrics(accepted_lengths, num_mtp_heads)
    metrics["num_steps"] = num_steps
    metrics["speedup_vs_autoregressive"] = total_generated / max(1, num_steps)

    return {
        "generated_ids": generated[:, 1:],  # Remove BOS
        "num_steps": num_steps,
        "metrics": metrics,
    }


@torch.no_grad()
def _mtp_generate_verified(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    bos_token_id: Optional[int],
    fallback_to_autoregressive: bool = False,
    fallback_after_steps: int = 1,
    fallback_min_emitted_length: float = 3.8,
) -> dict:
    """Main-head-constrained greedy MTP decoding with KV caching."""
    if input_ids.size(0) > 1:
        results = []
        for i in range(input_ids.size(0)):
            results.append(
                _mtp_generate_verified(
                    model=model,
                    input_ids=input_ids[i : i + 1],
                    attention_mask=attention_mask[i : i + 1],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    fallback_to_autoregressive=fallback_to_autoregressive,
                    fallback_after_steps=fallback_after_steps,
                    fallback_min_emitted_length=fallback_min_emitted_length,
                )
            )
        max_len = max(result["generated_ids"].size(1) for result in results)
        pad_id = pad_token_id if pad_token_id is not None else 0
        padded = []
        for result in results:
            ids = result["generated_ids"]
            if ids.size(1) < max_len:
                pad = torch.full(
                    (1, max_len - ids.size(1)),
                    pad_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, pad], dim=1)
            padded.append(ids)
        metrics = _aggregate_verified_metrics([result["metrics"] for result in results])
        return {
            "generated_ids": torch.cat(padded, dim=0),
            "num_steps": sum(result["num_steps"] for result in results) / len(results),
            "metrics": metrics,
        }

    device = input_ids.device
    if bos_token_id is None:
        bos_token_id = eos_token_id or 0

    h_mem, memory_attention_mask = model.encode(
        input_ids, attention_mask, return_attention_mask=True
    )
    
    generated = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
    generated_token_list = [int(bos_token_id)]
    current_input_ids = generated
    past_key_values = None
    
    num_steps = 0
    total_generated = 0
    accepted_lengths: List[int] = []
    emitted_lengths: List[int] = []
    num_mtp_heads = model.cfg.mtp_num_heads
    fallback_triggered = False
    fallback_after_mtp_steps = None

    while total_generated < max_new_tokens:
        decoder_states, past_key_values = model.decoder(
            input_ids=current_input_ids,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        last_hidden = decoder_states[:, -1:, :]
        
        main_logits = _apply_greedy_constraints_from_list(
            logits=model.lm_head(last_hidden).squeeze(1),
            prefix_tokens=generated_token_list,
            generated_count=total_generated,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )
        main_token = main_logits.argmax(dim=-1).unsqueeze(1)

        from ..models.mtp_heads import ParallelMTPHeads
        from ..models.mtp_cascaded import CascadedMTP

        if isinstance(model.mtp_module, ParallelMTPHeads):
            draft_results = model.mtp_module.get_draft_tokens_with_confidence(last_hidden)
        elif isinstance(model.mtp_module, CascadedMTP):
            draft_results = model.mtp_module.generate_draft(last_hidden, main_token)
        else:
            draft_results = []

        draft_tokens = [draft["token_ids"] for draft in draft_results]
        
        if not draft_tokens:
            generated = torch.cat([generated, main_token], dim=1)
            generated_token_list.append(int(main_token.item()))
            total_generated += 1
            current_input_ids = main_token
            accepted_lengths.append(1)
            emitted_lengths.append(1)
            num_steps += 1
            if eos_token_id is not None and main_token.item() == eos_token_id:
                break
            continue

        candidate = torch.cat([main_token] + draft_tokens, dim=1)
        remaining = max_new_tokens - total_generated
        candidate = candidate[:, :remaining]

        verify_states, new_past = model.decoder(
            input_ids=candidate,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        verify_logits = model.lm_head(verify_states)

        accepted_drafts = 0
        verifier_tokens = []
        candidate_token_list = [int(token) for token in candidate[0].tolist()]

        for pos in range(candidate.size(1)):
            constrained_verify = _apply_greedy_constraints_from_list(
                logits=verify_logits[:, pos, :],
                prefix_tokens=generated_token_list + candidate_token_list[: pos + 1],
                generated_count=total_generated + 1 + pos,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
            )
            v_tok = constrained_verify.argmax(dim=-1).unsqueeze(1)
            verifier_tokens.append(v_tok)

            if pos < candidate.size(1) - 1:
                if torch.equal(candidate[:, pos+1:pos+2], v_tok):
                    accepted_drafts += 1
                else:
                    break

        accepted_cand = candidate[:, :1 + accepted_drafts]
        corrected_tok = verifier_tokens[accepted_drafts]

        emitted_this_step = [accepted_cand]
        eos_found = False
        
        if eos_token_id is not None:
            for i in range(accepted_cand.size(1)):
                if accepted_cand[0, i].item() == eos_token_id:
                    accepted_cand = accepted_cand[:, :i+1]
                    emitted_this_step = [accepted_cand]
                    corrected_tok = None
                    eos_found = True
                    break
        
        if not eos_found:
            emitted_this_step.append(corrected_tok)
            if eos_token_id is not None and corrected_tok.item() == eos_token_id:
                eos_found = True

        emitted_tensor = torch.cat(emitted_this_step, dim=1)

        if total_generated + emitted_tensor.size(1) > max_new_tokens:
            emitted_tensor = emitted_tensor[:, :max_new_tokens - total_generated]
            corrected_tok = emitted_tensor[:, -1:]
            
        generated = torch.cat([generated, emitted_tensor], dim=1)
        generated_token_list.extend(int(token) for token in emitted_tensor[0].tolist())
        total_generated += emitted_tensor.size(1)
        
        accepted_lengths.append(1 + accepted_drafts)
        emitted_lengths.append(emitted_tensor.size(1))
        num_steps += 1

        if eos_found or total_generated >= max_new_tokens:
            break

        keep_length = generated.size(1) - emitted_tensor.size(1) + accepted_cand.size(1)

        past_key_values = _slice_decoder_cache(new_past, keep_length)

        current_input_ids = corrected_tok

        if (
            fallback_to_autoregressive
            and fallback_after_steps > 0
            and num_steps >= fallback_after_steps
            and total_generated < max_new_tokens
            and safe_average(emitted_lengths) < fallback_min_emitted_length
        ):
            fallback_triggered = True
            fallback_after_mtp_steps = num_steps
            (
                generated,
                generated_token_list,
                past_key_values,
                current_input_ids,
                total_generated,
                num_steps,
                eos_reached,
            ) = _finish_autoregressive_from_state(
                model=model,
                h_mem=h_mem,
                memory_attention_mask=memory_attention_mask,
                generated=generated,
                generated_token_list=generated_token_list,
                current_input_ids=current_input_ids,
                past_key_values=past_key_values,
                total_generated=total_generated,
                num_steps=num_steps,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
            )
            emitted_lengths.extend([1] * max(0, num_steps - len(emitted_lengths)))
            accepted_lengths.extend([1] * max(0, num_steps - len(accepted_lengths)))
            if eos_reached or total_generated >= max_new_tokens:
                break

    metrics = compute_acceptance_metrics(accepted_lengths, num_mtp_heads)
    metrics["num_steps"] = num_steps
    metrics["speedup_vs_autoregressive"] = total_generated / max(1, num_steps)
    metrics["average_emitted_length"] = sum(emitted_lengths) / max(1, len(emitted_lengths))
    metrics["emitted_tokens"] = total_generated
    metrics["verified_with_main"] = True
    metrics["fallback_to_autoregressive"] = fallback_triggered
    metrics["fallback_after_mtp_steps"] = fallback_after_mtp_steps

    return {
        "generated_ids": generated[:, 1:],
        "num_steps": num_steps,
        "metrics": metrics,
    }


def safe_average(values: List[int]) -> float:
    return sum(values) / max(1, len(values))


def _slice_decoder_cache(past_key_values, keep_length: int):
    """Slice decoder self-attention cache while preserving cross-attention cache."""
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "key_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :keep_length, :]
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :keep_length, :]
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values._seen_tokens = keep_length
        return past_key_values

    if isinstance(past_key_values, list):
        sliced = []
        for layer_past in past_key_values:
            if isinstance(layer_past, dict):
                layer_cache = {}
                self_cache = layer_past.get("self")
                if self_cache is not None:
                    k, v = self_cache
                    layer_cache["self"] = (
                        k[:, :, :keep_length, :].contiguous(),
                        v[:, :, :keep_length, :].contiguous(),
                    )
                if layer_past.get("cross") is not None:
                    layer_cache["cross"] = layer_past["cross"]
                sliced.append(layer_cache)
            else:
                k, v = layer_past
                sliced.append(
                    (
                        k[:, :, :keep_length, :],
                        v[:, :, :keep_length, :],
                    )
                )
        return sliced

    sliced_tuple = []
    for layer_past in past_key_values:
        k, v = layer_past
        sliced_tuple.append(
            (
                k[:, :, :keep_length, :],
                v[:, :, :keep_length, :],
            )
        )
    return tuple(sliced_tuple)


@torch.no_grad()
def _finish_autoregressive_from_state(
    model,
    h_mem: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    generated: torch.Tensor,
    generated_token_list: List[int],
    current_input_ids: torch.Tensor,
    past_key_values,
    total_generated: int,
    num_steps: int,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: Optional[int],
):
    """Continue with standard greedy decoding from the current verified state."""
    eos_reached = False
    while total_generated < max_new_tokens:
        decoder_states, past_key_values = model.decoder(
            input_ids=current_input_ids,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = _apply_greedy_constraints_from_list(
            logits=model.lm_head(decoder_states[:, -1, :]),
            prefix_tokens=generated_token_list,
            generated_count=total_generated,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        generated_token_list.append(int(next_token.item()))
        current_input_ids = next_token
        total_generated += 1
        num_steps += 1
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            eos_reached = True
            break

    return (
        generated,
        generated_token_list,
        past_key_values,
        current_input_ids,
        total_generated,
        num_steps,
        eos_reached,
    )


def _apply_greedy_constraints_from_list(
    logits: torch.Tensor,
    prefix_tokens: List[int],
    generated_count: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    """Single-sample fast path for deterministic decoding constraints."""
    constrained = logits

    if eos_token_id is not None and generated_count + 1 < min_new_tokens:
        constrained[:, eos_token_id] = float("-inf")

    if repetition_penalty and repetition_penalty != 1.0:
        unique_tokens = list(set(prefix_tokens))
        if unique_tokens:
            score = constrained[0, unique_tokens]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            constrained[0, unique_tokens] = score

    if no_repeat_ngram_size and len(prefix_tokens) >= no_repeat_ngram_size:
        n = no_repeat_ngram_size
        prefix_len = len(prefix_tokens)
        prefix = prefix_tokens[-(n - 1):]
        blocked_tokens = []
        for i in range(prefix_len - n + 1):
            if prefix_tokens[i : i + n - 1] == prefix:
                blocked_tokens.append(prefix_tokens[i + n - 1])
        if blocked_tokens:
            constrained[0, blocked_tokens] = float("-inf")

    return constrained


def _apply_greedy_constraints(
    logits: torch.Tensor,
    generated_prefix: torch.Tensor,
    generated_count: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    """Apply the same deterministic decoding constraints as autoregressive_generate."""
    constrained = logits.clone()
    bsz = constrained.size(0)

    if eos_token_id is not None and generated_count + 1 < min_new_tokens:
        constrained[:, eos_token_id] = float("-inf")

    if repetition_penalty and repetition_penalty != 1.0:
        for batch_idx in range(bsz):
            previous_tokens = set(generated_prefix[batch_idx].tolist())
            for token_id in previous_tokens:
                if constrained[batch_idx, token_id] < 0:
                    constrained[batch_idx, token_id] *= repetition_penalty
                else:
                    constrained[batch_idx, token_id] /= repetition_penalty

    if no_repeat_ngram_size and generated_prefix.size(1) >= no_repeat_ngram_size:
        for batch_idx in range(bsz):
            prefix = generated_prefix[batch_idx].tolist()
            ngram_prefix = tuple(prefix[-(no_repeat_ngram_size - 1):])
            blocked_tokens = []
            for i in range(len(prefix) - no_repeat_ngram_size + 1):
                ngram = tuple(prefix[i: i + no_repeat_ngram_size])
                if ngram[:-1] == ngram_prefix:
                    blocked_tokens.append(ngram[-1])
            if blocked_tokens:
                constrained[batch_idx, blocked_tokens] = float("-inf")

    return constrained


def _aggregate_verified_metrics(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = {
        "acceptance_rate",
        "average_accepted_length",
        "speedup_vs_autoregressive",
        "average_emitted_length",
        "emitted_tokens",
        "num_steps",
    }
    result = {
        key: sum(float(metrics.get(key, 0.0)) for metrics in metrics_list) / len(metrics_list)
        for key in keys
    }
    car_values = [metrics.get("cumulative_acceptance_rates", []) for metrics in metrics_list]
    if car_values and car_values[0]:
        result["cumulative_acceptance_rates"] = [
            sum(values[i] for values in car_values if i < len(values)) / len(car_values)
            for i in range(len(car_values[0]))
        ]
    result["verified_with_main"] = True
    return result
