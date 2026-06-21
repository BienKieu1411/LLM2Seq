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
) -> dict:
    """Main-head-constrained greedy MTP decoding."""
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
    num_steps = 0
    total_generated = 0
    accepted_lengths: List[int] = []
    emitted_lengths: List[int] = []
    num_mtp_heads = model.cfg.mtp_num_heads

    while total_generated < max_new_tokens:
        decoder_states, _ = model.decoder(
            input_ids=generated,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            use_cache=False,
        )
        last_hidden = decoder_states[:, -1:, :]
        main_logits = _apply_greedy_constraints(
            logits=model.lm_head(last_hidden).squeeze(1),
            generated_prefix=generated,
            generated_count=total_generated,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )
        main_token = main_logits.argmax(dim=-1)
        main_token = main_token.unsqueeze(1)

        from ..models.mtp_heads import ParallelMTPHeads
        from ..models.mtp_cascaded import CascadedMTP

        if isinstance(model.mtp_module, ParallelMTPHeads):
            draft_results = model.mtp_module.get_draft_tokens_with_confidence(last_hidden)
        elif isinstance(model.mtp_module, CascadedMTP):
            draft_results = model.mtp_module.generate_draft(last_hidden, main_token)
        else:
            draft_results = []

        draft_tokens = [draft["token_ids"] for draft in draft_results]
        candidate = torch.cat([main_token] + draft_tokens, dim=1)
        remaining = max_new_tokens - total_generated
        candidate = candidate[:, :remaining]

        verify_input = torch.cat([generated, candidate], dim=1)
        verify_states, _ = model.decoder(
            input_ids=verify_input,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            use_cache=False,
        )
        start = generated.size(1) - 1
        verify_logits = model.lm_head(verify_states[:, start : start + candidate.size(1), :])
        verifier_tokens: List[torch.Tensor] = []
        for pos in range(candidate.size(1)):
            prefix = verify_input[:, : start + pos + 1]
            constrained_logits = _apply_greedy_constraints(
                logits=verify_logits[:, pos, :],
                generated_prefix=prefix,
                generated_count=total_generated + pos,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_token_id,
            )
            verifier_tokens.append(constrained_logits.argmax(dim=-1, keepdim=True))
        verifier_tokens = torch.cat(verifier_tokens, dim=1)

        accepted_tokens: List[torch.Tensor] = []
        accepted_drafts = 0
        for pos in range(candidate.size(1)):
            verifier_token = verifier_tokens[:, pos : pos + 1]
            candidate_token = candidate[:, pos : pos + 1]
            if pos == 0:
                accepted_tokens.append(verifier_token)
            elif torch.equal(candidate_token, verifier_token):
                accepted_tokens.append(candidate_token)
                accepted_drafts += 1
            else:
                accepted_tokens.append(verifier_token)
                break

            if eos_token_id is not None and accepted_tokens[-1].item() == eos_token_id:
                break

        accepted_lengths.append(1 + accepted_drafts)
        emitted_lengths.append(len(accepted_tokens))
        for token in accepted_tokens:
            generated = torch.cat([generated, token], dim=1)
            total_generated += 1
            if eos_token_id is not None and token.item() == eos_token_id:
                break
            if total_generated >= max_new_tokens:
                break

        num_steps += 1
        if eos_token_id is not None and generated[:, -1].item() == eos_token_id:
            break

    metrics = compute_acceptance_metrics(accepted_lengths, num_mtp_heads)
    metrics["num_steps"] = num_steps
    metrics["speedup_vs_autoregressive"] = total_generated / max(1, num_steps)
    metrics["average_emitted_length"] = sum(emitted_lengths) / max(1, len(emitted_lengths))
    metrics["emitted_tokens"] = total_generated
    metrics["verified_with_main"] = True

    return {
        "generated_ids": generated[:, 1:],
        "num_steps": num_steps,
        "metrics": metrics,
    }


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
