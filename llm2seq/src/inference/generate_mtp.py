"""
MTP-based generation for LLM2Seq.

Uses Multi-Token Prediction heads to propose multiple tokens per step,
then accepts a prefix based on confidence-adaptive logic.
This can significantly reduce the number of decoding steps.
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
    confidence_threshold: float = 0.9,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
) -> dict:
    """
    MTP-accelerated generation.

    At each step:
    1. Run decoder to get hidden states.
    2. Main head predicts y_t.
    3. MTP heads predict y_{t+1}, ..., y_{t+K}.
    4. Confidence-adaptive accept selects a prefix.
    5. Append accepted tokens and continue.

    Args:
        model: LLM2Seq model with MTP module.
        input_ids: [B, S_src] — source token IDs.
        attention_mask: [B, S_src] — source attention mask.
        max_new_tokens: Maximum tokens to generate.
        confidence_threshold: Threshold for accepting MTP drafts.
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
    device = input_ids.device
    bsz = input_ids.size(0)

    if model.mtp_module is None:
        raise ValueError("Model does not have an MTP module. Use autoregressive_generate instead.")

    # Encode source once
    h_mem = model.encode(input_ids, attention_mask)

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
            encoder_attention_mask=attention_mask,
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
