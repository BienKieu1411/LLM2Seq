from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import RandomAccessJsonl
from .drafter import ParallelBlockDrafter


def greedy_margin_accepted_length_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Smoothly reward target top-1 margins across a greedy proposal block.

    The exact greedy accept rule is discrete: a proposed token is accepted
    only when it is the target argmax. This loss uses sigmoid of the draft's
    target-token-vs-best-competitor logit margin as a differentiable surrogate,
    then sums cumulative products to approximate accepted prefix length. These
    sigmoid values are *not* calibrated acceptance probabilities and this is
    not the paper's probability-mass EAL objective.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, block, vocab] and labels [batch, block]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("Every margin-loss window must contain at least one target token")
    # Block labels must be a contiguous prefix followed only by padding.
    if ((~valid).cummax(dim=1).values & valid).any():
        raise ValueError("Ignored labels must form a contiguous suffix of each block")
    vocab_size = logits.shape[-1]
    if ((labels[valid] < 0) | (labels[valid] >= vocab_size)).any():
        raise ValueError("A target label is outside the draft vocabulary")

    safe_labels = labels.clamp_min(0)
    scores = logits.float()
    target_scores = scores.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    competitors = scores.scatter(-1, safe_labels.unsqueeze(-1), -torch.inf)
    best_competitor = competitors.max(dim=-1).values
    margins = (target_scores - best_competitor) / temperature
    acceptance_surrogate = torch.sigmoid(margins)
    # P(prefix through position t) is the product of the per-token surrogates.
    # Padding contributes a neutral factor and is excluded from the objective.
    acceptance_surrogate = torch.where(valid, acceptance_surrogate, torch.ones_like(acceptance_surrogate))
    prefix_surrogate = torch.cumprod(acceptance_surrogate, dim=1)
    prefix_surrogate = prefix_surrogate * valid.to(prefix_surrogate.dtype)
    return -prefix_surrogate.sum(dim=1).mean()


def train_drafter(
    adapter,
    config: Dict[str, Any],
    trace_path: str | Path,
    output_path: Optional[str | Path] = None,
    max_steps: Optional[int] = None,
    resume_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    training = config.get("training", {})
    trace_path = Path(trace_path).expanduser().resolve()
    records = RandomAccessJsonl(trace_path)
    if not records:
        raise ValueError(f"Teacher trace file contains no examples: {trace_path}")
    output_path = Path(output_path or training.get("output_checkpoint", "./runs/drafter/last.pt"))
    if not output_path.is_absolute():
        output_path = (Path(config["_config_dir"]) / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and resume_path is None:
        raise FileExistsError(f"Refusing to overwrite an existing drafter checkpoint: {output_path}")
    step_limit = int(max_steps if max_steps is not None else training.get("steps", 1000))
    if step_limit < 1:
        raise ValueError("training.steps must be positive")

    seed = int(training.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if adapter.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    drafter = ParallelBlockDrafter.from_config(adapter, config).to(device=adapter.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        drafter.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    completed_steps = 0
    if resume_path:
        payload = torch.load(Path(resume_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        if payload.get("format") != "t5gemma2_parallel_block_drafter":
            raise ValueError("Resume file is not a supported parallel drafter checkpoint")
        saved_target = payload.get("target_manifest", {})
        current_target = adapter.target_manifest()
        if any(
            saved_target.get(field) != current_target.get(field)
            for field in ("target_fingerprint", "config_sha256", "tokenizer_sha256")
        ):
            raise ValueError("Resume drafter was trained against a different T5Gemma2 target")
        drafter.load_state_dict(payload["state_dict"], strict=True)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        completed_steps = int(payload.get("step", 0))

    generation = config["generation"]
    block_size = int(config["speculation"]["block_size"])
    log_every = max(1, int(training.get("log_every", 10)))
    save_every = max(1, int(training.get("save_every", 100)))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    loss_type = str(training.get("loss_type", "cross_entropy"))
    if loss_type not in {"cross_entropy", "greedy_margin_accepted_length"}:
        raise ValueError("training.loss_type must be 'cross_entropy' or 'greedy_margin_accepted_length'")
    margin_warmup_steps = max(0, int(training.get("margin_warmup_steps", 0)))
    margin_ce_weight = float(training.get("margin_ce_weight", 0.05))
    margin_temperature = float(training.get("margin_temperature", 1.0))
    if margin_ce_weight < 0:
        raise ValueError("training.margin_ce_weight must be nonnegative")
    if margin_temperature <= 0:
        raise ValueError("training.margin_temperature must be positive")
    loss_history: List[float] = []
    drafter.train()
    adapter.model.eval()

    for local_step in range(1, step_limit + 1):
        row = records[random.randrange(len(records))]
        source_values = row.get("source_ids")
        decoder_values = row.get("decoder_ids")
        if not isinstance(source_values, list) or not isinstance(decoder_values, list):
            raise ValueError("Trace rows must include source_ids and decoder_ids arrays")
        if not source_values or len(decoder_values) < 3:
            continue
        decoder_values = [int(value) for value in decoder_values]
        if decoder_values[0] != adapter.start_token_id:
            raise ValueError(f"Trace {row.get('id')} does not start with decoder_start_token_id")

        source_ids = torch.tensor([source_values], dtype=torch.long, device=adapter.device)
        source_mask = torch.ones_like(source_ids)
        source = adapter.encode_ids(source_ids, source_mask)
        decoder_inputs = torch.tensor([decoder_values[:-1]], dtype=torch.long, device=adapter.device)
        with torch.no_grad():
            target_features = adapter.decoder_features(source, decoder_inputs)

        upper_anchor = min(len(decoder_values) - 1, target_features.shape[1])
        if upper_anchor <= 1:
            continue
        anchor = random.randrange(1, upper_anchor)
        label_values = decoder_values[anchor + 1 : anchor + block_size + 1]
        if not label_values:
            continue
        labels = torch.full(
            (1, block_size),
            -100,
            dtype=torch.long,
            device=adapter.device,
        )
        labels[0, : len(label_values)] = torch.tensor(label_values, dtype=torch.long, device=adapter.device)
        history_start = max(0, anchor - int(config["draft"]["target_feature_window"]))
        history = target_features[:, history_start:anchor].detach()
        memory_config = config["source_memory"]
        prepared_source = drafter.prepare_source(
            source.hidden_states.detach(),
            source.attention_mask,
            source.fingerprint,
            mode=str(memory_config.get("mode", "full")),
            budget=int(memory_config.get("budget", 256)),
        )
        pending = torch.tensor([decoder_values[anchor]], dtype=torch.long, device=adapter.device)
        optimizer.zero_grad(set_to_none=True)
        draft_logits = drafter(
            pending_ids=pending,
            accepted_features=history,
            source=prepared_source,
            input_embedding=adapter.input_embedding,
            output_head=adapter.output_head,
            block_size=block_size,
        )
        cross_entropy = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )
        margin_loss = greedy_margin_accepted_length_loss(
            draft_logits,
            labels,
            temperature=margin_temperature,
        )
        margin_active = loss_type == "greedy_margin_accepted_length" and completed_steps >= margin_warmup_steps
        if margin_active:
            loss = margin_loss + margin_ce_weight * cross_entropy
        else:
            loss = cross_entropy
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite drafter loss at step {completed_steps + 1}")
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), max_grad_norm)
        optimizer.step()
        completed_steps += 1
        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)

        if completed_steps % log_every == 0 or local_step == 1:
            print(
                json.dumps(
                    {
                        "step": completed_steps,
                        "loss": loss_value,
                        "loss_type": "greedy_margin_accepted_length" if margin_active else "cross_entropy",
                        "cross_entropy": float(cross_entropy.detach().cpu()),
                        "accepted_length_surrogate": float((-margin_loss).detach().cpu()),
                        "margin_temperature": margin_temperature,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "trace_id": row.get("id"),
                        "anchor": anchor,
                        "block_size": block_size,
                    }
                ),
                flush=True,
            )
        if completed_steps % save_every == 0:
            _save_checkpoint(drafter, optimizer, adapter, output_path, completed_steps, config)

    _save_checkpoint(drafter, optimizer, adapter, output_path, completed_steps, config)
    return {
        "output_checkpoint": str(output_path),
        "steps": completed_steps,
        "mean_loss": sum(loss_history) / max(1, len(loss_history)),
        "trace_file": str(trace_path),
        "target": adapter.target_manifest(),
    }


def _save_checkpoint(drafter, optimizer, adapter, output_path: Path, step: int, config: Dict[str, Any]) -> None:
    payload = drafter.checkpoint_payload(adapter.target_manifest())
    payload.update(
        {
            "step": int(step),
            "optimizer_state": optimizer.state_dict(),
            "training_config": {
                "block_size": int(config["speculation"]["block_size"]),
                "loss": str(config.get("training", {}).get("loss_type", "cross_entropy")),
                "margin_warmup_steps": int(config.get("training", {}).get("margin_warmup_steps", 0)),
                "margin_ce_weight": float(config.get("training", {}).get("margin_ce_weight", 0.05)),
                "margin_temperature": float(config.get("training", {}).get("margin_temperature", 1.0)),
                "trace_file": str(config.get("training", {}).get("trace_file", "")),
            },
        }
    )
    temporary = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
