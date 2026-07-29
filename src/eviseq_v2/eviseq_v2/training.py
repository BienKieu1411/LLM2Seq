"""EviSeq V2 training entry point.

Extends the V1 training with:
- Evidence-focused hard contrastive learning (replaces document-level InfoNCE)
- Phase 3: BRIO-like candidate ranking fine-tune
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from . import runtime as stable
from .config import load_config
from .contrastive import exact_duplicate_mask, info_nce_loss
from .data import (
    CandidateRankingCollator,
    CandidateRankingDataset,
    Seq2SeqCollator,
    SummarizationDataset,
    decoder_seed_ids,
)
from .data_integrity import audit
from .model import EviSeq
from .parameter_manifest import build_parameter_manifest

LOGGER = logging.getLogger("eviseq_v2")

_TRAIN_AVERAGE_METRICS = (
    "loss",
    "loss_ce",
    "loss_salience",
    "loss_contrastive",
    "loss_evidence_contrastive",
    "prompt_retrieval_accuracy",
    "contrastive_candidates",
    "cross_residual_ratio",
    "bidirectional_gate_mean",
    "evidence_top1_accuracy",
    "positive_similarity",
    "hard_negative_similarity",
    "evidence_similarity_gap",
    "evidence_valid_examples",
)
_SALIENCE_COUNT_METRICS = (
    "salience_tp",
    "salience_predicted_count",
    "salience_gold_count",
    "salience_correct_pairs",
    "salience_pair_count",
)
_RANKING_METRICS = (
    "loss",
    "loss_ce",
    "loss_rank",
    "loss_salience",
    "candidate_pair_accuracy",
    "candidate_pair_count",
    "ranking_to_ce_ratio",
    "cross_residual_ratio",
    "bidirectional_gate_mean",
)


def _rounded(value: float) -> float:
    """Keep JSON logs readable without changing any training computation."""

    return round(float(value), 5)


def _salience_scores(totals: Dict[str, float]) -> tuple[float, float, float]:
    tp = totals.get("salience_tp", 0.0)
    predicted = totals.get("salience_predicted_count", 0.0)
    gold = totals.get("salience_gold_count", 0.0)
    precision = tp / predicted if predicted > 0.0 else 0.0
    recall = tp / gold if gold > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return precision, recall, f1


def _salience_ranking_accuracy(totals: Dict[str, float]) -> float:
    pairs = totals.get("salience_pair_count", 0.0)
    return totals.get("salience_correct_pairs", 0.0) / pairs if pairs > 0.0 else 0.0


def _contrastive_scale(
    stage: str,
    stage_epoch: int,
    batch_index: int,
    batches_per_epoch: int,
    warmup_epochs: int,
) -> float:
    """Linearly introduce InfoNCE during interface warm-up."""

    if warmup_epochs <= 0 or stage != "interface_warmup":
        return 1.0
    progress = float(stage_epoch - 1) + float(batch_index) / max(1, batches_per_epoch)
    return min(1.0, progress / float(warmup_epochs))


def _uses_virtual_gradcache(model: EviSeq, accumulation: int) -> bool:
    """A one-microbatch window already has the exact local InfoNCE matrix."""

    return model.alignment_head is not None and model.contrastive_across_accumulation and int(accumulation) > 1


def _evidence_contrastive_scale(
    stage: str,
    stage_epoch: int,
    batch_index: int,
    batches_per_epoch: int,
    warmup_epochs: int,
) -> float:
    """Linearly introduce evidence contrastive during interface warm-up.

    During warm-up: ramp from 0 to 1.0 over warmup_epochs
    During full fine-tune: always 1.0
    """

    if warmup_epochs <= 0 or stage != "interface_warmup":
        return 1.0
    progress = float(stage_epoch - 1) + float(batch_index) / max(1, batches_per_epoch)
    return min(1.0, progress / float(warmup_epochs))


def _capture_optimizer_moments(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Dict[str, Any]]:
    """Keep Adam state by parameter name across the warm-up/full-stage cut."""

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        names[id(parameter)]: state for parameter, state in optimizer.state.items() if id(parameter) in names and state
    }


def _restore_optimizer_moments(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    carried: Optional[Dict[str, Dict[str, Any]]],
) -> int:
    """Restore states only for warm-up parameters that remain trainable."""

    if not carried:
        return 0
    restored = 0
    for name, parameter in model.named_parameters():
        state = carried.pop(name, None)
        if state is None or not parameter.requires_grad:
            continue
        optimizer.state[parameter] = state
        restored += 1
    return restored


def _capture_rng_state(device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    """Capture dropout RNG so the no-grad cache pass can be replayed exactly."""

    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng_state(state: Dict[str, Optional[torch.Tensor]], device: torch.device) -> None:
    cpu = state.get("cpu")
    if cpu is None:
        raise RuntimeError("GradCache RNG state is missing the CPU generator")
    torch.random.set_rng_state(cpu)
    cuda = state.get("cuda")
    if device.type == "cuda":
        if cuda is None:
            raise RuntimeError("GradCache RNG state is missing the CUDA generator")
        torch.cuda.set_rng_state(cuda, device)


def _virtual_duplicate_mask(batches: list[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Find exact duplicate sources across all microbatches in one update."""

    if not batches:
        raise ValueError("A contrastive virtual batch cannot be empty")
    total = sum(int(batch["input_ids"].shape[0]) for batch in batches)
    width = max(int(batch["input_ids"].shape[1]) for batch in batches)
    template = batches[0]["input_ids"]
    ids = torch.zeros((total, width), dtype=template.dtype, device=template.device)
    masks = torch.zeros((total, width), dtype=torch.bool, device=template.device)
    offset = 0
    for batch in batches:
        rows, length = batch["input_ids"].shape
        ids[offset : offset + rows, :length] = batch["input_ids"]
        masks[offset : offset + rows, :length] = batch["attention_mask"].bool()
        offset += rows
    return exact_duplicate_mask(ids, masks)


def _build_virtual_contrastive_cache(
    model: EviSeq,
    batches: list[Dict[str, torch.Tensor]],
    device: torch.device,
    training: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Compute exact virtual-batch InfoNCE representation gradients.

    This is the GradCache two-pass decomposition: the first pass stores only
    normalized source/prompt representations; their loss gradients are then
    replayed through each microbatch in `_run_stage`.  Encoder/decoder
    activations are therefore never retained for the full virtual batch.
    """

    source_chunks: list[torch.Tensor] = []
    prompt_chunks: list[torch.Tensor] = []
    rng_states: list[Dict[str, Optional[torch.Tensor]]] = []
    with torch.no_grad():
        for batch in batches:
            rng_states.append(_capture_rng_state(device))
            with stable._autocast(device, training):
                representations = model(**batch, contrastive_mode="representations_only")
            source_chunks.append(representations["source_repr"].detach())
            prompt_chunks.append(representations["prompt_repr"].detach())
    post_cache_rng_state = _capture_rng_state(device)

    sizes = [int(value.shape[0]) for value in source_chunks]
    source = torch.cat(source_chunks, dim=0).requires_grad_(True)
    prompt = torch.cat(prompt_chunks, dim=0).requires_grad_(True)
    loss, accuracy = info_nce_loss(
        source,
        prompt,
        model.contrastive_temperature,
        duplicate_mask=_virtual_duplicate_mask(batches),
    )
    weighted = model.contrastive_weight * float(scale) * loss
    source_gradient, prompt_gradient = torch.autograd.grad(
        weighted,
        (source, prompt),
        allow_unused=True,
    )
    source_gradient = source_gradient if source_gradient is not None else torch.zeros_like(source)
    prompt_gradient = prompt_gradient if prompt_gradient is not None else torch.zeros_like(prompt)
    return {
        "rng_states": rng_states,
        "post_cache_rng_state": post_cache_rng_state,
        "source_gradients": list(source_gradient.split(sizes, dim=0)),
        "prompt_gradients": list(prompt_gradient.split(sizes, dim=0)),
        "loss": loss.detach(),
        "weighted_loss": weighted.detach(),
        "accuracy": accuracy.detach(),
        "effective_batch_size": int(source.shape[0]),
    }


@torch.no_grad()
def validation_loss(
    model: EviSeq,
    loader: DataLoader,
    device: torch.device,
    training: Dict[str, Any],
) -> Dict[str, float]:
    """Measure source retrieval without adding auxiliary loss to eval CE."""

    model.eval()
    names = (
        "loss_ce",
        "loss_salience",
        "prompt_retrieval_accuracy",
        "contrastive_candidates",
        "cross_residual_ratio",
        "bidirectional_gate_mean",
        "loss_evidence_contrastive",
        "evidence_top1_accuracy",
        "positive_similarity",
        "hard_negative_similarity",
        "evidence_similarity_gap",
    )
    totals = {name: 0.0 for name in names}
    totals.update({name: 0.0 for name in _SALIENCE_COUNT_METRICS})
    examples = 0
    for batch in loader:
        batch = stable._move(batch, device)
        with stable._autocast(device, training):
            outputs = model(**batch, compute_source_diagnostics=True)
        batch_size = int(batch["input_ids"].shape[0])
        for name in names:
            if name in outputs:
                totals[name] += float(outputs[name].detach().float()) * batch_size
        for name in _SALIENCE_COUNT_METRICS:
            if name in outputs:
                totals[name] += float(outputs[name].detach().float())
        examples += batch_size
    result = {f"eval_{name}": totals[name] / max(1, examples) for name in names}
    salience_precision, salience_recall, salience_f1 = _salience_scores(totals)
    result.update(
        {
            "eval_salience_precision": salience_precision,
            "eval_salience_recall": salience_recall,
            "eval_salience_f1": salience_f1,
            "eval_salience_ranking_accuracy": _salience_ranking_accuracy(totals),
        }
    )
    result["eval_examples"] = float(examples)
    return result


def _run_stage(
    model: EviSeq,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training: Dict[str, Any],
    stage: str,
    stage_epochs: int,
    epoch_offset: int,
    global_step: int,
) -> Tuple[int, int]:
    """V2 runner: evidence contrastive + optional document-level InfoNCE."""

    if stage_epochs <= 0:
        return epoch_offset, global_step
    model.set_training_stage(stage)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    use_virtual_gradcache = _uses_virtual_gradcache(model, accumulation)
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * stage_epochs)
    optimizer, scheduler = stable.build_optimizer(model, training, stage, total_steps)
    carried = getattr(model, "_carried_optimizer_moments", None)
    restored_moments = _restore_optimizer_moments(model, optimizer, carried)
    use_fp16_scaler = (
        device.type == "cuda" and bool(training.get("fp16", False)) and not bool(training.get("bf16", True))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    validation_every = int(training.get("validation_every_epochs", 0))
    optimizer.zero_grad(set_to_none=True)
    local_step = 0
    log_started = time.perf_counter()
    steps_since_log = 0
    examples_since_log = 0

    LOGGER.info(
        "Starting stage=%s epochs=%d trainable=%s restored_optimizer_states=%d evidence_hard_negatives=%d",
        stage,
        stage_epochs,
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
        restored_moments,
        model.evidence_hard_negatives if model.use_evidence_contrastive else 0,
    )
    for stage_epoch in range(1, stage_epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        metric_count = 0
        iterator = iter(loader)
        for window_start in range(1, len(loader) + 1, accumulation):
            window_size = min(accumulation, len(loader) - window_start + 1)
            batches = [stable._move(next(iterator), device) for _ in range(window_size)]
            window_end = window_start + window_size - 1

            # Document-level contrastive scale (V1)
            scale = _contrastive_scale(
                stage,
                stage_epoch,
                window_end,
                len(loader),
                model.contrastive_warmup_epochs,
            )
            model.set_contrastive_scale(scale)

            # Evidence contrastive scale (V2)
            evi_scale = _evidence_contrastive_scale(
                stage,
                stage_epoch,
                window_end,
                len(loader),
                model.evidence_contrastive_warmup_epochs,
            )
            model.set_evidence_contrastive_scale(evi_scale)

            # Document-level GradCache (V1, only if enabled)
            cache = None
            if use_virtual_gradcache:
                cache = _build_virtual_contrastive_cache(model, batches, device, training, scale)

            for microbatch_index, batch in enumerate(batches):
                if cache is not None:
                    _restore_rng_state(cache["rng_states"][microbatch_index], device)
                with stable._autocast(device, training):
                    outputs = model(
                        **batch,
                        contrastive_mode="deferred" if cache is not None else "local",
                    )
                    backward_loss = outputs["loss"] / window_size
                    if cache is not None:
                        source_surrogate = (
                            outputs["source_repr"].float() * cache["source_gradients"][microbatch_index]
                        ).sum()
                        prompt_surrogate = (
                            outputs["prompt_repr"].float() * cache["prompt_gradients"][microbatch_index]
                        ).sum()
                        backward_loss = backward_loss + source_surrogate + prompt_surrogate
                scaler.scale(backward_loss).backward()
                metric_count += 1
                examples_since_log += int(batch["input_ids"].shape[0])
                metrics = dict(outputs)
                if cache is not None:
                    weighted = cache["weighted_loss"]
                    metrics.update(
                        {
                            "loss": outputs["loss"].detach() + weighted,
                            "loss_contrastive": cache["loss"],
                            "prompt_retrieval_accuracy": cache["accuracy"],
                            "contrastive_candidates": outputs["loss"].new_tensor(cache["effective_batch_size"]),
                        }
                    )
                elif model.alignment_head is not None:
                    metrics["contrastive_candidates"] = outputs["loss"].new_tensor(batch["input_ids"].shape[0])
                for name in (*_TRAIN_AVERAGE_METRICS, *_SALIENCE_COUNT_METRICS):
                    value = metrics.get(name)
                    if value is not None and hasattr(value, "numel") and value.numel() == 1:
                        running[name] = running.get(name, 0.0) + float(value.detach().float())

            if cache is not None:
                _restore_rng_state(cache["post_cache_rng_state"], device)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            local_step += 1
            global_step += 1
            steps_since_log += 1
            if local_step % log_every == 0 or window_end == len(loader):
                divisor = max(1, metric_count)
                elapsed = max(1.0e-6, time.perf_counter() - log_started)
                _, _, salience_f1 = _salience_scores(running)
                payload = {
                    "stage": stage,
                    "epoch": epoch_offset + stage_epoch,
                    "step": global_step,
                    "loss": _rounded(running.get("loss", 0.0) / divisor),
                    "ce": _rounded(running.get("loss_ce", 0.0) / divisor),
                    "sal": _rounded(running.get("loss_salience", 0.0) / divisor),
                    "sal_f1": _rounded(salience_f1),
                    "sal_rank": _rounded(_salience_ranking_accuracy(running)),
                    # Evidence contrastive metrics (V2)
                    "evi_cl": _rounded(running.get("loss_evidence_contrastive", 0.0) / divisor),
                    "evi_acc": _rounded(running.get("evidence_top1_accuracy", 0.0) / divisor),
                    "evi_pos_sim": _rounded(running.get("positive_similarity", 0.0) / divisor),
                    "evi_neg_sim": _rounded(running.get("hard_negative_similarity", 0.0) / divisor),
                    "evi_gap": _rounded(running.get("evidence_similarity_gap", 0.0) / divisor),
                    "cross_res": _rounded(running.get("cross_residual_ratio", 0.0) / divisor),
                    "bidir": _rounded(running.get("bidirectional_gate_mean", 0.0) / divisor),
                    "step_s": _rounded(elapsed / max(1, steps_since_log)),
                    "samples_s": _rounded(examples_since_log / elapsed),
                    "grad": _rounded(float(grad_norm)),
                    "lr": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                if model.alignment_head is not None:
                    payload.update(
                        {
                            "doc_cl": _rounded(running.get("loss_contrastive", 0.0) / divisor),
                            "doc_cl_acc": _rounded(running.get("prompt_retrieval_accuracy", 0.0) / divisor),
                        }
                    )
                if stage == "interface_warmup":
                    payload["evi_scale"] = _rounded(evi_scale)
                    if model.alignment_head is not None:
                        payload["doc_cl_scale"] = _rounded(scale)
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                running.clear()
                metric_count = 0
                log_started = time.perf_counter()
                steps_since_log = 0
                examples_since_log = 0
        absolute_epoch = epoch_offset + stage_epoch
        LOGGER.info("completed epoch=%d stage=%s", absolute_epoch, stage)
        if validation_every > 0 and (absolute_epoch % validation_every == 0 or stage_epoch == stage_epochs):
            metrics = validation_loss(model, validation_loader, device, training)
            payload = {
                "epoch": absolute_epoch,
                "ce": _rounded(metrics["eval_loss_ce"]),
                "sal": _rounded(metrics["eval_loss_salience"]),
                "sal_f1": _rounded(metrics["eval_salience_f1"]),
                "sal_rank": _rounded(metrics["eval_salience_ranking_accuracy"]),
                "evi_cl": _rounded(metrics.get("eval_loss_evidence_contrastive", 0.0)),
                "evi_acc": _rounded(metrics.get("eval_evidence_top1_accuracy", 0.0)),
                "evi_gap": _rounded(metrics.get("eval_evidence_similarity_gap", 0.0)),
                "cross_res": _rounded(metrics["eval_cross_residual_ratio"]),
                "bidir": _rounded(metrics["eval_bidirectional_gate_mean"]),
                "examples": int(metrics["eval_examples"]),
            }
            if model.alignment_head is not None:
                payload["doc_cl_acc"] = _rounded(metrics["eval_prompt_retrieval_accuracy"])
            LOGGER.info("validation %s", json.dumps(payload, separators=(",", ":")))
    model._carried_optimizer_moments = (  # type: ignore[attr-defined]
        _capture_optimizer_moments(model, optimizer) if stage == "interface_warmup" else {}
    )
    return epoch_offset + stage_epochs, global_step


# ---------------------------------------------------------------------------
# Phase 3: Candidate ranking stage
# ---------------------------------------------------------------------------


def _run_ranking_stage(
    model: EviSeq,
    ranking_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training: Dict[str, Any],
    stage_epochs: int,
    epoch_offset: int,
    global_step: int,
) -> Tuple[int, int]:
    """Phase 3: CE on reference + BRIO-like pairwise ranking on candidates."""

    if stage_epochs <= 0:
        return epoch_offset, global_step

    stage = "ranking_finetune"
    model.set_training_stage(stage)
    accumulation = int(training.get("ranking_gradient_accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(ranking_loader) / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * stage_epochs)
    optimizer, scheduler = stable.build_optimizer(model, training, stage, total_steps)

    use_fp16_scaler = (
        device.type == "cuda" and bool(training.get("fp16", False)) and not bool(training.get("bf16", True))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    validation_every = int(training.get("validation_every_epochs", 0))
    optimizer.zero_grad(set_to_none=True)
    local_step = 0

    LOGGER.info(
        "Starting ranking_finetune epochs=%d trainable=%s ranking_weight=%.4f",
        stage_epochs,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        model.ranking_weight,
    )

    for stage_epoch in range(1, stage_epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        metric_count = 0
        iterator = iter(ranking_loader)
        processed_batches = 0
        interval_started = time.perf_counter()

        while processed_batches < len(ranking_loader):
            # Scale the final partial window by its real size, not by the
            # configured accumulation factor.
            window_size = min(accumulation, len(ranking_loader) - processed_batches)
            for _ in range(window_size):
                batch = stable._move(next(iterator), device)
                with stable._autocast(device, training):
                    outputs = model.forward_ranking(**batch)
                    loss = outputs["loss"] / window_size
                scaler.scale(loss).backward()
                processed_batches += 1
                metric_count += 1

                for name in _RANKING_METRICS:
                    value = outputs.get(name)
                    if value is not None and hasattr(value, "numel") and value.numel() == 1:
                        running[name] = running.get(name, 0.0) + float(value.detach().float())

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            local_step += 1
            global_step += 1

            if local_step % log_every == 0 or processed_batches == len(ranking_loader):
                divisor = max(1, metric_count)
                elapsed = max(1.0e-6, time.perf_counter() - interval_started)
                payload = {
                    "stage": stage,
                    "epoch": epoch_offset + stage_epoch,
                    "step": global_step,
                    "loss": _rounded(running.get("loss", 0.0) / divisor),
                    "ce": _rounded(running.get("loss_ce", 0.0) / divisor),
                    "rank": _rounded(running.get("loss_rank", 0.0) / divisor),
                    "sal": _rounded(running.get("loss_salience", 0.0) / divisor),
                    "pair_acc": _rounded(running.get("candidate_pair_accuracy", 0.0) / divisor),
                    "pairs": int(round(running.get("candidate_pair_count", 0.0))),
                    "rank_ce": _rounded(running.get("ranking_to_ce_ratio", 0.0) / divisor),
                    "cross_res": _rounded(running.get("cross_residual_ratio", 0.0) / divisor),
                    "bidir": _rounded(running.get("bidirectional_gate_mean", 0.0) / divisor),
                    "grad": _rounded(float(grad_norm)),
                    "step_s": _rounded(elapsed / divisor),
                    "lr": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                running.clear()
                metric_count = 0
                interval_started = time.perf_counter()

        absolute_epoch = epoch_offset + stage_epoch
        LOGGER.info("completed epoch=%d stage=%s", absolute_epoch, stage)
        if validation_every > 0 and (absolute_epoch % validation_every == 0 or stage_epoch == stage_epochs):
            metrics = validation_loss(model, validation_loader, device, training)
            payload = {
                "epoch": absolute_epoch,
                "ce": _rounded(metrics["eval_loss_ce"]),
                "sal_f1": _rounded(metrics["eval_salience_f1"]),
                "examples": int(metrics["eval_examples"]),
            }
            LOGGER.info("validation %s", json.dumps(payload, separators=(",", ":")))

    return epoch_offset + stage_epochs, global_step


def _parameter_component(name: str) -> str:
    if (
        name.startswith("adapter.")
        or name.startswith("alignment_head.")
        or name.startswith("evidence_contrastive_head.")
    ):
        return "adapter"
    if name.startswith("encoder.") and any(
        marker in name for marker in ("evidence_norm", "evidence_head", "evidence_view_gate", "generic_token_gate")
    ):
        return "adapter"
    if ".cross_attn" in name or name.endswith(".cross_gate") or name.endswith(".memory_router_logits"):
        return "cross_attention"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    raise ValueError(f"Unclassified trainable parameter: {name}")


def build_experiment(
    config: Dict[str, Any],
    *,
    include_train: bool = True,
):
    encoder_tokenizer, decoder_tokenizer = stable._tokenizers(config)
    model = EviSeq(config)
    if include_train:
        parameter_manifest = build_parameter_manifest(model, config)
        config.setdefault("_runtime", {})["parameter_manifest"] = parameter_manifest
        output_dir = Path(config["experiment"]["output_dir"])
        if not output_dir.is_dir():
            raise RuntimeError("Training output directory must be prepared before constructing the EviSeq experiment")
        temporary = output_dir / "parameter_manifest.json.tmp"
        temporary.write_text(
            json.dumps(parameter_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir / "parameter_manifest.json")
    data = config["data"]
    limits = config.get("limits", {})
    train_dataset = None
    if include_train:
        train_dataset = SummarizationDataset(
            data["train_file"],
            encoder_tokenizer,
            decoder_tokenizer,
            data,
            max_examples=int(limits.get("max_train_examples", 0)),
            precompute_evidence=bool(data.get("precompute_evidence", True)),
        )
    validation_dataset = SummarizationDataset(
        data["validation_file"],
        encoder_tokenizer,
        decoder_tokenizer,
        data,
        max_examples=int(limits.get("max_validation_examples", 0)),
        precompute_evidence=False,
    )
    return model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset


def train(config_path: str, overwrite_output_dir: bool = False) -> Path:
    config = load_config(config_path)
    report = audit(config)
    output_dir = Path(config["experiment"]["output_dir"])
    preflight_dir = output_dir.parent / "data_audits"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    (preflight_dir / f"{config['experiment']['name']}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    originals = {
        "load_config": stable.load_config,
        "LLM2SeqV2": stable.LLM2SeqV2,
        "build_experiment": stable.build_experiment,
        "_parameter_component": stable._parameter_component,
        "_run_stage": stable._run_stage,
        "validation_loss": stable.validation_loss,
        "LOGGER": stable.LOGGER,
    }
    stable.load_config = load_config
    stable.LLM2SeqV2 = EviSeq
    stable.build_experiment = build_experiment
    stable._parameter_component = _parameter_component
    stable._run_stage = _run_stage
    stable.validation_loss = validation_loss
    stable.LOGGER = LOGGER
    try:
        last_path = stable.train(config_path, overwrite_output_dir)

        # ``stable.train`` enriches its own config object with the exact
        # parameter manifest before writing the Phase-2 checkpoint.  Reload
        # that resolved sidecar here so the separately saved ranked checkpoint
        # embeds the identical manifest and can pass strict evaluation.
        phase2_config_path = output_dir / "resolved_config.yaml"
        config = load_config(phase2_config_path)

        # --- Phase 3: generate candidates and ranking fine-tune in one run ---
        ranking = config.get("ranking", {})
        ranking_enabled = bool(ranking.get("enabled", False))
        ranking_epochs = int(config.get("training", {}).get("ranking_finetune_epochs", 0))
        candidates_file = str(ranking.get("candidates_file", "")).strip()

        if ranking_enabled and ranking_epochs > 0:
            output_complete = output_dir / "COMPLETE"
            phase2_complete = output_dir / "PHASE2_COMPLETE"
            output_complete.replace(phase2_complete)
            phase3_marker = output_dir / "RUNNING"
            phase3_marker.write_text(f"pid={os.getpid()}\nstage=candidate_generation\n", encoding="utf-8")

            LOGGER.info(
                "Starting Phase 3: candidates=%d max_documents=%d output=%s",
                int(ranking["num_candidates"]),
                int(ranking.get("max_examples", 0)),
                candidates_file,
            )
            training = config["training"]

            # Candidate generation is part of the same atomic pipeline.  The
            # Phase-2 last.pt remains untouched as an explicit control.
            from .generate_candidates import generate_candidates

            candidate_stats = generate_candidates(
                str(phase2_config_path),
                str(last_path),
                candidates_file,
                max_examples=None,
                seed=int(training.get("seed", 42)),
                num_candidates=int(ranking["num_candidates"]),
            )
            if int(candidate_stats.get("generated", 0)) <= 0:
                raise RuntimeError("Candidate generation produced no ranking examples")
            phase3_marker.write_text(f"pid={os.getpid()}\nstage=ranking_finetune\n", encoding="utf-8")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reload checkpoint
            from .checkpoint import load_last_checkpoint

            model = EviSeq(config)
            phase2_payload = load_last_checkpoint(model, last_path)

            device = stable._device()
            model.to(device)

            encoder_tokenizer, decoder_tokenizer = stable._tokenizers(config)
            data = config["data"]

            # Build ranking dataset
            ranking_dataset = CandidateRankingDataset(
                candidates_file,
                encoder_tokenizer,
                decoder_tokenizer,
                data,
                max_candidates=int(ranking["num_candidates"]),
            )

            prompt_length = len(decoder_seed_ids(decoder_tokenizer, data))
            ranking_collator = CandidateRankingCollator(
                encoder_pad_id=encoder_tokenizer.pad_token_id,
                decoder_pad_id=decoder_tokenizer.pad_token_id,
                max_source_length=int(data["max_source_length"]),
                max_decoder_length=int(data["max_target_length"]) + prompt_length - 1,
            )

            ranking_loader = DataLoader(
                ranking_dataset,
                batch_size=int(training.get("ranking_batch_size", 2)),
                shuffle=True,
                collate_fn=ranking_collator,
                num_workers=int(training.get("num_workers", 4)),
                pin_memory=device.type == "cuda",
                persistent_workers=int(training.get("num_workers", 4)) > 0,
                drop_last=False,
            )

            # Build validation loader for periodic validation
            validation_dataset = SummarizationDataset(
                data["validation_file"],
                encoder_tokenizer,
                decoder_tokenizer,
                data,
                max_examples=int(config.get("limits", {}).get("max_validation_examples", 0)),
                precompute_evidence=False,
            )

            val_collator = Seq2SeqCollator(
                encoder_pad_id=encoder_tokenizer.pad_token_id,
                decoder_pad_id=decoder_tokenizer.pad_token_id,
                max_source_length=int(data["max_source_length"]),
                max_decoder_length=int(data["max_target_length"]) + prompt_length - 1,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(training.get("validation_batch_size", 32)),
                shuffle=False,
                collate_fn=val_collator,
                num_workers=int(training.get("validation_num_workers", 2)),
                pin_memory=device.type == "cuda",
                persistent_workers=int(training.get("validation_num_workers", 2)) > 0,
            )

            # Exact Phase-2 counters come from the checkpoint; never infer them
            # from dataset size or accumulation arithmetic.
            epoch_offset = int(phase2_payload["epoch"])
            global_step = int(phase2_payload["global_step"])

            epoch_offset, global_step = _run_ranking_stage(
                model,
                ranking_loader,
                validation_loader,
                device,
                training,
                ranking_epochs,
                epoch_offset,
                global_step,
            )

            # Preserve Phase-2 last.pt.  The final ranked model has its own
            # complete checkpoint directory and matching config sidecars.
            from .checkpoint import save_last_checkpoint

            manifest = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            manifest["ranking_candidates"] = {
                "path": str(Path(candidates_file).resolve()),
                "num_examples": int(candidate_stats["generated"]),
                "sha256": str(candidate_stats["candidate_file_sha256"]),
                "phase2_epoch": int(candidate_stats["checkpoint_epoch"]),
                "phase2_global_step": int(candidate_stats["checkpoint_global_step"]),
            }
            ranking_dir = output_dir / "ranking"
            ranking_dir.mkdir(parents=True, exist_ok=True)
            ranked_last_path = ranking_dir / "last.pt"
            save_last_checkpoint(model, ranked_last_path, config, epoch_offset, global_step, manifest)
            shutil.copy2(output_dir / "resolved_config.yaml", ranking_dir / "resolved_config.yaml")
            (ranking_dir / "data_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            parameter_manifest = output_dir / "parameter_manifest.json"
            if parameter_manifest.is_file():
                shutil.copy2(parameter_manifest, ranking_dir / parameter_manifest.name)
            completion = {
                "checkpoint": str(ranked_last_path),
                "phase2_checkpoint": str(last_path),
                "epoch": epoch_offset,
                "global_step": global_step,
                "candidate_examples": int(candidate_stats["generated"]),
                "candidates_per_document": int(ranking["num_candidates"]),
            }
            (ranking_dir / "COMPLETE").write_text(json.dumps(completion), encoding="utf-8")
            phase3_marker.unlink(missing_ok=True)
            output_complete.write_text(json.dumps(completion), encoding="utf-8")
            LOGGER.info(
                "Ranking fine-tune complete: phase2=%s ranked=%s epoch=%d step=%d",
                last_path,
                ranked_last_path,
                epoch_offset,
                global_step,
            )
            return ranked_last_path

        return last_path
    finally:
        for name, value in originals.items():
            setattr(stable, name, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args.config, args.overwrite_output_dir)


if __name__ == "__main__":
    main()
