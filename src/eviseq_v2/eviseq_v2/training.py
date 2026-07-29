"""EviSeq V2 training entry point.

Extends the V1 training with:
- Evidence-focused hard contrastive learning (replaces document-level InfoNCE)
- Phase 3: BRIO-like candidate ranking fine-tune
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

    LOGGER.info(
        "Starting stage=%s epochs=%d trainable=%s restored_optimizer_states=%d contrastive_batch=%d",
        stage,
        stage_epochs,
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
        restored_moments,
        int(training.get("batch_size", 1)) * accumulation if model.alignment_head is not None else 0,
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
            if model.alignment_head is not None and model.contrastive_across_accumulation:
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
            if local_step % log_every == 0 or window_end == len(loader):
                divisor = max(1, metric_count)
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
                    "cl": _rounded(running.get("loss_contrastive", 0.0) / divisor),
                    "cl_acc": _rounded(running.get("prompt_retrieval_accuracy", 0.0) / divisor),
                    "cl_n": int(round(running.get("contrastive_candidates", 0.0) / divisor)),
                    # Evidence contrastive metrics (V2)
                    "evi_cl": _rounded(running.get("loss_evidence_contrastive", 0.0) / divisor),
                    "evi_acc": _rounded(running.get("evidence_top1_accuracy", 0.0) / divisor),
                    "evi_pos_sim": _rounded(running.get("positive_similarity", 0.0) / divisor),
                    "evi_neg_sim": _rounded(running.get("hard_negative_similarity", 0.0) / divisor),
                    "evi_gap": _rounded(running.get("evidence_similarity_gap", 0.0) / divisor),
                    "cross_res": _rounded(running.get("cross_residual_ratio", 0.0) / divisor),
                    "bidir": _rounded(running.get("bidirectional_gate_mean", 0.0) / divisor),
                    "grad": _rounded(float(grad_norm)),
                    "lr": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                if stage == "interface_warmup":
                    payload["cl_scale"] = _rounded(scale)
                    payload["evi_scale"] = _rounded(evi_scale)
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                running.clear()
                metric_count = 0
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
                "cl_acc": _rounded(metrics["eval_prompt_retrieval_accuracy"]),
                "cl_n": int(round(metrics["eval_contrastive_candidates"])),
                "evi_cl": _rounded(metrics.get("eval_loss_evidence_contrastive", 0.0)),
                "evi_acc": _rounded(metrics.get("eval_evidence_top1_accuracy", 0.0)),
                "evi_gap": _rounded(metrics.get("eval_evidence_similarity_gap", 0.0)),
                "cross_res": _rounded(metrics["eval_cross_residual_ratio"]),
                "bidir": _rounded(metrics["eval_bidirectional_gate_mean"]),
                "examples": int(metrics["eval_examples"]),
            }
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
    accumulation = int(training.get("gradient_accumulation_steps", 1))
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
        accumulation_count = 0

        for batch_index, batch in enumerate(ranking_loader, start=1):
            batch = stable._move(batch, device)
            with stable._autocast(device, training):
                outputs = model.forward_ranking(**batch)
                loss = outputs["loss"] / accumulation
            scaler.scale(loss).backward()
            accumulation_count += 1
            metric_count += 1

            for name in _RANKING_METRICS:
                value = outputs.get(name)
                if value is not None and hasattr(value, "numel") and value.numel() == 1:
                    running[name] = running.get(name, 0.0) + float(value.detach().float())

            update = accumulation_count == accumulation or batch_index == len(ranking_loader)
            if not update:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation_count = 0
            local_step += 1
            global_step += 1

            if local_step % log_every == 0 or batch_index == len(ranking_loader):
                divisor = max(1, metric_count)
                payload = {
                    "stage": stage,
                    "epoch": epoch_offset + stage_epoch,
                    "step": global_step,
                    "loss": _rounded(running.get("loss", 0.0) / divisor),
                    "ce": _rounded(running.get("loss_ce", 0.0) / divisor),
                    "rank": _rounded(running.get("loss_rank", 0.0) / divisor),
                    "sal": _rounded(running.get("loss_salience", 0.0) / divisor),
                    "pair_acc": _rounded(running.get("candidate_pair_accuracy", 0.0) / divisor),
                    "cross_res": _rounded(running.get("cross_residual_ratio", 0.0) / divisor),
                    "bidir": _rounded(running.get("bidirectional_gate_mean", 0.0) / divisor),
                    "grad": _rounded(float(grad_norm)),
                    "lr": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                running.clear()
                metric_count = 0

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

        # --- Phase 3: Ranking fine-tune (if enabled) ---
        ranking = config.get("ranking", {})
        ranking_enabled = bool(ranking.get("enabled", False))
        ranking_epochs = int(config.get("training", {}).get("ranking_finetune_epochs", 0))
        candidates_file = str(ranking.get("candidates_file", "")).strip()

        if ranking_enabled and ranking_epochs > 0 and candidates_file:
            LOGGER.info("Starting Phase 3: ranking fine-tune from candidates=%s", candidates_file)
            training = config["training"]

            # Reload checkpoint
            from .checkpoint import load_last_checkpoint

            model = EviSeq(config)
            load_last_checkpoint(model, last_path)

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
                batch_size=max(1, int(training.get("batch_size", 32)) // 4),  # Smaller batch for ranking
                shuffle=True,
                collate_fn=ranking_collator,
                num_workers=int(training.get("num_workers", 4)),
                pin_memory=device.type == "cuda",
                drop_last=False,
            )

            # Build validation loader for periodic validation
            _, _, _, _, validation_dataset = build_experiment(config, include_train=False)
            from .data import decoder_seed_ids as _dsi

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
            )

            # Get epoch/step from checkpoint
            import yaml

            resolved = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
            warmup_epochs = int(resolved.get("training", {}).get("interface_warmup_epochs", 2))
            full_epochs = int(resolved.get("training", {}).get("full_finetune_epochs", 6))
            epoch_offset = warmup_epochs + full_epochs

            # Estimate global step
            from .data import read_jsonl

            train_count = len(
                read_jsonl(data["train_file"], max_examples=int(config.get("limits", {}).get("max_train_examples", 0)))
            )
            steps_per_epoch = math.ceil(train_count / int(training.get("batch_size", 32))) // int(
                training.get("gradient_accumulation_steps", 1)
            )
            global_step = steps_per_epoch * epoch_offset

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

            # Save updated checkpoint
            from .checkpoint import save_last_checkpoint
            from .data import dataset_fingerprint

            limits = config.get("limits", {})
            manifest = {
                split: dataset_fingerprint(
                    data[f"{split}_file"],
                    int(limits.get(f"max_{split}_examples", 0)),
                )
                for split in ("train", "validation", "test")
            }
            save_last_checkpoint(model, last_path, config, epoch_offset, global_step, manifest)
            LOGGER.info("Ranking fine-tune complete: last=%s epoch=%d step=%d", last_path, epoch_offset, global_step)

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
