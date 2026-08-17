"""EviSeq two-stage training entry point."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..configuration import load_config, resolve_data_path
from ..data.dataset import (
    Text2TextDataset,
)
from ..modeling.architecture import EviSeq
from . import engine as stable
from .checkpoint import save_configured_epoch_checkpoints
from .objectives import exact_duplicate_mask, info_nce_loss

LOGGER = logging.getLogger("eviseq")

_TRAIN_AVERAGE_METRICS = (
    "loss",
    "loss_ce",
    "loss_salience",
    "loss_contrastive",
    "loss_evidence_contrastive",
    "weighted_evidence_contrastive",
    "evidence_contrastive_to_ce_ratio",
    "evidence_hard_negatives",
    "prompt_retrieval_accuracy",
    "contrastive_examples",
    "cross_residual_ratio",
    "bridge_projection_residual_ratio",
    "positive_attention_prior",
    "negative_attention_prior",
    "positive_attention_prior_gap",
    "bridge_salience_gate",
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
    model: torch.nn.Module,
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
    raw_model = stable.unwrap_model(model)
    loss, accuracy = info_nce_loss(
        source,
        prompt,
        raw_model.contrastive_temperature,
        duplicate_mask=_virtual_duplicate_mask(batches),
    )
    weighted = raw_model.contrastive_weight * float(scale) * loss
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
        "contrastive_examples",
        "cross_residual_ratio",
        "bridge_projection_residual_ratio",
        "positive_attention_prior",
        "negative_attention_prior",
        "positive_attention_prior_gap",
        "bridge_salience_gate",
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
    checkpoint_dir: Path | None = None,
    checkpoint_config: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
    """Train evidence contrastive with optional document-level InfoNCE."""

    if stage_epochs <= 0:
        return epoch_offset, global_step
    raw_model = stable.unwrap_model(model)
    distributed = stable.distributed_context()
    raw_model.set_training_stage(stage)
    active_model: torch.nn.Module = raw_model
    if distributed.enabled:
        active_model = torch.nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    use_virtual_gradcache = _uses_virtual_gradcache(raw_model, accumulation)
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * stage_epochs)
    optimizer, scheduler = stable.build_optimizer(raw_model, training, stage, total_steps)
    carried = getattr(raw_model, "_carried_optimizer_moments", None)
    restored_moments = _restore_optimizer_moments(raw_model, optimizer, carried)
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
        f"{sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad):,}",
        restored_moments,
        raw_model.evidence_hard_negatives if raw_model.use_evidence_contrastive else 0,
    )
    for stage_epoch in range(1, stage_epochs + 1):
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch_offset + stage_epoch - 1)
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_offset + stage_epoch - 1)
        model.train()
        running: Dict[str, float] = {}
        metric_count = 0
        iterator = iter(loader)
        for window_start in range(1, len(loader) + 1, accumulation):
            window_size = min(accumulation, len(loader) - window_start + 1)
            batches = [stable._move(next(iterator), device) for _ in range(window_size)]
            window_end = window_start + window_size - 1

            # Document-level contrastive scale.
            scale = _contrastive_scale(
                stage,
                stage_epoch,
                window_end,
                len(loader),
                raw_model.contrastive_warmup_epochs,
            )
            raw_model.set_contrastive_scale(scale)

            # Evidence contrastive scale.
            evi_scale = _evidence_contrastive_scale(
                stage,
                stage_epoch,
                window_end,
                len(loader),
                raw_model.evidence_contrastive_warmup_epochs,
            )
            raw_model.set_evidence_contrastive_scale(evi_scale)

            # Document-level GradCache, only if enabled.
            cache = None
            if use_virtual_gradcache:
                cache = _build_virtual_contrastive_cache(active_model, batches, device, training, scale)

            for microbatch_index, batch in enumerate(batches):
                if cache is not None:
                    _restore_rng_state(cache["rng_states"][microbatch_index], device)
                is_last_microbatch = microbatch_index + 1 == window_size
                sync_context = (
                    active_model.no_sync()
                    if distributed.enabled and not is_last_microbatch
                    else nullcontext()
                )
                with sync_context:
                    with stable._autocast(device, training):
                        outputs = active_model(
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
                            "contrastive_examples": outputs["loss"].new_tensor(cache["effective_batch_size"]),
                        }
                    )
                elif raw_model.alignment_head is not None:
                    metrics["contrastive_examples"] = outputs["loss"].new_tensor(batch["input_ids"].shape[0])
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
                    "evi_cl": _rounded(running.get("loss_evidence_contrastive", 0.0) / divisor),
                    "evi_w": _rounded(running.get("weighted_evidence_contrastive", 0.0) / divisor),
                    "evi_ce_ratio": _rounded(running.get("evidence_contrastive_to_ce_ratio", 0.0) / divisor),
                    "evi_k": int(round(running.get("evidence_hard_negatives", 0.0) / divisor)),
                    "evi_acc": _rounded(running.get("evidence_top1_accuracy", 0.0) / divisor),
                    "evi_pos_sim": _rounded(running.get("positive_similarity", 0.0) / divisor),
                    "evi_neg_sim": _rounded(running.get("hard_negative_similarity", 0.0) / divisor),
                    "evi_gap": _rounded(running.get("evidence_similarity_gap", 0.0) / divisor),
                    "evi_queries": int(round(running.get("evidence_valid_examples", 0.0) / divisor)),
                    "cross_res": _rounded(running.get("cross_residual_ratio", 0.0) / divisor),
                    "bridge_proj_res": _rounded(running.get("bridge_projection_residual_ratio", 0.0) / divisor),
                    "attn_pos_prior": _rounded(running.get("positive_attention_prior", 0.0) / divisor),
                    "attn_prior_gap": _rounded(running.get("positive_attention_prior_gap", 0.0) / divisor),
                    "bridge_sal_gate": _rounded(running.get("bridge_salience_gate", 0.0) / divisor),
                    "bidir": _rounded(running.get("bidirectional_gate_mean", 0.0) / divisor),
                    "step_s": _rounded(elapsed / max(1, steps_since_log)),
                    "samples_s": _rounded(examples_since_log / elapsed),
                    "grad": _rounded(float(grad_norm)),
                    "lr": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                if raw_model.alignment_head is not None:
                    payload.update(
                        {
                            "doc_cl": _rounded(running.get("loss_contrastive", 0.0) / divisor),
                            "doc_cl_acc": _rounded(running.get("prompt_retrieval_accuracy", 0.0) / divisor),
                        }
                    )
                if stage == "interface_warmup":
                    payload["evi_scale"] = _rounded(evi_scale)
                    if raw_model.alignment_head is not None:
                        payload["doc_cl_scale"] = _rounded(scale)
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                running.clear()
                metric_count = 0
                log_started = time.perf_counter()
                steps_since_log = 0
                examples_since_log = 0
        absolute_epoch = epoch_offset + stage_epoch
        LOGGER.info("completed epoch=%d stage=%s", absolute_epoch, stage)
        save_best = bool((checkpoint_config or {}).get("checkpoint", {}).get("save_best", False))
        scheduled_validation = validation_every > 0 and (
            absolute_epoch % validation_every == 0 or stage_epoch == stage_epochs
        )
        metrics = None
        if distributed.is_main and (save_best or scheduled_validation):
            metrics = validation_loss(raw_model, validation_loader, device, training)
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
                "bridge_proj_res": _rounded(metrics.get("eval_bridge_projection_residual_ratio", 0.0)),
                "attn_pos_prior": _rounded(metrics.get("eval_positive_attention_prior", 0.0)),
                "attn_prior_gap": _rounded(metrics.get("eval_positive_attention_prior_gap", 0.0)),
                "bridge_sal_gate": _rounded(metrics.get("eval_bridge_salience_gate", 0.0)),
                "bidir": _rounded(metrics["eval_bidirectional_gate_mean"]),
                "examples": int(metrics["eval_examples"]),
            }
            if raw_model.alignment_head is not None:
                payload["doc_cl_acc"] = _rounded(metrics["eval_prompt_retrieval_accuracy"])
            LOGGER.info("validation %s", json.dumps(payload, separators=(",", ":")))
        if distributed.is_main and checkpoint_dir is not None and checkpoint_config is not None:
            saved = save_configured_epoch_checkpoints(
                raw_model,
                checkpoint_dir,
                checkpoint_config,
                absolute_epoch,
                global_step,
                metrics,
            )
            if saved:
                LOGGER.info("checkpoint %s", json.dumps(saved, separators=(",", ":")))
        stable.distributed_barrier(distributed)
    raw_model._carried_optimizer_moments = (  # type: ignore[attr-defined]
        _capture_optimizer_moments(raw_model, optimizer) if stage == "interface_warmup" else {}
    )
    del active_model
    return epoch_offset + stage_epochs, global_step


def _parameter_component(name: str) -> str:
    if (
        name.startswith("adapter.")
        or name.startswith("alignment_head.")
        or name.startswith("evidence_contrastive_head.")
        or name.startswith("prompt_conditioned_evidence_head.")
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
    data = config["data"]
    limits = config.get("limits", {})
    train_dataset = None
    if include_train:
        train_dataset = Text2TextDataset(
            resolve_data_path(data["train_file"], config),
            encoder_tokenizer,
            decoder_tokenizer,
            data,
            max_examples=int(limits.get("max_train_examples", 0)),
            precompute_evidence=bool(data.get("precompute_evidence", True)),
        )
    validation_dataset = Text2TextDataset(
        resolve_data_path(data["validation_file"], config),
        encoder_tokenizer,
        decoder_tokenizer,
        data,
        max_examples=int(limits.get("max_validation_examples", 0)),
        precompute_evidence=False,
    )
    return model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset


def train(
    config_path: str,
    overwrite_output_dir: bool = False,
    init_checkpoint: str = "",
    output_dir: str = "",
    allow_partial_init: bool = False,
) -> Path:
    originals = {
        "load_config": stable.load_config,
        "RuntimeModel": stable.RuntimeModel,
        "build_experiment": stable.build_experiment,
        "_parameter_component": stable._parameter_component,
        "_run_stage": stable._run_stage,
        "validation_loss": stable.validation_loss,
        "LOGGER": stable.LOGGER,
    }
    stable.load_config = load_config
    stable.RuntimeModel = EviSeq
    stable.build_experiment = build_experiment
    stable._parameter_component = _parameter_component
    stable._run_stage = _run_stage
    stable.validation_loss = validation_loss
    stable.LOGGER = LOGGER
    try:
        return stable.train(
            config_path,
            overwrite_output_dir,
            init_checkpoint,
            output_dir,
            allow_partial_init,
        )
    finally:
        for name, value in originals.items():
            setattr(stable, name, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-partial-init", action="store_true")
    args = parser.parse_args()
    train(
        args.config,
        args.overwrite_output_dir,
        args.init_checkpoint,
        args.output_dir,
        args.allow_partial_init,
    )


if __name__ == "__main__":
    main()
