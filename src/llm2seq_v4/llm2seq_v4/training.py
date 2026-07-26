"""Two-stage full fine-tuning for the prospective-summary bridge."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import save_last_checkpoint
from .config import load_config
from .data import (
    Seq2SeqCollator,
    SummarizationDataset,
    dataset_fingerprint,
    decoder_seed_ids,
)
from .model import LLM2SeqV4

LOGGER = logging.getLogger("llm2seq_v4")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_locked_data_manifest(
    config: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, str]:
    """Require exact T5Gemma data parity for every complete split.

    Smoke and pilot configs intentionally cap their splits, so those entries
    are marked partial rather than compared with a full-split fingerprint.
    """
    locked = config.get("benchmark", {}).get("data", {})
    limits = config.get("limits", {})
    status: Dict[str, str] = {}
    for split in ("train", "validation", "test"):
        if int(limits.get(f"max_{split}_examples", 0)) > 0:
            status[split] = "partial_not_comparable"
            continue
        expected = locked.get(split, {})
        actual = manifest.get(split, {})
        if not expected:
            status[split] = "unlocked"
            continue
        matches = int(actual.get("num_examples", -1)) == int(expected.get("num_examples", -2)) and str(
            actual.get("sha256", "")
        ) == str(expected.get("sha256", "missing"))
        if not matches:
            raise RuntimeError(
                f"{split} data differs from the locked T5Gemma split: actual={actual}, expected={expected}"
            )
        status[split] = "exact_match"
    return status


def verify_declared_parameter_budget(
    config: Dict[str, Any],
    parameter_summary: Dict[str, int],
) -> Dict[str, int | bool | str]:
    """Fail before training if even the conservative total misses the target."""

    target = config.get("benchmark", {}).get("parameters", {})
    target_count = int(target.get("target_declared_parameters", 0))
    total = int(parameter_summary.get("total", 0))
    alignment = int(parameter_summary.get("alignment_head", 0))
    deployable = total - alignment
    if total <= 0 or deployable <= 0:
        raise RuntimeError(f"Invalid parameter summary: {parameter_summary}")
    if target_count > 0 and total >= target_count:
        raise RuntimeError(
            f"Candidate total parameters ({total:,}) are not below the declared "
            f"{target.get('target_name', 'baseline')} target ({target_count:,})"
        )
    return {
        "target_name": str(target.get("target_name", "")),
        "target_declared_parameters": target_count,
        "candidate_total_parameters": total,
        "candidate_deployable_parameters": deployable,
        "total_below_declared_target": target_count <= 0 or total < target_count,
    }


def _tokenizers(config: Dict[str, Any]) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer

    model = config["model"]
    encoder_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(model.get("encoder_trust_remote_code", True)),
    }
    if model.get("encoder_revision") is not None:
        encoder_kwargs["revision"] = str(model["encoder_revision"])
    encoder = AutoTokenizer.from_pretrained(model["encoder_name"], **encoder_kwargs)
    decoder = AutoTokenizer.from_pretrained(model["decoder_name"], trust_remote_code=True)
    for tokenizer in (encoder, decoder):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither PAD nor EOS")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return encoder, decoder


def build_experiment(
    config: Dict[str, Any],
    *,
    include_train: bool = True,
) -> Tuple[LLM2SeqV4, Any, Any, SummarizationDataset | None, SummarizationDataset | None]:
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = LLM2SeqV4(config)
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
        precompute_evidence=bool(data.get("precompute_validation_evidence", False)),
    )
    return model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset


def _collator(
    config: Dict[str, Any],
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
) -> Seq2SeqCollator:
    data = config["data"]
    prompt_length = len(decoder_seed_ids(decoder_tokenizer, data))
    return Seq2SeqCollator(
        encoder_pad_id=encoder_tokenizer.pad_token_id,
        decoder_pad_id=decoder_tokenizer.pad_token_id,
        max_source_length=int(data["max_source_length"]),
        max_decoder_length=int(data["max_target_length"]) + prompt_length - 1,
    )


def _parameter_component(name: str) -> str:
    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("alignment_head."):
        return "adapter"  # alignment head trains at adapter LR
    if ".cross_attn" in name or name.endswith(".cross_gate") or ".memory_router" in name:
        return "cross_attention"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    raise ValueError(f"Unclassified trainable parameter: {name}")


def _component_lrs(training: Dict[str, Any], stage: str) -> Dict[str, float]:
    if stage == "interface_warmup":
        return {
            "adapter": float(training.get("warmup_adapter_lr", 1e-4)),
            "cross_attention": float(training.get("warmup_cross_attention_lr", 1e-4)),
        }
    return {
        "encoder": float(training.get("full_encoder_lr", 1.2e-5)),
        "adapter": float(training.get("full_adapter_lr", 3e-5)),
        "decoder": float(training.get("full_decoder_lr", 1e-5)),
        "cross_attention": float(training.get("full_cross_attention_lr", 5e-5)),
    }


def build_optimizer(
    model: LLM2SeqV4,
    training: Dict[str, Any],
    stage: str,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    rates = _component_lrs(training, stage)
    grouped: Dict[Tuple[str, bool], List[torch.nn.Parameter]] = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        component = _parameter_component(name)
        if component not in rates:
            raise RuntimeError(f"{component} is unexpectedly trainable during {stage}: {name}")
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or name.endswith("_gate")
            or name.endswith(".global_logits")
        )
        grouped.setdefault((component, no_decay), []).append(parameter)
    if not grouped:
        raise RuntimeError(f"No trainable parameters for stage {stage}")
    parameter_groups = []
    for (component, no_decay), parameters in sorted(grouped.items()):
        parameter_groups.append(
            {
                "params": parameters,
                "lr": rates[component],
                "initial_lr": rates[component],
                "weight_decay": 0.0 if no_decay else float(training.get("weight_decay", 0.03)),
                "component": component,
            }
        )
    optimizer = AdamW(
        parameter_groups,
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.95))),
        eps=float(training.get("adam_epsilon", 1e-8)),
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.08)))
    min_ratio = float(training.get("min_lr_ratio", 0.0))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_ratio + (1.0 - min_ratio) * cosine

    return optimizer, LambdaLR(optimizer, schedule)


def _autocast(device: torch.device, training: Dict[str, Any]):
    enabled = device.type == "cuda" and (bool(training.get("bf16", True)) or bool(training.get("fp16", False)))
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if bool(training.get("bf16", True)) else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _contrastive_scale(
    stage: str,
    stage_epoch: int,
    batch_index: int,
    batches_per_epoch: int,
    contrastive_warmup_epochs: int,
) -> float:
    """Linearly ramp contrastive objectives during interface warm-up only."""
    if contrastive_warmup_epochs <= 0:
        return 1.0
    if stage != "interface_warmup":
        return 1.0
    progress = float(stage_epoch - 1) + float(batch_index) / max(1, batches_per_epoch)
    return min(1.0, progress / float(contrastive_warmup_epochs))


def _linear_curriculum(
    progress_epochs: float,
    start: float,
    end: float,
    anneal_epochs: float,
) -> float:
    """Interpolate a train-only curriculum by fractional global epoch."""

    if anneal_epochs <= 0.0:
        return float(end)
    fraction = min(1.0, max(0.0, float(progress_epochs) / float(anneal_epochs)))
    return float(start) + fraction * (float(end) - float(start))


def _restore_optimizer_moments(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    carried: Optional[Dict[str, Dict[str, Any]]],
) -> int:
    """Restore Adam moments for interface parameters that survive a stage cut.

    Encoder/decoder parameters newly unfrozen in phase two correctly start with
    empty state.  Adapter and cross-attention moments are retained instead of
    being discarded after their warm-up, which otherwise creates a needless
    optimization discontinuity on the small WikiLingua training set.
    """

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


def _capture_optimizer_moments(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        names[id(parameter)]: state
        for parameter, state in optimizer.state.items()
        if id(parameter) in names and state
    }


@torch.no_grad()
def validation_loss(
    model: LLM2SeqV4,
    loader: DataLoader,
    device: torch.device,
    training: Dict[str, Any],
    max_examples: int = 0,
) -> Dict[str, float]:
    # Validation/test must never consume reference-derived evidence or the
    # stochastic plan-only path.  This explicit reset prevents state left by
    # the last training batch from leaking into held-out evaluation.
    if hasattr(model, "set_summary_curriculum"):
        model.set_summary_curriculum(plan_only_probability=0.0, oracle_evidence_mix=0.0)
    model.eval()
    metric_names = (
        "loss",
        "loss_ce",
        "loss_salience",
        "loss_response_alignment",
        "response_alignment_cosine",
        "response_alignment_accuracy",
        "response_alignment_valid_slots",
        "plan_only_rate",
        "summary_prefix_rms",
        "decoder_embedding_rms",
        "prefix_to_embedding_rms_ratio",
        "loss_contrastive",
        "prompt_retrieval_accuracy",
        "loss_source_swap",
        "source_swap_nll_gap",
        "source_swap_accuracy",
        "source_swap_negative_similarity",
        "loss_routing_balance",
        "memory_routing_entropy",
        "adaptive_routing_delta",
        "memory_route_lexical",
        "memory_route_semantic",
        "memory_route_summary",
        "cross_gate_mean",
        "cross_residual_ratio",
    )
    totals = {name: 0.0 for name in metric_names}
    examples = 0
    for batch in loader:
        if max_examples > 0:
            remaining = max_examples - examples
            if remaining <= 0:
                break
            if batch["input_ids"].shape[0] > remaining:
                batch = {name: value[:remaining] for name, value in batch.items()}
        batch = _move(batch, device)
        with _autocast(device, training):
            outputs = model(**batch, compute_source_diagnostics=True)
        batch_size = int(batch["input_ids"].shape[0])
        for name in totals:
            value = outputs.get(name)
            if value is not None:
                totals[name] += float(value.detach().float()) * batch_size
        examples += batch_size
    result = {f"eval_{name}": value / max(1, examples) for name, value in totals.items()}
    result["eval_examples"] = float(examples)
    return result


def _run_stage(
    model: LLM2SeqV4,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    stage: str,
    stage_epochs: int,
    epoch_offset: int,
    global_step: int,
    optimizer_moments: Optional[Dict[str, Dict[str, Any]]] = None,
    preserve_optimizer_moments: bool = False,
) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
    if stage_epochs <= 0:
        return epoch_offset, global_step, optimizer_moments or {}
    training = config["training"]
    objectives = config.get("objectives", {})
    contrastive_warmup_epochs = int(objectives.get("contrastive_warmup_epochs", 2))
    plan_only_start = float(objectives.get("plan_only_probability_start", 0.25))
    plan_only_end = float(objectives.get("plan_only_probability_end", 0.10))
    plan_only_anneal_epochs = float(objectives.get("plan_only_anneal_epochs", 5.0))
    oracle_start = float(objectives.get("oracle_evidence_mix_start", 1.0))
    oracle_end = float(objectives.get("oracle_evidence_mix_end", 0.0))
    oracle_anneal_epochs = float(objectives.get("oracle_evidence_anneal_epochs", 5.0))

    model.set_training_stage(stage)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * stage_epochs)
    optimizer, scheduler = build_optimizer(model, training, stage, total_steps)
    restored_moments = _restore_optimizer_moments(model, optimizer, optimizer_moments)
    fp16 = device.type == "cuda" and bool(training.get("fp16", False)) and not bool(training.get("bf16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    validation_every = int(training.get("validation_every_epochs", 0))
    validate_at_stage_end = bool(training.get("validate_at_stage_end", True))
    validation_max_examples = int(training.get("validation_max_examples", 256))
    optimizer.zero_grad(set_to_none=True)
    local_step = 0

    LOGGER.info(
        "Starting stage=%s epochs=%d trainable=%s restored_optimizer_states=%d",
        stage,
        stage_epochs,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        restored_moments,
    )
    for stage_epoch in range(1, stage_epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        accumulation_count = 0
        metric_count = 0
        for batch_index, batch in enumerate(loader, start=1):
            curriculum_progress = (
                float(epoch_offset + stage_epoch - 1)
                + float(batch_index - 1) / max(1, len(loader))
            )
            plan_only_probability = _linear_curriculum(
                curriculum_progress,
                plan_only_start,
                plan_only_end,
                plan_only_anneal_epochs,
            )
            oracle_evidence_mix = _linear_curriculum(
                curriculum_progress,
                oracle_start,
                oracle_end,
                oracle_anneal_epochs,
            )
            model.set_summary_curriculum(
                plan_only_probability=plan_only_probability,
                oracle_evidence_mix=oracle_evidence_mix,
            )
            scale = _contrastive_scale(
                stage,
                stage_epoch,
                batch_index,
                len(loader),
                contrastive_warmup_epochs,
            )
            model.set_contrastive_scale(scale)
            batch = _move(batch, device)
            with _autocast(device, training):
                outputs = model(**batch)
                loss = outputs["loss"] / accumulation
            scaler.scale(loss).backward()
            accumulation_count += 1
            metric_count += 1
            for name, value in outputs.items():
                if value.numel() == 1:
                    running[name] = running.get(name, 0.0) + float(value.detach().float())

            update = accumulation_count == accumulation or batch_index == len(loader)
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
            if local_step % log_every == 0 or local_step == total_steps:
                divisor = max(1, metric_count)
                payload = {
                    "stage": stage,
                    "epoch": epoch_offset + stage_epoch,
                    "stage_epoch": stage_epoch,
                    "step": global_step,
                    "stage_step": local_step,
                    "loss": running.get("loss", 0.0) / divisor,
                    "loss_ce": running.get("loss_ce", 0.0) / divisor,
                    "loss_salience": running.get("loss_salience", 0.0) / divisor,
                    "loss_response_alignment": running.get("loss_response_alignment", 0.0) / divisor,
                    "response_alignment_cosine": running.get("response_alignment_cosine", 0.0) / divisor,
                    "response_alignment_accuracy": running.get("response_alignment_accuracy", 0.0) / divisor,
                    "response_alignment_valid_slots": running.get("response_alignment_valid_slots", 0.0) / divisor,
                    "summary_prefix_rms": running.get("summary_prefix_rms", 0.0) / divisor,
                    "decoder_embedding_rms": running.get("decoder_embedding_rms", 0.0) / divisor,
                    "prefix_to_embedding_rms_ratio": running.get("prefix_to_embedding_rms_ratio", 0.0) / divisor,
                    "plan_only_rate": running.get("plan_only_rate", 0.0) / divisor,
                    "plan_only_probability": plan_only_probability,
                    "oracle_evidence_mix": oracle_evidence_mix,
                    "loss_contrastive": running.get("loss_contrastive", 0.0) / divisor,
                    "loss_source_swap": running.get("loss_source_swap", 0.0) / divisor,
                    "prompt_retrieval_accuracy": running.get("prompt_retrieval_accuracy", 0.0) / divisor,
                    "source_swap_nll_gap": running.get("source_swap_nll_gap", 0.0) / divisor,
                    "source_swap_accuracy": running.get("source_swap_accuracy", 0.0) / divisor,
                    "source_swap_negative_similarity": running.get("source_swap_negative_similarity", 0.0) / divisor,
                    "loss_routing_balance": running.get("loss_routing_balance", 0.0) / divisor,
                    "memory_routing_entropy": running.get("memory_routing_entropy", 0.0) / divisor,
                    "adaptive_routing_delta": running.get("adaptive_routing_delta", 0.0) / divisor,
                    "memory_route_lexical": running.get("memory_route_lexical", 0.0) / divisor,
                    "memory_route_semantic": running.get("memory_route_semantic", 0.0) / divisor,
                    "memory_route_summary": running.get("memory_route_summary", 0.0) / divisor,
                    "alignment_last_pool_weight": running.get("alignment_last_pool_weight", 0.0) / divisor,
                    "contrastive_scale": scale,
                    "cross_gate": running.get("cross_gate_mean", 0.0) / divisor,
                    "cross_residual_ratio": running.get("cross_residual_ratio", 0.0) / divisor,
                    "bidirectional_gate": running.get("bidirectional_gate_mean", 0.0) / divisor,
                    "branch_context_gate": running.get("branch_context_gate_mean", 0.0) / divisor,
                    "grad_norm": float(grad_norm),
                    "learning_rates": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False))
                running.clear()
                metric_count = 0
        absolute_epoch = epoch_offset + stage_epoch
        LOGGER.info("completed epoch=%d stage=%s", absolute_epoch, stage)
        periodic_validation = validation_every > 0 and absolute_epoch % validation_every == 0
        should_validate = periodic_validation or (validate_at_stage_end and stage_epoch == stage_epochs)
        if should_validate:
            metrics = validation_loss(
                model,
                validation_loader,
                device,
                training,
                max_examples=validation_max_examples,
            )
            validation_payload = {"epoch": absolute_epoch, "stage": stage, **metrics}
            LOGGER.info("validation %s", json.dumps(validation_payload))
            history_path = Path(config["experiment"]["output_dir"]) / "validation_history.jsonl"
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_payload, ensure_ascii=False) + "\n")
    carried = _capture_optimizer_moments(model, optimizer) if preserve_optimizer_moments else {}
    return epoch_offset + stage_epochs, global_step, carried


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to mix a new run with existing artifacts: {output_dir}. "
                "Pass --overwrite-output-dir for an intentional rerun."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "RUNNING"
    marker.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    return marker


def train(config_path: str, overwrite_output_dir: bool = False) -> Path:
    config = load_config(config_path)
    training = config["training"]
    output_dir = Path(config["experiment"]["output_dir"])
    marker = _prepare_output(output_dir, overwrite_output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )
    try:
        seed = int(training.get("seed", 42))
        _set_seed(seed)
        device = _device()
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(training.get("tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(training.get("tf32", True))
            torch.set_float32_matmul_precision("high")
        data = config["data"]
        limits = config.get("limits", {})
        manifest = {
            split: dataset_fingerprint(
                data[f"{split}_file"],
                int(limits.get(f"max_{split}_examples", 0)),
            )
            for split in ("train", "validation", "test")
        }
        manifest_status = verify_locked_data_manifest(config, manifest)
        LOGGER.info("Locked data parity: %s", json.dumps(manifest_status))
        model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset = build_experiment(config)
        assert train_dataset is not None
        parameter_summary = model.parameter_summary()
        parameter_budget = verify_declared_parameter_budget(config, parameter_summary)
        LOGGER.info("Parameter budget: %s", json.dumps(parameter_budget))
        model.to(device)
        collator = _collator(config, encoder_tokenizer, decoder_tokenizer)
        loader = DataLoader(
            train_dataset,
            batch_size=int(training.get("batch_size", 32)),
            shuffle=True,
            collate_fn=collator,
            num_workers=int(training.get("num_workers", 4)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(training.get("num_workers", 4)) > 0,
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(training.get("validation_batch_size", 32)),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(training.get("validation_num_workers", 2)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(training.get("validation_num_workers", 2)) > 0,
        )
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (output_dir / "data_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("device=%s model=%s", device, json.dumps(parameter_summary))
        LOGGER.info(
            "data train=%d validation=%d batch=%d accumulation=%d",
            len(train_dataset),
            len(validation_dataset),
            int(training.get("batch_size", 32)),
            int(training.get("gradient_accumulation_steps", 1)),
        )
        epoch, global_step = 0, 0
        epoch, global_step, interface_moments = _run_stage(
            model,
            loader,
            validation_loader,
            device,
            config,
            "interface_warmup",
            int(training.get("interface_warmup_epochs", 2)),
            epoch,
            global_step,
            preserve_optimizer_moments=True,
        )
        epoch, global_step, _ = _run_stage(
            model,
            loader,
            validation_loader,
            device,
            config,
            "full_finetune",
            int(training.get("full_finetune_epochs", 12)),
            epoch,
            global_step,
            optimizer_moments=interface_moments,
        )
        last_path = output_dir / "last.pt"
        save_last_checkpoint(model, last_path, config, epoch, global_step, manifest)
        marker.unlink(missing_ok=True)
        (output_dir / "COMPLETE").write_text(
            json.dumps({"checkpoint": str(last_path), "epoch": epoch, "global_step": global_step}),
            encoding="utf-8",
        )
        LOGGER.info("Training complete: last=%s epoch=%d step=%d", last_path, epoch, global_step)
        return last_path
    except Exception:
        LOGGER.exception("Training failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args.config, args.overwrite_output_dir)


if __name__ == "__main__":
    main()
