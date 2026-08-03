"""Stable two-stage EviSeq runtime with atomic epoch and final checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..configuration import load_config
from ..data.dataset import (
    LengthBucketBatchSampler,
    Seq2SeqCollator,
    Text2TextDataset,
    decoder_seed_ids,
)
from ..modeling.architecture import EviSeq as RuntimeModel
from .checkpoint import initialize_from_checkpoint, save_configured_epoch_checkpoints, save_last_checkpoint

LOGGER = logging.getLogger("eviseq.training.engine")


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


def _tokenizers(config: Dict[str, Any]) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer

    model = config["model"]
    encoder = AutoTokenizer.from_pretrained(model["encoder_name"], trust_remote_code=True)
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
) -> Tuple[RuntimeModel, Any, Any, Text2TextDataset | None, Text2TextDataset | None]:
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = RuntimeModel(config)
    data = config["data"]
    limits = config.get("limits", {})
    train_dataset = None
    if include_train:
        train_dataset = Text2TextDataset(
            data["train_file"],
            encoder_tokenizer,
            decoder_tokenizer,
            data,
            max_examples=int(limits.get("max_train_examples", 0)),
            precompute_evidence=bool(data.get("precompute_evidence", True)),
        )
    validation_dataset = Text2TextDataset(
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
    if ".cross_attn" in name or name.endswith(".cross_gate") or name.endswith(".memory_router_logits"):
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
        "encoder": float(training.get("full_encoder_lr", 8e-6)),
        "adapter": float(training.get("full_adapter_lr", 3e-5)),
        "decoder": float(training.get("full_decoder_lr", 1e-5)),
        "cross_attention": float(training.get("full_cross_attention_lr", 5e-5)),
    }


def build_optimizer(
    model: RuntimeModel,
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
                "weight_decay": 0.0 if no_decay else float(training.get("weight_decay", 0.01)),
                "component": component,
            }
        )
    optimizer = AdamW(
        parameter_groups,
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.95))),
        eps=float(training.get("adam_epsilon", 1e-8)),
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.05)))
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


@torch.no_grad()
def validation_loss(
    model: RuntimeModel,
    loader: DataLoader,
    device: torch.device,
    training: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_salience": 0.0}
    batches = 0
    for batch in loader:
        batch = _move(batch, device)
        with _autocast(device, training):
            outputs = model(**batch)
        for name in totals:
            totals[name] += float(outputs[name].detach().float())
        batches += 1
    return {f"eval_{name}": value / max(1, batches) for name, value in totals.items()}


def _run_stage(
    model: RuntimeModel,
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
    if stage_epochs <= 0:
        return epoch_offset, global_step
    model.set_training_stage(stage)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * stage_epochs)
    optimizer, scheduler = build_optimizer(model, training, stage, total_steps)
    fp16 = device.type == "cuda" and bool(training.get("fp16", False)) and not bool(training.get("bf16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    validation_every = int(training.get("validation_every_epochs", 0))
    optimizer.zero_grad(set_to_none=True)
    local_step = 0

    LOGGER.info(
        "Starting stage=%s epochs=%d trainable=%s",
        stage,
        stage_epochs,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    for stage_epoch in range(1, stage_epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        accumulation_count = 0
        metric_count = 0
        for batch_index, batch in enumerate(loader, start=1):
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
                    "cross_gate": running.get("cross_gate_mean", 0.0) / divisor,
                    "cross_residual_ratio": running.get("cross_residual_ratio", 0.0) / divisor,
                    "bidirectional_gate": running.get("bidirectional_gate_mean", 0.0) / divisor,
                    "grad_norm": float(grad_norm),
                    "learning_rates": {str(group["component"]): float(group["lr"]) for group in optimizer.param_groups},
                }
                LOGGER.info("train %s", json.dumps(payload, ensure_ascii=False))
                running.clear()
                metric_count = 0
        absolute_epoch = epoch_offset + stage_epoch
        LOGGER.info("completed epoch=%d stage=%s", absolute_epoch, stage)
        save_best = bool((checkpoint_config or {}).get("checkpoint", {}).get("save_best", False))
        scheduled_validation = validation_every > 0 and (
            absolute_epoch % validation_every == 0 or stage_epoch == stage_epochs
        )
        metrics = None
        if save_best or scheduled_validation:
            metrics = validation_loss(model, validation_loader, device, training)
            LOGGER.info("validation %s", json.dumps({"epoch": absolute_epoch, **metrics}))
        if checkpoint_dir is not None and checkpoint_config is not None:
            saved = save_configured_epoch_checkpoints(
                model,
                checkpoint_dir,
                checkpoint_config,
                absolute_epoch,
                global_step,
                metrics,
            )
            if saved:
                LOGGER.info("checkpoint %s", json.dumps(saved))
    return epoch_offset + stage_epochs, global_step


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to mix a new run with existing artifacts: {output_dir}. "
                "Pass --overwrite-output-dir for an intentional rerun."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def train(
    config_path: str,
    overwrite_output_dir: bool = False,
    init_checkpoint: str = "",
    output_dir_override: str = "",
    allow_partial_init: bool = False,
) -> Path:
    config = load_config(config_path)
    training = config["training"]
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override
    configured_init = str(training.get("init_checkpoint", "")).strip()
    initialization_path = str(init_checkpoint).strip() or configured_init
    if initialization_path:
        training["init_checkpoint"] = initialization_path
    if allow_partial_init:
        training["strict_initial_checkpoint"] = False
    output_dir = Path(config["experiment"]["output_dir"])
    _prepare_output(output_dir, overwrite_output_dir)
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
        model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset = build_experiment(config)
        assert train_dataset is not None
        if initialization_path:
            initialization = initialize_from_checkpoint(
                model,
                initialization_path,
                strict=bool(training.get("strict_initial_checkpoint", True)),
            )
            LOGGER.info(
                "initialized checkpoint=%s loaded_tensors=%d skipped_tensors=%d",
                initialization_path,
                initialization["loaded_tensors"],
                len(initialization["skipped_tensors"]),
            )
        model.to(device)
        collator = _collator(config, encoder_tokenizer, decoder_tokenizer)
        batch_size = int(training.get("batch_size", 32))
        loader_kwargs = {
            "collate_fn": collator,
            "num_workers": int(training.get("num_workers", 4)),
            "pin_memory": device.type == "cuda",
            "persistent_workers": int(training.get("num_workers", 4)) > 0,
        }
        length_bucketing = bool(training.get("length_bucketing", False))
        if length_bucketing:
            loader = DataLoader(
                train_dataset,
                batch_sampler=LengthBucketBatchSampler(
                    train_dataset.source_length_estimates(),
                    batch_size,
                    seed=int(training.get("seed", 42)),
                    bucket_size_multiplier=int(training.get("length_bucket_multiplier", 50)),
                ),
                **loader_kwargs,
            )
        else:
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                **loader_kwargs,
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
        LOGGER.info("device=%s model=%s", device, json.dumps(model.parameter_summary()))
        LOGGER.info(
            "data train=%d validation=%d batch=%d accumulation=%d length_bucketing=%s",
            len(train_dataset),
            len(validation_dataset),
            int(training.get("batch_size", 32)),
            int(training.get("gradient_accumulation_steps", 1)),
            length_bucketing,
        )
        epoch, global_step = 0, 0
        epoch, global_step = _run_stage(
            model,
            loader,
            validation_loader,
            device,
            training,
            "interface_warmup",
            int(training.get("interface_warmup_epochs", 3)),
            epoch,
            global_step,
            output_dir,
            config,
        )
        epoch, global_step = _run_stage(
            model,
            loader,
            validation_loader,
            device,
            training,
            "full_finetune",
            int(training.get("full_finetune_epochs", 12)),
            epoch,
            global_step,
            output_dir,
            config,
        )
        last_path = output_dir / "last.pt"
        save_last_checkpoint(model, last_path, config, epoch, global_step)
        LOGGER.info("Training complete: last=%s epoch=%d step=%d", last_path, epoch, global_step)
        return last_path
    except Exception:
        LOGGER.exception("Training failed")
        raise


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
