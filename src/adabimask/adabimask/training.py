"""Focused trainer for the AdaBiMask ablation matrix."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_checkpoint, save_checkpoint
from .config import QWEN35_MODEL_SIZES, apply_model_size, dump_config, load_config
from .data import DirectCollator, DirectSummarizationDataset, PromptedSeq2SeqDataset, Seq2SeqCollator
from .direct_baseline import DirectCausalBaseline
from .model import AdaBiMaskSeq2Seq

LOGGER = logging.getLogger("adabimask")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_seq2seq_data(config: Dict[str, Any], tokenizer: Any) -> Tuple[Any, Any, Any]:
    data = config.get("data", {}) or {}
    train_dataset = PromptedSeq2SeqDataset(data["train_file"], tokenizer, data)
    validation_dataset = PromptedSeq2SeqDataset(data["validation_file"], tokenizer, data)
    collator = Seq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=int(data.get("max_source_length", 3072)),
        max_target_length=int(data.get("max_target_length", 384)),
    )
    return train_dataset, validation_dataset, collator


def build_experiment(config: Dict[str, Any]) -> Tuple[nn.Module, Any, Any, Any, Any]:
    from transformers import AutoTokenizer

    model_config = config.get("model", {}) or {}
    data_config = config.get("data", {}) or {}
    model_name = str(model_config.get("encoder_name", "Qwen/Qwen3.5-0.8B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    if kind == "encoder_decoder":
        model = AdaBiMaskSeq2Seq(config, vocab_size=len(tokenizer))
        train_dataset, validation_dataset, collator = _build_seq2seq_data(config, tokenizer)
    elif kind == "direct_causal":
        model = DirectCausalBaseline(model_config)
        train_dataset = DirectSummarizationDataset(data_config["train_file"], tokenizer, data_config)
        validation_dataset = DirectSummarizationDataset(data_config["validation_file"], tokenizer, data_config)
        collator = DirectCollator(tokenizer.pad_token_id)
    else:
        raise ValueError(f"Unknown experiment.kind: {kind}")

    max_train_samples = int(data_config.get("max_train_samples", 0))
    max_validation_samples = int(data_config.get("max_validation_samples", 0))
    if max_train_samples > 0:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_validation_samples > 0:
        validation_dataset = Subset(
            validation_dataset,
            range(min(max_validation_samples, len(validation_dataset))),
        )
    return model, tokenizer, train_dataset, validation_dataset, collator


def _component_for_parameter(name: str) -> str:
    if "policy.gate_logits" in name:
        return "gate"
    if ".cross_attn" in name or name.endswith("cross_gate"):
        return "adaptor"
    if name.startswith("encoder.") or name.startswith("model."):
        return "encoder"
    if name.startswith("adaptor."):
        return "adaptor"
    return "decoder"


def build_optimizer(
    model: nn.Module,
    training_config: Dict[str, Any],
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    learning_rates = {
        "encoder": float(training_config.get("encoder_lr", 5e-5)),
        "adaptor": float(training_config.get("adaptor_lr", 2e-4)),
        "decoder": float(training_config.get("decoder_lr", 2e-4)),
        "gate": float(training_config.get("gate_lr", 1e-2)),
    }
    weight_decay = float(training_config.get("weight_decay", 0.01))
    grouped: Dict[Tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _component_for_parameter(name)
        no_decay = name.endswith("bias") or any(token in name.lower() for token in ("norm", "gate_logits"))
        grouped.setdefault((component, no_decay), []).append(parameter)
    groups = [
        {
            "params": parameters,
            "lr": learning_rates[component],
            "weight_decay": 0.0 if no_decay else weight_decay,
            "component": component,
        }
        for (component, no_decay), parameters in grouped.items()
    ]
    if not groups:
        raise ValueError("No trainable parameters")

    fused = bool(training_config.get("fused_optimizer", True)) and torch.cuda.is_available()
    optimizer = AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=fused)
    warmup_steps = int(total_steps * float(training_config.get("warmup_ratio", 0.05)))
    min_ratio = float(training_config.get("min_lr_ratio", 0.1))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return max(min_ratio, cosine)

    return optimizer, LambdaLR(optimizer, schedule)


def _forward(model: nn.Module, batch: Dict[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    if kind == "direct_causal":
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch.get("decoder_attention_mask"),
        labels=batch["labels"],
    )


def train(
    config_path: str,
    resume: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    model_size: Optional[str] = None,
) -> None:
    config = load_config(config_path)
    apply_model_size(config, model_size)
    training_config = config.get("training", {}) or {}
    experiment = config.get("experiment", {}) or {}
    kind = str(experiment.get("kind", "encoder_decoder"))
    output_dir = Path(str(experiment.get("output_dir", "runs/adabimask")))
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
        force=True,
    )
    seed = int(training_config.get("seed", 42))
    set_seed(seed)
    if bool(training_config.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, _, train_dataset, _, collator = build_experiment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if resume:
        payload = load_checkpoint(model, resume)
        LOGGER.info("Loaded trainable weights from %s (step=%s)", resume, payload.get("global_step", "unknown"))

    batch_size = int(training_config.get("batch_size", 32))
    grad_accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    workers = int(training_config.get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    epochs = int(training_config.get("epochs", 12))
    interface_epochs = int(training_config.get("interface_warmup_epochs", 1)) if kind == "encoder_decoder" else 0
    if resume and bool(training_config.get("skip_interface_warmup_on_resume", True)):
        interface_epochs = 0
    if not 0 <= interface_epochs < epochs:
        raise ValueError(f"interface_warmup_epochs must be in [0, epochs), got {interface_epochs}/{epochs}")
    steps_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
    total_steps = max(1, steps_per_epoch * epochs)
    if max_steps_override is not None:
        total_steps = min(total_steps, max_steps_override)

    stage = "interface_warmup" if interface_epochs > 0 else "full_finetune"
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
    interface_steps = min(total_steps, steps_per_epoch * interface_epochs)
    full_steps = max(1, total_steps - interface_steps)
    stage_steps = interface_steps if stage == "interface_warmup" else full_steps
    optimizer, scheduler = build_optimizer(model, training_config, max(1, stage_steps))

    use_bf16 = bool(training_config.get("bf16", True)) and device.type == "cuda"
    use_fp16 = bool(training_config.get("fp16", False)) and device.type == "cuda"
    if use_bf16 and use_fp16:
        raise ValueError("Choose bf16 or fp16, not both")
    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device.type, enabled=use_fp16)
    max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
    log_every = int(training_config.get("log_every_steps", 10))
    global_step = 0
    stage_step = 0
    completed_epoch = 0
    optimizer.zero_grad(set_to_none=True)
    dump_config(config, output_dir / "resolved_config.yaml")
    LOGGER.info("\n%s", model.summary())
    LOGGER.info(
        "Training %s: %d examples, %d epochs (%d interface + %d full), %d optimizer steps, effective batch=%d",
        kind,
        len(train_dataset),
        epochs,
        interface_epochs,
        epochs - interface_epochs,
        total_steps,
        batch_size * grad_accumulation,
    )

    for epoch in range(1, epochs + 1):
        target_stage = "interface_warmup" if epoch <= interface_epochs else "full_finetune"
        if target_stage != stage:
            stage = target_stage
            stage_step = 0
            if hasattr(model, "set_training_stage"):
                model.set_training_stage(stage)
            optimizer, scheduler = build_optimizer(model, training_config, full_steps)
            optimizer.zero_grad(set_to_none=True)
            LOGGER.info(
                "Switched to full fine-tuning: trainable_parameters=%d",
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            )
        model.train()
        running = 0.0
        micro_steps = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if stage == "full_finetune" and hasattr(model, "set_curriculum_progress"):
                model.set_curriculum_progress(stage_step / max(1, full_steps - 1))
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = _forward(model, batch, kind)
                loss = outputs["loss"] / grad_accumulation
            scaler.scale(loss).backward()
            running += float(outputs["loss"].item())
            micro_steps += 1

            should_step = batch_index % grad_accumulation == 0 or batch_index == len(train_loader)
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            stage_step += 1

            if global_step % log_every == 0:
                message: Dict[str, Any] = {
                    "epoch": epoch,
                    "step": global_step,
                    "stage": stage,
                    "loss": running / max(1, micro_steps),
                }
                if "loss_gate" in outputs:
                    message.update(
                        {
                            "loss_gate": float(outputs["loss_gate"].item()),
                            "loss_budget": float(outputs["loss_budget"].item()),
                            "loss_binary": float(outputs["loss_binary"].item()),
                        }
                    )
                LOGGER.info("train %s", json.dumps(message, ensure_ascii=False))
                running = 0.0
                micro_steps = 0
            if global_step >= total_steps:
                break
        completed_epoch = epoch
        LOGGER.info("completed epoch=%d stage=%s global_step=%d", epoch, stage, global_step)
        if (
            stage == "interface_warmup"
            and bool(training_config.get("stop_after_interface", False))
            and epoch >= interface_epochs
        ):
            LOGGER.info("Stopped after the requested shared interface warm-up")
            break
        if global_step >= total_steps:
            break

    final_path = output_dir / "final.pt"
    save_checkpoint(model, final_path, config, completed_epoch, global_step)
    LOGGER.info("Training complete: final_checkpoint=%s", final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Load trainable tensors from a warm-up/search checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test override")
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    args = parser.parse_args()
    train(
        args.config,
        resume=args.resume,
        max_steps_override=args.max_steps,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
