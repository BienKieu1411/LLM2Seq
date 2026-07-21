"""Two-stage trainer for EviBridge and its controlled baselines."""

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
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import QWEN35_MODEL_SIZES, apply_model_size, dump_config, load_config
from .data import DirectCollator, DirectSummarizationDataset, EvidenceSeq2SeqCollator, EvidenceSeq2SeqDataset
from .direct_baseline import DirectCausalBaseline
from .model import EviBridgeSeq2Seq

LOGGER = logging.getLogger("evibridge")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_experiment(config: Dict[str, Any]) -> Tuple[nn.Module, Any, Any, Any, Any]:
    from transformers import AutoTokenizer

    model_config = config.get("model", {}) or {}
    data_config = config.get("data", {}) or {}
    train_limit = int(data_config.get("max_train_samples", 0))
    validation_limit = int(data_config.get("max_validation_samples", 0))
    model_name = str(model_config.get("encoder_name", "Qwen/Qwen3.5-0.8B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    if kind == "encoder_decoder":
        model = EviBridgeSeq2Seq(config, vocab_size=len(tokenizer))
        train_dataset = EvidenceSeq2SeqDataset(
            data_config["train_file"],
            tokenizer,
            data_config,
            precompute_evidence=bool(data_config.get("precompute_evidence_on_load", True)),
            max_examples=train_limit,
        )
        validation_dataset = EvidenceSeq2SeqDataset(
            data_config["validation_file"],
            tokenizer,
            data_config,
            precompute_evidence=False,
            max_examples=validation_limit,
        )
        collator = EvidenceSeq2SeqCollator(
            tokenizer.pad_token_id,
            int(data_config.get("max_source_length", 3072)),
            int(data_config.get("max_target_length", 384)),
        )
    elif kind == "direct_causal":
        model = DirectCausalBaseline(model_config)
        train_dataset = DirectSummarizationDataset(
            data_config["train_file"], tokenizer, data_config, max_examples=train_limit
        )
        validation_dataset = DirectSummarizationDataset(
            data_config["validation_file"], tokenizer, data_config, max_examples=validation_limit
        )
        collator = DirectCollator(tokenizer.pad_token_id)
    else:
        raise ValueError(f"Unknown experiment.kind: {kind}")

    return model, tokenizer, train_dataset, validation_dataset, collator


def _component(name: str) -> str:
    if name.startswith("bridge.") or name.endswith("fusion_logits"):
        return "bridge"
    if ".cross_attn" in name or name.endswith("cross_gate"):
        return "interface"
    if name.startswith("encoder.") or name.startswith("model."):
        return "encoder"
    return "decoder"


def build_optimizer(
    model: nn.Module,
    training: Dict[str, Any],
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    rates = {
        "encoder": float(training.get("encoder_lr", 8e-6)),
        "decoder": float(training.get("decoder_lr", 1e-5)),
        "bridge": float(training.get("bridge_lr", 1e-4)),
        "interface": float(training.get("interface_lr", 1e-4)),
    }
    decay = float(training.get("weight_decay", 0.01))
    grouped: Dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _component(name)
        no_decay = name.endswith("bias") or "norm" in name.lower() or name.endswith("gate")
        grouped.setdefault((component, no_decay), []).append(parameter)
    groups = [
        {
            "params": parameters,
            "lr": rates[component],
            "weight_decay": 0.0 if no_decay else decay,
            "component": component,
        }
        for (component, no_decay), parameters in grouped.items()
    ]
    if not groups:
        raise ValueError("No trainable parameters")
    optimizer = AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    warmup = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    minimum = float(training.get("min_lr_ratio", 0.1))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return max(minimum, cosine)

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
        unit_ids=batch["unit_ids"],
        evidence_labels=batch["evidence_labels"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
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
    training = config.get("training", {}) or {}
    experiment = config.get("experiment", {}) or {}
    kind = str(experiment.get("kind", "encoder_decoder"))
    output_dir = Path(str(experiment.get("output_dir", "runs/evibridge/base")))
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
        force=True,
    )
    set_seed(int(training.get("seed", 42)))
    if bool(training.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, _, train_dataset, _, collator = build_experiment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if resume:
        payload = load_checkpoint(model, resume)
        LOGGER.info("Loaded %s at step %s", resume, payload.get("global_step", "unknown"))

    batch_size = int(training.get("batch_size", 32))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    epochs = int(training.get("epochs", 10))
    interface_epochs = int(training.get("interface_warmup_epochs", 2)) if kind == "encoder_decoder" else 0
    if resume and bool(training.get("skip_interface_warmup_on_resume", True)):
        interface_epochs = 0
    if not 0 <= interface_epochs < epochs:
        raise ValueError(f"interface_warmup_epochs must be in [0, epochs): {interface_epochs}/{epochs}")
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, steps_per_epoch * epochs)
    if max_steps_override is not None:
        total_steps = min(total_steps, max_steps_override)
    warm_steps = min(total_steps, steps_per_epoch * interface_epochs)
    full_steps = max(1, total_steps - warm_steps)

    stage = "interface_warmup" if interface_epochs else "full_finetune"
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
    optimizer, scheduler = build_optimizer(
        model,
        training,
        max(1, warm_steps if stage == "interface_warmup" else full_steps),
    )
    use_bf16 = bool(training.get("bf16", True)) and device.type == "cuda"
    use_fp16 = bool(training.get("fp16", False)) and device.type == "cuda"
    if use_bf16 and use_fp16:
        raise ValueError("Choose bf16 or fp16, not both")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device.type, enabled=use_fp16)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    global_step = 0
    completed_epoch = 0
    running: Dict[str, float] = {}
    micro_steps = 0
    optimizer.zero_grad(set_to_none=True)
    dump_config(config, output_dir / "resolved_config.yaml")
    LOGGER.info("\n%s", model.summary())
    LOGGER.info(
        "Training %s: %d examples, %d epochs (%d interface + %d full), %d steps, effective batch=%d",
        kind,
        len(train_dataset),
        epochs,
        interface_epochs,
        epochs - interface_epochs,
        total_steps,
        batch_size * accumulation,
    )

    for epoch in range(1, epochs + 1):
        target_stage = "interface_warmup" if epoch <= interface_epochs else "full_finetune"
        if target_stage != stage:
            stage = target_stage
            if hasattr(model, "set_training_stage"):
                model.set_training_stage(stage)
            optimizer, scheduler = build_optimizer(model, training, full_steps)
            optimizer.zero_grad(set_to_none=True)
            LOGGER.info("Switched to full fine-tuning; trainable=%d", sum(p.numel() for p in model.parameters() if p.requires_grad))
        model.train()
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16 or use_fp16):
                outputs = _forward(model, batch, kind)
                loss = outputs["loss"] / accumulation
            scaler.scale(loss).backward()
            micro_steps += 1
            for key in ("loss", "loss_ce", "loss_evidence", "loss_diversity"):
                if key in outputs:
                    running[key] = running.get(key, 0.0) + float(outputs[key].detach().item())
            if batch_index % accumulation and batch_index != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % log_every == 0:
                message = {"epoch": epoch, "step": global_step, "stage": stage}
                message.update({key: value / max(1, micro_steps) for key, value in running.items()})
                LOGGER.info("train %s", json.dumps(message, ensure_ascii=False))
                running.clear()
                micro_steps = 0
            if global_step >= total_steps:
                break
        completed_epoch = epoch
        LOGGER.info("completed epoch=%d stage=%s global_step=%d", epoch, stage, global_step)
        if global_step >= total_steps:
            break

    final_path = output_dir / "final.pt"
    save_checkpoint(model, final_path, config, completed_epoch, global_step)
    LOGGER.info("Training complete: %s", final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    args = parser.parse_args()
    train(args.config, args.resume, args.max_steps, args.model_size)


if __name__ == "__main__":
    main()
