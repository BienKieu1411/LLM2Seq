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
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import dump_config, load_config
from .data import DirectCollator, DirectSummarizationDataset
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
    try:
        from src.data.collator import Seq2SeqCollator
        from src.data.dataset import Seq2SeqDataset
    except ImportError as exc:
        raise ImportError(
            "Add sibling src/llm2seq to PYTHONPATH; src/adabimask/run.sh does this automatically"
        ) from exc

    data = config.get("data", {}) or {}
    common = {
        "tokenizer": tokenizer,
        "max_source_length": int(data.get("max_source_length", 3072)),
        "max_target_length": int(data.get("max_target_length", 384)),
        "source_prefix": str(data.get("source_prefix", "")),
    }
    train_dataset = Seq2SeqDataset(data["train_file"], **common)
    validation_dataset = Seq2SeqDataset(data["validation_file"], **common)
    collator = Seq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=common["max_source_length"],
        max_target_length=common["max_target_length"],
    )
    return train_dataset, validation_dataset, collator


def build_experiment(config: Dict[str, Any]) -> Tuple[nn.Module, Any, Any, Any, Any]:
    from transformers import AutoTokenizer

    model_config = config.get("model", {}) or {}
    data_config = config.get("data", {}) or {}
    model_name = str(model_config.get("encoder_name", "Qwen/Qwen3-0.6B-Base"))
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
    return model, tokenizer, train_dataset, validation_dataset, collator


def _component_for_parameter(name: str) -> str:
    if "policy.gate_logits" in name:
        return "gate"
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
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _component_for_parameter(name)
        no_decay = name.endswith("bias") or any(token in name.lower() for token in ("norm", "gate_logits"))
        groups.append(
            {
                "params": [parameter],
                "lr": learning_rates[component],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "component": component,
            }
        )
    if not groups:
        raise ValueError("No trainable parameters")

    optimizer = AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
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


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = _forward(model, batch, kind)
        total += float(outputs["loss"].item())
        count += 1
    return total / max(1, count)


def train(config_path: str, resume: Optional[str] = None, max_steps_override: Optional[int] = None) -> None:
    config = load_config(config_path)
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

    model, _, train_dataset, validation_dataset, collator = build_experiment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    LOGGER.info("\n%s", model.summary())

    if resume:
        payload = load_checkpoint(model, resume)
        LOGGER.info("Loaded trainable weights from %s (step=%s)", resume, payload.get("global_step", "unknown"))

    batch_size = int(training_config.get("batch_size", 1))
    grad_accumulation = int(training_config.get("gradient_accumulation_steps", 16))
    workers = int(training_config.get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    epochs = int(training_config.get("epochs", 4))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
    total_steps = max(1, steps_per_epoch * epochs)
    if max_steps_override is not None:
        total_steps = min(total_steps, max_steps_override)
    optimizer, scheduler = build_optimizer(model, training_config, total_steps)

    use_bf16 = bool(training_config.get("bf16", True)) and device.type == "cuda"
    use_fp16 = bool(training_config.get("fp16", False)) and device.type == "cuda"
    if use_bf16 and use_fp16:
        raise ValueError("Choose bf16 or fp16, not both")
    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device.type, enabled=use_fp16)
    max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
    log_every = int(training_config.get("log_every_steps", 10))

    best_eval = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    dump_config(config, output_dir / "resolved_config.yaml")
    LOGGER.info(
        "Training %s: %d examples, %d epochs, %d optimizer steps, effective batch=%d",
        kind,
        len(train_dataset),
        epochs,
        total_steps,
        batch_size * grad_accumulation,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        micro_steps = 0
        for batch_index, batch in enumerate(train_loader, start=1):
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

            if global_step % log_every == 0:
                message: Dict[str, Any] = {
                    "epoch": epoch,
                    "step": global_step,
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

        eval_loss = evaluate_loss(model, validation_loader, device, kind, amp_dtype, use_amp)
        LOGGER.info("epoch=%d eval_loss=%.6f", epoch, eval_loss)
        save_checkpoint(model, output_dir / "last.pt", config, epoch, global_step, eval_loss)
        if eval_loss < best_eval:
            best_eval = eval_loss
            save_checkpoint(model, output_dir / "best.pt", config, epoch, global_step, eval_loss)
        if global_step >= total_steps:
            break

    LOGGER.info("Training complete: best_eval_loss=%.6f output=%s", best_eval, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Load trainable tensors from a warm-up/search checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test override")
    args = parser.parse_args()
    train(args.config, resume=args.resume, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
