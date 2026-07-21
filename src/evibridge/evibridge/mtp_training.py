"""Phase 3: train only EviBridge future-token draft blocks."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .config import QWEN35_MODEL_SIZES, apply_model_size, dump_config, load_config
from .mtp import future_prediction_loss, save_mtp_checkpoint
from .training import build_experiment


LOGGER = logging.getLogger("evibridge.phase3")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, scale)


def train_mtp(
    config_path: str,
    checkpoint_path: str,
    output_path: str | None,
    model_size: str | None = None,
    max_samples: int | None = None,
    epochs_override: int | None = None,
) -> None:
    config = load_config(config_path)
    apply_model_size(config, model_size)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        config.setdefault("data", {})["max_train_samples"] = int(max_samples)
    mtp_config: Dict[str, object] = config.get("mtp", {}) or {}
    phase3: Dict[str, object] = config.get("phase3", {}) or {}
    experiment = config.get("experiment", {}) or {}
    output = Path(output_path) if output_path else Path(
        str(phase3.get("output", Path(str(experiment.get("output_dir", "runs/evibridge"))) / "phase3_mtp.pt"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output.parent / "phase3_mtp.log", encoding="utf-8")],
        force=True,
    )
    seed = int(phase3.get("seed", (config.get("training", {}) or {}).get("seed", 42)))
    _seed(seed)
    if bool(phase3.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, tokenizer, train_dataset, _, collator = build_experiment(config)
    load_checkpoint(model, checkpoint_path)
    predictor = model.enable_mtp()
    model.set_mtp_only_trainable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictor.train()
    predictor.draft_head.prepare(model.lm_head)

    batch_size = int(phase3.get("batch_size", 32))
    accumulation = int(phase3.get("gradient_accumulation_steps", 1))
    workers = int(phase3.get("num_workers", 4))
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    epochs = int(phase3.get("epochs", 3)) if epochs_override is None else int(epochs_override)
    if epochs < 1:
        raise ValueError("Phase 3 needs at least one epoch")
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = max(1, epochs * steps_per_epoch)
    optimizer = AdamW(
        predictor.parameters(),
        lr=float(phase3.get("learning_rate", 3e-4)),
        betas=(0.9, 0.95),
        weight_decay=float(phase3.get("weight_decay", 0.01)),
        fused=bool(phase3.get("fused_optimizer", True)) and device.type == "cuda",
    )
    scheduler = _scheduler(optimizer, total_steps, float(phase3.get("warmup_ratio", 0.05)))
    use_bf16 = bool(phase3.get("bf16", True)) and device.type == "cuda"
    use_fp16 = bool(phase3.get("fp16", False)) and device.type == "cuda"
    if use_bf16 and use_fp16:
        raise ValueError("Choose phase3.bf16 or phase3.fp16, not both")
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device.type, enabled=use_fp16)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs PAD or EOS for phase 3")
    log_every = int(phase3.get("log_every_steps", 10))
    maximum_grad_norm = float(phase3.get("max_grad_norm", 1.0))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    dump_config(config, output.parent / "phase3_resolved_config.yaml")
    LOGGER.info(
        "Phase 3: %d examples, %d epochs, %d draft depths, effective batch=%d, trainable=%d",
        len(train_dataset),
        epochs,
        predictor.num_draft_tokens,
        batch_size * accumulation,
        sum(parameter.numel() for parameter in predictor.parameters()),
    )

    for epoch in range(1, epochs + 1):
        running: Dict[str, float] = {}
        micro_steps = 0
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with torch.no_grad(), autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=use_bf16 or use_fp16,
            ):
                bridge = model.encode(
                    batch["input_ids"],
                    batch["attention_mask"],
                    unit_ids=batch["unit_ids"],
                    return_bridge_output=True,
                )
                decoder_states, _ = model.decoder(
                    input_ids=batch["decoder_input_ids"],
                    encoder_hidden_states=bridge.memory,
                    encoder_attention_mask=bridge.memory_mask,
                    attention_mask=batch["decoder_attention_mask"],
                    use_cache=False,
                )
            with autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=use_bf16 or use_fp16,
            ):
                loss, metrics = future_prediction_loss(
                    predictor,
                    decoder_states,
                    batch["labels"],
                    model.decoder.embed_tokens,
                    model.lm_head,
                    int(pad_token_id),
                    mtp_config,
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            micro_steps += 1
            for name, value in metrics.items():
                running[name] = running.get(name, 0.0) + float(value.item())
            if batch_index % accumulation and batch_index != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), maximum_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % log_every == 0:
                averaged = {name: value / max(1, micro_steps) for name, value in running.items()}
                LOGGER.info(
                    "phase3 %s",
                    json.dumps({"epoch": epoch, "step": global_step, **averaged}, ensure_ascii=False),
                )
                running.clear()
                micro_steps = 0
        LOGGER.info("completed phase3 epoch=%d global_step=%d", epoch, global_step)

    save_mtp_checkpoint(
        predictor,
        output,
        mtp_config,
        checkpoint_path,
        epochs,
        global_step,
    )
    LOGGER.info("Phase 3 complete: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-size", choices=sorted(QWEN35_MODEL_SIZES), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_mtp(
        args.config,
        args.checkpoint,
        args.output,
        args.model_size,
        args.max_samples,
        args.epochs,
    )


if __name__ == "__main__":
    main()
