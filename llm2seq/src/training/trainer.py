"""
LLM2Seq Training Loop.

Full training pipeline:
1. Load config → build model → build dataset → train.
2. Supports --config and --resume.
3. Mixed precision, gradient accumulation, gradient checkpointing.
4. Per-component learning rates for encoder, adaptor, decoder.
5. Eval loop with metric logging and checkpoint saving.

Usage:
    python -m llm2seq.src.training.trainer --config llm2seq/configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import yaml

# Local imports
from ..models.llm2seq_model import LLM2Seq, LLM2SeqConfig
from ..data.dataset import Seq2SeqDataset
from ..data.collator import Seq2SeqCollator
from .scheduler import build_optimizer_and_scheduler

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: LLM2SeqConfig, vocab_size: int) -> LLM2Seq:
    """Build LLM2Seq model from config."""
    model = LLM2Seq(cfg=cfg, vocab_size=vocab_size)
    logger.info(model.summary())
    return model


def train(
    config_path: str,
    resume_from: Optional[str] = None,
) -> None:
    """
    Main training function.

    Args:
        config_path: Path to YAML config file.
        resume_from: Path to checkpoint to resume from (overrides config).
    """
    # Load config
    raw_cfg = load_config(config_path)
    cfg = LLM2SeqConfig(raw_cfg)
    training_cfg = raw_cfg.get("training", {})
    data_cfg = raw_cfg.get("data", {})
    eval_cfg = raw_cfg.get("evaluation", {})
    project_cfg = raw_cfg.get("project", {})

    # Output directory
    output_dir = project_cfg.get("output_dir", "runs/llm2seq_default")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
        ],
    )

    # Seed
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Tokenizer (from encoder model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Build model
    model = build_model(cfg, vocab_size)
    model = model.to(device)

    # Gradient checkpointing
    if training_cfg.get("gradient_checkpointing", False):
        if hasattr(model.encoder.model, "gradient_checkpointing_enable"):
            model.encoder.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for encoder")

    # Dataset
    train_file = data_cfg.get("train_file", "data/processed/train.jsonl")
    eval_file = data_cfg.get("eval_file", "data/processed/eval.jsonl")

    train_dataset = Seq2SeqDataset(
        data_path=train_file,
        tokenizer=tokenizer,
        max_source_length=data_cfg.get("max_source_length", 512),
        max_target_length=data_cfg.get("max_target_length", 256),
    )
    eval_dataset = Seq2SeqDataset(
        data_path=eval_file,
        tokenizer=tokenizer,
        max_source_length=data_cfg.get("max_source_length", 512),
        max_target_length=data_cfg.get("max_target_length", 256),
    )

    collator = Seq2SeqCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=data_cfg.get("max_source_length", 512),
        max_target_length=data_cfg.get("max_target_length", 256),
    )

    batch_size = training_cfg.get("batch_size", 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        encoder_lr=float(training_cfg.get("encoder_lr", 1e-5)),
        adaptor_lr=float(training_cfg.get("adaptor_lr", 2e-4)),
        decoder_lr=float(training_cfg.get("decoder_lr", 2e-4)),
        warmup_steps=training_cfg.get("warmup_steps", 1000),
        max_steps=training_cfg.get("max_steps", 50000),
    )

    # Mixed precision
    use_fp16 = training_cfg.get("fp16", True) and device.type == "cuda"
    use_bf16 = training_cfg.get("bf16", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_fp16)
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)

    # Training params
    grad_accum_steps = training_cfg.get("grad_accum_steps", 8)
    max_steps = training_cfg.get("max_steps", 50000)
    log_every = training_cfg.get("log_every_steps", 50)
    eval_every = eval_cfg.get("eval_every_steps", 1000)
    save_every = eval_cfg.get("save_every_steps", 2000)

    # Resume from checkpoint
    global_step = 0
    best_eval_loss = float("inf")
    checkpoint_path = resume_from or training_cfg.get("resume_from", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if "best_eval_loss" in ckpt:
            best_eval_loss = ckpt["best_eval_loss"]
        logger.info(f"Resumed at step {global_step}")

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False)

    # Training loop
    logger.info(f"Starting training for {max_steps} steps...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {grad_accum_steps}")
    logger.info(f"  Effective batch size: {batch_size * grad_accum_steps}")
    logger.info(f"  Mixed precision: fp16={use_fp16}, bf16={use_bf16}")

    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_mtp = 0.0
    step_count = 0

    epoch = 0
    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= max_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_fp16 or use_bf16)):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch.get("decoder_attention_mask"),
                    labels=batch["labels"],
                    teacher_logits=batch.get("teacher_logits"),
                    teacher_topk_indices=batch.get("teacher_topk_indices"),
                )
                loss = outputs["loss"] / grad_accum_steps

            # Backward
            scaler.scale(loss).backward()

            # Accumulate metrics
            running_loss += outputs["loss"].item()
            running_ce += outputs["loss_ce"].item()
            running_kd += outputs.get("loss_kd", torch.tensor(0.0)).item()
            running_mtp += outputs.get("loss_mtp", torch.tensor(0.0)).item()
            step_count += 1

            # Optimizer step
            if step_count % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    avg_loss = running_loss / step_count
                    avg_ce = running_ce / step_count
                    avg_kd = running_kd / step_count
                    avg_mtp = running_mtp / step_count
                    lr = optimizer.param_groups[-1]["lr"]

                    logger.info(
                        f"Step {global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                        f"KD: {avg_kd:.4f} | MTP: {avg_mtp:.4f} | "
                        f"LR: {lr:.2e} | Epoch: {epoch}"
                    )

                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    running_mtp = 0.0
                    step_count = 0

                # Evaluation
                if global_step % eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, device, amp_dtype, use_fp16 or use_bf16)
                    logger.info(f"Step {global_step} | Eval Loss: {eval_loss:.4f}")
                    model.train()

                    # Save best
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(
                            model, optimizer, global_step, best_eval_loss,
                            os.path.join(output_dir, "best.pt"),
                        )
                        logger.info(f"New best eval loss: {best_eval_loss:.4f}")

                # Periodic save
                if global_step % save_every == 0:
                    save_checkpoint(
                        model, optimizer, global_step, best_eval_loss,
                        os.path.join(output_dir, f"checkpoint_{global_step}.pt"),
                    )

    # Final save
    save_checkpoint(
        model, optimizer, global_step, best_eval_loss,
        os.path.join(output_dir, "final.pt"),
    )
    logger.info(f"Training complete. Best eval loss: {best_eval_loss:.4f}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in eval_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch.get("decoder_attention_mask"),
                labels=batch["labels"],
            )

        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    best_eval_loss: float,
    path: str,
) -> None:
    """Save training checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "best_eval_loss": best_eval_loss,
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="LLM2Seq Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
