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
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import yaml

# Local imports
from ..models.llm2seq_model import LLM2Seq, LLM2SeqConfig
from ..data.dataset import Seq2SeqDataset
from ..data.collator import Seq2SeqCollator
from .scheduler import build_optimizer_and_scheduler

logger = logging.getLogger(__name__)


class HuggingFaceRunPusher:
    """Small helper for uploading durable training artifacts to Hugging Face."""

    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        hf_cfg = cfg.get("huggingface", {})
        self.enabled = bool(hf_cfg.get("enabled", False))
        self.output_dir = Path(output_dir)
        self.repo_id = hf_cfg.get("repo_id") or os.environ.get("HF_REPO_ID")
        self.repo_type = hf_cfg.get("repo_type", "model")
        self.path_in_repo = str(hf_cfg.get("path_in_repo", self.output_dir.name)).strip("/")
        self.push_each_epoch = bool(hf_cfg.get("push_each_epoch", False))
        self.push_final_best = bool(hf_cfg.get("push_final_best", False))
        self.keep_local_epoch_checkpoints = int(hf_cfg.get("keep_local_epoch_checkpoints", 1))
        self.fail_on_error = bool(hf_cfg.get("fail_on_error", True))
        self.token = os.environ.get("HF_TOKEN")
        self.api = None

    def setup(self) -> None:
        if not self.enabled:
            return
        if not self.repo_id:
            raise EnvironmentError("huggingface.enabled=true but HF_REPO_ID is not set.")
        if not self.token:
            raise EnvironmentError("huggingface.enabled=true but HF_TOKEN is not set.")
        from huggingface_hub import HfApi, create_repo

        self.api = HfApi(token=self.token)
        create_repo(self.repo_id, token=self.token, repo_type=self.repo_type, exist_ok=True)
        logger.info(
            "Hugging Face uploads enabled: repo=%s path=%s",
            self.repo_id,
            self.path_in_repo,
        )

    def upload_file(self, path: str | Path, path_in_repo: str, commit_message: str) -> None:
        if not self.enabled:
            return
        if self.api is None:
            self.setup()
        path = Path(path)
        if not path.exists():
            logger.warning("Skip HF upload because file does not exist: %s", path)
            return
        assert self.api is not None
        remote_path = f"{self.path_in_repo}/{path_in_repo}".strip("/")
        logger.info("Uploading to HF: %s -> %s/%s", path, self.repo_id, remote_path)
        try:
            self.api.upload_file(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token,
                path_or_fileobj=str(path),
                path_in_repo=remote_path,
                commit_message=commit_message,
            )
        except Exception:
            logger.exception("HF upload failed for %s", path)
            if self.fail_on_error:
                raise

    def upload_epoch_checkpoint(self, path: str | Path, epoch: int, global_step: int) -> None:
        if not (self.enabled and self.push_each_epoch):
            return
        self.upload_file(
            path,
            f"epochs/epoch_{epoch:03d}_step_{global_step}.pt",
            f"Upload epoch {epoch} checkpoint at step {global_step}",
        )
        self.upload_file(
            self.output_dir / "config.yaml",
            "config.yaml",
            f"Upload config after epoch {epoch}",
        )
        self.upload_file(
            self.output_dir / "train.log",
            "train.log",
            f"Upload train log after epoch {epoch}",
        )
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            self.upload_file(
                best_path,
                "best.pt",
                f"Upload current best checkpoint after epoch {epoch}",
            )

    def upload_final_artifacts(self) -> None:
        if not (self.enabled and self.push_final_best):
            return
        for filename in ("best.pt", "final.pt", "config.yaml", "train.log"):
            self.upload_file(
                self.output_dir / filename,
                filename,
                "Upload final LLM2Seq training artifacts",
            )


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


def apply_trainable_policy(model: LLM2Seq, training_cfg: Dict[str, Any]) -> None:
    """Apply stage-specific parameter freezing before optimizer creation."""
    model.save_frozen_encoder_in_checkpoint = bool(training_cfg.get("save_frozen_encoder", False))

    if not training_cfg.get("freeze_non_mtp", False):
        return

    if model.mtp_module is None:
        raise ValueError("training.freeze_non_mtp=true requires features.use_mtp=true.")

    for param in model.parameters():
        param.requires_grad = False

    trainable_names = []
    for name, param in model.mtp_module.named_parameters():
        if name.startswith("blocks.") or name.startswith("heads."):
            param.requires_grad = True
            trainable_names.append(f"mtp_module.{name}")

    if not trainable_names:
        raise ValueError("training.freeze_non_mtp=true found no trainable MTP block/head parameters.")

    logger.info("Applied freeze_non_mtp policy: trainable parameter roots are MTP blocks/heads only")
    logger.info("Trainable MTP tensors: %s", ", ".join(trainable_names[:12]))
    if len(trainable_names) > 12:
        logger.info("... plus %d more MTP tensors", len(trainable_names) - 12)
    logger.info("Trainable params after policy: %s", f"{model.get_trainable_params():,}")


def get_allowed_missing_prefixes(stage: str, context: str) -> tuple[str, ...]:
    """Return checkpoint keys that may be absent for a specific transition."""
    if context == "resume" and stage == "h200_phase2_full_encoder":
        # Phase 1 checkpoints intentionally omit the frozen encoder.
        return ("encoder.",)
    if context == "resume" and stage == "h200_phase3_mtp_self_distill":
        # Phase 2 checkpoints do not contain the newly added MTP module.
        return ("mtp_module.",)
    return ()


def load_model_state_checked(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    stage: str,
    context: str,
) -> None:
    """Load a checkpoint and fail on unexpected missing weights."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_prefixes = get_allowed_missing_prefixes(stage, context)
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]

    if bad_missing or unexpected:
        preview_missing = ", ".join(bad_missing[:20])
        preview_unexpected = ", ".join(unexpected[:20])
        raise RuntimeError(
            f"Checkpoint load mismatch for stage={stage}, context={context}. "
            f"Bad missing keys ({len(bad_missing)}): {preview_missing}. "
            f"Unexpected keys ({len(unexpected)}): {preview_unexpected}."
        )

    if missing:
        logger.info(
            "Checkpoint load allowed %d missing keys for stage=%s context=%s, prefixes=%s",
            len(missing),
            stage,
            context,
            allowed_prefixes,
        )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors after loading a CPU checkpoint."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def set_training_mode(model: LLM2Seq, training_cfg: Dict[str, Any]) -> None:
    """Set train/eval modes while keeping frozen phase-3 teacher path stable."""
    model.train()
    if training_cfg.get("freeze_non_mtp", False):
        model.encoder.eval()
        model.adaptor.eval()
        model.decoder.eval()
        model.lm_head.eval()
        if model.mtp_module is not None:
            model.mtp_module.train()


def validate_config(raw_cfg: Dict[str, Any]) -> None:
    """Fail fast for incompatible training-stage/model combinations."""
    model_cfg = raw_cfg.get("model", {})
    features_cfg = raw_cfg.get("features", {})
    mtp_cfg = raw_cfg.get("mtp", {})
    training_cfg = raw_cfg.get("training", {})
    stage = str(training_cfg.get("stage", ""))

    if stage.startswith("h200") and model_cfg.get("use_lora_for_encoder", False):
        raise ValueError(
            "H200 configs are designed for frozen/full encoder training, not LoRA. "
            "Remove model.use_lora_for_encoder or set it to false."
        )

    if stage == "h200_phase2_full_encoder" and not model_cfg.get("encoder_trainable", False):
        raise ValueError(
            "h200_phase2_full_encoder must set model.encoder_trainable: true "
            "so the full encoder is fine-tuned."
        )

    if stage == "h200_phase3_mtp_self_distill":
        if not features_cfg.get("use_mtp", False):
            raise ValueError("h200_phase3_mtp_self_distill must set features.use_mtp: true.")
        if not mtp_cfg.get("self_distillation", False):
            raise ValueError("h200_phase3_mtp_self_distill must set mtp.self_distillation: true.")
        if not mtp_cfg.get("train_only", False):
            raise ValueError("h200_phase3_mtp_self_distill must set mtp.train_only: true.")
        if not training_cfg.get("freeze_non_mtp", False):
            raise ValueError("h200_phase3_mtp_self_distill must set training.freeze_non_mtp: true.")
        if not training_cfg.get("save_frozen_encoder", False):
            raise ValueError(
                "h200_phase3_mtp_self_distill must set training.save_frozen_encoder: true "
                "so the phase-2 fine-tuned encoder is preserved in phase-3 checkpoints."
            )


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
    validate_config(raw_cfg)
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
        if training_cfg.get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info("Enabled TF32 matmul/cudnn for CUDA")
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

    apply_trainable_policy(model, training_cfg)

    # Dataset
    train_file = data_cfg.get("train_file", "data/processed/train.jsonl")
    eval_file = data_cfg.get("eval_file", "data/processed/eval.jsonl")

    train_dataset = Seq2SeqDataset(
        data_path=train_file,
        tokenizer=tokenizer,
        max_source_length=data_cfg.get("max_source_length", 512),
        max_target_length=data_cfg.get("max_target_length", 256),
        source_prefix=data_cfg.get("source_prefix", ""),
    )
    eval_dataset = Seq2SeqDataset(
        data_path=eval_file,
        tokenizer=tokenizer,
        max_source_length=data_cfg.get("max_source_length", 512),
        max_target_length=data_cfg.get("max_target_length", 256),
        source_prefix=data_cfg.get("source_prefix", ""),
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

    # Training params
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 8))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    num_train_epochs_cfg = training_cfg.get("num_train_epochs")
    if num_train_epochs_cfg is not None:
        num_train_epochs = int(num_train_epochs_cfg)
        if num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be > 0")
        max_steps = steps_per_epoch * num_train_epochs
    else:
        max_steps = int(training_cfg.get("max_steps", 50000))
        num_train_epochs = max(1, math.ceil(max_steps / steps_per_epoch))

    warmup_steps = training_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.03))
        warmup_steps = int(max_steps * warmup_ratio)
    warmup_steps = int(warmup_steps)

    log_every = int(training_cfg.get("log_every_steps", 50))
    eval_every_epochs = eval_cfg.get("eval_every_epochs")
    if eval_every_epochs is not None:
        eval_every = max(1, steps_per_epoch * int(eval_every_epochs))
    else:
        eval_every = int(eval_cfg.get("eval_every_steps", 1000))
    save_every_epochs = eval_cfg.get("save_every_epochs")
    if save_every_epochs is not None:
        save_every = max(1, steps_per_epoch * int(save_every_epochs))
    else:
        save_every = int(eval_cfg.get("save_every_steps", 2000))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        encoder_lr=float(training_cfg.get("encoder_lr", 1e-5)),
        adaptor_lr=float(training_cfg.get("adaptor_lr", 2e-4)),
        decoder_lr=float(training_cfg.get("decoder_lr", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.1)),
    )

    # Mixed precision
    use_fp16 = training_cfg.get("fp16", True) and device.type == "cuda"
    use_bf16 = training_cfg.get("bf16", False) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_fp16)
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)

    # Resume from checkpoint
    global_step = 0
    best_eval_loss = float("inf")
    checkpoint_path = resume_from or training_cfg.get("resume_from", None)
    stage = str(training_cfg.get("stage", ""))
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        load_model_state_checked(
            model=model,
            state_dict=ckpt["model_state_dict"],
            stage=stage,
            context="resume",
        )
        if "optimizer_state_dict" in ckpt and not training_cfg.get("skip_optimizer_resume", False):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        elif "optimizer_state_dict" in ckpt:
            logger.info("Skipped optimizer state resume because training.skip_optimizer_resume=true")
        if "global_step" in ckpt and not training_cfg.get("reset_global_step_on_resume", False):
            global_step = ckpt["global_step"]
        elif "global_step" in ckpt:
            logger.info("Reset global step because training.reset_global_step_on_resume=true")
        if "best_eval_loss" in ckpt and not training_cfg.get("reset_best_eval_loss_on_resume", False):
            best_eval_loss = ckpt["best_eval_loss"]
        elif "best_eval_loss" in ckpt:
            logger.info("Reset best eval loss because training.reset_best_eval_loss_on_resume=true")
        logger.info(f"Resumed at step {global_step}")

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False)

    hf_pusher = HuggingFaceRunPusher(raw_cfg, output_dir)
    hf_pusher.setup()

    # Training loop
    logger.info(f"Train file: {train_file} ({len(train_dataset):,} examples)")
    logger.info(f"Eval file: {eval_file} ({len(eval_dataset):,} examples)")
    logger.info(f"Starting training for {num_train_epochs} epochs ({max_steps:,} optimizer steps)...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {grad_accum_steps}")
    logger.info(f"  Effective batch size: {batch_size * grad_accum_steps}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Eval every: {eval_every} optimizer steps")
    logger.info(f"  Save every: {save_every} optimizer steps")
    logger.info(f"  Mixed precision: fp16={use_fp16}, bf16={use_bf16}")

    set_training_mode(model, training_cfg)
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    running_mtp = 0.0
    running_mtp_ce = 0.0
    running_mtp_kl = 0.0
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
            running_mtp_ce += outputs.get("loss_mtp_ce", torch.tensor(0.0)).item()
            running_mtp_kl += outputs.get("loss_mtp_kl", torch.tensor(0.0)).item()
            step_count += 1

            # Optimizer step
            if step_count % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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
                    avg_mtp_ce = running_mtp_ce / step_count
                    avg_mtp_kl = running_mtp_kl / step_count
                    lr = optimizer.param_groups[-1]["lr"]

                    logger.info(
                        f"Step {global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                        f"KD: {avg_kd:.4f} | MTP: {avg_mtp:.4f} | "
                        f"MTP_CE: {avg_mtp_ce:.4f} | MTP_KL: {avg_mtp_kl:.4f} | "
                        f"LR: {lr:.2e} | Epoch: {epoch}/{num_train_epochs}"
                    )

                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    running_mtp = 0.0
                    running_mtp_ce = 0.0
                    running_mtp_kl = 0.0
                    step_count = 0

                # Evaluation
                if global_step % eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, device, amp_dtype, use_fp16 or use_bf16)
                    logger.info(f"Step {global_step} | Eval Loss: {eval_loss:.4f}")
                    set_training_mode(model, training_cfg)

                    # Save best
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(
                            model, optimizer, global_step, best_eval_loss,
                            os.path.join(output_dir, "best.pt"),
                            include_optimizer=False,
                        )
                        logger.info(f"New best eval loss: {best_eval_loss:.4f}")

                # Periodic save
                if global_step % save_every == 0:
                    epoch_ckpt_path = os.path.join(output_dir, f"epoch_{epoch:03d}.pt")
                    save_checkpoint(
                        model, optimizer, global_step, best_eval_loss,
                        epoch_ckpt_path,
                        include_optimizer=False,
                    )
                    hf_pusher.upload_epoch_checkpoint(epoch_ckpt_path, epoch, global_step)
                    cleanup_epoch_checkpoints(
                        output_dir,
                        keep_last=hf_pusher.keep_local_epoch_checkpoints,
                    )

                    save_checkpoint(
                        model, optimizer, global_step, best_eval_loss,
                        os.path.join(output_dir, f"checkpoint_{global_step}.pt"),
                        include_optimizer=True,
                    )
                    cleanup_periodic_checkpoints(output_dir, keep_last=1)

    # Final save
    save_checkpoint(
        model, optimizer, global_step, best_eval_loss,
        os.path.join(output_dir, "final.pt"),
        include_optimizer=False,
    )
    hf_pusher.upload_final_artifacts()
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
    include_optimizer: bool = False,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": get_compact_state_dict(model),
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "compact_checkpoint": True,
    }
    if include_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def get_compact_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Save only trainable/generator weights.

    The LLM2Vec encoder is frozen and can be reloaded from Hugging Face, so
    storing it in every checkpoint wastes several GB on Kaggle.
    """
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    save_frozen_encoder = bool(getattr(model, "save_frozen_encoder_in_checkpoint", False))
    compact = {}
    for name, tensor in model.state_dict().items():
        if save_frozen_encoder or not name.startswith("encoder.") or name in trainable_names:
            compact[name] = tensor.detach().cpu()
    return compact


def cleanup_periodic_checkpoints(output_dir: str, keep_last: int = 1) -> None:
    """Keep only the newest periodic checkpoint_N.pt files."""
    paths = []
    for path in Path(output_dir).glob("checkpoint_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        paths.append((step, path))
    paths.sort()
    for _, path in paths[:-keep_last]:
        try:
            path.unlink()
            logger.info(f"Deleted old checkpoint: {path}")
        except OSError as exc:
            logger.warning(f"Could not delete old checkpoint {path}: {exc}")


def cleanup_epoch_checkpoints(output_dir: str, keep_last: int = 1) -> None:
    """Keep only the newest local epoch_N.pt checkpoints after HF upload."""
    if keep_last < 0:
        return
    paths = []
    for path in Path(output_dir).glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        paths.append((epoch, path))
    paths.sort()
    for _, path in paths[:-keep_last]:
        try:
            path.unlink()
            logger.info(f"Deleted old local epoch checkpoint: {path}")
        except OSError as exc:
            logger.warning(f"Could not delete old epoch checkpoint {path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="LLM2Seq Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume)


if __name__ == "__main__":
    main()
