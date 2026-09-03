"""Fourteen-epoch full fine-tuning for the direct Qwen3-0.6B control."""

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
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from llm2seq_v2.checkpoint import save_last_checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import load_config
from .data import CausalCollator, DirectCausalDataset
from .provenance import (
    audit_splits,
    data_manifest,
    parameter_manifest,
    resolve_from_src,
    tokenizer_manifest,
)

LOGGER = logging.getLogger("direct_qwen")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype(value: str) -> torch.dtype:
    lowered = str(value).lower()
    if lowered in {"float32", "fp32"}:
        return torch.float32
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def load_tokenizer_and_model(
    config: Dict[str, Any],
    *,
    evaluation: bool = False,
    tokenizer_factory: Any = None,
    model_factory: Any = None,
) -> Tuple[Any, torch.nn.Module]:
    """Local-only loader; factories make its no-network contract unit-testable."""

    if tokenizer_factory is None or model_factory is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_factory = tokenizer_factory or AutoTokenizer
        model_factory = model_factory or AutoModelForCausalLM
    model_config = config["model"]
    if model_config.get("local_files_only") is not True:
        raise RuntimeError("Refusing a model load without local_files_only=true")
    name = str(model_config["name_or_path"])
    common = {
        "local_files_only": True,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    tokenizer = tokenizer_factory.from_pretrained(name, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer has neither PAD nor EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if evaluation else "right"

    dtype_name = model_config.get("eval_torch_dtype" if evaluation else "torch_dtype", "bfloat16")
    model_kwargs = {
        **common,
        "dtype": _dtype(str(dtype_name)),
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
    }
    model = model_factory.from_pretrained(name, **model_kwargs)
    if not evaluation and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        model.to(dtype=torch.float32)
    if hasattr(model, "config"):
        model.config.use_cache = bool(evaluation)
    return tokenizer, model


def model_context_length(model: torch.nn.Module) -> int:
    model_config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "max_seq_len", "n_positions"):
        value = getattr(model_config, name, None)
        if value is not None and int(value) > 0:
            return int(value)
    raise RuntimeError("Cannot verify the direct model context length")


def prepare_full_finetune(model: torch.nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if bool(config["model"].get("gradient_checkpointing", True)):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if enable is None:
            raise RuntimeError("Configured gradient checkpointing is unsupported by this model")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
    manifest = parameter_manifest(model, config)
    if not manifest["full_finetune"]:
        raise RuntimeError("Direct control must have 100% trainable parameters")
    low_precision = [
        f"{name}:{parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    ]
    if low_precision:
        raise RuntimeError("Full fine-tuning requires FP32 master parameters: " + ", ".join(low_precision[:20]))
    return manifest


def build_optimizer(
    model: torch.nn.Module,
    training: Dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, LambdaLR]:
    decay = []
    no_decay = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        target = no_decay if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() else decay
        target.append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("No trainable direct-Qwen parameters")
    learning_rate = float(training["learning_rate"])
    groups = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": float(training.get("weight_decay", 0.01)),
            }
        )
    if no_decay:
        groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0})
    optimizer = AdamW(
        groups,
        lr=learning_rate,
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.95))),
        eps=float(training.get("adam_epsilon", 1e-8)),
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    minimum = float(training.get("min_lr_ratio", 0.0))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return minimum + (1.0 - minimum) * cosine

    return optimizer, LambdaLR(optimizer, schedule)


def _autocast(target: torch.device, training: Dict[str, Any]):
    enabled = target.type == "cuda" and (bool(training.get("bf16", True)) or bool(training.get("fp16", False)))
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if bool(training.get("bf16", True)) else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _prepare_output(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to mix a direct-Qwen run with existing artifacts: {path}. "
                "Pass --overwrite-output-dir for an intentional rerun."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "RUNNING"
    marker.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    return marker


def train(config_path: str, overwrite_output_dir: bool = False) -> Path:
    config = load_config(config_path)
    training = config["training"]
    output_dir = resolve_from_src(config["experiment"]["output_dir"])
    split_audit = audit_splits(config)
    marker = _prepare_output(output_dir, overwrite_output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
        force=True,
    )
    try:
        seed = int(training.get("seed", 42))
        set_seed(seed)
        target = device()
        if target.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(training.get("tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(training.get("tf32", True))
            torch.set_float32_matmul_precision("high")

        tokenizer, model = load_tokenizer_and_model(config, evaluation=False)
        context_length = model_context_length(model)
        required_context = int(config["model"]["minimum_context_length"])
        if context_length < required_context:
            raise RuntimeError(f"Model context {context_length} is below required {required_context}")
        parameter_report = prepare_full_finetune(model, config)
        tokenizer_report = tokenizer_manifest(tokenizer, config)
        model.to(target)

        data = config["data"]
        limits = config.get("limits", {})
        train_dataset = DirectCausalDataset(
            resolve_from_src(data["train_file"]),
            tokenizer,
            data,
            max_sequence_length=context_length,
            max_examples=int(limits.get("max_train_examples", 0)),
        )
        collator = CausalCollator(tokenizer.pad_token_id)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=int(training.get("num_workers", 4)),
            pin_memory=target.type == "cuda",
            persistent_workers=int(training.get("num_workers", 4)) > 0,
            drop_last=False,
        )
        epochs = int(training["num_train_epochs"])
        accumulation = int(training["gradient_accumulation_steps"])
        optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
        total_steps = max(1, optimizer_steps_per_epoch * epochs)
        optimizer, scheduler = build_optimizer(model, training, total_steps)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=target.type == "cuda" and bool(training.get("fp16", False)),
        )
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        log_every = int(training.get("log_every_steps", 10))
        max_grad_norm = float(training.get("max_grad_norm", 1.0))

        manifests = data_manifest(config)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (output_dir / "data_manifest.json").write_text(
            json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "data_audit.json").write_text(
            json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "parameter_manifest.json").write_text(
            json.dumps(parameter_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "tokenizer_manifest.json").write_text(
            json.dumps(tokenizer_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "device=%s examples=%d epochs=%d batch=%d accumulation=%d optimizer_steps=%d parameters=%s",
            target,
            len(train_dataset),
            epochs,
            int(training["batch_size"]),
            accumulation,
            total_steps,
            json.dumps(parameter_report),
        )

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_microbatches = 0
            accumulation_count = 0
            window_size = accumulation
            for batch_index, batch in enumerate(loader, start=1):
                if accumulation_count == 0:
                    window_size = min(accumulation, len(loader) - batch_index + 1)
                batch = {name: value.to(target, non_blocking=True) for name, value in batch.items()}
                with _autocast(target, training):
                    outputs = model(**batch, use_cache=False)
                    if outputs.loss is None or outputs.loss.ndim != 0:
                        raise RuntimeError("Causal LM did not return a scalar supervised loss")
                    loss = outputs.loss / window_size
                scaler.scale(loss).backward()
                running_loss += float(outputs.loss.detach().float())
                running_microbatches += 1
                accumulation_count += 1
                if accumulation_count != window_size:
                    continue
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulation_count = 0
                global_step += 1
                if global_step % log_every == 0 or global_step == total_steps:
                    LOGGER.info(
                        "train %s",
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "loss_ce": running_loss / max(1, running_microbatches),
                                "grad_norm": float(grad_norm),
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            }
                        ),
                    )
                    running_loss = 0.0
                    running_microbatches = 0
            LOGGER.info("completed epoch=%d", epoch)

        last_path = output_dir / "last.pt"
        save_last_checkpoint(model, last_path, config, epochs, global_step, manifests)
        marker.unlink(missing_ok=True)
        (output_dir / "COMPLETE").write_text(
            json.dumps(
                {
                    "checkpoint": str(last_path),
                    "epoch": epochs,
                    "global_step": global_step,
                    "checkpoint_role": "last",
                }
            ),
            encoding="utf-8",
        )
        LOGGER.info("Training complete: last=%s epoch=%d step=%d", last_path, epochs, global_step)
        return last_path
    except Exception:
        LOGGER.exception("Direct-Qwen training failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args.config, args.overwrite_output_dir)


if __name__ == "__main__":
    main()
