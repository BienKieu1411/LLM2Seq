"""Two-stage trainer for GenBridge and its controlled baselines."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from .checkpoint import load_checkpoint, save_checkpoint
from .config import MODEL_PROFILES, apply_model_size, dump_config, load_config
from .data import (
    DirectCollator,
    DirectSummarizationDataset,
    EvidenceSeq2SeqCollator,
    EvidenceSeq2SeqDataset,
    decoder_prompt_ids,
    jsonl_fingerprint,
)
from .direct_baseline import DirectCausalBaseline
from .model import GenBridgeSeq2Seq

LOGGER = logging.getLogger("genbridge")


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle examples, then sort only within large random length buckets.

    This retains stochastic batches while avoiding the worst padding waste of
    random batching for long documents.  It is deliberately single-process:
    the current paper setup uses one B200.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int,
        bucket_multiplier: int = 50,
    ):
        if not lengths:
            raise ValueError("LengthBucketBatchSampler requires at least one example")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size = self.batch_size * max(1, int(bucket_multiplier))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(len(self.lengths), generator=generator).tolist()
        batches = []
        for start in range(0, len(shuffled), self.bucket_size):
            bucket = shuffled[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__, reverse=True)
            batches.extend(
                bucket[offset : offset + self.batch_size] for offset in range(0, len(bucket), self.batch_size)
            )
        order = torch.randperm(len(batches), generator=generator).tolist()
        yield from (batches[index] for index in order)


def assert_tokenizers_compatible(source_tokenizer: Any, decoder_tokenizer: Any) -> None:
    """Require one exact token-to-id space before sharing source/target ids."""

    source_vocab = source_tokenizer.get_vocab()
    decoder_vocab = decoder_tokenizer.get_vocab()
    if source_vocab != decoder_vocab:
        raise ValueError(
            "Mixed encoder/decoder checkpoints require identical token-to-id vocabularies. "
            "Use two explicit tokenization streams for genuinely different tokenizers."
        )
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        source_id = getattr(source_tokenizer, field, None)
        decoder_id = getattr(decoder_tokenizer, field, None)
        if source_id != decoder_id:
            raise ValueError(f"Mixed encoder/decoder tokenizers disagree on {field}: {source_id} != {decoder_id}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_experiment(
    config: Dict[str, Any],
    load_datasets: bool = True,
) -> Tuple[nn.Module, Any, Any, Any, Any]:
    from transformers import AutoTokenizer

    model_config = config.get("model", {}) or {}
    decoder_config = config.get("decoder", {}) or {}
    data_config = config.get("data", {}) or {}
    train_limit = int(data_config.get("max_train_samples", 0))
    validation_limit = int(data_config.get("max_validation_samples", 0))
    encoder_name = str(model_config.get("encoder_name", "Qwen/Qwen3-0.6B"))
    kind = str((config.get("experiment", {}) or {}).get("kind", "encoder_decoder"))
    decoder_name = str(decoder_config.get("pretrained_name", encoder_name))
    # Target ids are consumed by the decoder LM head, so its tokenizer is the
    # canonical shared tokenizer. Mixed Qwen3 scales are allowed only after an
    # exact vocabulary and special-token check against the source tokenizer.
    tokenizer_name = decoder_name if kind == "encoder_decoder" else encoder_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if kind == "encoder_decoder" and encoder_name != decoder_name:
        source_tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        assert_tokenizers_compatible(source_tokenizer, tokenizer)
        del source_tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    if kind == "encoder_decoder":
        model = GenBridgeSeq2Seq(config, vocab_size=len(tokenizer))
        if not load_datasets:
            return model, tokenizer, None, None, None
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
            precompute_evidence=bool(data_config.get("precompute_validation_evidence_on_load", True)),
            max_examples=validation_limit,
        )
        collator = EvidenceSeq2SeqCollator(
            tokenizer.pad_token_id,
            int(data_config.get("max_source_length", 3072)),
            int(data_config.get("max_target_length", 384)),
            decoder_prompt_length=len(decoder_prompt_ids(tokenizer, data_config)),
        )
    elif kind == "direct_causal":
        model = DirectCausalBaseline(model_config)
        if not load_datasets:
            return model, tokenizer, None, None, None
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
    if name.startswith("bridge.") or name.endswith("fusion_logits") or name.endswith("summary_tokens"):
        return "bridge"
    if ".cross_attn" in name or ".plan_gate" in name or name.endswith("cross_gate"):
        return "interface"
    if name.startswith("encoder.") or name.startswith("model."):
        return "encoder"
    return "decoder"


def build_optimizer(
    model: nn.Module,
    training: Dict[str, Any],
    total_steps: int,
    stage: str,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    if stage == "interface_warmup":
        learning_rate = float(training.get("adapter_warmup_lr", 1e-4))
    elif stage == "full_finetune":
        learning_rate = float(training.get("full_lr", 1e-5))
    else:
        raise ValueError(f"Unknown optimizer stage: {stage}")
    rates = {name: learning_rate for name in ("encoder", "decoder", "bridge", "interface")}
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
    optimizer_name = str(training.get("optimizer", "adamw_torch"))
    betas = (
        float(training.get("adam_beta1", 0.9)),
        float(training.get("adam_beta2", 0.95)),
    )
    epsilon = float(training.get("adam_epsilon", 1e-8))
    if optimizer_name == "adamw_torch":
        optimizer = AdamW(
            groups,
            betas=betas,
            eps=epsilon,
            fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
        )
    elif optimizer_name == "adamw_8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise ImportError("The 4B full-finetune profile requires bitsandbytes for 8-bit optimizer states") from exc
        optimizer = AdamW8bit(groups, betas=betas, eps=epsilon)
    else:
        raise ValueError("training.optimizer must be adamw_torch or adamw_8bit")
    warmup = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    minimum = float(training.get("min_lr_ratio", 0.0))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("training.min_lr_ratio must be in [0, 1]")

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


@torch.inference_mode()
def evaluate_teacher_forced(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    """Measure held-out teacher-forced loss without autoregressive decoding.

    Checkpoint selection uses token-weighted CE rather than the combined
    auxiliary objective.  CE is comparable across batches with different
    target lengths and directly measures summary-token prediction, while the
    salience/planning losses remain diagnostics rather than deciding which
    checkpoint is called best.
    """

    model.eval()
    ce_sum = 0.0
    supervised_tokens = 0
    example_count = 0
    component_sums: Dict[str, float] = {}
    for batch in loader:
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = _forward(model, batch, kind)
        labels = batch["labels"][:, 1:] if kind == "direct_causal" else batch["labels"]
        token_count = int(labels.ne(-100).sum().item())
        if token_count <= 0:
            raise ValueError("Validation batch contains no supervised summary tokens")
        batch_examples = int(labels.shape[0])
        loss_ce = outputs.get("loss_ce", outputs["loss"])
        ce_sum += float(loss_ce.detach().float().item()) * token_count
        supervised_tokens += token_count
        example_count += batch_examples
        for key in (
            "loss",
            "loss_salience",
            "loss_plan_alignment",
            "loss_plan_diversity",
        ):
            if key in outputs:
                component_sums[key] = component_sums.get(key, 0.0) + (
                    float(outputs[key].detach().float().item()) * batch_examples
                )
    if supervised_tokens == 0 or example_count == 0:
        raise ValueError("Validation loader is empty")
    metrics = {
        "eval_loss_ce": ce_sum / supervised_tokens,
        "eval_supervised_tokens": float(supervised_tokens),
        "eval_examples": float(example_count),
    }
    metrics.update({f"eval_{key}": value / example_count for key, value in component_sums.items()})
    return metrics


def train(
    config_path: str,
    resume: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    model_size: Optional[str] = None,
    overwrite_output_dir: bool = False,
) -> None:
    config = load_config(config_path)
    apply_model_size(config, model_size)
    training = config.get("training", {}) or {}
    experiment = config.get("experiment", {}) or {}
    kind = str(experiment.get("kind", "encoder_decoder"))
    output_dir = Path(str(experiment.get("output_dir", "runs/genbridge/base")))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    legacy_final_path = output_dir / "final.pt"
    running_marker = output_dir / "RUNNING"
    canonical_checkpoints = (best_path, last_path, legacy_final_path)
    if resume and Path(resume).resolve() in {path.resolve() for path in canonical_checkpoints}:
        raise ValueError(
            "Cannot resume from a canonical checkpoint into its own output directory because a new run "
            "must never coexist with a stale final result. Use a different output_dir."
        )
    occupied = [path for path in (*canonical_checkpoints, running_marker) if path.exists()]
    if occupied and not overwrite_output_dir:
        raise FileExistsError(
            "Refusing to mix a new run with existing artifacts: "
            + ", ".join(str(path) for path in occupied)
            + ". Use a fresh experiment.output_dir or pass --overwrite-output-dir."
        )
    if overwrite_output_dir:
        for path in occupied:
            path.unlink()
    running_marker.write_text(
        json.dumps(
            {"config": str(Path(config_path).resolve()), "status": "running"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    data_config = config.get("data", {}) or {}
    data_manifest: Dict[str, Any] = {}
    for split, file_key, limit_key in (
        ("train", "train_file", "max_train_samples"),
        ("validation", "validation_file", "max_validation_samples"),
        ("test", "test_file", "max_test_samples"),
    ):
        dataset_path = data_config.get(file_key)
        if dataset_path and Path(str(dataset_path)).exists():
            data_manifest[split] = jsonl_fingerprint(
                str(dataset_path),
                max_examples=int(data_config.get(limit_key, 0)),
            )
    if data_manifest:
        config["_data_manifest"] = data_manifest
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    set_seed(int(training.get("seed", 42)))
    if bool(training.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, _, train_dataset, validation_dataset, collator = build_experiment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if bool(training.get("require_fp32_master_weights", True)):
        low_precision = [
            f"{name}:{parameter.dtype}"
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if low_precision:
            raise RuntimeError(
                "Full fine-tuning requires FP32 master parameters; use model.dtype=float32. "
                "Low-precision tensors: " + ", ".join(low_precision[:20])
            )
    if resume:
        payload = load_checkpoint(model, resume)
        LOGGER.info("Loaded %s at step %s", resume, payload.get("global_step", "unknown"))

    batch_size = int(training.get("batch_size", 32))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    workers = int(training.get("num_workers", 4))
    length_sampler = None
    loader_kwargs = {
        "num_workers": workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if bool(training.get("group_by_length", True)):
        lengths = [len(str(example.get("source", ""))) for example in train_dataset.examples]
        length_sampler = LengthBucketBatchSampler(
            lengths,
            batch_size=batch_size,
            seed=int(training.get("seed", 42)),
            bucket_multiplier=int(training.get("length_bucket_multiplier", 50)),
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=length_sampler,
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    generation = config.get("generation", {}) or {}
    eval_batch_size = int(training.get("eval_batch_size", generation.get("batch_size", 8)))
    if eval_batch_size <= 0:
        raise ValueError("training.eval_batch_size must be positive")
    eval_workers = int(training.get("eval_num_workers", workers))
    eval_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_workers > 0,
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
        stage,
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
    if data_manifest:
        LOGGER.info("Data manifest: %s", json.dumps(data_manifest, ensure_ascii=False))
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
    LOGGER.info(
        "Checkpoint selection: %d validation examples, batch=%d, metric=eval_loss_ce (minimize)",
        len(validation_dataset),
        eval_batch_size,
    )
    LOGGER.info(
        "Precision: FP32 master parameters, %s autocast compute",
        "BF16" if use_bf16 else "FP16" if use_fp16 else "FP32",
    )

    best_metric = math.inf
    best_epoch = 0
    last_validation_metrics: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        if length_sampler is not None:
            length_sampler.set_epoch(epoch)
        target_stage = "interface_warmup" if epoch <= interface_epochs else "full_finetune"
        if target_stage != stage:
            stage = target_stage
            if hasattr(model, "set_training_stage"):
                model.set_training_stage(stage)
            optimizer, scheduler = build_optimizer(model, training, full_steps, stage)
            optimizer.zero_grad(set_to_none=True)
            LOGGER.info(
                "Switched to full fine-tuning; trainable=%d",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        model.train()
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16 or use_fp16):
                outputs = _forward(model, batch, kind)
                loss = outputs["loss"] / accumulation
            scaler.scale(loss).backward()
            micro_steps += 1
            for key in (
                "loss",
                "loss_ce",
                "loss_salience",
                "loss_plan_alignment",
                "loss_plan_diversity",
                "cross_gate_mean",
                "cross_residual_ratio",
                "plan_gate_mean",
                "token_adapter_gate",
                "plan_adapter_gate",
                "unit_broadcast_gate",
                "salience_probability_mean",
                "salience_predicted_positive_rate",
                "salience_precision",
                "salience_recall",
            ):
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
        last_validation_metrics = evaluate_teacher_forced(
            model,
            eval_loader,
            device,
            kind,
            amp_dtype,
            use_bf16 or use_fp16,
        )
        selection_value = float(last_validation_metrics["eval_loss_ce"])
        if not math.isfinite(selection_value):
            raise RuntimeError(f"Non-finite validation CE at epoch {epoch}: {selection_value}")
        LOGGER.info(
            "validation %s",
            json.dumps(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "stage": stage,
                    **last_validation_metrics,
                },
                ensure_ascii=False,
            ),
        )
        if selection_value < best_metric:
            best_metric = selection_value
            best_epoch = epoch
            save_checkpoint(
                model,
                best_path,
                config,
                epoch,
                global_step,
                validation_metrics=last_validation_metrics,
                checkpoint_role="best",
            )
            LOGGER.info(
                "New best checkpoint: %s (epoch=%d eval_loss_ce=%.8f)",
                best_path,
                epoch,
                best_metric,
            )
        if global_step >= total_steps:
            break

    save_checkpoint(
        model,
        last_path,
        config,
        completed_epoch,
        global_step,
        validation_metrics=last_validation_metrics,
        checkpoint_role="last",
    )
    if best_epoch == 0 or not best_path.exists():
        raise RuntimeError("Training completed without producing best.pt")
    running_marker.unlink(missing_ok=True)
    LOGGER.info(
        "Training complete: best=%s (epoch=%d eval_loss_ce=%.8f), last=%s (epoch=%d)",
        best_path,
        best_epoch,
        best_metric,
        last_path,
        completed_epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model-size", choices=sorted(MODEL_PROFILES), default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(
        args.config,
        args.resume,
        args.max_steps,
        args.model_size,
        args.overwrite_output_dir,
    )


if __name__ == "__main__":
    main()
