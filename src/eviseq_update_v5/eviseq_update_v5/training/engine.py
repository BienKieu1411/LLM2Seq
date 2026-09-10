from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..data.copy_alignment import COPY_INPUT_KEYS
from ..distributed import run_on_main, world_size
from .checkpoint import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer, set_stage_trainability

LOGGER = logging.getLogger("eviseq_update_v5.train")


def global_cosine_factor(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    """Global-step cosine schedule shared by warm-up and full fine-tuning."""

    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    if warmup_steps and step < warmup_steps:
        return float(step) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


class GlobalCosineScheduler:
    """Small serializable scheduler whose clock never resets at a stage boundary."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0, step: int = 0):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.global_step = int(step)
        self.base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
        self._apply()

    def _apply(self) -> None:
        factor = global_cosine_factor(self.global_step, self.total_steps, self.warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.global_step += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "global_step": self.global_step,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("total_steps", self.total_steps)) != self.total_steps:
            raise ValueError("Scheduler total_steps mismatch on resume")
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.global_step = int(state.get("global_step", 0))
        if len(state.get("base_lrs", self.base_lrs)) != len(self.base_lrs):
            raise ValueError("Scheduler parameter-group count mismatch on resume")
        self._apply()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_bar(fraction: float, width: int = 18) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(width * fraction))
    if filled >= width:
        return "[" + "=" * width + "]"
    if filled == 0:
        return "[" + "." * width + "]"
    return "[" + "=" * (filled - 1) + ">" + "." * (width - filled) + "]"


def _stage_label(stage: str) -> str:
    return {"interface_warmup": "warmup", "full_finetune": "full"}.get(stage, stage)


def _peak_vram_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)


class _LossOnlyModel(torch.nn.Module):
    """Expose only the backward root so DDP can identify unused parameters."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        loss = self.model(*args, **kwargs).loss_ce
        if loss is None:
            raise RuntimeError("AFMR training requires decoder labels")
        return loss


class AFMRTrainer:
    def __init__(self, model: torch.nn.Module, config: dict[str, Any], device: torch.device | str):
        self.model = model.to(device=device, dtype=torch.float32)
        self.config = config
        self.device = torch.device(device)
        self.use_bf16 = self.device.type == "cuda" and config["model"].get("compute_dtype", "bfloat16") == "bfloat16"
        LOGGER.info(
            "precision | parameters=FP32 | optimizer=FP32 | compute=%s", "BF16 autocast" if self.use_bf16 else "FP32"
        )
        self.global_step = 0
        self.best_metric: float | None = None
        self.stage_optimizer_step = 0
        self.epoch = 0
        self.scheduler = None
        self.metrics_path = Path(self.config["experiment"]["output_dir"]) / "training_metrics.jsonl"
        self._fit_started_at: float | None = None
        self._elapsed_before_fit = 0.0
        self._loss_model = _LossOnlyModel(self.model)
        self._ddp_model = None

    def _configure_distributed(self) -> None:
        # Warmup and full finetuning have different trainable parameters.
        # Rebuild the reducer after changing requires_grad at each stage.
        self._ddp_model = None
        if world_size() > 1:
            self._ddp_model = DistributedDataParallel(
                self._loss_model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

    def _write_metric(self, record: dict[str, Any]) -> None:
        def write():
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        run_on_main(write)

    def _elapsed_train_seconds(self) -> float:
        active = 0.0 if self._fit_started_at is None else time.monotonic() - self._fit_started_at
        return self._elapsed_before_fit + max(0.0, active)

    @torch.no_grad()
    def _calibrate_tiny_semantic_reader(self, train_loader: Iterable[dict[str, Any]]) -> None:
        """Calibrate the candidate residual on a deterministic prompt-only batch."""

        reader = getattr(self.model.decoder, "semantic_reader", None)
        if reader is None or reader.output_init != "tiny_rms_1e-3" or reader._tiny_calibrated:
            return
        batch = _move(next(iter(train_loader)), self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            prompt_ids = batch["decoder_prompt_ids"]
            prompt_mask = batch["decoder_prompt_mask"]
            output_budget = torch.full(
                (prompt_ids.shape[0],),
                int(self.config["generation"].get("max_new_tokens", 256)),
                device=self.device,
                dtype=torch.float32,
            )
            bridge = self.model.encode_source(
                batch["input_ids"],
                batch["attention_mask"],
                batch["source_content_mask"],
                prompt_ids,
                prompt_mask,
                output_budget,
                **{key: batch[key] for key in COPY_INPUT_KEYS if key in batch},
            )
            position_ids = prompt_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(~prompt_mask.bool(), 0)
            outputs = self.model.decoder.backbone(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=False,
                position_ids=position_ids,
                return_dict=True,
                encoder_hidden_states=bridge.memory,
                encoder_attention_mask=bridge.memory_mask,
                encoder_attention_bias=bridge.source_bias,
                encoder_value_states=bridge.value_memory,
            )
            hidden = outputs.last_hidden_state.float()
            _, diagnostics = reader.read(hidden, bridge.semantic_state)
            raw = reader.output(diagnostics["u"].float())
            valid = prompt_mask.bool()
            numerator = torch.sqrt(raw.square().mean(-1)).masked_select(valid).sum()
            denominator = torch.sqrt(hidden.square().mean(-1)).masked_select(valid).sum()
            count = valid.sum().to(numerator.dtype)
            totals = torch.stack((numerator, denominator, count))
            if world_size() > 1:
                dist.all_reduce(totals)
            if totals[0] > 0 and totals[1] > 0 and totals[2] > 0:
                scale = 1.0e-3 * totals[1] / totals[0]
                reader.output.weight.mul_(scale.to(reader.output.weight.dtype))
                reader._tiny_calibrated = True
                measured = torch.sqrt(reader.output(diagnostics["u"].float()).square().mean(-1))
                measured = measured.masked_select(valid).sum() / denominator.clamp_min(1.0e-12)
                LOGGER.info("semantic tiny calibration | prompt_rms_ratio=%.6g", float(measured))
        finally:
            self.model.train(was_training)

    def _run_epoch(
        self,
        loader: Iterable[dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        stage: str,
        train: bool,
        global_epoch: int | None = None,
        total_epochs: int | None = None,
        total_training_steps: int | None = None,
    ) -> dict[str, float]:
        self.model.train(train)
        if train and world_size() > 1 and self._ddp_model is None:
            self._configure_distributed()
        forward_model = self._ddp_model if train and self._ddp_model is not None else self._loss_model
        accum = int(self.config["training"]["gradient_accumulation_steps"])
        ce_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        token_total = 0
        iterator = iter(loader)
        epoch_step = 0
        epoch_steps = math.ceil(len(loader) / accum) if train else len(loader)
        global_epoch = int(global_epoch or self.epoch or 1)
        total_epochs = int(total_epochs or global_epoch)
        total_training_steps = int(total_training_steps or epoch_steps * total_epochs)
        if train and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        epoch_started_at = time.monotonic()
        while window := list(islice(iterator, accum if train else 1)):
            started = time.monotonic()
            counts = [int(raw["labels"][:, 1:].ne(-100).sum()) for raw in window]
            window_tokens = sum(counts)
            examples = sum(int(raw.get("example_count", raw["input_ids"].shape[0])) for raw in window)
            if train and world_size() > 1:
                totals = torch.tensor([window_tokens, examples], device=self.device, dtype=torch.long)
                dist.all_reduce(totals)
                window_tokens, examples = totals.tolist()
            if train:
                optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros_like(ce_sum)
            for microstep, (raw_batch, tokens) in enumerate(zip(window, counts)):
                batch = _move(raw_batch, self.device)
                sync = (
                    self._ddp_model.no_sync()
                    if train and self._ddp_model is not None and microstep < len(window) - 1
                    else nullcontext()
                )
                with sync, torch.set_grad_enabled(train):
                    with torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext():
                        loss_ce = forward_model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["source_content_mask"],
                            batch["decoder_prompt_ids"],
                            batch["decoder_prompt_mask"],
                            batch["decoder_input_ids"],
                            batch.get("decoder_attention_mask"),
                            batch.get("labels"),
                            return_logits=False,
                            **{key: batch[key] for key in COPY_INPUT_KEYS if key in batch},
                        )
                        # DDP averages rank gradients. Undo that averaging to
                        # obtain the mean over all real target tokens globally.
                        scale = world_size() if train else 1
                        loss = loss_ce * (scale * tokens / max(1, window_tokens))
                    if train:
                        loss.backward()
                step_loss += loss_ce.detach().double() * tokens
                ce_sum += loss_ce.detach().double() * tokens
                token_total += tokens
                del loss_ce, batch, loss
            if train:
                max_grad_norm = self.config["training"].get("max_grad_norm")
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(max_grad_norm) if max_grad_norm is not None else math.inf,
                    error_if_nonfinite=True,
                )
                post_clip_sq = torch.zeros((), device=self.device, dtype=torch.float32)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        post_clip_sq = post_clip_sq + parameter.grad.detach().float().square().sum()
                post_clip_norm = post_clip_sq.sqrt()
                clip_coefficient = (
                    min(1.0, float(max_grad_norm) / max(float(pre_clip_norm), 1.0e-12))
                    if max_grad_norm is not None
                    else 1.0
                )
                learning_rates = ",".join(
                    dict.fromkeys(
                        f"{group.get('name', i)}:{group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)
                    )
                )
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.stage_optimizer_step += 1
                epoch_step += 1
                if self.global_step % int(self.config["training"]["log_every_steps"]) == 0:
                    if world_size() > 1:
                        dist.all_reduce(step_loss)
                    step_loss /= max(1, window_tokens)
                    window_elapsed = time.monotonic() - started
                    epoch_elapsed = time.monotonic() - epoch_started_at
                    epoch_progress = epoch_step / max(1, epoch_steps)
                    total_progress = (global_epoch - 1 + epoch_progress) / max(1, total_epochs)
                    epoch_eta = epoch_elapsed * (1.0 - epoch_progress) / max(epoch_progress, 1.0e-9)
                    remaining_steps = max(0, total_epochs - global_epoch) * epoch_steps + max(
                        0, epoch_steps - epoch_step
                    )
                    total_eta = epoch_elapsed / max(1, epoch_step) * remaining_steps
                    total_elapsed = self._elapsed_train_seconds()
                    peak_vram_gib = _peak_vram_gib(self.device)
                    if peak_vram_gib is not None and world_size() > 1:
                        peak = torch.tensor(peak_vram_gib, device=self.device)
                        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                        peak_vram_gib = float(peak)
                    record = {
                        "type": "step",
                        "stage": stage,
                        "epoch": global_epoch,
                        "step": self.global_step,
                        "epoch_step": epoch_step,
                        "epoch_steps": epoch_steps,
                        "epoch_progress": round(epoch_progress, 6),
                        "epoch_percent": round(100.0 * epoch_progress, 3),
                        "total_progress": round(total_progress, 6),
                        "total_percent": round(100.0 * total_progress, 3),
                        "ce": round(float(step_loss), 6),
                        "grad_norm": round(float(pre_clip_norm), 6),
                        "pre_clip_grad_norm": round(float(pre_clip_norm), 6),
                        "post_clip_grad_norm": round(float(post_clip_norm), 6),
                        "clip_coefficient": round(float(clip_coefficient), 8),
                        "clipped": bool(clip_coefficient < 1.0),
                        "max_grad_norm": max_grad_norm,
                        "learning_rate": {
                            group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)
                        },
                        "seconds": round(window_elapsed, 4),
                        "window_seconds": round(window_elapsed, 4),
                        "epoch_elapsed_seconds": round(epoch_elapsed, 4),
                        "total_elapsed_seconds": round(total_elapsed, 4),
                        "epoch_eta_seconds": round(epoch_eta, 4),
                        "total_eta_seconds": round(total_eta, 4),
                        "peak_vram_gib": peak_vram_gib,
                        "examples": examples,
                        "tokens": window_tokens,
                        "world_size": world_size(),
                        "examples_per_second": round(examples / max(window_elapsed, 1e-9), 4),
                        "tokens_per_second": round(window_tokens / max(window_elapsed, 1e-9), 4),
                    }
                    self._write_metric(record)
                    LOGGER.info(
                        "[train] stage=%s | epoch=%d/%d | epoch_progress=%s %5.1f%% | step=%d/%d | "
                        "total_step=%d/%d | CE=%.5f | grad=%.4f | lr=%s | elapsed=%s | epoch_eta=%s | "
                        "total_eta=%s | vram=%s | ex/s=%.2f | tok/s=%.0f",
                        _stage_label(stage),
                        global_epoch,
                        total_epochs,
                        _progress_bar(epoch_progress),
                        100.0 * epoch_progress,
                        epoch_step,
                        epoch_steps,
                        self.global_step,
                        total_training_steps,
                        float(step_loss),
                        float(pre_clip_norm),
                        learning_rates,
                        _format_duration(total_elapsed),
                        _format_duration(epoch_eta),
                        _format_duration(total_eta),
                        f"{peak_vram_gib:.2f}GiB" if peak_vram_gib is not None else "NA",
                        examples / max(window_elapsed, 1e-9),
                        window_tokens / max(window_elapsed, 1e-9),
                    )
        totals = torch.stack((ce_sum, ce_sum.new_tensor(token_total)))
        if world_size() > 1:
            dist.all_reduce(totals)
        ce = float(totals[0]) / max(1, float(totals[1]))
        return {"loss": ce, "ce": ce}

    def fit(self, train_loader, validation_loader=None, resume_checkpoint: str | None = None) -> None:
        resume_info = None
        if resume_checkpoint:
            resume_info = load_checkpoint(
                resume_checkpoint,
                self.model,
                config=self.config,
                validate_training_protocol=True,
            )
            self.global_step = int(resume_info.get("step") or 0)
            self.best_metric = resume_info.get("best_metric")
            self._elapsed_before_fit = float(resume_info.get("elapsed_train_seconds") or 0.0)
        else:
            self._elapsed_before_fit = 0.0
        self._fit_started_at = time.monotonic()
        if resume_checkpoint is None:
            self._calibrate_tiny_semantic_reader(train_loader)
        training = self.config["training"]
        stages = (
            ("interface_warmup", int(training["interface_warmup_epochs"])),
            ("full_finetune", int(training["full_finetune_epochs"])),
        )
        steps_per_epoch = math.ceil(len(train_loader) / int(training["gradient_accumulation_steps"]))
        total_epochs = sum(epochs for _, epochs in stages)
        total_training_steps = max(1, steps_per_epoch * total_epochs)
        batch_manifest_hash = None
        if resume_info is not None:
            checkpoint_epoch = int(resume_info.get("epoch") or 0)
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(checkpoint_epoch)
            saved_manifest_hash = (resume_info.get("training_spec") or {}).get("global_batch_manifest_hash")
            active_manifest_hash = getattr(train_loader.batch_sampler, "manifest_hash", None)
            if saved_manifest_hash and active_manifest_hash != saved_manifest_hash:
                raise ValueError("Checkpoint global batch manifest does not match the active dataset/sampler")
        stage_order = {name: index for index, (name, _) in enumerate(stages)}
        if resume_info and resume_info.get("stage") not in stage_order:
            raise ValueError("Checkpoint has no recognized training stage")
        carried_state: dict[torch.Tensor, dict[str, Any]] = {}
        for stage, epochs in stages:
            if epochs <= 0 or (resume_info and stage_order[stage] < stage_order[resume_info["stage"]]):
                continue
            set_stage_trainability(self.model, stage)
            optimizer = build_optimizer(self.model, self.config, stage)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter in carried_state:
                        optimizer.state[parameter] = carried_state[parameter]
            carried_state = {}
            self.scheduler = GlobalCosineScheduler(
                optimizer,
                total_training_steps,
                int(training.get("warmup_steps", 0)),
                step=self.global_step,
            )
            start_epoch = 1
            self.stage_optimizer_step = 0
            if resume_info and resume_info["stage"] == stage:
                load_checkpoint(
                    resume_checkpoint,
                    self.model,
                    optimizer,
                    self.config,
                    scheduler=self.scheduler,
                    restore_rng=False,
                    validate_training_protocol=True,
                )
                start_epoch = int(resume_info.get("stage_epoch") or 0) + 1
                self.stage_optimizer_step = max(0, start_epoch - 1) * steps_per_epoch
            self._configure_distributed()
            for epoch in range(start_epoch, epochs + 1):
                self.epoch = epoch + (stages[0][1] if stage == "full_finetune" else 0)
                global_epoch = self.epoch
                if hasattr(train_loader.batch_sampler, "set_epoch"):
                    train_loader.batch_sampler.set_epoch(global_epoch)
                batch_manifest_hash = getattr(train_loader.batch_sampler, "manifest_hash", None)
                metrics = self._run_epoch(
                    train_loader,
                    optimizer,
                    stage,
                    True,
                    global_epoch,
                    total_epochs,
                    total_training_steps,
                )
                LOGGER.info(
                    "[train] epoch %d/%d complete | stage=%s | CE=%.5f | elapsed=%s",
                    global_epoch,
                    total_epochs,
                    _stage_label(stage),
                    metrics["ce"],
                    _format_duration(self._elapsed_train_seconds()),
                )
                self._write_metric(
                    {
                        "type": "epoch",
                        "split": "train",
                        "stage": stage,
                        "epoch": global_epoch,
                        "total_epochs": total_epochs,
                        "epoch_percent": 100.0,
                        "total_percent": round(100.0 * global_epoch / max(1, total_epochs), 3),
                        "ce": metrics["ce"],
                        "total_elapsed_seconds": round(self._elapsed_train_seconds(), 4),
                    }
                )
                validation = None
                if validation_loader is not None:
                    validation = self._run_epoch(
                        validation_loader,
                        optimizer,
                        stage,
                        False,
                        global_epoch,
                        total_epochs,
                        total_training_steps,
                    )
                    LOGGER.info(
                        "[validation] epoch %d/%d | CE=%.5f | elapsed=%s",
                        global_epoch,
                        total_epochs,
                        validation["ce"],
                        _format_duration(self._elapsed_train_seconds()),
                    )
                    self._write_metric(
                        {
                            "type": "epoch",
                            "split": "validation",
                            "stage": stage,
                            "epoch": global_epoch,
                            "total_epochs": total_epochs,
                            "total_elapsed_seconds": round(self._elapsed_train_seconds(), 4),
                            "ce": validation["ce"],
                        }
                    )
                output_dir = self.config["experiment"]["output_dir"]
                if (
                    validation is not None
                    and bool(training.get("save_best", False))
                    and (self.best_metric is None or validation["loss"] < self.best_metric)
                ):
                    self.best_metric = float(validation["loss"])
                    save_checkpoint(
                        f"{output_dir}/best.pt",
                        self.model,
                        optimizer,
                        self.config,
                        epoch=global_epoch,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        stage=stage,
                        stage_epoch=epoch,
                        elapsed_train_seconds=self._elapsed_train_seconds(),
                        scheduler=self.scheduler,
                        global_batch_manifest_hash=batch_manifest_hash,
                        global_scheduler_step=self.global_step,
                    )
                if bool(training.get("save_each_epoch", True)):
                    save_checkpoint(
                        f"{output_dir}/epoch_{global_epoch:03d}.pt",
                        self.model,
                        optimizer,
                        self.config,
                        epoch=global_epoch,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        stage=stage,
                        stage_epoch=epoch,
                        elapsed_train_seconds=self._elapsed_train_seconds(),
                        scheduler=self.scheduler,
                        global_batch_manifest_hash=batch_manifest_hash,
                        global_scheduler_step=self.global_step,
                    )
                save_checkpoint(
                    f"{output_dir}/last.pt",
                    self.model,
                    optimizer,
                    self.config,
                    epoch=global_epoch,
                    step=self.global_step,
                    best_metric=self.best_metric,
                    stage=stage,
                    stage_epoch=epoch,
                    elapsed_train_seconds=self._elapsed_train_seconds(),
                    scheduler=self.scheduler,
                    global_batch_manifest_hash=batch_manifest_hash,
                    global_scheduler_step=self.global_step,
                )
            carried_state = dict(optimizer.state)
        LOGGER.info(
            "[train] complete | epochs=%d/%d | optimizer_steps=%d/%d | total_elapsed=%s",
            total_epochs,
            total_epochs,
            self.global_step,
            total_training_steps,
            _format_duration(self._elapsed_train_seconds()),
        )
