from __future__ import annotations

import json
import logging
import math
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .optimizer import build_optimizer, set_stage_trainability

LOGGER = logging.getLogger("eviseq_afmr.train")


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


class AFMRTrainer:
    def __init__(self, model: torch.nn.Module, config: dict[str, Any], device: torch.device | str):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.global_step = 0
        self.best_metric: float | None = None
        self.stage_optimizer_step = 0
        self.epoch = 0
        self.scheduler = None
        self.metrics_path = Path(self.config["experiment"]["output_dir"]) / "training_metrics.jsonl"

    def _write_metric(self, record: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_epoch(
        self, loader: Iterable[dict[str, Any]], optimizer: torch.optim.Optimizer, stage: str, train: bool
    ) -> dict[str, float]:
        self.model.train(train)
        accum = int(self.config["training"]["gradient_accumulation_steps"])
        ce_sum = torch.zeros((), device=self.device)
        token_total = 0
        iterator = iter(loader)
        epoch_step = 0
        epoch_steps = math.ceil(len(loader) / accum) if train else len(loader)
        while window := list(islice(iterator, accum if train else 1)):
            started = time.monotonic()
            counts = [int(raw["labels"][:, 1:].ne(-100).sum()) for raw in window]
            window_tokens = sum(counts)
            if train:
                optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros_like(ce_sum)
            for raw_batch, tokens in zip(window, counts):
                batch = _move(raw_batch, self.device)
                with torch.set_grad_enabled(train):
                    output = self.model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["source_content_mask"],
                        batch["decoder_prompt_ids"],
                        batch["decoder_prompt_mask"],
                        batch["decoder_input_ids"],
                        batch.get("decoder_attention_mask"),
                        batch.get("labels"),
                        return_logits=False,
                    )
                    if output.loss_ce is None:
                        raise RuntimeError("AFMR training requires decoder labels")
                    loss = output.loss_ce * tokens / max(1, window_tokens)
                    if train:
                        loss.backward()
                step_loss += loss.detach()
                ce_sum += output.loss_ce.detach() * tokens
                token_total += tokens
                del output, batch, loss
            if train:
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["training"]["max_grad_norm"]), error_if_nonfinite=True
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
                    elapsed = time.monotonic() - started
                    examples = sum(int(raw["input_ids"].shape[0]) for raw in window)
                    record = {
                        "type": "step",
                        "stage": stage,
                        "epoch": self.epoch,
                        "step": self.global_step,
                        "epoch_step": epoch_step,
                        "epoch_steps": epoch_steps,
                        "epoch_progress": round(epoch_step / max(1, epoch_steps), 6),
                        "ce": round(float(step_loss), 6),
                        "grad_norm": round(float(grad), 6),
                        "learning_rate": {
                            group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)
                        },
                        "seconds": round(elapsed, 4),
                        "examples": examples,
                        "tokens": window_tokens,
                        "examples_per_second": round(examples / max(elapsed, 1e-9), 4),
                        "tokens_per_second": round(window_tokens / max(elapsed, 1e-9), 4),
                    }
                    self._write_metric(record)
                    LOGGER.info(
                        "[train] stage=%s | epoch=%d (%d/%d) | step=%d | CE=%.5f | grad=%.4f | lr=%s | %.2fs | %d ex | %.1f tok/s",
                        stage,
                        self.epoch,
                        epoch_step,
                        epoch_steps,
                        self.global_step,
                        float(step_loss),
                        float(grad),
                        learning_rates,
                        elapsed,
                        examples,
                        window_tokens / max(elapsed, 1e-9),
                    )
        ce = float(ce_sum) / max(1, token_total)
        return {"loss": ce, "ce": ce}

    def fit(self, train_loader, validation_loader=None, resume_checkpoint: str | None = None) -> None:
        resume_info = None
        if resume_checkpoint:
            resume_info = load_checkpoint(resume_checkpoint, self.model, config=self.config)
            self.global_step = int(resume_info.get("step") or 0)
            self.best_metric = resume_info.get("best_metric")
        training = self.config["training"]
        stages = (
            ("interface_warmup", int(training["interface_warmup_epochs"])),
            ("full_finetune", int(training["full_finetune_epochs"])),
        )
        steps_per_epoch = math.ceil(len(train_loader) / int(training["gradient_accumulation_steps"]))
        stage_order = {name: index for index, (name, _) in enumerate(stages)}
        if resume_info and resume_info.get("stage") not in stage_order:
            raise ValueError("Checkpoint has no recognized training stage")
        carried_state = {}
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
            total_steps = max(1, steps_per_epoch * epochs)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step, total=total_steps: max(0.0, 1.0 - step / total)
            )
            start_epoch = 1
            self.stage_optimizer_step = 0
            if resume_info and resume_info["stage"] == stage:
                load_checkpoint(resume_checkpoint, self.model, optimizer, self.config, scheduler=self.scheduler)
                start_epoch = int(resume_info.get("stage_epoch") or 0) + 1
                self.stage_optimizer_step = max(0, start_epoch - 1) * steps_per_epoch
                for group, base_lr in zip(optimizer.param_groups, self.scheduler.base_lrs):
                    group["lr"] = base_lr * max(0.0, 1.0 - self.stage_optimizer_step / total_steps)
            for epoch in range(start_epoch, epochs + 1):
                self.epoch = epoch + (stages[0][1] if stage == "full_finetune" else 0)
                metrics = self._run_epoch(train_loader, optimizer, stage, True)
                LOGGER.info(
                    "[train] epoch=%d stage=%s | CE=%.5f",
                    self.epoch,
                    stage,
                    metrics["ce"],
                )
                self._write_metric(
                    {"type": "epoch", "split": "train", "stage": stage, "epoch": self.epoch, "ce": metrics["ce"]}
                )
                validation = None
                if validation_loader is not None:
                    validation = self._run_epoch(validation_loader, optimizer, stage, False)
                    LOGGER.info(
                        "[validation] epoch=%d stage=%s | CE=%.5f",
                        self.epoch,
                        stage,
                        validation["ce"],
                    )
                    self._write_metric(
                        {
                            "type": "epoch",
                            "split": "validation",
                            "stage": stage,
                            "epoch": self.epoch,
                            "ce": validation["ce"],
                        }
                    )
                output_dir = self.config["experiment"]["output_dir"]
                global_epoch = epoch + (stages[0][1] if stage == "full_finetune" else 0)
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
                        scheduler=self.scheduler,
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
                        scheduler=self.scheduler,
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
                    scheduler=self.scheduler,
                )
            carried_state = dict(optimizer.state)
