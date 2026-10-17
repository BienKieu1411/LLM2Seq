"""Gold-only adaptation to the same per-example word prompts used by GRPO."""

from __future__ import annotations

import gc
import json
import logging
import math
from pathlib import Path

import torch
import torch.distributed as dist
from eviseq_afmr.training.checkpoint import save_checkpoint
from eviseq_afmr.training.optimizer import build_optimizer
from transformers import Trainer, TrainingArguments

from .config import cosine_multiplier
from .data import LengthCollator


def warmup_steps(size, settings, world):
    phase, cfg = settings["ce_warmup"], settings["training"]
    if phase["max_steps"]:
        return phase["max_steps"]
    if not phase["fraction"]:
        return 0
    updates = math.ceil(size / (cfg["prompt_batch_size"] * cfg["gradient_accumulation_steps"] * world))
    return max(1, math.ceil(phase["fraction"] * updates))


class GoldActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, batch):
        count = batch["labels"][:, 1:].ne(-100).sum()
        loss = self.actor.policy(batch, temperature=1.0, ce=True)
        return {"loss_sum": loss * count}


class GoldTrainer(Trainer):
    def __init__(self, *, settings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        # We explicitly consume the globally accumulated gold-token count.
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is None:
            raise RuntimeError("CE warm-up requires the global gold-token denominator")
        outputs = model(batch=inputs["batch"])
        loss = outputs["loss_sum"] * self.accelerator.num_processes / num_items_in_batch.clamp_min(1)
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            cfg = self.settings
            warmup = round(num_training_steps * cfg["warmup_ratio"])
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer,
                lambda step: cosine_multiplier(step, num_training_steps, warmup, cfg["min_lr_ratio"]),
            )
        return self.lr_scheduler


def adapt(actor, dataset, collator, settings, base, *, use_cpu, dtype, processor, world):
    """Update the actor only; discard this stage's optimizer before any rollout."""
    steps = warmup_steps(len(dataset), settings, world)
    if not steps:
        return {"steps": 0, "enabled": False}
    cfg, output = settings["training"], Path(settings["output_dir"])
    logging.getLogger(__name__).info(
        "[ce_warmup] steps=%d | sources=%d | per-sample word prompts | no generation", steps, len(dataset)
    )
    gold_collator = LengthCollator(
        (collator.encoder_tokenizer, collator.decoder_tokenizer),
        collator.data,
        collator.length,
        grounded_copy=collator.grounded_copy,
        include_targets=True,
    )

    def collate(rows):
        batch = gold_collator(rows)
        return {"batch": batch, "labels": batch["labels"]}

    args = TrainingArguments(
        output_dir=str(output / "ce_warmup_logs"),
        max_steps=steps,
        per_device_train_batch_size=cfg["prompt_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["decoder_lr"],
        max_grad_norm=cfg["max_grad_norm"],
        use_cpu=use_cpu,
        bf16=not use_cpu and dtype == "bfloat16",
        fp16=not use_cpu and dtype == "float16",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        ddp_find_unused_parameters=True,
        ddp_timeout=86400,
        gradient_checkpointing=False,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        logging_steps=cfg["log_every_steps"],
        disable_tqdm=True,
        dataloader_num_workers=cfg["num_workers"],
    )
    optimizer = build_optimizer(actor.policy.seam, base, "full_finetune")
    trainer = GoldTrainer(
        model=GoldActor(actor),
        args=args,
        settings=cfg,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=processor,
        optimizers=(optimizer, None),
    )
    try:
        result = trainer.train()
        seam = trainer.accelerator.unwrap_model(trainer.model).actor.policy.seam
        seam.decoder.clear_cross_cache()
        summary = {
            "enabled": True,
            "steps": trainer.state.global_step,
            "training_sources": len(dataset),
            "loss": result.training_loss,
            "history": trainer.state.log_history,
        }
        if trainer.accelerator.is_main_process:
            save_checkpoint(
                output / "length_ce_warmup.pt",
                seam,
                None,
                base,
                epoch=0,
                step=trainer.state.global_step,
                stage="length_ce_warmup",
            )
            (output / "ce_warmup_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if use_cpu and dist.is_initialized():
            dist.barrier()
        else:
            trainer.accelerator.wait_for_everyone()
    finally:
        trainer.accelerator.free_memory()
        del trainer, optimizer
        actor.zero_grad(set_to_none=True)
        gc.collect()
        if not use_cpu:
            torch.cuda.empty_cache()
    return summary
